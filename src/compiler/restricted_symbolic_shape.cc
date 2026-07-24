#include "kxc/compiler/restricted_symbolic_shape.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "kxc/relay/op.h"

#ifndef KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
#define KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE 0
#endif

namespace kxc::api::experimental::restricted_symbolic_shape::v1 {
namespace shape =
    kxc::api::experimental::shape_specialization::v1;

struct RestrictedDispatchDecision::Impl final {
    DispatchKind kind;
    shape::GraphTemplate graph;
    shape::ExactOracle exact_oracle;
    std::optional<shape::GuardedShapeProfile> guarded_profile;
};

namespace {

[[noreturn]] void Reject(const std::string& message) {
    throw std::invalid_argument("RestrictedSymbolicShapeAdapter: " + message);
}

void RequireEnabled() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
    throw std::runtime_error("RestrictedSymbolicShapeAdapter is disabled; configure with "
                             "-DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON");
#endif
}

bool ValidContract(const shape::ConcreteTensorShapeContract& value) {
    if (value.logical.size() != value.physical.size() ||
        value.logical.size() != value.valid.size() ||
        (!value.axis_names.empty() &&
         value.axis_names.size() != value.logical.size())) {
        return false;
    }
    for (size_t i = 0; i < value.logical.size(); ++i) {
        if (value.logical[i] < 0 || value.physical[i] < value.logical[i] ||
            value.valid[i] < 0 || value.valid[i] > value.logical[i]) return false;
    }
    return true;
}

void VerifyContracts(const std::vector<shape::UnitSpecializationRequest>& requests) {
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        if (request.ordered_call_index != i || request.call_locator.value().empty() ||
            !shape::MatchesExactSignatureDigest(request.signature_digest,
                                                request.ordered_inputs,
                                                request.ordered_outputs)) {
            Reject("invalid exact request routing or signature");
        }
        for (const auto& value : request.ordered_inputs) if (!ValidContract(value)) Reject("invalid exact input extent contract");
        for (const auto& value : request.ordered_outputs) if (!ValidContract(value)) Reject("invalid exact output extent contract");
    }
}

void VerifyContracts(const std::vector<shape::GuardedUnitSpecializationRequest>& requests) {
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        if (request.ordered_call_index != i || request.call_locator.value().empty() ||
            request.guard_canonical.empty()) Reject("invalid guarded request routing or authority");
        for (const auto& value : request.ordered_inputs) if (!ValidContract(value)) Reject("invalid guarded input logical/physical/valid extent contract");
        for (const auto& value : request.ordered_outputs) if (!ValidContract(value)) Reject("invalid guarded output logical/physical/valid extent contract");
    }
}

const shape::NamedTensorContract& Named(const shape::GraphTemplate& graph,
                                        const std::string& name) {
    const auto find = [&name](const std::vector<shape::NamedTensorContract>& values)
        -> const shape::NamedTensorContract* {
        const auto it = std::find_if(values.begin(), values.end(), [&name](const auto& value) {
            return value.name == name;
        });
        return it == values.end() ? nullptr : &*it;
    };
    if (const auto* result = find(graph.shape_program().inputs())) return *result;
    if (const auto* result = find(graph.shape_program().outputs())) return *result;
    Reject("frozen graph has an unknown unit value");
}

shape::ConcreteTensorShapeContract StaticValue(const shape::GraphTemplate& graph,
                                                const std::string& name) {
    return Named(graph, name).contract.Evaluate(shape::BindingSet());
}

void RequireSameShape(const shape::ConcreteTensorShapeContract& left,
                      const shape::ConcreteTensorShapeContract& right) {
    if (!(left == right) || !ValidContract(left)) Reject("broadcast or non-exact shape contract is unsupported");
}

void ValidateFrozenUnits(const shape::GraphTemplate& graph,
                         const std::vector<std::string>& operations) {
    if (graph.ordered_units().size() != operations.size() || operations.empty()) {
        Reject("frozen graph does not preserve the restricted call sequence");
    }
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& unit = graph.ordered_units()[i];
        if (unit.output_value_names.size() != 1) Reject("multiple outputs are unsupported");
        const auto output = StaticValue(graph, unit.output_value_names[0]);
        if (operations[i] == "relu" || operations[i] == "sqrt") {
            if (unit.input_value_names.size() != 1) Reject("shape-transparent unary arity mismatch");
            RequireSameShape(StaticValue(graph, unit.input_value_names[0]), output);
        } else if (operations[i] == "add" || operations[i] == "mul") {
            if (unit.input_value_names.size() != 2) Reject("binary arity mismatch");
            RequireSameShape(StaticValue(graph, unit.input_value_names[0]), output);
            RequireSameShape(StaticValue(graph, unit.input_value_names[1]), output);
        } else {
            Reject("unsupported frozen operation");
        }
    }
}

void CollectOperations(const Expr& expression, const std::set<const Object*>& parameters,
                       std::set<const Object*>* seen, std::vector<std::string>* operations) {
    if (expression.As<VarNode>()) {
        if (!parameters.count(expression.get())) Reject("graph references a non-parameter variable");
        return;
    }
    const auto* call = expression.As<CallNode>();
    if (!call || !seen->insert(expression.get()).second) Reject("only a tree of restricted calls is supported");
    const auto* op = call->op.As<relay::OpNode>();
    if (!op || (op->name != "relu" && op->name != "nn_relu" && op->name != "sqrt" &&
                op->name != "add" && op->name != "mul")) {
        Reject("unsupported Relay operation");
    }
    const size_t arity = (op->name == "add" || op->name == "mul") ? 2 : 1;
    if (call->args.size() != arity) Reject("unsupported Relay operation arity");
    for (const Expr& argument : call->args) CollectOperations(argument, parameters, seen, operations);
    operations->push_back(op->name == "nn_relu" ? "relu" : op->name);
}

std::vector<std::string> ValidateSyntax(const Function& function) {
    if (!function.defined() || function->params.empty()) Reject("a nonempty fixed-rank parameter list is required");
    std::set<const Object*> parameters;
    for (const Var& parameter : function->params) {
        const auto* type = parameter->type_annotation.As<TensorTypeNode>();
        if (!type || type->shape.empty()) Reject("every input must have a fixed non-scalar tensor rank");
        for (int64_t extent : type->shape) if (extent < 0) Reject("legacy -1 or negative input extent is unsupported");
        if (!parameters.insert(parameter.get()).second) Reject("duplicate input parameter");
    }
    std::set<const Object*> seen;
    std::vector<std::string> operations;
    CollectOperations(function->body, parameters, &seen, &operations);
    return operations;
}

std::vector<int64_t> Dimensions(const shape::NamedTensorContract& value) {
    return value.contract.Evaluate(shape::BindingSet()).logical;
}

shape::NamedTensorContract Overlay(const shape::NamedTensorContract& source,
                                   const std::map<std::vector<int64_t>, std::vector<shape::DimExpr>>& shapes) {
    const auto found = shapes.find(Dimensions(source));
    const std::vector<shape::DimExpr>& dimensions = found == shapes.end()
        ? source.contract.logical().dimensions() : found->second;
    return {source.name, shape::TensorShapeContract(
        shape::LogicalShape(dimensions, source.contract.logical().axis_names()),
        shape::PhysicalCapacity(dimensions),
        shape::ValidExtent(dimensions))};
}

void VerifyBindings(const shape::GraphTemplate& graph, const shape::BindingSet& bindings) {
    const auto& symbols = graph.shape_program().declared_symbols();
    if (bindings.bindings().size() != symbols.size()) Reject("bindings must explicitly bind every overlay symbol exactly once");
    for (const std::string& symbol : symbols) if (!bindings.Find(symbol).has_value()) Reject("unbound overlay symbol");
}

std::vector<shape::UnitSpecializationRequest> RebuildExact(
    const RestrictedDispatchDecision::Impl& decision) {
    const auto requests = shape::MakeExactSpecializationRequests(decision.graph, decision.exact_oracle);
    if (requests.empty()) Reject("decision has no exact requests");
    VerifyContracts(requests);
    const auto& exact_key = decision.exact_oracle.profile().key();
    for (const auto& request : requests) {
        if (!(request.shape_profile_key == exact_key)) Reject("exact request is detached from decision oracle");
    }
    return requests;
}

std::vector<shape::GuardedUnitSpecializationRequest> RebuildGuarded(
    const RestrictedDispatchDecision::Impl& decision) {
    if (!decision.guarded_profile) return {};
    const auto requests = shape::MakeGuardedSpecializationRequests(decision.graph, *decision.guarded_profile);
    const auto exact = RebuildExact(decision);
    if (requests.size() != exact.size()) Reject("guarded decision cardinality mismatch");
    VerifyContracts(requests);
    const auto expected = decision.kind == DispatchKind::kBucket
        ? shape::GuardedProfileKind::kBucket : shape::GuardedProfileKind::kPolymorphic;
    for (const auto& request : requests) {
        if (request.kind != expected ||
            !(request.exact_oracle_key == decision.exact_oracle.profile().key())) {
            Reject("guarded request is detached from decision authority");
        }
    }
    return requests;
}

void VerifyDecision(const RestrictedDispatchDecision::Impl& decision) {
    (void)RebuildExact(decision);
    const auto guarded = RebuildGuarded(decision);
    if ((decision.kind == DispatchKind::kExact) != guarded.empty()) {
        Reject("decision kind and guarded profile disagree");
    }
}

std::string ExactUnitBoundaryIdentity(const shape::UnitSpecializationRequest& request) {
    return "restricted.exact-unit-boundary.v2|" +
           request.unit_semantic_key.canonical_bytes() + "|" +
           request.signature_digest.value();
}

std::string GuardedUnitBoundaryIdentity(const shape::GuardedUnitSpecializationRequest& request) {
    return "restricted.guarded-unit-boundary.v2|" +
           request.unit_semantic_key.canonical_bytes() + "|" +
           request.specialization_canonical;
}

std::string ProofToken(const std::string& operation) {
    if (operation == "relu" || operation == "sqrt" || operation == "add" || operation == "mul") {
        return "restricted." + operation + ".equal-shape.v1";
    }
    Reject("unsupported polymorphic unit operation");
}

void VerifyPolymorphicProofs(const std::vector<std::string>& operations,
                             const shape::GraphTemplate& graph,
                             const shape::PolymorphicPolicy& policy) {
    const auto& proofs = policy.allowlist_proofs();
    if (proofs.size() != operations.size() || proofs.size() != graph.ordered_units().size()) {
        Reject("polymorphic proof cardinality does not match frozen operations");
    }
    for (size_t index = 0; index < proofs.size(); ++index) {
        if (proofs[index].ordered_unit_index != index ||
            !(proofs[index].unit_semantic_key == graph.ordered_units()[index].semantic_key) ||
            proofs[index].proof != ProofToken(operations[index])) {
            Reject("polymorphic proof is not an allowlisted restricted operation token");
        }
    }
}

void VerifyCounters(const shape_exact::v1::PreparedGraphTemplate& representative,
                    const shape_exact::v1::ShapeExactPreparationCounters& frozen) {
    const auto& now = representative.counters();
    if (now.execution_contract_resolutions != frozen.execution_contract_resolutions ||
        now.relay_graph_pipelines != frozen.relay_graph_pipelines ||
        now.capability_boundary_checks != frozen.capability_boundary_checks ||
        now.value_graph_builds != frozen.value_graph_builds || now.partitions != frozen.partitions) {
        Reject("profile minting mutated prepared pipeline or partition counters");
    }
}

}  // namespace

struct PreparedRestrictedSymbolicTemplate::Impl final {
    shape_exact::v1::PreparedGraphTemplate representative;
    shape::GraphTemplate graph;
    std::vector<InputAxisSymbol> symbols;
    std::vector<std::string> operations;
    shape_exact::v1::ShapeExactPreparationCounters counters;
};

PreparedRestrictedSymbolicTemplate::PreparedRestrictedSymbolicTemplate() = default;
PreparedRestrictedSymbolicTemplate::PreparedRestrictedSymbolicTemplate(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
PreparedRestrictedSymbolicTemplate::~PreparedRestrictedSymbolicTemplate() = default;
PreparedRestrictedSymbolicTemplate::PreparedRestrictedSymbolicTemplate(const PreparedRestrictedSymbolicTemplate&) = default;
PreparedRestrictedSymbolicTemplate& PreparedRestrictedSymbolicTemplate::operator=(const PreparedRestrictedSymbolicTemplate&) = default;
PreparedRestrictedSymbolicTemplate::PreparedRestrictedSymbolicTemplate(PreparedRestrictedSymbolicTemplate&&) noexcept = default;
PreparedRestrictedSymbolicTemplate& PreparedRestrictedSymbolicTemplate::operator=(PreparedRestrictedSymbolicTemplate&&) noexcept = default;
const shape::GraphTemplate& PreparedRestrictedSymbolicTemplate::graph_template() const { if (!impl_) Reject("prepared template is undefined"); return impl_->graph; }
const std::vector<InputAxisSymbol>& PreparedRestrictedSymbolicTemplate::input_axis_symbols() const { if (!impl_) Reject("prepared template is undefined"); return impl_->symbols; }
const shape_exact::v1::PreparedGraphTemplate& PreparedRestrictedSymbolicTemplate::representative() const { if (!impl_) Reject("prepared template is undefined"); return impl_->representative; }

bool RestrictedSymbolicShapeAdapter::IsEnabled() noexcept {
#if KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
    return true;
#else
    return false;
#endif
}

PreparedRestrictedSymbolicTemplate RestrictedSymbolicShapeAdapter::Prepare(
    Function representative, CompileConfig config, std::vector<InputAxisSymbol> input_axis_symbols) {
    RequireEnabled();
    const std::vector<std::string> operations = ValidateSyntax(representative);
    if (input_axis_symbols.empty()) Reject("at least one explicit input-axis symbol is required");
    std::sort(input_axis_symbols.begin(), input_axis_symbols.end(), [](const auto& a, const auto& b) {
        return std::tie(a.parameter_index, a.axis, a.symbol) < std::tie(b.parameter_index, b.axis, b.symbol);
    });
    std::set<std::pair<size_t, size_t>> axes;
    std::set<std::string> names;
    std::map<std::string, InputAxisSymbol> definitions;
    for (const auto& symbol : input_axis_symbols) {
        if (symbol.symbol.empty() || symbol.lower < 0 || symbol.upper < symbol.lower || symbol.divisible_by <= 0 ||
            !axes.insert({symbol.parameter_index, symbol.axis}).second ||
            symbol.parameter_index >= representative->params.size()) Reject("duplicate, out-of-range, or invalid input-axis symbol");
        const auto existing_definition = definitions.find(symbol.symbol);
        if (existing_definition != definitions.end() &&
            (existing_definition->second.lower != symbol.lower ||
             existing_definition->second.upper != symbol.upper ||
             existing_definition->second.divisible_by != symbol.divisible_by)) {
            Reject("repeated overlay symbol has inconsistent bounds");
        }
        definitions.emplace(symbol.symbol, symbol);
        names.insert(symbol.symbol);
        const auto* type = representative->params[symbol.parameter_index]->type_annotation.As<TensorTypeNode>();
        if (!type || symbol.axis >= type->shape.size() || type->shape[symbol.axis] < symbol.lower ||
            type->shape[symbol.axis] > symbol.upper || type->shape[symbol.axis] % symbol.divisible_by != 0) {
            Reject("input-axis symbol does not cover its frozen representative extent");
        }
    }
    // Freeze through the production exact adapter before exposing any symbolic overlay.
    shape_exact::v1::PreparedGraphTemplate frozen =
        shape_exact::v1::ProductionExactShapeAdapter::PrepareGraphTemplate(representative, config);
    const shape::GraphTemplate& static_graph = frozen.graph_template();
    ValidateFrozenUnits(static_graph, operations);
    if (static_graph.shape_program().inputs().size() != representative->params.size()) {
        Reject("restricted graph cannot contain constants or hidden inputs");
    }
    std::map<std::vector<int64_t>, std::vector<shape::DimExpr>> shape_overlays;
    for (size_t parameter = 0; parameter < representative->params.size(); ++parameter) {
        const auto static_shape = Dimensions(static_graph.shape_program().inputs()[parameter]);
        std::vector<shape::DimExpr> dimensions;
        for (int64_t extent : static_shape) dimensions.push_back(shape::DimExpr::Const(extent));
        for (const auto& symbol : input_axis_symbols) if (symbol.parameter_index == parameter) dimensions[symbol.axis] = shape::DimExpr::Symbol(symbol.symbol);
        const auto existing = shape_overlays.find(static_shape);
        if (existing != shape_overlays.end() && existing->second != dimensions) {
            Reject("equal-shape inputs must share an identical explicit overlay");
        }
        shape_overlays.emplace(static_shape, std::move(dimensions));
    }
    std::vector<shape::Constraint> constraints;
    for (const auto& [name, symbol] : definitions) {
        const shape::DimExpr expression = shape::DimExpr::Symbol(name);
        constraints.push_back(shape::Constraint::Range(expression, symbol.lower, symbol.upper));
        constraints.push_back(shape::Constraint::DivisibleBy(expression, symbol.divisible_by));
    }
    std::vector<shape::NamedTensorContract> inputs, outputs;
    for (const auto& value : static_graph.shape_program().inputs()) inputs.push_back(Overlay(value, shape_overlays));
    for (const auto& value : static_graph.shape_program().outputs()) outputs.push_back(Overlay(value, shape_overlays));
    shape::GraphTemplate graph(static_graph.key(), shape::ShapeProgram(
        std::vector<std::string>(names.begin(), names.end()), std::move(inputs), std::move(outputs),
        std::move(constraints)), static_graph.ordered_units());
    const auto counters = frozen.counters();
    return PreparedRestrictedSymbolicTemplate(std::make_shared<PreparedRestrictedSymbolicTemplate::Impl>(
        PreparedRestrictedSymbolicTemplate::Impl{std::move(frozen), std::move(graph),
            std::move(input_axis_symbols), operations, counters}));
}

RestrictedDispatchDecision::RestrictedDispatchDecision(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}

DispatchKind RestrictedDispatchDecision::kind() const noexcept {
    return impl_ ? impl_->kind : DispatchKind::kExact;
}

const shape::ExactOracle& RestrictedDispatchDecision::exact_oracle() const {
    if (!impl_) Reject("decision is undefined");
    VerifyDecision(*impl_);
    return impl_->exact_oracle;
}

std::vector<shape::UnitSpecializationRequest> RestrictedDispatchDecision::exact_requests() const {
    if (!impl_) Reject("decision is undefined");
    VerifyDecision(*impl_);
    return RebuildExact(*impl_);
}

std::vector<shape::GuardedUnitSpecializationRequest> RestrictedDispatchDecision::guarded_requests() const {
    if (!impl_) Reject("decision is undefined");
    VerifyDecision(*impl_);
    return RebuildGuarded(*impl_);
}

const shape::GraphTemplate& RestrictedDispatchDecision::graph_template() const {
    if (!impl_) Reject("decision is undefined");
    VerifyDecision(*impl_);
    return impl_->graph;
}

const shape::GuardedShapeProfile* RestrictedDispatchDecision::guarded_profile() const {
    if (!impl_) Reject("decision is undefined");
    VerifyDecision(*impl_);
    return impl_->guarded_profile ? &*impl_->guarded_profile : nullptr;
}

const shape::BucketPolicy* RestrictedDispatchDecision::bucket_policy() const {
    const auto* profile = guarded_profile();
    return profile ? profile->bucket_policy() : nullptr;
}

const shape::PolymorphicPolicy* RestrictedDispatchDecision::polymorphic_policy() const {
    const auto* profile = guarded_profile();
    return profile ? profile->polymorphic_policy() : nullptr;
}

RestrictedDispatchDecision RestrictedSymbolicShapeAdapter::MintExact(
    const PreparedRestrictedSymbolicTemplate& prepared, const shape::BindingSet& bindings) {
    RequireEnabled();
    if (!prepared.impl_) Reject("prepared template is undefined");
    VerifyBindings(prepared.impl_->graph, bindings);
    const shape::ExactOracle oracle = shape::InstantiateExactProfile(prepared.impl_->graph, bindings);
    RestrictedDispatchDecision result(std::make_shared<RestrictedDispatchDecision::Impl>(
        RestrictedDispatchDecision::Impl{DispatchKind::kExact, prepared.impl_->graph, oracle, std::nullopt}));
    VerifyDecision(*result.impl_);
    VerifyCounters(prepared.impl_->representative, prepared.impl_->counters);
    return result;
}

RestrictedDispatchDecision RestrictedSymbolicShapeAdapter::MintBucket(
    const PreparedRestrictedSymbolicTemplate& prepared, const shape::BindingSet& bindings,
    const shape::BucketPolicy& policy) {
    RestrictedDispatchDecision exact = MintExact(prepared, bindings);
    const shape::GuardedShapeProfile profile = shape::BuildBucketProfile(
        prepared.impl_->graph, exact.impl_->exact_oracle, policy);
    RestrictedDispatchDecision result(std::make_shared<RestrictedDispatchDecision::Impl>(
        RestrictedDispatchDecision::Impl{DispatchKind::kBucket, prepared.impl_->graph,
            exact.impl_->exact_oracle, profile}));
    VerifyDecision(*result.impl_);
    VerifyCounters(prepared.impl_->representative, prepared.impl_->counters);
    return result;
}

RestrictedDispatchDecision RestrictedSymbolicShapeAdapter::MintPolymorphic(
    const PreparedRestrictedSymbolicTemplate& prepared, const shape::BindingSet& bindings,
    const shape::PolymorphicPolicy& policy) {
    RestrictedDispatchDecision exact = MintExact(prepared, bindings);
    VerifyPolymorphicProofs(prepared.impl_->operations, prepared.impl_->graph, policy);
    const shape::GuardedShapeProfile profile = shape::BuildPolymorphicProfile(
        prepared.impl_->graph, exact.impl_->exact_oracle, policy);
    RestrictedDispatchDecision result(std::make_shared<RestrictedDispatchDecision::Impl>(
        RestrictedDispatchDecision::Impl{DispatchKind::kPolymorphic, prepared.impl_->graph,
            exact.impl_->exact_oracle, profile}));
    VerifyDecision(*result.impl_);
    VerifyCounters(prepared.impl_->representative, prepared.impl_->counters);
    return result;
}

std::vector<size_t> RestrictedSymbolicShapeAdapter::ChangedUnitIndices(
    const RestrictedDispatchDecision& previous, const RestrictedDispatchDecision& next) {
    if (!previous.impl_ || !next.impl_) Reject("decision is undefined");
    VerifyDecision(*previous.impl_);
    VerifyDecision(*next.impl_);
    const auto previous_exact = RebuildExact(*previous.impl_);
    const auto next_exact = RebuildExact(*next.impl_);
    if (previous_exact.size() != next_exact.size()) Reject("decision unit cardinality mismatch");
    const auto previous_guarded = RebuildGuarded(*previous.impl_);
    const auto next_guarded = RebuildGuarded(*next.impl_);
    std::vector<size_t> changed;
    for (size_t i = 0; i < previous_exact.size(); ++i) {
        const auto& previous_semantic = previous.kind() == DispatchKind::kExact
            ? previous_exact[i].unit_semantic_key
            : previous_guarded[i].unit_semantic_key;
        const auto& next_semantic = next.kind() == DispatchKind::kExact
            ? next_exact[i].unit_semantic_key
            : next_guarded[i].unit_semantic_key;
        if (!(previous_semantic == next_semantic)) Reject("decision unit semantic contexts are incomparable");
        const std::string left = previous.kind() == DispatchKind::kExact
            ? ExactUnitBoundaryIdentity(previous_exact[i])
            : GuardedUnitBoundaryIdentity(previous_guarded[i]);
        const std::string right = next.kind() == DispatchKind::kExact
            ? ExactUnitBoundaryIdentity(next_exact[i])
            : GuardedUnitBoundaryIdentity(next_guarded[i]);
        if (left != right) changed.push_back(i);
    }
    return changed;
}

}  // namespace kxc::api::experimental::restricted_symbolic_shape::v1
