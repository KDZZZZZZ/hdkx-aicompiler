#include "kxc/compiler/restricted_symbolic_shape.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "../internal/dynamic_shape_contract.h"
#include "../internal/relay_snapshot.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/runtime/ndarray.h"

#ifndef KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
#define KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE 0
#endif

namespace kxc::api::experimental::restricted_symbolic_shape::v1 {
namespace shape =
    kxc::api::experimental::shape_specialization::v1;

struct RestrictedDispatchDecision::Impl final {
    shape::GraphTemplate graph;
    shape::ExactOracle exact_oracle;
};

struct BoundedCompileRequest::Impl final {
    Function representative;
    Function logical_boundary_function;
    shape::GraphTemplate graph;
    shape::ExactOracle representative_oracle;
    CompileConfig config;
    uint32_t applicability_version{kBoundedCompileApplicabilityVersion};
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
    return ContractDefect(value) == nullptr;
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
        for (const auto& value : request.ordered_inputs) {
            if (!ValidContract(value)) Reject("invalid exact input extent contract");
        }
        for (const auto& value : request.ordered_outputs) {
            if (!ValidContract(value)) Reject("invalid exact output extent contract");
        }
    }
}

std::string ValueName(size_t id) {
    return "value." + std::to_string(id);
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

const shape::NamedTensorContract& ParameterContract(
    const shape::GraphTemplate& graph, size_t parameter_index) {
    const std::string name = ValueName(parameter_index);
    const auto& inputs = graph.shape_program().inputs();
    const auto found = std::find_if(inputs.begin(), inputs.end(),
        [&name](const auto& input) { return input.name == name; });
    if (found == inputs.end()) {
        Reject("GraphTemplate has no value for a representative parameter");
    }
    return *found;
}

shape::ConcreteTensorShapeContract StaticValue(const shape::GraphTemplate& graph,
                                                const std::string& name) {
    return Named(graph, name).contract.Evaluate(shape::BindingSet());
}

void RequireSameShape(const shape::ConcreteTensorShapeContract& left,
                      const shape::ConcreteTensorShapeContract& right) {
    if (!(left == right) || !ValidContract(left)) {
        Reject("broadcast or non-exact shape contract is unsupported");
    }
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
                       std::set<const Object*>* seen, std::vector<std::string>* operations,
                       std::vector<std::string>* registry_operations,
                       bool* has_operation_attrs) {
    if (expression.As<VarNode>()) {
        if (!parameters.count(expression.get())) Reject("graph references a non-parameter variable");
        return;
    }
    const auto* call = expression.As<CallNode>();
    if (!call || !seen->insert(expression.get()).second) {
        Reject("only a tree of restricted calls is supported");
    }
    const auto* op = call->op.As<relay::OpNode>();
    if (!op || (op->name != "relu" && op->name != "nn_relu" && op->name != "sqrt" &&
                op->name != "add" && op->name != "mul")) {
        Reject("unsupported Relay operation");
    }
    const size_t arity = (op->name == "add" || op->name == "mul") ? 2 : 1;
    if (call->args.size() != arity) Reject("unsupported Relay operation arity");
    *has_operation_attrs = *has_operation_attrs || call->attrs.defined();
    for (const Expr& argument : call->args) {
        CollectOperations(argument, parameters, seen, operations,
                          registry_operations, has_operation_attrs);
    }
    operations->push_back(op->name == "nn_relu" ? "relu" : op->name);
    // 物化必须用注册表原名重放，不能用规范化别名。
    registry_operations->push_back(op->name);
}

struct RestrictedSyntax final {
    std::vector<std::string> operations;           // 规范化名，供 frozen unit 校验
    std::vector<std::string> registry_operations;  // 注册表原名，供物化重放
    bool has_operation_attrs{false};
};

RestrictedSyntax ValidateSyntax(const Function& function) {
    if (!function.defined() || function->params.empty()) {
        Reject("a nonempty fixed-rank parameter list is required");
    }
    std::set<const Object*> parameters;
    for (const Var& parameter : function->params) {
        const auto* type = parameter->type_annotation.As<TensorTypeNode>();
        if (!type || type->shape.empty()) {
            Reject("every input must have a fixed non-scalar tensor rank");
        }
        for (int64_t extent : type->shape) {
            if (extent < 0) Reject("legacy -1 or negative input extent is unsupported");
        }
        if (!parameters.insert(parameter.get()).second) Reject("duplicate input parameter");
    }
    std::set<const Object*> seen;
    RestrictedSyntax result;
    CollectOperations(function->body, parameters, &seen, &result.operations,
                      &result.registry_operations,
                      &result.has_operation_attrs);
    return result;
}

std::vector<int64_t> Dimensions(const shape::NamedTensorContract& value) {
    return value.contract.Evaluate(shape::BindingSet()).logical;
}

shape::NamedTensorContract Overlay(
    const shape::NamedTensorContract& source,
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
    if (bindings.bindings().size() != symbols.size()) {
        Reject("bindings must explicitly bind every overlay symbol exactly once");
    }
    for (const std::string& symbol : symbols) {
        if (!bindings.Find(symbol).has_value()) Reject("unbound overlay symbol");
    }
}

std::vector<shape::UnitSpecializationRequest> RebuildExact(
    const RestrictedDispatchDecision::Impl& decision) {
    const auto requests = shape::MakeExactSpecializationRequests(
        decision.graph, decision.exact_oracle);
    if (requests.empty()) Reject("decision has no exact requests");
    VerifyContracts(requests);
    const auto& exact_key = decision.exact_oracle.profile().key();
    for (const auto& request : requests) {
        if (!(request.shape_profile_key == exact_key)) {
            Reject("exact request is detached from decision oracle");
        }
    }
    return requests;
}

void VerifyDecision(const RestrictedDispatchDecision::Impl& decision) {
    (void)RebuildExact(decision);
}

std::string ExactUnitBoundaryIdentity(const shape::UnitSpecializationRequest& request) {
    return "restricted.exact-unit-boundary.v2|" +
           request.unit_semantic_key.canonical_bytes() + "|" +
           request.signature_digest.value();
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
    shape_exact::v1::ShapeExactPreparationCounters counters;
    // 物化重放所需：注册表原名的有序调用序列与参数 dtype 快照。
    std::vector<std::string> registry_operations;
    std::vector<std::string> input_dtypes;
    bool has_operation_attrs{false};
    Function representative_snapshot;
    CompileConfig config;
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

BoundedCompileRequest::BoundedCompileRequest(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {
    if (!impl_) Reject("bounded compile request is undefined");
}
BoundedCompileRequest::~BoundedCompileRequest() = default;
BoundedCompileRequest::BoundedCompileRequest(const BoundedCompileRequest&) = default;
BoundedCompileRequest& BoundedCompileRequest::operator=(const BoundedCompileRequest&) = default;
BoundedCompileRequest::BoundedCompileRequest(BoundedCompileRequest&&) noexcept = default;
BoundedCompileRequest& BoundedCompileRequest::operator=(BoundedCompileRequest&&) noexcept = default;
Function BoundedCompileRequest::representative() const {
    if (!impl_) Reject("bounded compile request is undefined");
    return internal::CloneRelaySnapshot(impl_->representative);
}
Function BoundedCompileRequest::logical_boundary_function() const {
    if (!impl_) Reject("bounded compile request is undefined");
    return internal::CloneRelaySnapshot(impl_->logical_boundary_function);
}
const shape::GraphTemplate& BoundedCompileRequest::graph_template() const {
    if (!impl_) Reject("bounded compile request is undefined");
    return impl_->graph;
}
const shape::ExactOracle& BoundedCompileRequest::representative_oracle() const {
    if (!impl_) Reject("bounded compile request is undefined");
    return impl_->representative_oracle;
}
const CompileConfig& BoundedCompileRequest::compile_config() const {
    if (!impl_) Reject("bounded compile request is undefined");
    return impl_->config;
}
const Target& BoundedCompileRequest::target() const {
    return compile_config()->target;
}
uint32_t BoundedCompileRequest::applicability_version() const noexcept {
    return impl_ ? impl_->applicability_version : 0;
}

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
    config.Validate();
    CompileConfig frozen_config = config;
    const RestrictedSyntax syntax = ValidateSyntax(representative);
    Function representative_snapshot =
        internal::CloneRelaySnapshot(representative);
    const std::vector<std::string>& operations = syntax.operations;
    std::vector<std::string> input_dtypes;
    input_dtypes.reserve(representative_snapshot->params.size());
    for (const Var& parameter : representative_snapshot->params) {
        input_dtypes.push_back(
            parameter->type_annotation.As<TensorTypeNode>()->dtype);
    }
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
            symbol.parameter_index >= representative_snapshot->params.size()) {
            Reject("duplicate, out-of-range, or invalid input-axis symbol");
        }
        const auto existing_definition = definitions.find(symbol.symbol);
        if (existing_definition != definitions.end() &&
            (existing_definition->second.lower != symbol.lower ||
             existing_definition->second.upper != symbol.upper ||
             existing_definition->second.divisible_by != symbol.divisible_by)) {
            Reject("repeated overlay symbol has inconsistent bounds");
        }
        definitions.emplace(symbol.symbol, symbol);
        names.insert(symbol.symbol);
        const auto* type = representative_snapshot->params[symbol.parameter_index]->type_annotation.As<TensorTypeNode>();
        if (!type || symbol.axis >= type->shape.size() || type->shape[symbol.axis] < symbol.lower ||
            type->shape[symbol.axis] > symbol.upper || type->shape[symbol.axis] % symbol.divisible_by != 0) {
            Reject("input-axis symbol does not cover its frozen representative extent");
        }
    }
    shape_exact::v1::PreparedGraphTemplate frozen =
        shape_exact::v1::ProductionExactShapeAdapter::PrepareGraphTemplate(
            representative_snapshot, frozen_config);
    const shape::GraphTemplate& static_graph = frozen.graph_template();
    ValidateFrozenUnits(static_graph, operations);
    if (static_graph.shape_program().inputs().size() != representative_snapshot->params.size()) {
        Reject("restricted graph cannot contain constants or hidden inputs");
    }
    std::map<std::vector<int64_t>, std::vector<shape::DimExpr>> shape_overlays;
    for (size_t parameter = 0; parameter < representative_snapshot->params.size(); ++parameter) {
        const auto static_shape = Dimensions(
            ParameterContract(static_graph, parameter));
        std::vector<shape::DimExpr> dimensions;
        for (int64_t extent : static_shape) dimensions.push_back(shape::DimExpr::Const(extent));
        for (const auto& symbol : input_axis_symbols) {
            if (symbol.parameter_index == parameter) dimensions[symbol.axis] = shape::DimExpr::Symbol(symbol.symbol);
        }
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
        PreparedRestrictedSymbolicTemplate::Impl{
            std::move(frozen), std::move(graph),
            std::move(input_axis_symbols), counters, syntax.registry_operations,
            std::move(input_dtypes), syntax.has_operation_attrs,
            std::move(representative_snapshot),
            std::move(frozen_config)}));
}

RestrictedDispatchDecision::RestrictedDispatchDecision(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}

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

const shape::GraphTemplate& RestrictedDispatchDecision::graph_template() const {
    if (!impl_) Reject("decision is undefined");
    VerifyDecision(*impl_);
    return impl_->graph;
}

RestrictedDispatchDecision RestrictedSymbolicShapeAdapter::MintExact(
    const PreparedRestrictedSymbolicTemplate& prepared, const shape::BindingSet& bindings) {
    RequireEnabled();
    if (!prepared.impl_) Reject("prepared template is undefined");
    VerifyBindings(prepared.impl_->graph, bindings);
    const shape::ExactOracle oracle = shape::InstantiateExactProfile(prepared.impl_->graph, bindings);
    RestrictedDispatchDecision result(std::make_shared<RestrictedDispatchDecision::Impl>(
        RestrictedDispatchDecision::Impl{prepared.impl_->graph, oracle}));
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
    std::vector<size_t> changed;
    for (size_t i = 0; i < previous_exact.size(); ++i) {
        if (!(previous_exact[i].unit_semantic_key == next_exact[i].unit_semantic_key)) {
            Reject("decision unit semantic contexts are incomparable");
        }
        if (ExactUnitBoundaryIdentity(previous_exact[i]) !=
            ExactUnitBoundaryIdentity(next_exact[i])) {
            changed.push_back(i);
        }
    }
    return changed;
}

namespace {

// 决策必须由同一个 prepared 模板铸出（key 与模板全文一致）。
void RequireDecisionOwnership(
    const shape::GraphTemplate& prepared_graph,
    const RestrictedDispatchDecision::Impl& decision) {
    if (!(decision.graph.key() == prepared_graph.key()) ||
        decision.graph.CanonicalBytes() != prepared_graph.CanonicalBytes()) {
        Reject("decision does not belong to this prepared template");
    }
}

// ShapeProgram::outputs() 按 BuildTemplate 语义收录全部计算值（含被后续
// unit 消费的中间值）；图输出是其中未被任何 unit 消费的那些。
std::vector<shape::NamedTensorContract> GraphOutputs(
    const shape::GraphTemplate& graph) {
    std::set<std::string> consumed;
    for (const auto& unit : graph.ordered_units()) {
        consumed.insert(unit.input_value_names.begin(),
                        unit.input_value_names.end());
    }
    std::vector<shape::NamedTensorContract> outputs;
    for (const auto& value : graph.shape_program().outputs()) {
        if (!consumed.count(value.name)) outputs.push_back(value);
    }
    return outputs;
}

using ReplayShape = std::function<Array<int64_t>(
    size_t, const shape::NamedTensorContract&)>;

Function ReplayRestrictedFunction(
    const shape::GraphTemplate& graph,
    const std::vector<std::string>& registry_operations,
    const std::vector<std::string>& input_dtypes,
    const ReplayShape& replay_shape) {
    if (graph.shape_program().inputs().size() != input_dtypes.size()) {
        Reject("restricted template parameter dtype cardinality drifted");
    }
    Array<Var> params;
    std::map<std::string, Expr> values;
    for (size_t i = 0; i < input_dtypes.size(); ++i) {
        const auto& named = ParameterContract(graph, i);
        Var parameter(named.name, TensorType(replay_shape(i, named),
                                             input_dtypes[i]));
        params.push_back(parameter);
        values.emplace(named.name, parameter);
    }
    const auto& units = graph.ordered_units();
    if (units.size() != registry_operations.size()) {
        Reject("frozen unit sequence does not match the recorded call replay");
    }
    for (size_t i = 0; i < units.size(); ++i) {
        Array<Expr> args;
        for (const std::string& input : units[i].input_value_names) {
            const auto found = values.find(input);
            if (found == values.end()) Reject("unit consumes an unknown value");
            args.push_back(found->second);
        }
        const Call call(relay::Op::Get(registry_operations[i]), args);
        if (units[i].output_value_names.size() != 1 ||
            !values.emplace(units[i].output_value_names[0], call).second) {
            Reject("frozen unit output wiring is not a single-assignment tree");
        }
    }
    const auto graph_outputs = GraphOutputs(graph);
    if (graph_outputs.size() != 1) {
        Reject("restricted template must have exactly one graph output");
    }
    const auto output = values.find(graph_outputs[0].name);
    if (output == values.end()) Reject("graph output was never produced");
    return Function(params, output->second);
}

Array<int64_t> BoundedBoundaryShape(
    const shape::NamedTensorContract& named) {
    Array<int64_t> dimensions;
    for (const shape::DimExpr& expression :
         named.contract.logical().dimensions()) {
        if (expression.kind() == shape::DimExpr::Kind::kConst) {
            dimensions.push_back(expression.Evaluate(shape::BindingSet()));
        } else if (expression.kind() == shape::DimExpr::Kind::kSymbol &&
                   expression.Symbols().size() == 1) {
            dimensions.push_back(-1);
        } else {
            Reject("bounded boundary accepts only direct Symbol or Const axes");
        }
    }
    return dimensions;
}

}  // namespace

BoundedCompileRequest RestrictedSymbolicShapeAdapter::MintBoundedCompileRequest(
    const PreparedRestrictedSymbolicTemplate& prepared) {
    RequireEnabled();
    if (!prepared.impl_) Reject("prepared template is undefined");
    prepared.impl_->config.Validate();
    const Target& target = prepared.impl_->config->target;
    if (target->kind != "llvm" || target->device_type != kCPU ||
        target->device_id != 0) {
        Reject("bounded v1 admits only CPU/LLVM targets");
    }
    if (prepared.impl_->has_operation_attrs) {
        Reject("bounded v1 operation attrs are unsupported");
    }
    if (Compiler::BuildGraphSemanticKey(prepared.impl_->representative_snapshot) !=
        prepared.impl_->graph.key()) {
        Reject("representative Function semantics differ from GraphTemplate");
    }

    // This is the only GraphTemplate + UnitSkeleton contract producer. Calling
    // it here makes unsupported arithmetic/broadcast/rank forms fail before a
    // request can exist; no contract is caller-authored.
    (void)internal::BuildDynamicUnitShapeContracts(prepared.impl_->graph);
    std::vector<std::vector<int64_t>> representative_shapes;
    representative_shapes.reserve(
        prepared.impl_->representative_snapshot->params.size());
    for (const Var& parameter :
         prepared.impl_->representative_snapshot->params) {
        const auto* type = parameter->type_annotation.As<TensorTypeNode>();
        representative_shapes.emplace_back(type->shape.begin(),
                                           type->shape.end());
    }
    shape::ExactOracle representative_oracle = shape::InstantiateExactProfile(
        prepared.impl_->graph,
        BindingsFromInputShapes(prepared, representative_shapes));
    (void)shape::MakeExactSpecializationRequests(
        prepared.impl_->graph, representative_oracle);

    Function logical_boundary = relay::InferTypePass(
        ReplayRestrictedFunction(
            prepared.impl_->graph, prepared.impl_->registry_operations,
            prepared.impl_->input_dtypes,
            [](size_t, const shape::NamedTensorContract& named) {
                return BoundedBoundaryShape(named);
            }));
    BoundedCompileRequest request(
        std::make_shared<BoundedCompileRequest::Impl>(
            BoundedCompileRequest::Impl{
                internal::CloneRelaySnapshot(prepared.impl_->representative_snapshot),
                internal::CloneRelaySnapshot(logical_boundary),
                prepared.impl_->graph, std::move(representative_oracle),
                prepared.impl_->config,
                kBoundedCompileApplicabilityVersion}));
    VerifyCounters(prepared.impl_->representative, prepared.impl_->counters);
    return request;
}

Function RestrictedSymbolicShapeAdapter::MaterializeExactFunction(
    const PreparedRestrictedSymbolicTemplate& prepared,
    const RestrictedDispatchDecision& decision) {
    RequireEnabled();
    if (!prepared.impl_ || !decision.impl_) {
        Reject("prepared template or decision is undefined");
    }
    VerifyDecision(*decision.impl_);
    RequireDecisionOwnership(prepared.impl_->graph, *decision.impl_);
    const shape::ExactShapeProfile& profile =
        decision.impl_->exact_oracle.profile();
    return ReplayRestrictedFunction(
        prepared.impl_->graph, prepared.impl_->registry_operations,
        prepared.impl_->input_dtypes,
        [&profile](size_t, const shape::NamedTensorContract& named) {
            Array<int64_t> dimensions;
            for (int64_t extent : profile.Value(named.name).contract.logical) {
                dimensions.push_back(extent);
            }
            return dimensions;
        });
}

DispatchKey RestrictedSymbolicShapeAdapter::ExactDispatchKey(
    const PreparedRestrictedSymbolicTemplate& prepared,
    const RestrictedDispatchDecision& decision) {
    RequireEnabled();
    if (!prepared.impl_ || !decision.impl_) {
        Reject("prepared template or decision is undefined");
    }
    VerifyDecision(*decision.impl_);
    RequireDecisionOwnership(prepared.impl_->graph, *decision.impl_);
    return BuildStaticExactDispatchKey(
        prepared.impl_->graph.key(),
        decision.impl_->exact_oracle.profile().key());
}

void RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(
    const PreparedRestrictedSymbolicTemplate& prepared,
    const RestrictedDispatchDecision& decision,
    const CompiledGraph& compiled) {
    RequireEnabled();
    if (!prepared.impl_ || !decision.impl_) {
        Reject("prepared template or decision is undefined");
    }
    VerifyDecision(*decision.impl_);
    RequireDecisionOwnership(prepared.impl_->graph, *decision.impl_);
    if (!compiled.defined()) {
        throw std::runtime_error(
            "restricted exact variant: compiled graph is undefined");
    }
    const runtime::ExecutablePlan& plan = compiled.plan();
    const auto& units = prepared.impl_->graph.ordered_units();
    if (plan.calls().size() != units.size()) {
        throw std::runtime_error(
            "restricted exact variant: call count differs from template units");
    }
    const shape::ExactShapeProfile& profile =
        decision.impl_->exact_oracle.profile();
    const auto& program = prepared.impl_->graph.shape_program();
    std::map<int64_t, runtime::ValueSpec> specs;
    for (const auto& value : plan.values()) specs.emplace(value->value_id, value);
    const auto check_boundary =
        [&](const Array<int64_t>& ids,
            const std::vector<shape::NamedTensorContract>& named,
            const char* what) {
            if (ids.size() != named.size()) {
                throw std::runtime_error(
                    std::string("restricted exact variant: ") + what +
                    " arity differs from the decision");
            }
            for (size_t i = 0; i < ids.size(); ++i) {
                const auto found = specs.find(ids[i]);
                if (found == specs.end()) {
                    throw std::runtime_error(
                        std::string("restricted exact variant: ") + what +
                        " references an unknown plan value");
                }
                const auto& expected =
                    profile.Value(named[i].name).contract.logical;
                const Array<int64_t> actual = found->second.shape();
                bool same = actual.size() == expected.size();
                for (size_t axis = 0; same && axis < expected.size(); ++axis) {
                    same = actual[axis] == expected[axis];
                }
                if (!same) {
                    throw std::runtime_error(
                        std::string("restricted exact variant: ") + what +
                        " boundary shape differs from the decision");
                }
            }
        };
    std::vector<shape::NamedTensorContract> parameter_contracts;
    parameter_contracts.reserve(prepared.impl_->input_dtypes.size());
    for (size_t i = 0; i < prepared.impl_->input_dtypes.size(); ++i) {
        parameter_contracts.push_back(
            ParameterContract(prepared.impl_->graph, i));
    }
    check_boundary(plan.input_value_ids(), parameter_contracts, "input");
    check_boundary(plan.output_value_ids(), GraphOutputs(prepared.impl_->graph),
                   "output");
    for (size_t i = 0; i < plan.input_value_ids().size(); ++i) {
        const DLDataType expected =
            runtime::DataTypeFromString(prepared.impl_->input_dtypes[i]);
        const DLDataType actual = specs.at(plan.input_value_ids()[i])->dtype;
        if (actual.code != expected.code || actual.bits != expected.bits ||
            actual.lanes != expected.lanes) {
            throw std::runtime_error(
                "restricted exact variant: input dtype differs from the "
                "representative");
        }
    }
    // 语义绑定：边界一致不构成授权——同 shape/dtype/call 数但算子不同的
    // 图（如 sqrt 换 relu）必须被拒绝。产物身份必须等于按本决策物化出的
    // concrete Function 的语义身份；类型（含输出 dtype）是身份的一部分。
    const Function expected_function =
        MaterializeExactFunction(prepared, decision);
    if (!(compiled.graph_semantic_key() ==
          Compiler::BuildGraphSemanticKey(expected_function))) {
        throw std::runtime_error(
            "restricted exact variant: compiled graph semantic identity does "
            "not match the decision's materialized function");
    }
}

shape::BindingSet RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
    const PreparedRestrictedSymbolicTemplate& prepared,
    const std::vector<std::vector<int64_t>>& input_shapes) {
    RequireEnabled();
    if (!prepared.impl_) Reject("prepared template is undefined");
    const auto& program = prepared.impl_->graph.shape_program();
    if (input_shapes.size() != program.inputs().size()) {
        Reject("input shape count differs from the template parameters");
    }
    std::map<std::string, int64_t> bound;
    for (const InputAxisSymbol& symbol : prepared.impl_->symbols) {
        if (symbol.parameter_index >= input_shapes.size() ||
            symbol.axis >= input_shapes[symbol.parameter_index].size()) {
            Reject("input rank does not cover an overlay axis");
        }
        const int64_t extent = input_shapes[symbol.parameter_index][symbol.axis];
        const auto existing = bound.emplace(symbol.symbol, extent);
        if (!existing.second && existing.first->second != extent) {
            Reject("conflicting extents for a shared overlay symbol");
        }
    }
    std::vector<shape::Binding> bindings_list;
    bindings_list.reserve(bound.size());
    for (const auto& [name, value] : bound) bindings_list.push_back({name, value});
    const shape::BindingSet bindings(std::move(bindings_list));
    // 非 overlay 静态轴与 rank 必须与模板求值结果一致。
    for (size_t i = 0; i < program.inputs().size(); ++i) {
        const auto evaluated =
            ParameterContract(prepared.impl_->graph, i)
                .contract.Evaluate(bindings);
        if (evaluated.logical != input_shapes[i]) {
            Reject("input shape does not match the template contract");
        }
    }
    return bindings;
}

}  // namespace kxc::api::experimental::restricted_symbolic_shape::v1
