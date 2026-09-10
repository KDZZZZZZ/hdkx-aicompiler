#include "kxc/compiler/restricted_symbolic_shape.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "../internal/dynamic_shape_contract.h"
#include "../internal/logical_value.h"
#include "../internal/relay_snapshot.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/runtime/ndarray.h"
#include "shape_value_resolver.h"

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
    // 与模板 unit 顺序平行的形状值 value 表达式覆盖（nullopt = 推导）。
    std::vector<std::optional<shape_resolution::EncodedExpr>>
        unit_value_expressions;
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
        if (operations[i] == "relu" || operations[i] == "sqrt" || operations[i] == "neg" ||
            operations[i] == "softmax" || operations[i] == "sigmoid" || operations[i] == "cast" || operations[i] == "trilu") {
            if (unit.input_value_names.size() != 1) Reject("shape-transparent unary arity mismatch");
            RequireSameShape(StaticValue(graph, unit.input_value_names[0]), output);
        } else if (operations[i] == "masked_softmax") {
            if (unit.input_value_names.size() != 2) Reject("masked_softmax requires data and mask");
            RequireSameShape(StaticValue(graph, unit.input_value_names[0]), output);
            if (!ValidContract(StaticValue(graph, unit.input_value_names[1]))) {
                Reject("invalid frozen masked_softmax mask");
            }
        } else if (operations[i] == "add" || operations[i] == "mul" ||
                   operations[i] == "divide" || operations[i] == "pow") {
            // Call arity is validated before partitioning. The physical unit
            // boundary deduplicates a logical call such as mul(x, x).
            if (unit.input_value_names.empty() || unit.input_value_names.size() > 2) {
                Reject("binary input boundary mismatch");
            }
            if (!ValidContract(output)) Reject("invalid frozen broadcast output");
            // Symbolic broadcast proof comes from the resolver; ordinary
            // InferType has validated the representative's shapes and dtype.
        } else if (operations[i] == "matmul" || operations[i] == "transpose" ||
                   operations[i] == "reduce_mean" || operations[i] == "slice" ||
                   operations[i] == "concatenate" || operations[i] == "gather") {
            // The resolver proves symbolic axis relationships; ordinary
            // InferType validates the frozen representative and its attrs.
            const size_t arity = operations[i] == "slice" ? 3 :
                (operations[i] == "matmul" || operations[i] == "concatenate" ||
                 operations[i] == "gather" ? 2 : 1);
            if (unit.input_value_names.empty() ||
                unit.input_value_names.size() > arity || !ValidContract(output)) {
                Reject("invalid frozen attention operator contract");
            }
        } else if (operations[i] == "shape_of") {
            // M3 形状值：输出是 int64[rank(input)] 的固定长度向量；长度与
            // dtype 契约在 unit 合同与 partition 校验中再行钉死。
            if (unit.input_value_names.size() != 1) {
                Reject("shape_of expects exactly one data input");
            }
            const auto input = StaticValue(graph, unit.input_value_names[0]);
            if (!ValidContract(output) || output.logical.size() != 1 ||
                input.logical.empty() ||
                output.logical[0] != static_cast<int64_t>(input.logical.size())) {
                Reject("shape_of must materialize one int64 element per input axis");
            }
        } else if (operations[i] == "shape_expr") {
            if (unit.input_value_names.size() != 1) {
                Reject("shape_expr expects exactly one source input");
            }
            if (!ValidContract(output) || output.logical.size() != 1 ||
                output.logical[0] < 1) {
                Reject("shape_expr must materialize a nonempty int64 vector");
            }
        } else if (operations[i] == "constant_of_shape") {
            if (unit.input_value_names.empty() || unit.input_value_names.size() > 2) {
                Reject("constant_of_shape expects shape or data+shape");
            }
            const auto control = StaticValue(graph, unit.input_value_names.back());
            if (!ValidContract(output) || !ValidContract(control) || control.logical.size() != 1 ||
                control.logical[0] < 1 || output.logical.size() != static_cast<size_t>(control.logical[0])) {
                Reject("constant_of_shape output rank must equal control length");
            }
        } else if (operations[i] == "reshape_dynamic") {
            if (unit.input_value_names.size() != 2) {
                Reject("reshape_dynamic expects data and a control shape value");
            }
            const auto control = StaticValue(graph, unit.input_value_names[1]);
            if (!ValidContract(output) || !ValidContract(control) ||
                control.logical.size() != 1 || control.logical[0] < 1 ||
                output.logical.size() != static_cast<size_t>(control.logical[0])) {
                Reject("reshape_dynamic output rank must equal its control length");
            }
        } else if (operations[i] == "expand_dynamic") {
            if (unit.input_value_names.size() != 2) {
                Reject("expand expects data and a control shape value");
            }
            const auto data = StaticValue(graph, unit.input_value_names[0]);
            const auto control = StaticValue(graph, unit.input_value_names[1]);
            if (!ValidContract(output) || !ValidContract(control) ||
                control.logical.size() != 1 || control.logical[0] < 1 ||
                output.logical.size() != data.logical.size() ||
                output.logical.size() != static_cast<size_t>(control.logical[0])) {
                Reject("expand output rank must equal data rank and control length");
            }
        } else if (operations[i] == "squeeze" || operations[i] == "unsqueeze") {
            if (unit.input_value_names.size() != 1) {
                Reject(operations[i] + " expects exactly one data input");
            }
            if (!ValidContract(output) || output.logical.empty()) {
                Reject(operations[i] + " must produce a rank >= 1 output");
            }
        } else {
            Reject("unsupported frozen operation");
        }
    }
}

bool IsRestrictedSyntaxOp(const std::string& name) {
    return name == "relu" || name == "nn_relu" || name == "sqrt" || name == "sigmoid" || name == "neg" || name == "slice" ||
           name == "add" || name == "subtract" || name == "mul" || name == "divide" || name == "pow" ||
           name == "cast" || name == "reduce_mean" || name == "matmul" ||
           name == "transpose" || name == "softmax" || name == "masked_softmax" || name == "shape_of" ||
           name == "gather" || name == "concatenate" ||
           // shape_expr 是解析器折叠 gather/concat 链后的单元形态，只出现在
           // 重写后的快照里；constant_of_shape 由受限目标形状驱动。
           name == "shape_expr" || name == "constant_of_shape" || name == "trilu" ||
           name == "reshape_dynamic" || name == "expand_dynamic" ||
           name == "squeeze" || name == "unsqueeze";
}

size_t RestrictedSyntaxArity(const std::string& name) {
    if (name == "add" || name == "subtract" || name == "mul" || name == "divide" || name == "pow" ||
        name == "matmul" || name == "gather" || name == "masked_softmax" ||
        name == "concatenate" || name == "reshape_dynamic" || name == "expand_dynamic") {
        return 2;
    }
    return 1;
}

void CollectOperations(const Expr& expression, const std::set<const Object*>& parameters,
                       std::set<const Object*>* seen, std::vector<std::string>* operations,
                       std::vector<std::string>* registry_operations) {
    if (expression.As<VarNode>()) {
        if (!parameters.count(expression.get())) Reject("graph references a non-parameter variable");
        return;
    }
    if (expression.As<ConstantNode>()) {
        // 常量（Gather 索引等）由受限形状解析器裁决；此处只允许通过。
        return;
    }
    if (!seen->insert(expression.get()).second) return;
    if (const auto* tuple = expression.As<TupleNode>()) {
        if (tuple->fields.empty()) Reject("empty result tuples are unsupported");
        for (const Expr& field : tuple->fields) {
            CollectOperations(field, parameters, seen, operations,
                              registry_operations);
        }
        return;
    }
    const auto* call = expression.As<CallNode>();
    if (!call) Reject("unsupported restricted graph expression");
    const auto* op = call->op.As<relay::OpNode>();
    if (!op || !IsRestrictedSyntaxOp(op->name)) {
        Reject("unsupported Relay operation");
    }
    if (call->args.size() != RestrictedSyntaxArity(op->name) &&
        !((op->name == "constant_of_shape" || op->name == "slice") &&
          (call->args.size() == 2 || (op->name == "slice" && call->args.size() == 3)))) {
        Reject("unsupported Relay operation arity");
    }
    for (const Expr& argument : call->args) {
        CollectOperations(argument, parameters, seen, operations,
                          registry_operations);
    }
    operations->push_back(op->name == "nn_relu" ? "relu" : op->name);
    // 物化必须用注册表原名重放，不能用规范化别名。
    registry_operations->push_back(op->name);
}

struct RestrictedSyntax final {
    std::vector<std::string> operations;           // 规范化名，供 frozen unit 校验
    std::vector<std::string> registry_operations;  // 注册表原名，供物化重放
};

// 参数形态：固定非标量 rank、无遗留负 extent、无重复。形状值解析器读取
// 这些标注，因此必须在解析之前校验。
std::set<const Object*> ValidateParameters(const Function& function) {
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
    return parameters;
}

RestrictedSyntax ValidateSyntax(const Function& function) {
    const std::set<const Object*> parameters = ValidateParameters(function);
    std::set<const Object*> seen;
    RestrictedSyntax result;
    CollectOperations(function->body, parameters, &seen, &result.operations,
                      &result.registry_operations);
    return result;
}

std::vector<int64_t> Dimensions(const shape::NamedTensorContract& value) {
    return value.contract.Evaluate(shape::BindingSet()).logical;
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
    // 单元合同所需的注册表原名；原始调用接线由 Relay
    // 快照持有，不能从去重后的 unit boundary 重建逻辑参数。
    std::vector<std::string> registry_operations;
    std::vector<std::optional<shape_resolution::EncodedExpr>>
        unit_value_expressions;
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

const std::vector<std::optional<shape_resolution::EncodedExpr>>&
BoundedCompileRequest::unit_value_expressions() const {
    if (!impl_) Reject("bounded compile request is undefined");
    return impl_->unit_value_expressions;
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
    Function representative, CompileConfig config, std::vector<InputAxisSymbol> input_axis_symbols,
    std::vector<TensorType> representative_outputs) {
    RequireEnabled();
    config.Validate();
    CompileConfig frozen_config = config;
    (void)ValidateParameters(representative);
    Function representative_snapshot =
        internal::CloneRelaySnapshot(representative);
    // M3 受限形状值解析：链式证明、越界/元素总数/广播拒绝、链折叠与
    // 目标表达式投影。此后快照只含受支持的单元形态。
    const shape_resolution::Resolution resolution =
        shape_resolution::ResolveShapeValues(representative_snapshot,
                                             input_axis_symbols);
    representative_snapshot = resolution.rewritten;
    // 形状链折叠后校验受支持的算子与逻辑元数；共享调用只记录一次。
    (void)ValidateSyntax(representative_snapshot);
    if (!representative_outputs.empty()) {
        const Function typed = relay::InferTypePass(representative_snapshot);
        const auto actual = internal::FlattenLogicalTensorTypes(
            typed->body.checked_type(), "restricted imported outputs");
        if (actual.size() != representative_outputs.size()) {
            Reject("declared imported output count differs from the resolved graph");
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!TypeEqual(actual[i], representative_outputs[i])) {
                Reject("declared imported output " + std::to_string(i) +
                       " differs from the resolved graph");
            }
        }
    }
    const std::vector<std::string>& operations = [&] {
        std::vector<std::string> normalized;
        normalized.reserve(resolution.registry_operations.size());
        for (const std::string& name : resolution.registry_operations) {
            normalized.push_back(name == "nn_relu" ? "relu" : name);
        }
        return normalized;
    }();
    const std::vector<std::string>& registry_operations =
        resolution.registry_operations;
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
    if (static_graph.shape_program().inputs().size() < representative_snapshot->params.size()) {
        Reject("restricted graph has missing parameters");
    }
    std::vector<shape::NamedTensorContract> inputs, outputs;
    std::set<std::string> parameter_values;
    for (size_t parameter = 0; parameter < representative_snapshot->params.size(); ++parameter) {
        const auto& source = ParameterContract(static_graph, parameter);
        parameter_values.insert(source.name);
        const auto static_shape = Dimensions(source);
        std::vector<shape::DimExpr> dimensions;
        for (int64_t extent : static_shape) dimensions.push_back(shape::DimExpr::Const(extent));
        for (const auto& symbol : input_axis_symbols) {
            if (symbol.parameter_index == parameter) dimensions[symbol.axis] = shape::DimExpr::Symbol(symbol.symbol);
        }
        // Equal representative sizes do not imply equal symbolic dimensions:
        // query and key lengths may coincide in the sample and vary separately.
        inputs.push_back({source.name, shape::TensorShapeContract(
            shape::LogicalShape(dimensions, source.contract.logical().axis_names()),
            shape::PhysicalCapacity(dimensions), shape::ValidExtent(dimensions))});
    }
    // ShapeProgram sorts canonical names lexically: value.10 precedes value.2.
    // Distinguish parameters by their ValueGraph names, never by list position.
    for (const auto& input : static_graph.shape_program().inputs()) {
        if (!parameter_values.count(input.name)) inputs.push_back(input);
    }
    std::vector<shape::Constraint> constraints;
    for (const auto& [name, symbol] : definitions) {
        const shape::DimExpr expression = shape::DimExpr::Symbol(name);
        constraints.push_back(shape::Constraint::Range(expression, symbol.lower, symbol.upper));
        constraints.push_back(shape::Constraint::DivisibleBy(expression, symbol.divisible_by));
    }
    // 输出合同由解析器的符号维表达式驱动：链折叠后的目标（如
    // reshape_dynamic / expand / squeeze / unsqueeze）不再能由静态形状键
    // 匹配近似表达。
    if (resolution.unit_output_dimensions.size() != static_graph.ordered_units().size()) {
        Reject("dimension proofs do not match the frozen unit sequence");
    }
    std::map<std::string, std::vector<shape::DimExpr>> dimensions_by_value;
    for (size_t i = 0; i < static_graph.ordered_units().size(); ++i) {
        dimensions_by_value.emplace(
            static_graph.ordered_units()[i].output_value_names[0],
            resolution.unit_output_dimensions[i]);
    }
    for (const auto& value : static_graph.shape_program().outputs()) {
        const auto found = dimensions_by_value.find(value.name);
        if (found == dimensions_by_value.end()) {
            Reject("the restricted resolver has no dimension proof for " +
                   value.name);
        }
        const auto& dims = found->second;
        outputs.push_back({value.name, shape::TensorShapeContract(
            shape::LogicalShape(dims, value.contract.logical().axis_names()),
            shape::PhysicalCapacity(dims), shape::ValidExtent(dims))});
    }
    shape::GraphTemplate graph(static_graph.key(), shape::ShapeProgram(
        std::vector<std::string>(names.begin(), names.end()), std::move(inputs), std::move(outputs),
        std::move(constraints)), static_graph.ordered_units());
    const auto counters = frozen.counters();
    return PreparedRestrictedSymbolicTemplate(std::make_shared<PreparedRestrictedSymbolicTemplate::Impl>(
        PreparedRestrictedSymbolicTemplate::Impl{
            std::move(frozen), std::move(graph),
            std::move(input_axis_symbols), counters, registry_operations,
            resolution.unit_value_expressions,
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

using ReplayShape = std::function<Array<int64_t>(
    size_t, const shape::NamedTensorContract&)>;

Function MaterializeRestrictedFunction(
    const shape::GraphTemplate& graph,
    const Function& representative,
    const ReplayShape& replay_shape) {
    if (graph.shape_program().inputs().size() < representative->params.size()) {
        Reject("restricted template parameter cardinality drifted");
    }
    // The snapshot is the sole authority for Call arguments and function
    // results. Retyping its private parameters preserves DAG sharing, repeated
    // arguments, nested result tuples, and outputs that also have consumers.
    Function result = internal::CloneRelaySnapshot(representative);
    for (size_t i = 0; i < result->params.size(); ++i) {
        const auto& named = ParameterContract(graph, i);
        auto* parameter = const_cast<VarNode*>(result->params[i].operator->());
        const auto* type = parameter->type_annotation.As<TensorTypeNode>();
        parameter->type_annotation = TensorType(replay_shape(i, named), type->dtype);
    }
    return relay::InferTypePass(result);
}

Array<int64_t> BoundedBoundaryShape(
    const shape::NamedTensorContract& named) {
    Array<int64_t> dimensions;
    for (const shape::DimExpr& expression :
         named.contract.logical().dimensions()) {
        if (expression.kind() == shape::DimExpr::Kind::kConst) {
            dimensions.push_back(expression.Evaluate(shape::BindingSet()));
        } else if ((expression.kind() == shape::DimExpr::Kind::kSymbol ||
                    expression.kind() == shape::DimExpr::Kind::kAdd) &&
                   expression.Symbols().size() == 1) {
            dimensions.push_back(-1);
        } else {
            Reject("bounded boundary accepts Const or a single symbol with a constant offset");
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
    if (!((target->kind == "llvm" && target->device_type == kCPU && target->device_id == 0) ||
          (target->kind == "cuda" && target->device_type == kCUDA))) {
        Reject("bounded compilation requires CPU:0/LLVM or CUDA");
    }
    // M3：attrs 策略由受限形状解析器在准备时强制（attr-free 算子不带属性，
    // 受控算子的目标表达式是链式证明的唯一投影），此处无需再整体拒绝。
    if (Compiler::BuildGraphSemanticKey(prepared.impl_->representative_snapshot) !=
        prepared.impl_->graph.key()) {
        Reject("representative Function semantics differ from GraphTemplate");
    }

    // This is the only GraphTemplate + UnitSkeleton contract producer. Calling
    // it here makes unsupported arithmetic/broadcast/rank forms fail before a
    // request can exist; no contract is caller-authored.
    (void)internal::BuildDynamicUnitShapeContracts(
        prepared.impl_->graph, prepared.impl_->registry_operations,
        internal::DecodeValueExpressionOverrides(
            prepared.impl_->unit_value_expressions));
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

    Function logical_boundary =
        MaterializeRestrictedFunction(
            prepared.impl_->graph, prepared.impl_->representative_snapshot,
            [](size_t, const shape::NamedTensorContract& named) {
                return BoundedBoundaryShape(named);
            });
    BoundedCompileRequest request(
        std::make_shared<BoundedCompileRequest::Impl>(
            BoundedCompileRequest::Impl{
                internal::CloneRelaySnapshot(prepared.impl_->representative_snapshot),
                internal::CloneRelaySnapshot(logical_boundary),
                prepared.impl_->graph, std::move(representative_oracle),
                prepared.impl_->unit_value_expressions,
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
    return MaterializeRestrictedFunction(
        prepared.impl_->graph, prepared.impl_->representative_snapshot,
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
    const Function expected_function =
        MaterializeExactFunction(prepared, decision);
    std::map<int64_t, runtime::ValueSpec> specs;
    for (const auto& value : plan.values()) specs.emplace(value->value_id, value);
    const auto check_boundary =
        [&](const Array<int64_t>& ids,
            const std::vector<Type>& types,
            const char* what) {
            if (ids.size() != types.size()) {
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
                const auto* type = types[i].As<TensorTypeNode>();
                if (!type) Reject("materialized boundary must have tensor leaves");
                const auto& expected = type->shape;
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
                const DLDataType expected_dtype = runtime::DataTypeFromString(type->dtype);
                const DLDataType actual_dtype = found->second->dtype;
                if (actual_dtype.code != expected_dtype.code ||
                    actual_dtype.bits != expected_dtype.bits ||
                    actual_dtype.lanes != expected_dtype.lanes) {
                    throw std::runtime_error(
                        std::string("restricted exact variant: ") + what +
                        " dtype differs from the materialized function");
                }
            }
        };
    std::vector<Type> parameter_types;
    for (const Var& parameter : expected_function->params) {
        parameter_types.push_back(parameter->type_annotation);
    }
    check_boundary(plan.input_value_ids(), parameter_types, "input");
    check_boundary(plan.output_value_ids(), internal::FlattenLogicalTensorTypes(
        expected_function.checked_type(), "restricted exact variant outputs"), "output");
    // 语义绑定：边界一致不构成授权——同 shape/dtype/call 数但算子不同的
    // 图（如 sqrt 换 relu）必须被拒绝。产物身份必须等于按本决策物化出的
    // concrete Function 的语义身份；类型（含输出 dtype）是身份的一部分。
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
    const size_t parameter_count = prepared.impl_->representative_snapshot->params.size();
    if (input_shapes.size() != parameter_count) {
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
    for (size_t i = 0; i < parameter_count; ++i) {
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
