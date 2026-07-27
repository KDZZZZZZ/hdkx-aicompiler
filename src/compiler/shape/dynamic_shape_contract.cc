/*! \file src/compiler/shape/dynamic_shape_contract.cc
 * \brief Bounded compile admission and unit-local shape-contract producer.
 */

#include "../internal/dynamic_shape_contract.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "../internal/execution_contract.h"
#include "kxc/relay/transforms/infer_type.h"
#include "support/canonical.h"

namespace kxc::api::internal {

class BoundedCompilePreparationAccess final {
public:
    static BoundedLogicalShapeAdmission MintLogicalShapeAdmission() {
        return BoundedLogicalShapeAdmission();
    }
};

namespace {

using DimExpr = specialization::DimExpr;
using NamedTensorContract = specialization::NamedTensorContract;

[[noreturn]] void Reject(const std::string& message) {
    throw std::invalid_argument("bounded compile preparation: " + message);
}

struct DirectDimension final {
    bool is_constant{false};
    std::uint64_t constant{0};
    std::string symbol;
};

DirectDimension ReadDirectDimension(const DimExpr& expression) {
    if (expression.kind() == DimExpr::Kind::kConst) {
        const int64_t value = expression.Evaluate(specialization::BindingSet());
        if (value < 0) Reject("a constant dimension is negative");
        return DirectDimension{true, static_cast<std::uint64_t>(value), {}};
    }
    if (expression.kind() != DimExpr::Kind::kSymbol) {
        Reject("only direct Symbol and Const dimensions are supported");
    }
    const std::vector<std::string> symbols = expression.Symbols();
    if (symbols.size() != 1 || symbols.front().empty()) {
        Reject("a direct symbolic dimension is malformed");
    }
    return DirectDimension{false, 0, symbols.front()};
}

bool SameDimensions(const std::vector<DimExpr>& left,
                    const std::vector<DimExpr>& right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
}

const NamedTensorContract& FindNamed(
    const specialization::GraphTemplate& graph, const std::string& name) {
    const auto find = [&name](const std::vector<NamedTensorContract>& values)
        -> const NamedTensorContract* {
        const auto found = std::find_if(
            values.begin(), values.end(), [&name](const auto& value) {
                return value.name == name;
            });
        return found == values.end() ? nullptr : &*found;
    };
    if (const auto* value = find(graph.shape_program().inputs())) return *value;
    if (const auto* value = find(graph.shape_program().outputs())) return *value;
    Reject("a unit references an unknown template value '" + name + "'");
}

struct SymbolBounds final {
    bool has_range{false};
    std::uint64_t lower{0};
    std::uint64_t upper{0};
    bool has_divisibility{false};
    std::uint64_t divisible_by{1};
};

using BoundsBySymbol = std::map<std::string, SymbolBounds>;

std::string DirectSymbol(const DimExpr& expression, const char* context) {
    const DirectDimension direct = ReadDirectDimension(expression);
    if (direct.is_constant) {
        Reject(std::string(context) + " must reference a direct Symbol");
    }
    return direct.symbol;
}

void ValidateNonemptyDomain(const std::string& symbol,
                            const SymbolBounds& bounds) {
    const std::uint64_t remainder = bounds.lower % bounds.divisible_by;
    const std::uint64_t increment =
        remainder == 0 ? 0 : bounds.divisible_by - remainder;
    if (increment > bounds.upper - bounds.lower) {
        Reject("symbol '" + symbol + "' has an empty bounded domain");
    }
}

BoundsBySymbol AnalyzeTemplate(
    const specialization::GraphTemplate& graph) {
    graph.Verify();
    const auto& program = graph.shape_program();
    BoundsBySymbol bounds;
    for (const std::string& symbol : program.declared_symbols()) {
        bounds.emplace(symbol, SymbolBounds{});
    }

    for (const specialization::Constraint& constraint : program.constraints()) {
        switch (constraint.kind()) {
            case specialization::Constraint::Kind::kRange: {
                const std::string symbol =
                    DirectSymbol(constraint.left(), "a range constraint");
                SymbolBounds& value = bounds.at(symbol);
                if (value.has_range) {
                    Reject("a symbol has multiple range constraints");
                }
                value.has_range = true;
                value.lower = static_cast<std::uint64_t>(constraint.lower());
                value.upper = static_cast<std::uint64_t>(constraint.upper());
                break;
            }
            case specialization::Constraint::Kind::kDivisibleBy: {
                const std::string symbol =
                    DirectSymbol(constraint.left(), "a divisibility constraint");
                SymbolBounds& value = bounds.at(symbol);
                if (value.has_divisibility) {
                    Reject("a symbol has multiple divisibility constraints");
                }
                value.has_divisibility = true;
                value.divisible_by =
                    static_cast<std::uint64_t>(constraint.divisor());
                break;
            }
            case specialization::Constraint::Kind::kEq:
                Reject("ShapeProgram equality constraints are unsupported; use repeated direct input symbols");
            case specialization::Constraint::Kind::kBroadcastCompatible:
                Reject("broadcast constraints are unsupported");
        }
    }

    std::set<std::string> input_symbols;
    const auto validate_values = [&](const std::vector<NamedTensorContract>& values,
                                     bool is_input) {
        for (const NamedTensorContract& named : values) {
            const auto& logical = named.contract.logical().dimensions();
            const auto& physical = named.contract.physical().dimensions();
            const auto& valid = named.contract.valid().dimensions();
            if (logical.empty() || !SameDimensions(logical, physical) ||
                !SameDimensions(logical, valid)) {
                Reject("every value must have a fixed nonzero rank and logical == physical == valid expressions");
            }
            for (const DimExpr& expression : logical) {
                const DirectDimension direct = ReadDirectDimension(expression);
                if (!direct.is_constant && is_input) {
                    input_symbols.insert(direct.symbol);
                }
            }
        }
    };
    validate_values(program.inputs(), true);
    validate_values(program.outputs(), false);

    for (auto& [symbol, value] : bounds) {
        if (!value.has_range) {
            Reject("symbol '" + symbol + "' has no finite range");
        }
        if (!value.has_divisibility) value.divisible_by = 1;
        if (value.divisible_by == 0 || value.lower > value.upper) {
            Reject("symbol '" + symbol + "' has invalid bounds");
        }
        if (!input_symbols.count(symbol)) {
            Reject("symbol '" + symbol + "' is not anchored by a graph input axis");
        }
        ValidateNonemptyDomain(symbol, value);
    }
    return bounds;
}

void AppendExpression(support::CanonicalBytesEncoder* encoder,
                      const DynamicShapeExpr& expression) {
    encoder->IntegerField("expression_kind",
                          static_cast<std::uint8_t>(expression.kind()));
    if (expression.kind() == DynamicShapeExpr::Kind::kConst) {
        encoder->IntegerField("constant", expression.constant());
    } else {
        encoder->IntegerField("input_index", expression.input_index());
        encoder->IntegerField("axis", expression.axis());
    }
}

std::string ContractCanonicalBytes(
    const UnitSemanticKey& representative_unit_semantic_key,
    const std::vector<std::vector<DynamicInputAxisGuard>>& input_guards,
    const std::vector<std::vector<DynamicShapeExpr>>& output_expressions,
    const std::vector<DynamicShapeExpr>& runtime_extent_expressions) {
    support::CanonicalBytesEncoder encoder("dynamic-unit-shape-contract-v1");
    encoder.IntegerField("version", kDynamicUnitShapeContractVersion);
    encoder.Field("representative_unit_semantics",
                  representative_unit_semantic_key.canonical_bytes());
    encoder.IntegerField("input_count", input_guards.size());
    for (std::size_t input = 0; input < input_guards.size(); ++input) {
        encoder.IntegerField("input_index", input);
        encoder.IntegerField("input_rank", input_guards[input].size());
        for (const DynamicInputAxisGuard& guard : input_guards[input]) {
            encoder.IntegerField("axis", guard.axis);
            encoder.IntegerField("lower", guard.lower);
            encoder.IntegerField("upper", guard.upper);
            encoder.IntegerField("divisible_by", guard.divisible_by);
            encoder.BoolField("has_exact", guard.exact.has_value());
            if (guard.exact) encoder.IntegerField("exact", *guard.exact);
            encoder.BoolField("has_equal_to", guard.equal_to.has_value());
            if (guard.equal_to) {
                encoder.IntegerField("equal_input",
                                     guard.equal_to->input_index);
                encoder.IntegerField("equal_axis", guard.equal_to->axis);
            }
        }
    }
    encoder.IntegerField("output_count", output_expressions.size());
    for (std::size_t output = 0; output < output_expressions.size(); ++output) {
        encoder.IntegerField("output_index", output);
        encoder.IntegerField("output_rank", output_expressions[output].size());
        for (const DynamicShapeExpr& expression : output_expressions[output]) {
            AppendExpression(&encoder, expression);
        }
    }
    encoder.IntegerField("runtime_extent_count",
                         runtime_extent_expressions.size());
    for (const DynamicShapeExpr& expression : runtime_extent_expressions) {
        AppendExpression(&encoder, expression);
    }
    return std::move(encoder).Take();
}

std::string ValueName(ValueId id) {
    return "value." + std::to_string(id);
}

bool SameIds(const Array<ValueId>& left, const Array<ValueId>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index] != right[index]) return false;
    }
    return true;
}

std::vector<std::string> ValueNames(const Array<ValueId>& ids) {
    std::vector<std::string> result;
    result.reserve(ids.size());
    for (ValueId id : ids) result.push_back(ValueName(id));
    return result;
}

bool IsSupportedOperation(const std::string& name) {
    return name == "relu" || name == "nn_relu" || name == "sqrt" ||
           name == "add" || name == "mul";
}

int64_t LogicalBoundaryExtent(const DimExpr& expression) {
    const DirectDimension direct = ReadDirectDimension(expression);
    if (direct.is_constant) {
        if (direct.constant >
            static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
            Reject("a constant dimension exceeds int64");
        }
        return static_cast<int64_t>(direct.constant);
    }
    return -1;
}

void ValidateValueContracts(
    const specialization::GraphTemplate& graph,
    const specialization::ExactOracle& representative_oracle,
    const PartitionedGraph& representative,
    const PartitionedGraph& bounded) {
    const auto& static_values = representative.value_graph.values;
    const auto& dynamic_values = bounded.value_graph.values;
    if (static_values.size() != dynamic_values.size() ||
        graph.shape_program().inputs().size() +
                graph.shape_program().outputs().size() !=
            static_values.size() ||
        !SameIds(representative.input_value_ids, bounded.input_value_ids) ||
        !SameIds(representative.constant_value_ids,
                 bounded.constant_value_ids) ||
        !SameIds(representative.output_value_ids, bounded.output_value_ids)) {
        Reject("Function, GraphTemplate, and bounded ValueGraph cardinality differ");
    }

    for (std::size_t index = 0; index < static_values.size(); ++index) {
        const LogicalValueContract& fixed = static_values[index];
        const LogicalValueContract& dynamic = dynamic_values[index];
        if (fixed.id != static_cast<ValueId>(index) || dynamic.id != fixed.id ||
            fixed.origin != dynamic.origin ||
            fixed.origin == LogicalValueOrigin::kConstant ||
            fixed.device != dynamic.device) {
            Reject("representative and bounded logical value identities differ");
        }
        const std::string name = ValueName(fixed.id);
        const NamedTensorContract& named = FindNamed(graph, name);
        const auto* fixed_type = fixed.checked_type.As<TensorTypeNode>();
        const auto* dynamic_type = dynamic.checked_type.As<TensorTypeNode>();
        if (!fixed_type || !dynamic_type ||
            fixed_type->dtype != dynamic_type->dtype ||
            fixed_type->shape.size() != dynamic_type->shape.size() ||
            fixed_type->shape.size() !=
                named.contract.logical().dimensions().size()) {
            Reject("value dtype or fixed rank differs across Function, template, and bounded graph");
        }
        const auto& exact = representative_oracle.profile().Value(name).contract;
        if (ContractDefect(exact) != nullptr || !IsExactContract(exact) ||
            exact.logical.size() != fixed_type->shape.size()) {
            Reject("representative oracle has an invalid value contract");
        }
        for (std::size_t axis = 0; axis < fixed_type->shape.size(); ++axis) {
            if (fixed_type->shape[axis] != exact.logical[axis] ||
                dynamic_type->shape[axis] != LogicalBoundaryExtent(
                    named.contract.logical().dimensions()[axis])) {
                Reject("value extent differs across representative proof, template, and bounded graph");
            }
        }
    }
}

void ValidateUnits(
    const specialization::GraphTemplate& graph,
    const specialization::ExactOracle& representative_oracle,
    const PartitionedGraph& representative,
    const PartitionedGraph& bounded,
    const std::vector<DynamicUnitShapeContract>& contracts) {
    const auto requests = specialization::MakeExactSpecializationRequests(
        graph, representative_oracle);
    if (representative.units.size() != graph.ordered_units().size() ||
        bounded.units.size() != graph.ordered_units().size() ||
        contracts.size() != graph.ordered_units().size() ||
        requests.size() != graph.ordered_units().size()) {
        Reject("partition unit cardinality differs from GraphTemplate");
    }

    for (std::size_t index = 0; index < graph.ordered_units().size(); ++index) {
        const auto& skeleton = graph.ordered_units()[index];
        const PrimitiveUnit& fixed = representative.units[index];
        const PrimitiveUnit& dynamic = bounded.units[index];
        const DynamicUnitShapeContract& contract = contracts[index];
        if (fixed.id != static_cast<PrimitiveUnitId>(index) ||
            dynamic.id != fixed.id || fixed.semantic_key != skeleton.semantic_key ||
            contract.representative_unit_semantic_key() != skeleton.semantic_key ||
            requests[index].unit_semantic_key != skeleton.semantic_key ||
            skeleton.call_locator.value() != ValueName(fixed.output_value_ids[0]) ||
            skeleton.input_value_names !=
                ValueNames(fixed.boundary_input_value_ids) ||
            skeleton.output_value_names != ValueNames(fixed.output_value_ids) ||
            !SameIds(fixed.argument_value_ids, dynamic.argument_value_ids) ||
            !SameIds(fixed.boundary_input_value_ids,
                     dynamic.boundary_input_value_ids) ||
            !SameIds(fixed.output_value_ids, dynamic.output_value_ids)) {
            Reject("partition unit routing or representative semantics differ from GraphTemplate");
        }
        if (!IsSupportedOperation(fixed.call.spec.name) ||
            fixed.call.spec.name != dynamic.call.spec.name ||
            relay::SerializeOperatorSpec(fixed.call.spec) !=
                relay::SerializeOperatorSpec(dynamic.call.spec) ||
            (fixed.call.attrs.defined() ? relay::SerializeAttrs(fixed.call.attrs)
                                        : std::string("<none>")) !=
                (dynamic.call.attrs.defined()
                     ? relay::SerializeAttrs(dynamic.call.attrs)
                     : std::string("<none>")) ||
            fixed.output_value_ids.size() != 1 ||
            contract.local_input_guards().size() !=
                dynamic.boundary_input_value_ids.size() ||
            contract.output_shape_expressions().size() !=
                dynamic.output_value_ids.size()) {
            Reject("bounded unit operator, dtype/rank boundary, or contract mapping differs");
        }
        for (std::size_t input = 0;
             input < dynamic.boundary_input_value_ids.size(); ++input) {
            const auto* type = bounded.value_graph.values[
                static_cast<std::size_t>(dynamic.boundary_input_value_ids[input])]
                                   .checked_type.As<TensorTypeNode>();
            if (!type || contract.local_input_guards()[input].size() !=
                             type->shape.size()) {
                Reject("unit-local input guard rank differs from the partition value");
            }
        }
        for (std::size_t output = 0;
             output < dynamic.output_value_ids.size(); ++output) {
            const auto* type = bounded.value_graph.values[
                static_cast<std::size_t>(dynamic.output_value_ids[output])]
                                   .checked_type.As<TensorTypeNode>();
            if (!type || contract.output_shape_expressions()[output].size() !=
                             type->shape.size()) {
                Reject("unit-local output expression rank differs from the partition value");
            }
        }
    }
}

}  // namespace

DynamicShapeExpr::DynamicShapeExpr(Kind kind, std::uint64_t constant,
                                   std::size_t input_index,
                                   std::size_t axis)
    : kind_(kind), constant_(constant), input_index_(input_index), axis_(axis) {}

DynamicShapeExpr DynamicShapeExpr::Const(std::uint64_t value) {
    return DynamicShapeExpr(Kind::kConst, value, 0, 0);
}

DynamicShapeExpr DynamicShapeExpr::InputAxis(std::size_t input_index,
                                              std::size_t axis) {
    return DynamicShapeExpr(Kind::kInputAxis, 0, input_index, axis);
}

DynamicShapeExpr::Kind DynamicShapeExpr::kind() const noexcept { return kind_; }

std::uint64_t DynamicShapeExpr::constant() const {
    if (kind_ != Kind::kConst) {
        throw std::logic_error("DynamicShapeExpr is not a constant");
    }
    return constant_;
}

std::size_t DynamicShapeExpr::input_index() const {
    if (kind_ != Kind::kInputAxis) {
        throw std::logic_error("DynamicShapeExpr is not an input axis");
    }
    return input_index_;
}

std::size_t DynamicShapeExpr::axis() const {
    if (kind_ != Kind::kInputAxis) {
        throw std::logic_error("DynamicShapeExpr is not an input axis");
    }
    return axis_;
}

bool DynamicShapeExpr::operator==(const DynamicShapeExpr& other) const noexcept {
    return kind_ == other.kind_ && constant_ == other.constant_ &&
           input_index_ == other.input_index_ && axis_ == other.axis_;
}

DynamicUnitShapeContract::DynamicUnitShapeContract(
    UnitSemanticKey representative_unit_semantic_key,
    std::vector<std::vector<DynamicInputAxisGuard>> local_input_guards,
    std::vector<std::vector<DynamicShapeExpr>> output_shape_expressions,
    std::vector<DynamicShapeExpr> runtime_extent_expressions)
    : representative_unit_semantic_key_(
          std::move(representative_unit_semantic_key)),
      local_input_guards_(std::move(local_input_guards)),
      output_shape_expressions_(std::move(output_shape_expressions)),
      runtime_extent_expressions_(std::move(runtime_extent_expressions)) {
    if (!representative_unit_semantic_key_.defined() ||
        local_input_guards_.empty() || output_shape_expressions_.size() != 1 ||
        runtime_extent_expressions_ != output_shape_expressions_.front()) {
        Reject("a DynamicUnitShapeContract is incomplete");
    }
    canonical_bytes_ = ContractCanonicalBytes(
        representative_unit_semantic_key_, local_input_guards_,
        output_shape_expressions_, runtime_extent_expressions_);
}

std::uint32_t DynamicUnitShapeContract::version() const noexcept {
    return version_;
}

const UnitSemanticKey&
DynamicUnitShapeContract::representative_unit_semantic_key() const noexcept {
    return representative_unit_semantic_key_;
}

const std::vector<std::vector<DynamicInputAxisGuard>>&
DynamicUnitShapeContract::local_input_guards() const noexcept {
    return local_input_guards_;
}

const std::vector<std::vector<DynamicShapeExpr>>&
DynamicUnitShapeContract::output_shape_expressions() const noexcept {
    return output_shape_expressions_;
}

const std::vector<DynamicShapeExpr>&
DynamicUnitShapeContract::runtime_extent_expressions() const noexcept {
    return runtime_extent_expressions_;
}

const std::string& DynamicUnitShapeContract::canonical_bytes() const noexcept {
    return canonical_bytes_;
}

DynamicUnitShapeContract BuildDynamicUnitShapeContract(
    const specialization::GraphTemplate& graph,
    std::size_t ordered_unit_index) {
    const BoundsBySymbol bounds = AnalyzeTemplate(graph);
    if (ordered_unit_index >= graph.ordered_units().size()) {
        Reject("ordered unit index is out of range");
    }
    const specialization::UnitSkeleton& unit =
        graph.ordered_units()[ordered_unit_index];
    if (unit.input_value_names.empty() || unit.output_value_names.size() != 1) {
        Reject("bounded v1 requires unit inputs and exactly one output");
    }

    std::map<std::string, DynamicInputAxisReference> anchors;
    std::vector<std::vector<DynamicInputAxisGuard>> input_guards;
    input_guards.reserve(unit.input_value_names.size());
    for (std::size_t input = 0; input < unit.input_value_names.size(); ++input) {
        const auto& dimensions =
            FindNamed(graph, unit.input_value_names[input])
                .contract.logical().dimensions();
        std::vector<DynamicInputAxisGuard> guards;
        guards.reserve(dimensions.size());
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
            const DirectDimension direct = ReadDirectDimension(dimensions[axis]);
            DynamicInputAxisGuard guard;
            guard.axis = axis;
            if (direct.is_constant) {
                guard.lower = direct.constant;
                guard.upper = direct.constant;
                guard.exact = direct.constant;
            } else {
                const SymbolBounds& symbol_bounds = bounds.at(direct.symbol);
                guard.lower = symbol_bounds.lower;
                guard.upper = symbol_bounds.upper;
                guard.divisible_by = symbol_bounds.divisible_by;
                const DynamicInputAxisReference current{input, axis};
                const auto inserted = anchors.emplace(direct.symbol, current);
                if (!inserted.second) {
                    if (inserted.first->second.input_index >= input) {
                        Reject("a repeated symbol must reference a strictly prior unit input");
                    }
                    guard.equal_to = inserted.first->second;
                }
            }
            guards.push_back(std::move(guard));
        }
        input_guards.push_back(std::move(guards));
    }

    std::vector<std::vector<DynamicShapeExpr>> output_expressions;
    for (const std::string& output_name : unit.output_value_names) {
        const auto& dimensions = FindNamed(graph, output_name)
                                     .contract.logical().dimensions();
        std::vector<DynamicShapeExpr> expressions;
        expressions.reserve(dimensions.size());
        for (const DimExpr& dimension : dimensions) {
            const DirectDimension direct = ReadDirectDimension(dimension);
            if (direct.is_constant) {
                expressions.push_back(DynamicShapeExpr::Const(direct.constant));
                continue;
            }
            const auto anchor = anchors.find(direct.symbol);
            if (anchor == anchors.end()) {
                Reject("a unit output symbol is not anchored by a local input axis");
            }
            expressions.push_back(DynamicShapeExpr::InputAxis(
                anchor->second.input_index, anchor->second.axis));
        }
        output_expressions.push_back(std::move(expressions));
    }
    std::vector<DynamicShapeExpr> runtime_extent_expressions =
        output_expressions.front();
    return DynamicUnitShapeContract(
        unit.semantic_key, std::move(input_guards),
        std::move(output_expressions),
        std::move(runtime_extent_expressions));
}

std::vector<DynamicUnitShapeContract> BuildDynamicUnitShapeContracts(
    const specialization::GraphTemplate& graph) {
    (void)AnalyzeTemplate(graph);
    if (graph.ordered_units().empty()) {
        Reject("a bounded graph requires at least one unit");
    }
    std::vector<DynamicUnitShapeContract> result;
    result.reserve(graph.ordered_units().size());
    for (std::size_t index = 0; index < graph.ordered_units().size(); ++index) {
        result.push_back(BuildDynamicUnitShapeContract(graph, index));
    }
    return result;
}

BoundedCompilePreparation::BoundedCompilePreparation(
    restricted::BoundedCompileRequest request,
    PartitionedGraph partitioned_graph,
    std::vector<DynamicUnitShapeContract> unit_shape_contracts)
    : request_(std::move(request)),
      partitioned_graph_(std::move(partitioned_graph)),
      unit_shape_contracts_(std::move(unit_shape_contracts)) {}

std::uint32_t BoundedCompilePreparation::version() const noexcept {
    return version_;
}

const restricted::BoundedCompileRequest&
BoundedCompilePreparation::request() const noexcept {
    return request_;
}

const PartitionedGraph&
BoundedCompilePreparation::partitioned_graph() const noexcept {
    return partitioned_graph_;
}

const std::vector<DynamicUnitShapeContract>&
BoundedCompilePreparation::unit_shape_contracts() const noexcept {
    return unit_shape_contracts_;
}

BoundedCompilePreparation PrepareBoundedCompile(
    const restricted::BoundedCompileRequest& request) {
    if (request.applicability_version() !=
        restricted::kBoundedCompileApplicabilityVersion) {
        Reject("bounded applicability version is unsupported");
    }
    request.compile_config().Validate();
    const Target& target = request.target();
    if (!target.defined() || target->kind != "llvm" ||
        target->device_type != kCPU || target->device_id != 0 ||
        CanonicalTargetSnapshot(target) != CanonicalTargetSnapshot(
            request.compile_config()->target)) {
        Reject("bounded v1 requires the request's immutable CPU/LLVM target snapshot");
    }
    const specialization::GraphTemplate& graph = request.graph_template();
    graph.Verify();
    if (Compiler::BuildGraphSemanticKey(request.representative()) !=
        graph.key()) {
        Reject("representative Function does not own the GraphTemplate semantics");
    }
    (void)specialization::MakeExactSpecializationRequests(
        graph, request.representative_oracle());

    const Device device(target->device_type, target->device_id);
    Function representative =
        relay::InferTypePass(request.representative());
    PartitionedGraph representative_partition = PartitionValueGraph(
        BuildValueGraph(representative, device));

    Function logical_boundary =
        relay::InferTypePass(request.logical_boundary_function());
    const BoundedLogicalShapeAdmission admission =
        BoundedCompilePreparationAccess::MintLogicalShapeAdmission();
    PartitionedGraph bounded_partition = PartitionValueGraph(
        BuildBoundedValueGraph(logical_boundary, device, admission));

    std::vector<DynamicUnitShapeContract> contracts =
        BuildDynamicUnitShapeContracts(graph);
    ValidateValueContracts(graph, request.representative_oracle(),
                           representative_partition, bounded_partition);
    ValidateUnits(graph, request.representative_oracle(),
                  representative_partition, bounded_partition, contracts);
    return BoundedCompilePreparation(
        request, std::move(bounded_partition), std::move(contracts));
}

}  // namespace kxc::api::internal
