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
#include "../internal/relay_program.h"
#include "runtime/internal/module_invocation_contract.h"
#include "kxc/relay/transforms/infer_type.h"
#include "shape_value_resolver.h"
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
    std::uint64_t offset{0};
};

// Recognize only Symbol + nonnegative constant. DimExpr owns evaluation and
// canonical equality; sampling alone never establishes this proof.
DirectDimension ReadOffsetDimension(const DimExpr& expression) {
    if (expression.kind() == DimExpr::Kind::kConst) {
        return {true, static_cast<std::uint64_t>(expression.Evaluate(specialization::BindingSet())), {}};
    }
    const auto symbols = expression.Symbols();
    if (symbols.size() != 1) Reject("a derived dimension must be Symbol plus a nonnegative constant");
    const auto offset = shape_resolution::ProveNonnegativeConstantOffset(expression,DimExpr::Symbol(symbols[0]));
    if (!offset) {
        Reject("a derived dimension must be Symbol plus a nonnegative constant");
    }
    return {false, 0, symbols[0], static_cast<std::uint64_t>(*offset)};
}

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
            if (!SameDimensions(logical, physical) ||
                !SameDimensions(logical, valid)) {
                Reject("every value must have a fixed rank and logical == physical == valid expressions (value '" +
                       named.name + "', logical rank " +
                       std::to_string(logical.size()) + ", physical rank " +
                       std::to_string(physical.size()) + ", valid rank " +
                       std::to_string(valid.size()) + ")");
            }
            for (const DimExpr& expression : logical) {
                const DirectDimension direct = is_input ? ReadDirectDimension(expression)
                                                        : ReadOffsetDimension(expression);
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
    std::vector<kxc::shape::experimental::v1::Binding> upper_bindings;
    for (const auto& [symbol, value] : bounds) {
        upper_bindings.push_back({symbol, static_cast<int64_t>(value.upper)});
    }
    const specialization::BindingSet uppers(std::move(upper_bindings));
    for (const auto& named : program.outputs()) {
        for (const auto& dimension : named.contract.logical().dimensions()) {
            if (dimension.kind() == DimExpr::Kind::kAdd &&
                dimension.Evaluate(uppers) > std::numeric_limits<int32_t>::max()) {
                Reject("derived dimension exceeds the int32 loop domain");
            }
        }
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
        encoder->IntegerField("offset", expression.offset());
    }
}

std::string ContractCanonicalBytes(
    const UnitSemanticKey& representative_unit_semantic_key,
    const std::vector<std::vector<DynamicInputAxisGuard>>& input_guards,
    const std::vector<std::vector<DynamicShapeExpr>>& output_expressions,
    const std::vector<std::vector<DynamicShapeExpr>>& output_value_expressions,
    const std::vector<DynamicShapeExpr>& runtime_extent_expressions) {
    support::CanonicalBytesEncoder encoder("dynamic-unit-shape-contract-v3");
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
        encoder.IntegerField("output_value_count",
                             output_value_expressions[output].size());
        for (const DynamicShapeExpr& expression :
             output_value_expressions[output]) {
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
    return name == "relu" || name == "nn_relu" || name == "sqrt" || name == "sigmoid" ||
           name == "neg" || name == "slice" || name == "concatenate" ||
           name == "add" || name == "mul" || name == "divide" || name == "pow" ||
           name == "cast" || name == "reduce_mean" || name == "matmul" ||
           name == "transpose" || name == "softmax" || name == "masked_softmax" || name == "shape_of" || name == "gather" ||
           name == "shape_expr" || name == "reshape_dynamic" ||
           name == "constant_of_shape" || name == "trilu" ||
           name == "expand_dynamic" || name == "squeeze" || name == "unsqueeze";
}

// M3 形状值算子：输出是物化受限表达式的 int64 行向量，其元素来源与输出
// 形状是两个不同的事实，所以 unit 合同要分别记录 shape 表达式与 value 表达式。
bool ProducesShapeValue(const std::string& name) {
    return name == "shape_of" || name == "shape_expr";
}

// shape_of 的 value 表达式可从输入轴推导；shape_expr 必须由解析器覆盖提供。
bool DerivesValueExpressionsFromInput(const std::string& name) {
    return name == "shape_of";
}

int64_t LogicalBoundaryExtent(const DimExpr& expression) {
    const DirectDimension direct = ReadOffsetDimension(expression);
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
            Reject("value dtype or fixed rank differs across Function, template, and bounded graph at " +
                   name + " (fixed=" + TypeToString(fixed.checked_type) +
                   ", bounded=" + TypeToString(dynamic.checked_type) +
                   ", contract_rank=" +
                   std::to_string(named.contract.logical().dimensions().size()) + ")");
        }
        const auto& exact = representative_oracle.profile().Value(name).contract;
        if (ContractDefect(exact) != nullptr || !IsExactContract(exact) ||
            exact.logical.size() != fixed_type->shape.size()) {
            Reject("representative oracle has an invalid value contract");
        }
        for (std::size_t axis = 0; axis < fixed_type->shape.size(); ++axis) {
            if (fixed.origin == LogicalValueOrigin::kConstant &&
                named.contract.logical().dimensions()[axis].kind() != DimExpr::Kind::kConst) {
                Reject("constant values cannot carry dynamic shape expressions");
            }
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
            if (ProducesShapeValue(fixed.call.spec.name) &&
                (type->shape.size() != 1 || type->dtype != "int64")) {
                Reject("a shape-value output must be a rank-1 int64 vector");
            }
        }
    }
}

std::vector<runtime::GraphInputAxisGuard> BuildGraphInputGuards(
    const specialization::GraphTemplate& graph,
    const PartitionedGraph& bounded) {
    const BoundsBySymbol bounds = AnalyzeTemplate(graph);
    std::map<std::string, runtime::GraphInputAxisReference> anchors;
    std::vector<runtime::GraphInputAxisGuard> guards;
    for (std::size_t input = 0; input < bounded.input_value_ids.size(); ++input) {
        const ValueId value_id = bounded.input_value_ids[input];
        if (value_id < 0 ||
            static_cast<std::size_t>(value_id) >=
                bounded.value_graph.values.size()) {
            Reject("a graph input value id is out of range");
        }
        const auto* type = bounded.value_graph.values[
            static_cast<std::size_t>(value_id)].checked_type.As<TensorTypeNode>();
        const auto& dimensions =
            FindNamed(graph, ValueName(value_id)).contract.logical().dimensions();
        if (!type || type->shape.size() != dimensions.size()) {
            Reject("a graph input rank differs from its ShapeProgram contract");
        }
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
            const DirectDimension direct = ReadDirectDimension(dimensions[axis]);
            if (type->shape[axis] != -1) {
                if (!direct.is_constant ||
                    direct.constant !=
                        static_cast<std::uint64_t>(type->shape[axis])) {
                    Reject("a static graph input axis differs from its ShapeProgram contract");
                }
                continue;
            }
            if (direct.is_constant) {
                Reject("a wildcard graph input axis lacks a direct Symbol");
            }
            const SymbolBounds& symbol_bounds = bounds.at(direct.symbol);
            runtime::GraphInputAxisGuard guard;
            guard.input_index = input;
            guard.axis = axis;
            guard.lower = static_cast<std::int64_t>(symbol_bounds.lower);
            guard.upper = static_cast<std::int64_t>(symbol_bounds.upper);
            guard.divisible_by =
                static_cast<std::int64_t>(symbol_bounds.divisible_by);
            const runtime::GraphInputAxisReference current{input, axis};
            const auto inserted = anchors.emplace(direct.symbol, current);
            if (!inserted.second) guard.equal_to = inserted.first->second;
            guards.push_back(std::move(guard));
        }
    }
    return guards;
}

ModuleShapeExpr ToModuleShapeExpr(const DynamicShapeExpr& expression) {
    if (expression.kind() == DynamicShapeExpr::Kind::kConst) {
        return ModuleShapeExpr::Const(expression.constant());
    }
    auto axis = ModuleShapeExpr::InputAxis(expression.input_index(), expression.axis());
    return expression.offset() == 0 ? axis
        : ModuleShapeExpr::Add(std::move(axis), ModuleShapeExpr::Const(expression.offset()));
}

const LogicalValueContract& ContractValue(
    const std::vector<LogicalValueContract>& values, ValueId value_id,
    const char* context) {
    if (value_id < 0 ||
        static_cast<std::size_t>(value_id) >= values.size() ||
        values[static_cast<std::size_t>(value_id)].id != value_id) {
        Reject(std::string(context) + " references an invalid value id");
    }
    return values[static_cast<std::size_t>(value_id)];
}

std::size_t MaximumOutputBytes(
    const std::vector<DynamicShapeExpr>& expressions,
    const std::vector<std::vector<DynamicInputAxisGuard>>& input_guards,
    DLDataType dtype) {
    if (dtype.bits == 0 || dtype.bits % 8 != 0 || dtype.lanes == 0) {
        Reject("a dynamic output dtype is not byte addressable");
    }
    std::size_t elements = 1;
    for (const DynamicShapeExpr& expression : expressions) {
        std::uint64_t extent = 0;
        if (expression.kind() == DynamicShapeExpr::Kind::kConst) {
            extent = expression.constant();
        } else {
            if (expression.input_index() >= input_guards.size() ||
                expression.axis() >=
                    input_guards[expression.input_index()].size()) {
                Reject("a dynamic output expression references an unknown input axis");
            }
            extent = input_guards[expression.input_index()]
                                 [expression.axis()].upper;
            if (extent > std::numeric_limits<std::uint64_t>::max() - expression.offset()) {
                Reject("a dynamic output extent overflows uint64");
            }
            extent += expression.offset();
        }
        if (extent > std::numeric_limits<std::size_t>::max() ||
            (extent != 0 &&
             elements > std::numeric_limits<std::size_t>::max() /
                            static_cast<std::size_t>(extent))) {
            Reject("a dynamic output maximum element count overflows size_t");
        }
        elements *= static_cast<std::size_t>(extent);
    }
    const std::size_t item_bytes =
        static_cast<std::size_t>(dtype.bits / 8) * dtype.lanes;
    if (item_bytes == 0 ||
        elements > std::numeric_limits<std::size_t>::max() / item_bytes) {
        Reject("a dynamic output maximum byte count overflows size_t");
    }
    return elements * item_bytes;
}

}  // namespace

// 有序 extent ABI：先按输出轴顺序收录输出 shape 表达式中的动态项，再按
// 输入/轴顺序收录尚未出现的动态输入轴（kernel 体读取它们做索引），最后
// 收录形状值输出尚未出现的动态元素表达式。去重保持首次出现顺序，因此
// 既有 elementwise 单元的 ABI 顺序不变。
std::vector<DynamicShapeExpr> OrderedRuntimeExtentExpressions(
    const UnitSemanticKey& semantic_key,
    const std::vector<std::vector<DynamicInputAxisGuard>>& input_guards,
    const std::vector<std::vector<DynamicShapeExpr>>& output_shape_expressions,
    const std::vector<std::vector<DynamicShapeExpr>>&
        output_value_expressions) {
    (void)semantic_key;
    std::vector<DynamicShapeExpr> extents;
    const auto append = [&extents](const DynamicShapeExpr& value) {
        if (value.kind() != DynamicShapeExpr::Kind::kInputAxis) return;
        const auto expression = DynamicShapeExpr::InputAxis(value.input_index(), value.axis());
        for (const DynamicShapeExpr& existing : extents) {
            if (existing == expression) return;
        }
        extents.push_back(expression);
    };
    for (const DynamicShapeExpr& expression :
         output_shape_expressions.front()) {
        append(expression);
    }
    for (std::size_t input = 0; input < input_guards.size(); ++input) {
        for (const DynamicInputAxisGuard& guard : input_guards[input]) {
            if (guard.exact) continue;
            const DynamicInputAxisReference reference =
                guard.equal_to.value_or(
                    DynamicInputAxisReference{input, guard.axis});
            append(DynamicShapeExpr::InputAxis(reference.input_index,
                                               reference.axis));
        }
    }
    for (const DynamicShapeExpr& expression :
         output_value_expressions.front()) {
        append(expression);
    }
    return extents;
}

DynamicShapeExpr::DynamicShapeExpr(Kind kind, std::uint64_t constant,
                                   std::size_t input_index,
                                   std::size_t axis, std::uint64_t offset)
    : kind_(kind), constant_(constant), input_index_(input_index), axis_(axis), offset_(offset) {}

DynamicShapeExpr DynamicShapeExpr::Const(std::uint64_t value) {
    return DynamicShapeExpr(Kind::kConst, value, 0, 0);
}

DynamicShapeExpr DynamicShapeExpr::InputAxis(std::size_t input_index,
                                              std::size_t axis, std::uint64_t offset) {
    return DynamicShapeExpr(Kind::kInputAxis, 0, input_index, axis, offset);
}

std::uint64_t DynamicShapeExpr::offset() const noexcept { return offset_; }

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
           input_index_ == other.input_index_ && axis_ == other.axis_ && offset_ == other.offset_;
}

DynamicUnitShapeContract::DynamicUnitShapeContract(
    UnitSemanticKey representative_unit_semantic_key,
    std::vector<std::vector<DynamicInputAxisGuard>> local_input_guards,
    std::vector<std::vector<DynamicShapeExpr>> output_shape_expressions,
    std::vector<std::vector<DynamicShapeExpr>> output_value_expressions,
    std::vector<DynamicShapeExpr> runtime_extent_expressions)
    : representative_unit_semantic_key_(
          std::move(representative_unit_semantic_key)),
      local_input_guards_(std::move(local_input_guards)),
      output_shape_expressions_(std::move(output_shape_expressions)),
      output_value_expressions_(std::move(output_value_expressions)),
      runtime_extent_expressions_(std::move(runtime_extent_expressions)) {
    std::vector<DynamicShapeExpr> expected_runtime_extents =
        OrderedRuntimeExtentExpressions(
            representative_unit_semantic_key_, local_input_guards_,
            output_shape_expressions_, output_value_expressions_);
    if (!representative_unit_semantic_key_.defined() ||
        local_input_guards_.empty() || output_shape_expressions_.size() != 1 ||
        output_value_expressions_.size() !=
            output_shape_expressions_.size() ||
        runtime_extent_expressions_ != expected_runtime_extents) {
        Reject("a DynamicUnitShapeContract is incomplete");
    }
    canonical_bytes_ = ContractCanonicalBytes(
        representative_unit_semantic_key_, local_input_guards_,
        output_shape_expressions_, output_value_expressions_,
        runtime_extent_expressions_);
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

const std::vector<std::vector<DynamicShapeExpr>>&
DynamicUnitShapeContract::output_value_expressions() const noexcept {
    return output_value_expressions_;
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
    std::size_t ordered_unit_index, const std::string& operator_name,
    const std::optional<std::vector<DynamicShapeExpr>>& value_expression_override) {
    if (!IsSupportedOperation(operator_name)) {
        Reject("bounded unit operator is unsupported: " + operator_name);
    }
    const BoundsBySymbol bounds = AnalyzeTemplate(graph);
    if (ordered_unit_index >= graph.ordered_units().size()) {
        Reject("ordered unit index is out of range");
    }
    const specialization::UnitSkeleton& unit =
        graph.ordered_units()[ordered_unit_index];
    if (unit.input_value_names.empty() || unit.output_value_names.size() != 1) {
        Reject("bounded v1 requires unit inputs and exactly one output");
    }

    std::map<std::string, std::pair<DirectDimension, DynamicInputAxisReference>> anchors;
    std::vector<std::vector<DynamicInputAxisGuard>> input_guards;
    input_guards.reserve(unit.input_value_names.size());
    for (std::size_t input = 0; input < unit.input_value_names.size(); ++input) {
        const auto& dimensions =
            FindNamed(graph, unit.input_value_names[input])
                .contract.logical().dimensions();
        std::vector<DynamicInputAxisGuard> guards;
        guards.reserve(dimensions.size());
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
            const DirectDimension direct = ReadOffsetDimension(dimensions[axis]);
            DynamicInputAxisGuard guard;
            guard.axis = axis;
            if (direct.is_constant) {
                guard.lower = direct.constant;
                guard.upper = direct.constant;
                guard.exact = direct.constant;
            } else {
                const SymbolBounds& symbol_bounds = bounds.at(direct.symbol);
                guard.lower = dimensions[axis].Evaluate(specialization::BindingSet(
                    {{direct.symbol, static_cast<int64_t>(symbol_bounds.lower)}}));
                guard.upper = dimensions[axis].Evaluate(specialization::BindingSet(
                    {{direct.symbol, static_cast<int64_t>(symbol_bounds.upper)}}));
                guard.divisible_by = direct.offset == 0 ? symbol_bounds.divisible_by : 1;
                const DynamicInputAxisReference current{input, axis};
                const auto inserted = anchors.emplace(dimensions[axis].CanonicalString(), std::make_pair(direct, current));
                if (!inserted.second) {
                    const auto& prior = inserted.first->second.second;
                    if (prior.input_index > input ||
                        (prior.input_index == input && prior.axis >= axis)) {
                        Reject("a repeated symbol must reference a prior unit input axis");
                    }
                    guard.equal_to = prior;
                }
            }
            guards.push_back(std::move(guard));
        }
        input_guards.push_back(std::move(guards));
    }

    const auto anchored_expression =
        [&anchors](const DirectDimension& direct) -> DynamicShapeExpr {
        if (direct.is_constant) {
            return DynamicShapeExpr::Const(direct.constant);
        }
        // Prefer an identical local dimension, then a smaller offset of the
        // same symbol. No subtraction or inverse input binding is admitted.
        for (bool exact : {true, false}) {
            for (const auto& item : anchors) {
                const auto& [dimension, reference] = item.second;
                if (dimension.symbol == direct.symbol && dimension.offset <= direct.offset &&
                    (!exact || dimension.offset == direct.offset)) {
                    return DynamicShapeExpr::InputAxis(reference.input_index, reference.axis,
                        direct.offset - dimension.offset);
                }
            }
        }
        Reject("a unit output symbol is not anchored by a local input axis");
    };

    std::vector<std::vector<DynamicShapeExpr>> output_expressions;
    for (const std::string& output_name : unit.output_value_names) {
        const auto& dimensions = FindNamed(graph, output_name)
                                     .contract.logical().dimensions();
        std::vector<DynamicShapeExpr> expressions;
        expressions.reserve(dimensions.size());
        for (const DimExpr& dimension : dimensions) {
            const DirectDimension direct = ReadOffsetDimension(dimension);
            const auto expression = anchored_expression(direct);
            if (expression.offset() != 0 && operator_name != "concatenate" &&
                operator_name != "constant_of_shape" && operator_name != "reshape_dynamic" &&
                operator_name != "expand_dynamic") {
                Reject("a constant offset in an output dimension requires concatenate or an explicit target shape");
            }
            expressions.push_back(expression);
        }
        output_expressions.push_back(std::move(expressions));
    }

    // 形状值输出：物化受限表达式的 int64 行向量。shape_of 的元素表达式
    // 直接取输入轴；shape_expr 必须由解析器覆盖提供（唯一表达式来源）。
    std::vector<std::vector<DynamicShapeExpr>> output_value_expressions(
        unit.output_value_names.size());
    if (ProducesShapeValue(operator_name)) {
        if (unit.input_value_names.size() != 1) {
            Reject("a shape-value producer expects exactly one data input");
        }
        std::vector<DynamicShapeExpr>& value_expressions =
            output_value_expressions.front();
        if (value_expression_override) {
            value_expressions = *value_expression_override;
        } else if (DerivesValueExpressionsFromInput(operator_name)) {
            const auto& input_dimensions =
                FindNamed(graph, unit.input_value_names[0])
                    .contract.logical().dimensions();
            value_expressions.reserve(input_dimensions.size());
            for (const DimExpr& dimension : input_dimensions) {
                value_expressions.push_back(
                    anchored_expression(ReadOffsetDimension(dimension)));
            }
            const auto& output_dimensions =
                FindNamed(graph, unit.output_value_names[0])
                    .contract.logical().dimensions();
            if (output_dimensions.size() != 1 ||
                output_dimensions.front() != DimExpr::Const(static_cast<int64_t>(
                                                  input_dimensions.size()))) {
                Reject("a shape_of output must be a rank-1 vector with one "
                       "element per input axis");
            }
        } else {
            Reject("a " + operator_name +
                   " unit requires resolver-provided value expressions");
        }
        const auto& output_dimensions =
            FindNamed(graph, unit.output_value_names[0])
                .contract.logical().dimensions();
        if (output_dimensions.size() != 1 ||
            value_expressions.size() !=
                static_cast<size_t>(output_dimensions.front().Evaluate(
                    specialization::BindingSet()))) {
            Reject("a shape-value output must be a rank-1 vector whose length "
                   "equals its value expression count");
        }
    }

    const std::vector<DynamicShapeExpr> runtime_extent_expressions =
        OrderedRuntimeExtentExpressions(
            unit.semantic_key, input_guards, output_expressions,
            output_value_expressions);
    return DynamicUnitShapeContract(
        unit.semantic_key, std::move(input_guards),
        std::move(output_expressions), std::move(output_value_expressions),
        std::move(runtime_extent_expressions));
}

std::vector<DynamicUnitShapeContract> BuildDynamicUnitShapeContracts(
    const specialization::GraphTemplate& graph,
    const std::vector<std::string>& operator_names,
    const std::vector<std::optional<std::vector<DynamicShapeExpr>>>&
        value_expression_overrides) {
    (void)AnalyzeTemplate(graph);
    if (graph.ordered_units().empty()) {
        Reject("a bounded graph requires at least one unit");
    }
    if (operator_names.size() != graph.ordered_units().size() ||
        (!value_expression_overrides.empty() &&
         value_expression_overrides.size() != graph.ordered_units().size())) {
        Reject("operator name count differs from the ordered unit sequence");
    }
    std::vector<DynamicUnitShapeContract> result;
    result.reserve(graph.ordered_units().size());
    for (std::size_t index = 0; index < graph.ordered_units().size(); ++index) {
        std::optional<std::vector<DynamicShapeExpr>> override;
        if (!value_expression_overrides.empty()) {
            override = value_expression_overrides[index];
        }
        result.push_back(BuildDynamicUnitShapeContract(
            graph, index, operator_names[index], override));
    }
    return result;
}

BoundedCompilePreparation::BoundedCompilePreparation(
    restricted::BoundedCompileRequest request,
    PartitionedGraph partitioned_graph,
    std::vector<DynamicUnitShapeContract> unit_shape_contracts,
    std::vector<runtime::GraphInputAxisGuard> graph_input_guards)
    : request_(std::move(request)),
      partitioned_graph_(std::move(partitioned_graph)),
      unit_shape_contracts_(std::move(unit_shape_contracts)),
      graph_input_guards_(std::move(graph_input_guards)) {}

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

const std::vector<runtime::GraphInputAxisGuard>&
BoundedCompilePreparation::graph_input_guards() const noexcept {
    return graph_input_guards_;
}

BoundedCompilePreparation PrepareBoundedCompile(
    const restricted::BoundedCompileRequest& request) {
    if (request.applicability_version() !=
        restricted::kBoundedCompileApplicabilityVersion) {
        Reject("bounded applicability version is unsupported");
    }
    request.compile_config().Validate();
    const Target& target = request.target();
    if (!target.defined() ||
        !((target->kind == "llvm" && target->device_type == kCPU && target->device_id == 0) ||
          (target->kind == "cuda" && target->device_type == kCUDA)) ||
        CanonicalTargetSnapshot(target) != CanonicalTargetSnapshot(
            request.compile_config()->target)) {
        Reject("bounded compilation requires the request's immutable CPU:0/LLVM or CUDA target snapshot");
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
    // Use the same declared production pipeline as exact preparation. ANF
    // hoists calls before atomic constants at their consumers, so inferring
    // types on the original DAG alone can assign different ValueIds.
    Function representative = PrepareRelayProgram(
        request.representative(), request.compile_config(),
        ControlFlowPolicy::StaticOnly()).typed_anf();
    PartitionedGraph representative_partition = PartitionValueGraph(
        BuildValueGraph(representative, device));

    Function logical_boundary = PrepareRelayProgram(
        request.logical_boundary_function(), request.compile_config(),
        ControlFlowPolicy::StaticOnly()).typed_anf();
    const BoundedLogicalShapeAdmission admission =
        BoundedCompilePreparationAccess::MintLogicalShapeAdmission();
    PartitionedGraph bounded_partition = PartitionValueGraph(
        BuildBoundedValueGraph(logical_boundary, device, admission));

    std::vector<DynamicUnitShapeContract> contracts;
    {
        if (representative_partition.units.size() !=
            graph.ordered_units().size()) {
            Reject("partition unit cardinality differs from GraphTemplate");
        }
        std::vector<std::string> operator_names;
        operator_names.reserve(representative_partition.units.size());
        for (const PrimitiveUnit& unit : representative_partition.units) {
            operator_names.push_back(std::string(unit.call.spec.name));
        }
        contracts = BuildDynamicUnitShapeContracts(
            graph, operator_names,
            DecodeValueExpressionOverrides(
                request.unit_value_expressions()));
    }
    ValidateValueContracts(graph, request.representative_oracle(),
                           representative_partition, bounded_partition);
    ValidateUnits(graph, request.representative_oracle(),
                  representative_partition, bounded_partition, contracts);
    std::vector<runtime::GraphInputAxisGuard> graph_input_guards =
        BuildGraphInputGuards(graph, bounded_partition);
    return BoundedCompilePreparation(
        request, std::move(bounded_partition), std::move(contracts),
        std::move(graph_input_guards));
}

std::shared_ptr<const ModuleInvocationContract>
BuildDynamicModuleInvocationContract(
    const DynamicUnitShapeContract& shape_contract,
    const PrimitiveUnit& unit,
    const std::vector<LogicalValueContract>& values) {
    if (shape_contract.local_input_guards().size() !=
            unit.boundary_input_value_ids.size() ||
        shape_contract.output_shape_expressions().size() !=
            unit.output_value_ids.size()) {
        Reject("a unit and its dynamic invocation contract have different arity");
    }

    std::vector<ModuleInputContract> inputs;
    inputs.reserve(unit.boundary_input_value_ids.size());
    for (std::size_t input = 0;
         input < unit.boundary_input_value_ids.size(); ++input) {
        const LogicalValueContract& value = ContractValue(
            values, unit.boundary_input_value_ids[input],
            "a dynamic module input");
        if (value.origin == LogicalValueOrigin::kConstant) {
            // Constants are an ABI role, not Invoke data inputs. The existing
            // ValueGraph boundary is [data inputs][constants], so input-axis
            // references retain their indices when the suffix is omitted.
            const auto* type = value.checked_type.As<TensorTypeNode>();
            const auto& guards = shape_contract.local_input_guards()[input];
            if (!type || guards.size() != type->shape.size()) {
                Reject("constant guard rank differs from its frozen tensor");
            }
            for (size_t axis = 0; axis < guards.size(); ++axis) {
                if (type->shape[axis] < 0 || !guards[axis].exact || guards[axis].equal_to ||
                    *guards[axis].exact != static_cast<uint64_t>(type->shape[axis])) {
                    Reject("constant input guards must match static payload extents");
                }
            }
            continue;
        }
        if (input != inputs.size()) Reject("constant inputs must follow data inputs in the unit ABI");
        ModuleInputContract module_input;
        for (const DynamicInputAxisGuard& source :
             shape_contract.local_input_guards()[input]) {
            ModuleAxisGuard guard;
            guard.axis = source.axis;
            guard.lower = source.lower;
            guard.upper = source.upper;
            guard.divisible_by = source.divisible_by;
            guard.exact = source.exact;
            if (source.equal_to) {
                guard.equal_to = ModuleAxisReference{
                    source.equal_to->input_index, source.equal_to->axis};
            }
            module_input.axis_guards.push_back(std::move(guard));
        }
        inputs.push_back(std::move(module_input));
    }

    std::vector<ModuleTensorContract> outputs;
    std::size_t run_byte_budget = 0;
    for (std::size_t output = 0;
         output < unit.output_value_ids.size(); ++output) {
        const LogicalValueContract& value = ContractValue(
            values, unit.output_value_ids[output],
            "a dynamic module output");
        const auto* type = value.checked_type.As<TensorTypeNode>();
        if (!type || type->shape.size() !=
                         shape_contract.output_shape_expressions()[output].size()) {
            Reject("a dynamic module output rank differs from its shape contract");
        }
        ModuleTensorContract module_output;
        for (const DynamicShapeExpr& expression :
             shape_contract.output_shape_expressions()[output]) {
            module_output.logical.push_back(ToModuleShapeExpr(expression));
        }
        module_output.physical = module_output.logical;
        module_output.valid = module_output.logical;
        module_output.max_bytes = MaximumOutputBytes(
            shape_contract.output_shape_expressions()[output],
            shape_contract.local_input_guards(),
            runtime::DataTypeFromString(type->dtype));
        if (run_byte_budget >
            std::numeric_limits<std::size_t>::max() -
                module_output.max_bytes) {
            Reject("a dynamic module run budget overflows size_t");
        }
        run_byte_budget += module_output.max_bytes;
        outputs.push_back(std::move(module_output));
    }

    std::vector<ModuleRuntimeExtentScalar> scalars;
    scalars.reserve(shape_contract.runtime_extent_expressions().size());
    for (const DynamicShapeExpr& expression :
         shape_contract.runtime_extent_expressions()) {
        scalars.push_back(ModuleRuntimeExtentScalar{
            ToModuleShapeExpr(expression)});
    }
    // ModuleInvocationContract uses zero as "unspecified"; one byte is the
    // smallest finite cap for a graph whose bounded outputs are all empty.
    run_byte_budget = std::max<std::size_t>(run_byte_budget, 1);
    return std::make_shared<const ModuleInvocationContract>(
        std::move(inputs), std::move(outputs), std::move(scalars),
        run_byte_budget);
}

runtime::ExecutablePlan BuildDynamicExecutablePlan(
    const BoundedCompilePreparation& preparation) {
    const PartitionedGraph& graph = preparation.partitioned_graph();
    ValidatePartition(graph);
    Array<runtime::ValueSpec> value_specs;
    for (const LogicalValueContract& value : graph.value_graph.values) {
        const auto* type = value.checked_type.As<TensorTypeNode>();
        if (!type) Reject("a dynamic plan value has no TensorType");
        const bool is_graph_output =
            std::find(graph.output_value_ids.begin(),
                      graph.output_value_ids.end(), value.id) !=
            graph.output_value_ids.end();
        value_specs.push_back(runtime::ValueSpec(
            value.id, value.id, type->shape,
            runtime::DataTypeFromString(type->dtype), value.device,
            value.origin == LogicalValueOrigin::kParameter,
            value.origin == LogicalValueOrigin::kConstant,
            is_graph_output));
    }
    runtime::ExecutablePlan plan(
        value_specs, graph.calls, graph.input_value_ids,
        graph.constant_value_ids, graph.output_value_ids, {},
        runtime::ExecutablePlanMode::kDynamicFreshOutputV1,
        preparation.graph_input_guards());
    plan.Validate();
    return plan;
}


std::vector<std::optional<std::vector<DynamicShapeExpr>>>
DecodeValueExpressionOverrides(
    const std::vector<std::optional<
        shape_resolution::EncodedExpr>>& encoded) {
    using shape_resolution::kExprKindConst;
    using shape_resolution::kExprKindInputAxis;
    std::vector<std::optional<std::vector<DynamicShapeExpr>>> decoded;
    decoded.reserve(encoded.size());
    for (const auto& entry : encoded) {
        if (!entry) {
            decoded.push_back(std::nullopt);
            continue;
        }
        std::vector<DynamicShapeExpr> expressions;
        for (std::size_t i = 0; i < entry->kinds.size(); ++i) {
            if (entry->kinds[i] == kExprKindConst) {
                expressions.push_back(
                    DynamicShapeExpr::Const(entry->values[i]));
            } else if (entry->kinds[i] == kExprKindInputAxis) {
                expressions.push_back(DynamicShapeExpr::InputAxis(entry->values[i], entry->axes[i]));
            } else if (entry->kinds[i] == shape_resolution::kExprKindInputAxisOffset && entry->values[i] >= 0) {
                expressions.push_back(DynamicShapeExpr::InputAxis(0,entry->axes[i],entry->values[i]));
            } else Reject("shape value expression kind or offset is invalid");
        }
        decoded.push_back(std::move(expressions));
    }
    return decoded;
}

}  // namespace kxc::api::internal
