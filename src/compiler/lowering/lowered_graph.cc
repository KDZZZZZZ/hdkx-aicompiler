/*! \file src/compiler/lowering/lowered_graph.cc
 * \brief Lowers each ordinary compute Call through boundary-only TE placeholders.
 */

#include "../internal/lowered_graph.h"

#include <any>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "../internal/te_to_tir.h"
#include "kxc/compiler/capability.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/transforms/infer_type.h"

namespace kxc::api::internal {
namespace {

const ValueInfo& GetValue(const ValueGraph& graph, int64_t value_id) {
    if (value_id < 0 || static_cast<size_t>(value_id) >= graph.values.size() ||
        graph.values[static_cast<size_t>(value_id)].value_id != value_id) {
        throw std::invalid_argument("CompilationUnit references an invalid value id");
    }
    return graph.values[static_cast<size_t>(value_id)];
}

const CallInfo& GetCall(const ValueGraph& graph, const Expr& call) {
    for (const auto& record : graph.calls) {
        if (record.call.get() == call.get()) return record;
    }
    throw std::invalid_argument("CompilationUnit Call is missing from ValueGraph");
}

tir::DataType TIRDataType(const TensorTypeNode* type) {
    if (!type) throw std::invalid_argument("Boundary value must have TensorType");
    const DLDataType dtype = runtime::DataTypeFromString(type->dtype);
    if (dtype.code == kDLFloat) return tir::DataType::Float(dtype.bits, dtype.lanes);
    if (dtype.code == kDLInt) return tir::DataType::Int(dtype.bits, dtype.lanes);
    if (dtype.code == kDLUInt) return tir::DataType::UInt(dtype.bits, dtype.lanes);
    if (dtype.code == kDLBool) return tir::DataType::Bool();
    throw std::invalid_argument("Unsupported boundary dtype: " + type->dtype);
}

Array<tir::PrimExpr> TEShape(const TensorTypeNode* type) {
    if (!type) throw std::invalid_argument("Boundary value must have TensorType");
    Array<tir::PrimExpr> shape;
    for (int64_t dimension : type->shape) {
        shape.push_back(tir::IntImm(dimension, tir::DataType::Int(64)));
    }
    return shape;
}

te::Tensor MakeBoundaryTensor(const ValueInfo& value) {
    const auto* tensor_type = value.checked_type.As<TensorTypeNode>();
    const std::string prefix =
        value.origin == ValueOrigin::kConstant ? "const_value_" : "value_";
    return te::placeholder(TEShape(tensor_type), TIRDataType(tensor_type),
                           prefix + std::to_string(value.value_id));
}

relay::Attrs CallAttrs(const CallNode* call) {
    return call->attrs.defined() ? relay::Attrs(call->attrs) : relay::Attrs();
}

void ValidateInputArity(const relay::OperatorSpec& spec, size_t actual) {
    if (spec.input_arity.num_inputs >= 0) {
        if (actual != static_cast<size_t>(spec.input_arity.num_inputs)) {
            throw std::invalid_argument("Operator input arity mismatch for op: " +
                                        spec.name);
        }
        return;
    }
    if (spec.input_arity.min_inputs < 0 || spec.input_arity.max_inputs < 0 ||
        actual < static_cast<size_t>(spec.input_arity.min_inputs) ||
        actual > static_cast<size_t>(spec.input_arity.max_inputs)) {
        throw std::invalid_argument("Operator variable input arity mismatch for op: " +
                                    spec.name);
    }
}

void ValidateAttrs(const relay::OperatorSpec& spec, const CallNode* call) {
    if (!call->attrs.defined()) return;
    if (spec.attrs_type_key.empty()) {
        throw std::invalid_argument("Operator Call defines attrs outside its schema: " +
                                    spec.name);
    }
    const std::string actual(call->attrs.get()->GetTypeKey());
    const std::string expected_node = spec.attrs_type_key + "Node";
    if (actual != spec.attrs_type_key && actual != expected_node) {
        throw std::invalid_argument("Operator attrs type mismatch for op: " + spec.name);
    }
}

const relay::OperatorSpec& ValidateUnitOperator(const ValueGraph& graph,
                                                const CompilationUnit& unit) {
    const auto* call = unit.call.As<CallNode>();
    const auto* op = call ? call->op.As<relay::OpNode>() : nullptr;
    if (!call || !op || !op->has_spec) {
        throw std::invalid_argument(
            "CompilationUnit requires a registered specified operator Call");
    }
    relay::ValidateOperatorSpec(op->spec);
    if (!IsOrdinaryCompute(op->spec.lowering_kind)) {
        throw std::invalid_argument(
            "Special execution Call cannot enter ordinary unit lowering: " +
            op->name);
    }
    ValidateInputArity(op->spec, call->args.size());
    ValidateAttrs(op->spec, call);

    const auto relation_it = op->attrs.find(op->spec.type_relation_key);
    if (relation_it == op->attrs.end()) {
        throw std::invalid_argument("Operator type relation binding is missing: " +
                                    op->name);
    }
    const auto* relation = std::any_cast<relay::FInferType>(&relation_it->second);
    if (!relation) {
        throw std::invalid_argument("Operator type relation binding has wrong type: " +
                                    op->name);
    }
    Array<Type> argument_types;
    for (const auto& argument : call->args) {
        if (!argument.checked_type().defined()) {
            throw std::invalid_argument("Unit argument has stale or missing checked_type for op: " +
                                        op->name);
        }
        argument_types.push_back(argument.checked_type());
    }
    const Type inferred = (*relation)(CallAttrs(call), argument_types);
    if (!TypeEqual(inferred, unit.call.checked_type())) {
        throw std::invalid_argument("Operator checked_type is stale for op: " + op->name);
    }

    const CallInfo& record = GetCall(graph, unit.call);
    if (record.output_value_ids.size() !=
        static_cast<size_t>(op->spec.output_arity)) {
        throw std::invalid_argument("Operator output arity mismatch for op: " + op->name);
    }
    if (op->spec.lowering_kind == relay::OperatorLoweringKind::kSingleTE) {
        if (!unit.call.checked_type().As<TensorTypeNode>() ||
            op->spec.lowering_key != "FRelayToTE" ||
            op->attrs.find(op->spec.lowering_key) == op->attrs.end() ||
            !std::any_cast<relay::FRelayToTE>(
                &op->attrs.at(op->spec.lowering_key))) {
            throw std::invalid_argument("Single-output lowering contract mismatch for op: " +
                                        op->name);
        }
    } else {
        if (!unit.call.checked_type().As<TupleTypeNode>() ||
            op->spec.lowering_key != "FRelayToTEMulti" ||
            op->attrs.find(op->spec.lowering_key) == op->attrs.end() ||
            !std::any_cast<relay::FRelayToTEMulti>(
                &op->attrs.at(op->spec.lowering_key))) {
            throw std::invalid_argument("Multi-output lowering contract mismatch for op: " +
                                        op->name);
        }
    }
    return op->spec;
}

Array<te::Tensor> InvokeCurrentCall(const CallNode* call,
                                    const relay::OperatorSpec& spec,
                                    const Array<te::Tensor>& logical_inputs,
                                    const Type& output_type) {
    const auto* op = call->op.As<relay::OpNode>();
    const relay::Attrs attrs = CallAttrs(call);
    if (spec.lowering_kind == relay::OperatorLoweringKind::kSingleTE) {
        const auto* lower = std::any_cast<relay::FRelayToTE>(
            &op->attrs.at(spec.lowering_key));
        te::Tensor output = (*lower)(attrs, logical_inputs, output_type);
        if (!output.defined()) {
            throw std::invalid_argument("Operator lowering returned undefined tensor: " +
                                        op->name);
        }
        return {output};
    }
    const auto* lower = std::any_cast<relay::FRelayToTEMulti>(
        &op->attrs.at(spec.lowering_key));
    Array<te::Tensor> outputs = (*lower)(attrs, logical_inputs, output_type);
    if (outputs.size() != static_cast<size_t>(spec.output_arity)) {
        throw std::invalid_argument("Multi-output lowering result count mismatch for op: " +
                                    op->name);
    }
    for (const auto& output : outputs) {
        if (!output.defined()) {
            throw std::invalid_argument("Multi-output lowering returned undefined tensor: " +
                                        op->name);
        }
    }
    return outputs;
}

bool ReadIntAttr(const tir::PrimFunc& function, const char* key,
                 int64_t* value) {
    const String attr_key(key);
    if (!function->attrs.count(attr_key)) return false;
    const auto* integer = function->attrs.at(attr_key).As<tir::IntImmNode>();
    if (!integer) return false;
    *value = integer->value;
    return true;
}

String ReadStringAttr(const tir::PrimFunc& function, const char* key) {
    const String attr_key(key);
    if (!function->attrs.count(attr_key)) return String();
    return String(function->attrs.at(attr_key));
}

}  // namespace

relay::LoweredFunction LowerCompilationUnit(const ValueGraph& graph,
                                            const CompilationUnit& unit) {
    const relay::OperatorSpec& spec = ValidateUnitOperator(graph, unit);
    const auto* call = unit.call.As<CallNode>();

    std::unordered_map<int64_t, te::Tensor> boundary_tensors;
    Array<te::Tensor> abi_inputs;
    std::vector<relay::internal::ConstantTensor> constants;
    for (int64_t value_id : unit.input_value_ids) {
        const ValueInfo& value = GetValue(graph, value_id);
        te::Tensor tensor = MakeBoundaryTensor(value);
        boundary_tensors.emplace(value_id, tensor);
        if (value.origin == ValueOrigin::kConstant) {
            const auto* constant = value.source.As<ConstantNode>();
            if (!constant || !constant->data.defined()) {
                throw std::invalid_argument(
                    "Constant boundary value has no NDArray payload");
            }
            constants.push_back(relay::internal::ConstantTensor{
                tensor,
                String("relay.constant.v" + std::to_string(value_id)),
                constant->data});
        } else {
            abi_inputs.push_back(tensor);
        }
    }

    Array<te::Tensor> logical_inputs;
    for (const auto& argument : call->args) {
        const auto ids_it = graph.value_ids_by_expr.find(argument.get());
        if (ids_it == graph.value_ids_by_expr.end()) {
            throw std::invalid_argument(
                "Unit argument is absent from the stable ValueGraph");
        }
        for (int64_t value_id : ids_it->second) {
            const auto tensor_it = boundary_tensors.find(value_id);
            if (tensor_it == boundary_tensors.end()) {
                throw std::invalid_argument(
                    "Unit lowering attempted to read outside its boundary map");
            }
            logical_inputs.push_back(tensor_it->second);
        }
    }

    const Array<te::Tensor> outputs =
        InvokeCurrentCall(call, spec, logical_inputs, unit.call.checked_type());
    if (outputs.size() != unit.output_value_ids.size()) {
        throw std::invalid_argument(
            "Unit TE outputs do not match stable output value ids");
    }
    return relay::internal::LowerTensorGraphToTIR(
        abi_inputs, constants, outputs,
        relay::internal::PrimFuncIdentity{
            unit.symbol, unit.unit_id, String(spec.name), spec.schema_version,
            unit.structural_hash});
}

LoweredGraph LowerGraph(Function function, Device device, Target target) {
    if (!function.defined() || !device.defined()) {
        throw std::invalid_argument(
            "LowerGraph requires a defined Function and Device");
    }
    function = relay::InferTypePass(function);
    if (!target.defined()) target = BuildTarget(device);
    CapabilityVerifier::Require(CapabilityRequest{
        function, target, "graph", "", CapabilityBoundary::kPrePartition,
        CapabilityMode::kStaticExact, true});
    LoweredGraph result;
    result.partitioned =
        PartitionValueGraph(BuildValueGraph(function));

    for (const CompilationUnit& unit : result.partitioned.units) {
        relay::LoweredFunction lowered =
            LowerCompilationUnit(result.partitioned.value_graph, unit);
        const auto* call = unit.call.As<CallNode>();
        const auto* op = call->op.As<relay::OpNode>();
        result.primitives.push_back(
            LoweredPrimitive{unit.unit_id,
                             unit.symbol,
                             String(op->name + "@v" +
                                    std::to_string(op->spec.schema_version)),
                             unit.structural_hash,
                             lowered});
        for (const auto& binding : lowered.constants()) {
            if (result.constants.count(binding->key) &&
                result.constants.at(binding->key).get() != binding->value.get()) {
                throw std::invalid_argument(
                    "Stable constant key resolved to different payloads");
            }
            result.constants.Set(binding->key, binding->value);
        }
    }

    Array<runtime::ValueSpec> value_specs;
    for (const ValueInfo& value : result.partitioned.value_graph.values) {
        const auto* type = value.checked_type.As<TensorTypeNode>();
        value_specs.push_back(runtime::ValueSpec(
            value.value_id, value.value_id, type->shape,
            runtime::DataTypeFromString(type->dtype), device,
            value.origin == ValueOrigin::kParameter,
            value.origin == ValueOrigin::kConstant, value.is_graph_output));
    }
    result.plan = runtime::ExecutablePlan(
        value_specs, result.partitioned.calls,
        result.partitioned.input_value_ids,
        result.partitioned.constant_value_ids,
        result.partitioned.output_value_ids);
    ValidateLoweredGraph(result);
    return result;
}

void ValidateLoweredGraph(const LoweredGraph& graph) {
    ValidatePartition(graph.partitioned);
    graph.plan.Validate();
    if (graph.primitives.size() != graph.partitioned.units.size()) {
        throw std::invalid_argument(
            "LoweredGraph requires one primitive per CompilationUnit");
    }
    for (size_t index = 0; index < graph.primitives.size(); ++index) {
        const LoweredPrimitive& primitive = graph.primitives[index];
        const CompilationUnit& unit = graph.partitioned.units[index];
        primitive.lowered.Validate();
        const tir::PrimFunc& function = primitive.lowered->prim_func;
        int64_t unit_id = -1;
        if (primitive.unit_id != unit.unit_id ||
            !(primitive.symbol == unit.symbol) ||
            !(primitive.structural_hash == unit.structural_hash)) {
            throw std::invalid_argument(
                "Lowered primitive record drifted from its CompilationUnit");
        }
        if (!ReadIntAttr(function, "kxc.unit_id", &unit_id) ||
            unit_id != unit.unit_id) {
            throw std::invalid_argument("PrimFunc unit id metadata mismatch");
        }
        if (!(ReadStringAttr(function, "global_symbol") == unit.symbol)) {
            throw std::invalid_argument("PrimFunc symbol metadata mismatch");
        }
        const String actual_operator_identity =
            ReadStringAttr(function, "kxc.operator_identity");
        if (!(actual_operator_identity == primitive.operator_identity)) {
            throw std::invalid_argument(
                "PrimFunc operator identity metadata mismatch: expected " +
                std::string(primitive.operator_identity) + ", got " +
                std::string(actual_operator_identity));
        }
        if (!(ReadStringAttr(function, "kxc.structural_hash") ==
              unit.structural_hash)) {
            throw std::invalid_argument("PrimFunc structural hash metadata mismatch");
        }
    }
}

}  // namespace kxc::api::internal

namespace kxc::relay {

Array<LoweredFunction> LowerOperatorCallsToTIR(Function function) {
    const api::internal::LoweredGraph graph =
        api::internal::LowerGraph(std::move(function), Device::CPU());
    Array<LoweredFunction> result;
    for (const auto& primitive : graph.primitives) {
        result.push_back(primitive.lowered);
    }
    return result;
}

}  // namespace kxc::relay
