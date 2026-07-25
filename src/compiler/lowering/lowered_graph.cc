/*! \file src/compiler/lowering/lowered_graph.cc
 * \brief Lowers each ordinary compute Call through boundary-only TE placeholders.
 */

#include "../internal/lowered_graph.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../internal/te_to_tir.h"
#include "kxc/relay/transforms/infer_type.h"

namespace kxc::api::internal {
namespace {

const ValueInfo& GetValue(const std::vector<LogicalValueContract>& values,
                          int64_t value_id) {
    if (value_id < 0 || static_cast<size_t>(value_id) >= values.size() ||
        values[static_cast<size_t>(value_id)].id != value_id) {
        throw std::invalid_argument("PrimitiveUnit references an invalid value id");
    }
    return values[static_cast<size_t>(value_id)];
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
    for (size_t axis = 0; axis < type->shape.size(); ++axis) {
        const int64_t dimension = type->shape[axis];
        if (dimension < 0) {
            throw std::invalid_argument(
                "per-unit lowering requires non-negative static dimensions; axis " +
                std::to_string(axis) + " is " + std::to_string(dimension));
        }
        shape.push_back(tir::IntImm(dimension, tir::DataType::Int(64)));
    }
    relay::internal::ValidateStaticLoweringTensor(
        shape, TIRDataType(type), "per-unit lowering boundary TensorType");
    return shape;
}

te::Tensor MakeBoundaryTensor(const ValueInfo& value) {
    const auto* tensor_type = value.checked_type.As<TensorTypeNode>();
    const std::string prefix =
        value.origin == ValueOrigin::kConstant ? "const_value_" : "value_";
    return te::placeholder(TEShape(tensor_type), TIRDataType(tensor_type),
                           prefix + std::to_string(value.id));
}

const ResolvedRelayCall& ValidateUnitOperator(
    const std::vector<LogicalValueContract>& values,
    const PrimitiveUnit& unit) {
    ValidatePrimitiveUnit(unit, values);
    return unit.call;
}

Array<te::Tensor> InvokeCurrentCall(const ResolvedRelayCall& resolved,
                                    const Array<te::Tensor>& logical_inputs,
                                    const Type& output_type) {
    if (std::holds_alternative<relay::FRelayToTE>(resolved.lowering)) {
        const auto& lower = std::get<relay::FRelayToTE>(
            resolved.lowering);
        te::Tensor output =
            lower(resolved.attrs, logical_inputs, output_type);
        if (!output.defined()) {
            throw std::invalid_argument("Operator lowering returned undefined tensor: " +
                                        resolved.spec.name);
        }
        return {output};
    }
    const auto& lower = std::get<relay::FRelayToTEMulti>(
        resolved.lowering);
    Array<te::Tensor> outputs =
        lower(resolved.attrs, logical_inputs, output_type);
    if (outputs.size() != resolved.output_leaf_types.size()) {
        throw std::invalid_argument("Multi-output lowering result count mismatch for op: " +
                                    resolved.spec.name);
    }
    for (const auto& output : outputs) {
        if (!output.defined()) {
            throw std::invalid_argument("Multi-output lowering returned undefined tensor: " +
                                        resolved.spec.name);
        }
    }
    return outputs;
}

void ValidateTEOutputContracts(
    const std::vector<LogicalValueContract>& values,
    const PrimitiveUnit& unit, const Array<te::Tensor>& outputs,
    const std::string& op_name) {
    if (outputs.size() != unit.output_value_ids.size()) {
        throw std::invalid_argument(
            "Unit TE output count does not match stable output values for op: " +
            op_name);
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const te::Tensor& output = outputs[i];
        if (!output.defined()) {
            throw std::invalid_argument(
                "Unit TE output is undefined at index " + std::to_string(i) +
                " for op: " + op_name);
        }
        const ValueInfo& value = GetValue(values, unit.output_value_ids[i]);
        const auto* expected = value.checked_type.As<TensorTypeNode>();
        if (!expected) {
            throw std::invalid_argument(
                "Stable unit output must have TensorType for op: " + op_name);
        }
        if (output->dtype != TIRDataType(expected)) {
            throw std::invalid_argument(
                "Unit TE output dtype mismatch at index " +
                std::to_string(i) + " for op: " + op_name);
        }
        if (output->shape.size() != expected->shape.size()) {
            throw std::invalid_argument(
                "Unit TE output rank mismatch at index " +
                std::to_string(i) + " for op: " + op_name);
        }
        for (std::size_t axis = 0; axis < expected->shape.size(); ++axis) {
            const auto* extent = output->shape[axis].As<tir::IntImmNode>();
            if (!extent) {
                throw std::invalid_argument(
                    "Unit TE output shape is not static at index " +
                    std::to_string(i) + ", axis " + std::to_string(axis) +
                    " for op: " + op_name);
            }
            if (expected->shape[axis] < 0 ||
                extent->value != expected->shape[axis]) {
                throw std::invalid_argument(
                    "Unit TE output shape mismatch at index " +
                    std::to_string(i) + ", axis " + std::to_string(axis) +
                    " for op: " + op_name);
            }
        }
    }
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

void FreezeConstantPayloads(ValueGraph* graph) {
    for (ValueInfo& value : graph->values) {
        if (value.origin != ValueOrigin::kConstant) continue;
        const auto* constant = value.source.As<ConstantNode>();
        if (!constant || !constant->data.defined()) {
            throw std::invalid_argument(
                "Prepared static graph constant has no NDArray payload");
        }
        value.source = Constant(constant->data.CopyTo(constant->data.device()));
    }
}

}  // namespace

relay::LoweredFunction LowerPrimitiveUnit(
    const std::vector<LogicalValueContract>& values,
    const PrimitiveUnit& unit) {
    const ResolvedRelayCall& resolved =
        ValidateUnitOperator(values, unit);
    const relay::OperatorSpec& spec = resolved.spec;
    const auto* call = unit.call.call.As<CallNode>();

    std::unordered_map<int64_t, te::Tensor> boundary_tensors;
    Array<te::Tensor> abi_inputs;
    std::vector<relay::internal::ConstantTensor> constants;
    for (int64_t value_id : unit.boundary_input_value_ids) {
        const ValueInfo& value = GetValue(values, value_id);
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
    for (const int64_t value_id : unit.argument_value_ids) {
        const auto tensor_it = boundary_tensors.find(value_id);
        if (tensor_it == boundary_tensors.end()) {
            throw std::invalid_argument(
                "PrimitiveUnit argument is outside its boundary map");
        }
        logical_inputs.push_back(tensor_it->second);
    }

    const Array<te::Tensor> outputs =
        InvokeCurrentCall(
            resolved, logical_inputs, unit.call.call.checked_type());
    ValidateTEOutputContracts(values, unit, outputs, spec.name);
    return relay::internal::LowerTensorGraphToTIR(
        abi_inputs, constants, outputs,
        relay::internal::PrimFuncIdentity{
            unit.symbol, unit.id, String(spec.name), spec.schema_version,
            String(unit.semantic_key.digest())});
}

PreparedStaticGraph PrepareStaticGraph(Function function, Device device,
                                       Target target,
                                       String pipeline_fingerprint) {
    if (!function.defined() || !device.defined()) {
        throw std::invalid_argument(
            "PrepareStaticGraph requires a defined Function and Device");
    }
    if (!target.defined()) target = BuildTarget(device);
    if (!target.As<TargetNode>() ||
        target->device_type != device.device_type() ||
        target->device_id != device.device_id()) {
        throw std::invalid_argument(
            "PrepareStaticGraph requires matching Target and Device identity");
    }
    PreparedStaticGraph prepared;
    prepared.capability_boundary_checks = 0;
    ValueGraph value_graph = BuildValueGraph(function, device);
    prepared.value_graph_builds = 1;
    FreezeConstantPayloads(&value_graph);
    prepared.partitioned = PartitionValueGraph(std::move(value_graph));
    prepared.partitions = 1;
    prepared.device = std::move(device);
    prepared.target = std::move(target);
    prepared.pipeline_fingerprint = std::move(pipeline_fingerprint);
    return prepared;
}

LoweredGraph LowerPreparedStaticGraph(const PreparedStaticGraph& prepared) {
    if (!prepared.device.defined() ||
        !prepared.target.As<TargetNode>() ||
        prepared.target->device_type != prepared.device.device_type() ||
        prepared.target->device_id != prepared.device.device_id()) {
        throw std::invalid_argument(
            "LowerPreparedStaticGraph requires bound Device and Target identity");
    }
    ValidatePartition(prepared.partitioned);
    LoweredGraph result;
    result.partitioned = prepared.partitioned;

    for (const PrimitiveUnit& unit : result.partitioned.units) {
        relay::LoweredFunction lowered =
            LowerPrimitiveUnit(result.partitioned.value_graph.values, unit);
        const ResolvedRelayCall& resolved = unit.call;
        result.primitives.push_back(
            LoweredPrimitive{unit.id,
                             unit.symbol,
                             String(resolved.spec.name + "@v" +
                                    std::to_string(
                                        resolved.spec.schema_version)),
                             unit.semantic_key,
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

    result.plan = BuildStaticExecutablePlan(prepared);
    ValidateLoweredGraph(result);
    return result;
}

runtime::ExecutablePlan BuildStaticExecutablePlan(
    const PreparedStaticGraph& prepared) {
    ValidatePartition(prepared.partitioned);
    Array<runtime::ValueSpec> value_specs;
    for (const ValueInfo& value : prepared.partitioned.value_graph.values) {
        const auto* type = value.checked_type.As<TensorTypeNode>();
        const bool is_graph_output =
            std::find(prepared.partitioned.output_value_ids.begin(),
                      prepared.partitioned.output_value_ids.end(),
                      value.id) != prepared.partitioned.output_value_ids.end();
        value_specs.push_back(runtime::ValueSpec(
            value.id, value.id, type->shape,
            runtime::DataTypeFromString(type->dtype), value.device,
            value.origin == ValueOrigin::kParameter,
            value.origin == ValueOrigin::kConstant, is_graph_output));
    }
    runtime::ExecutablePlan plan(
        value_specs, prepared.partitioned.calls,
        prepared.partitioned.input_value_ids,
        prepared.partitioned.constant_value_ids,
        prepared.partitioned.output_value_ids);
    plan.Validate();
    return plan;
}

LoweredGraph LowerGraph(Function function, Device device, Target target,
                        String pipeline_fingerprint) {
    if (!target.defined()) target = BuildTarget(device);
    PreparedStaticGraph prepared = PrepareStaticGraph(
        std::move(function), device, std::move(target),
        std::move(pipeline_fingerprint));
    return LowerPreparedStaticGraph(prepared);
}

void ValidateLoweredGraph(const LoweredGraph& graph) {
    ValidatePartition(graph.partitioned);
    graph.plan.Validate();
    if (graph.primitives.size() != graph.partitioned.units.size()) {
        throw std::invalid_argument(
            "LoweredGraph requires one primitive per PrimitiveUnit");
    }
    for (size_t index = 0; index < graph.primitives.size(); ++index) {
        const LoweredPrimitive& primitive = graph.primitives[index];
        const PrimitiveUnit& unit = graph.partitioned.units[index];
        primitive.lowered.Validate();
        const tir::PrimFunc& function = primitive.lowered->prim_func;
        int64_t unit_id = -1;
        if (primitive.unit_id != unit.id ||
            !(primitive.symbol == unit.symbol) ||
            primitive.semantic_key != unit.semantic_key) {
            throw std::invalid_argument(
                "Lowered primitive record drifted from its PrimitiveUnit");
        }
        if (!ReadIntAttr(function, "kxc.unit_id", &unit_id) ||
            unit_id != unit.id) {
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
              String(unit.semantic_key.digest()))) {
            throw std::invalid_argument("PrimFunc structural hash metadata mismatch");
        }
    }
}

}  // namespace kxc::api::internal
