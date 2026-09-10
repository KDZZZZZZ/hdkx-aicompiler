/*! \file src/compiler/lowering/lowered_graph.cc
 * \brief Lowers primitive Calls and static regions through frozen TE candidates.
 */

#include "../internal/lowered_graph.h"
#include "../internal/dynamic_shape_contract.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../internal/te_to_tir.h"
#include "kxc/relay/transforms/infer_type.h"

#ifndef KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
#define KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH 0
#endif

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

#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
std::size_t RuntimeExtentIndex(
    const DynamicUnitShapeContract& contract,
    const DynamicShapeExpr& expression) {
    const auto& ordered = contract.runtime_extent_expressions();
    const auto found = std::find(ordered.begin(), ordered.end(), expression);
    if (found == ordered.end()) {
        throw std::invalid_argument(
            "dynamic input axis is absent from the runtime extent order");
    }
    return static_cast<std::size_t>(std::distance(ordered.begin(), found));
}

Array<tir::Var> RuntimeExtentBuffers(
    const DynamicUnitShapeContract& contract) {
    Array<tir::Var> buffers;
    for (std::size_t index = 0;
         index < contract.runtime_extent_expressions().size(); ++index) {
        const DynamicShapeExpr& expression =
            contract.runtime_extent_expressions()[index];
        if (expression.kind() != DynamicShapeExpr::Kind::kInputAxis) {
            throw std::invalid_argument(
                "runtime extent order may contain only dynamic InputAxis expressions");
        }
        buffers.push_back(tir::Var(
            "runtime_extent_" + std::to_string(index),
            tir::DataType::UInt(64)));
    }
    return buffers;
}

Array<tir::PrimExpr> DynamicInputTEShape(
    const TensorTypeNode* type, std::size_t input_index,
    const DynamicUnitShapeContract& contract,
    const Array<tir::Var>& runtime_extent_buffers) {
    if (!type || input_index >= contract.local_input_guards().size()) {
        throw std::invalid_argument(
            "dynamic boundary input has no fixed-rank shape contract");
    }
    const auto& guards = contract.local_input_guards()[input_index];
    if (type->shape.size() != guards.size()) {
        throw std::invalid_argument(
            "dynamic boundary input rank differs from its shape contract");
    }
    Array<tir::PrimExpr> shape;
    for (std::size_t axis = 0; axis < guards.size(); ++axis) {
        const DynamicInputAxisGuard& guard = guards[axis];
        if (guard.axis != axis) {
            throw std::invalid_argument(
                "dynamic boundary input guards are not in axis order");
        }
        if (guard.exact) {
            if (type->shape[axis] < 0 ||
                static_cast<std::uint64_t>(type->shape[axis]) !=
                    *guard.exact) {
                throw std::invalid_argument(
                    "static boundary axis differs from its exact shape guard");
            }
            shape.push_back(tir::IntImm(
                static_cast<std::int64_t>(*guard.exact),
                tir::DataType::Int(64)));
            continue;
        }
        if (type->shape[axis] != codegen::kDynamicDimension) {
            throw std::invalid_argument(
                "dynamic boundary axis is not represented by -1");
        }
        const DynamicInputAxisReference reference =
            guard.equal_to.value_or(
                DynamicInputAxisReference{input_index, axis});
        const std::size_t extent_index = RuntimeExtentIndex(
            contract, DynamicShapeExpr::InputAxis(
                          reference.input_index, reference.axis));
        shape.push_back(relay::internal::LoadRuntimeExtent(
            runtime_extent_buffers[extent_index]));
    }
    return shape;
}
#endif  // KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH

te::Tensor MakeBoundaryTensor(const ValueInfo& value) {
    const auto* tensor_type = value.checked_type.As<TensorTypeNode>();
    const std::string prefix =
        value.origin == ValueOrigin::kConstant ? "const_value_" : "value_";
    return te::placeholder(TEShape(tensor_type), TIRDataType(tensor_type),
                           prefix + std::to_string(value.id));
}

#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
te::Tensor MakeDynamicBoundaryTensor(
    const ValueInfo& value, std::size_t input_index,
    const DynamicUnitShapeContract& contract,
    const Array<tir::Var>& runtime_extent_buffers) {
    if (value.origin == ValueOrigin::kConstant) {
        return MakeBoundaryTensor(value);
    }
    const auto* tensor_type = value.checked_type.As<TensorTypeNode>();
    return te::placeholder(
        DynamicInputTEShape(tensor_type, input_index, contract,
                            runtime_extent_buffers),
        TIRDataType(tensor_type), "value_" + std::to_string(value.id));
}
#endif  // KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH

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
            int64_t extent = 0;
            if (!relay::internal::EvaluateStaticLoweringInt64(
                    output->shape[axis], &extent)) {
                throw std::invalid_argument(
                    "Unit TE output shape is not static at index " +
                    std::to_string(i) + ", axis " + std::to_string(axis) +
                    " for op: " + op_name);
            }
            if (expected->shape[axis] < 0 ||
                extent != expected->shape[axis]) {
                throw std::invalid_argument(
                    "Unit TE output shape mismatch at index " +
                    std::to_string(i) + ", axis " + std::to_string(axis) +
                    " for op: " + op_name);
            }
        }
    }
}

#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
void ValidateDynamicTEOutputContracts(
    const std::vector<LogicalValueContract>& values,
    const PrimitiveUnit& unit, const Array<te::Tensor>& outputs,
    const DynamicUnitShapeContract& contract,
    const Array<tir::Var>& runtime_extent_buffers) {
    if (outputs.size() != unit.output_value_ids.size() ||
        outputs.size() != contract.output_shape_expressions().size()) {
        throw std::invalid_argument(
            "dynamic TE output count differs from its unit shape contract");
    }
    for (std::size_t output_index = 0; output_index < outputs.size();
         ++output_index) {
        const te::Tensor& output = outputs[output_index];
        const ValueInfo& value =
            GetValue(values, unit.output_value_ids[output_index]);
        const auto* type = value.checked_type.As<TensorTypeNode>();
        const auto& expected =
            contract.output_shape_expressions()[output_index];
        if (!output.defined() || !type || output->dtype != TIRDataType(type) ||
            output->shape.size() != expected.size() ||
            type->shape.size() != expected.size()) {
            throw std::invalid_argument(
                "dynamic TE output dtype or fixed rank differs from its shape contract");
        }
        for (std::size_t axis = 0; axis < expected.size(); ++axis) {
            const DynamicShapeExpr& expression = expected[axis];
            if (expression.kind() == DynamicShapeExpr::Kind::kConst) {
                std::int64_t extent = -1;
                if (!relay::internal::EvaluateStaticLoweringInt64(
                        output->shape[axis], &extent) || extent < 0 ||
                    static_cast<std::uint64_t>(extent) !=
                        expression.constant() ||
                    type->shape[axis] != extent) {
                    throw std::invalid_argument(
                        "dynamic TE output static axis differs from its contract");
                }
                continue;
            }
            std::size_t actual_extent = 0;
            int64_t actual_offset = 0;
            if (type->shape[axis] != codegen::kDynamicDimension ||
                !relay::internal::MatchRuntimeExtentOffset(
                    output->shape[axis], runtime_extent_buffers,
                    &actual_extent, &actual_offset) ||
                static_cast<uint64_t>(actual_offset) != expression.offset() ||
                actual_extent != RuntimeExtentIndex(contract, DynamicShapeExpr::InputAxis(
                    expression.input_index(), expression.axis()))) {
                throw std::invalid_argument(
                    "dynamic TE output axis differs from runtime extent order");
            }
        }
    }
}
#endif  // KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH

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

namespace {

relay::LoweredFunction LowerPrimitiveUnitImpl(
    const std::vector<LogicalValueContract>& values,
    const PrimitiveUnit& unit, const Target& target,
    const DynamicUnitShapeContract* shape_contract,
    const std::string& tir_pipeline_canonical) {
    if (shape_contract && unit.producer) {
        throw std::invalid_argument("Static TE region cannot enter bounded dynamic lowering");
    }
    const ResolvedRelayCall& resolved = ValidateUnitOperator(values, unit);
    const relay::OperatorSpec& spec = resolved.spec;
    Array<tir::Var> runtime_extent_buffers;
    std::vector<int64_t> runtime_extent_upper_bounds;
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
    if (shape_contract) {
        runtime_extent_buffers = RuntimeExtentBuffers(*shape_contract);
        for (const DynamicShapeExpr& expression :
             shape_contract->runtime_extent_expressions()) {
            const auto upper = shape_contract->local_input_guards()
                .at(expression.input_index()).at(expression.axis()).upper;
            if (upper > static_cast<uint64_t>(
                            std::numeric_limits<int32_t>::max())) {
                throw std::invalid_argument(
                    "bounded extent exceeds the int32 loop domain");
            }
            runtime_extent_upper_bounds.push_back(static_cast<int64_t>(upper));
        }
        if (shape_contract->local_input_guards().size() !=
            unit.boundary_input_value_ids.size()) {
            throw std::invalid_argument(
                "dynamic unit input arity differs from its shape contract");
        }
    }
#else
    if (shape_contract) {
        throw std::logic_error("bounded dynamic lowering is disabled");
    }
#endif

    std::unordered_map<int64_t, te::Tensor> boundary_tensors;
    Array<te::Tensor> abi_inputs;
    std::vector<relay::internal::ConstantTensor> constants;
    for (std::size_t input_index = 0;
         input_index < unit.boundary_input_value_ids.size(); ++input_index) {
        const int64_t value_id = unit.boundary_input_value_ids[input_index];
        const ValueInfo& value = GetValue(values, value_id);
        te::Tensor tensor;
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
        tensor = shape_contract
                     ? MakeDynamicBoundaryTensor(
                           value, input_index, *shape_contract,
                           runtime_extent_buffers)
                     : MakeBoundaryTensor(value);
#else
        tensor = MakeBoundaryTensor(value);
#endif
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
    if (unit.producer) {
        Array<te::Tensor> producer_inputs;
        for (const ValueId id : unit.producer->argument_value_ids) {
            producer_inputs.push_back(boundary_tensors.at(id));
        }
        const auto producer_outputs = InvokeCurrentCall(unit.producer->call, producer_inputs,
            unit.producer->call.call.checked_type());
        if (producer_outputs.size() != 1) throw std::invalid_argument("TE region producer output arity drifted");
        const auto& value = GetValue(values, unit.producer->output_value_ids[0]);
        const auto* type = value.checked_type.As<TensorTypeNode>();
        relay::internal::ValidateStaticLoweringTensor(producer_outputs[0]->shape,
            producer_outputs[0]->dtype, "TE region producer");
        if (producer_outputs[0]->dtype != TIRDataType(type) ||
            producer_outputs[0]->shape.size() != type->shape.size()) {
            throw std::invalid_argument("TE region producer type drifted");
        }
        for (size_t axis = 0; axis < type->shape.size(); ++axis) {
            int64_t extent = -1;
            if (!relay::internal::EvaluateStaticLoweringInt64(producer_outputs[0]->shape[axis], &extent) ||
                extent != type->shape[axis]) throw std::invalid_argument("TE region producer shape drifted");
        }
        boundary_tensors.emplace(value.id, producer_outputs[0]);
    }
    for (const int64_t value_id : unit.argument_value_ids) {
        const auto tensor_it = boundary_tensors.find(value_id);
        if (tensor_it == boundary_tensors.end()) {
            throw std::invalid_argument(
                "PrimitiveUnit argument is outside its boundary map");
        }
        logical_inputs.push_back(tensor_it->second);
    }

    const Array<te::Tensor> outputs = InvokeCurrentCall(
        resolved, logical_inputs, unit.call.call.checked_type());
    te::Schedule schedule;
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
    schedule = shape_contract && !runtime_extent_buffers.empty()
                   ? relay::internal::BuildBoundedDynamicTESchedule(
                         outputs, target, runtime_extent_buffers)
                   : relay::internal::BuildDefaultTESchedule(outputs, target);
    if (shape_contract) {
        ValidateDynamicTEOutputContracts(
            values, unit, outputs, *shape_contract,
            runtime_extent_buffers);
    } else {
        ValidateTEOutputContracts(values, unit, outputs, spec.name);
    }
#else
    schedule = relay::internal::BuildDefaultTESchedule(outputs, target);
    ValidateTEOutputContracts(values, unit, outputs, spec.name);
#endif
    Array<te::Tensor> metadata_only_inputs;
    const std::string operator_name = spec.name;
    if (operator_name == "shape_of" || operator_name == "shape_expr" || operator_name == "constant_of_shape") {
        metadata_only_inputs = abi_inputs;
    } else if ((operator_name == "reshape_dynamic" || operator_name == "expand_dynamic") &&
               logical_inputs.size() == 2) {
        // The shape resolver already proved/projected this control dependency
        // into attrs. Its physical value is retained in the ABI, not read by TE.
        for (const auto& input : abi_inputs) {
            if (input.get() == logical_inputs[1].get()) metadata_only_inputs.push_back(input);
        }
    } else if (operator_name == "slice" && logical_inputs.size() >= 2) {
        const auto* attrs = resolved.attrs.As<relay::SliceAttrsNode>();
        if (attrs && attrs->prefix_axis >= 0) {
            // Prepared prefix/window forms use one or two shape-only anchors.
            // Keep both in the ABI for extent loads while excluding them from
            // the data dependency walk used by TE scheduling.
            for (std::size_t input_index = 1; input_index < logical_inputs.size(); ++input_index) {
                for (const auto& input : abi_inputs) {
                    if (input.get() == logical_inputs[input_index].get()) {
                        metadata_only_inputs.push_back(input);
                    }
                }
            }
        }
    }
    const relay::internal::PrimFuncIdentity identity{
        unit.symbol, unit.id, String(unit.producer ? "add+sqrt" : spec.name), spec.schema_version,
        String(unit.semantic_key.digest())};
    if (!shape_contract) {
        Array<te::Tensor> constant_tensors;
        for (const auto& constant : constants) constant_tensors.push_back(constant.tensor);
        const te::Program program(abi_inputs, constant_tensors, outputs, schedule,
                                  target, tir_pipeline_canonical, metadata_only_inputs);
        return relay::internal::LowerProgramToTIR(
            program, constants, target, tir_pipeline_canonical, identity);
    }
    return relay::internal::LowerTensorGraphToTIR(
        abi_inputs, constants, outputs, schedule, target,
        identity,
        runtime_extent_buffers, {}, runtime_extent_upper_bounds, metadata_only_inputs);
}

}  // namespace

relay::LoweredFunction LowerPrimitiveUnit(
    const std::vector<LogicalValueContract>& values,
    const PrimitiveUnit& unit, const Target& target,
    const std::string& tir_pipeline_canonical) {
    return LowerPrimitiveUnitImpl(values, unit, target, nullptr, tir_pipeline_canonical);
}

#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
relay::LoweredFunction LowerPrimitiveUnit(
    const std::vector<LogicalValueContract>& values,
    const PrimitiveUnit& unit, const Target& target,
    const DynamicUnitShapeContract& shape_contract) {
    return LowerPrimitiveUnitImpl(values, unit, target, &shape_contract, {});
}
#endif  // KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH

PreparedStaticGraph PrepareStaticGraph(Function function, Device device,
                                       Target target,
                                       String pipeline_fingerprint,
                                       bool fuse_static_add_sqrt) {
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
    prepared.partitioned = PartitionValueGraph(std::move(value_graph), fuse_static_add_sqrt);
    prepared.partitions = 1;
    prepared.device = std::move(device);
    prepared.target = std::move(target);
    prepared.pipeline_fingerprint = std::move(pipeline_fingerprint);
    return prepared;
}

runtime::ExecutablePlan BuildStaticExecutablePlan(
    const PreparedStaticGraph& prepared) {
    ValidatePartition(prepared.partitioned);
    Array<runtime::ValueSpec> value_specs;
    std::unordered_set<ValueId> internal_values;
    for (const auto& unit : prepared.partitioned.units) {
        if (unit.producer) {
            for (const ValueId id : unit.producer->output_value_ids) internal_values.insert(id);
        }
    }
    for (const ValueInfo& value : prepared.partitioned.value_graph.values) {
        if (internal_values.count(value.id)) continue;
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

}  // namespace kxc::api::internal
