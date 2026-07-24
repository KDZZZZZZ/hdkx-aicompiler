/*! \file src/compiler/lowering/lowered_graph.cc
 * \brief Lowers each ordinary compute Call through boundary-only TE placeholders.
 */

#include "../internal/lowered_graph.h"

#include <any>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
    if (!relation || !*relation) {
        throw std::invalid_argument("Operator type relation binding has wrong type or is empty: " +
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
        const auto lowering = op->attrs.find(op->spec.lowering_key);
        const auto* lower =
            lowering == op->attrs.end()
                ? nullptr
                : std::any_cast<relay::FRelayToTE>(&lowering->second);
        if (!unit.call.checked_type().As<TensorTypeNode>() ||
            op->spec.lowering_key != "FRelayToTE" || !lower || !*lower) {
            throw std::invalid_argument("Single-output lowering contract mismatch for op: " +
                                        op->name);
        }
    } else {
        const auto lowering = op->attrs.find(op->spec.lowering_key);
        const auto* lower =
            lowering == op->attrs.end()
                ? nullptr
                : std::any_cast<relay::FRelayToTEMulti>(&lowering->second);
        if (!unit.call.checked_type().As<TupleTypeNode>() ||
            op->spec.lowering_key != "FRelayToTEMulti" || !lower || !*lower) {
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

void ValidateTEOutputContracts(const ValueGraph& graph,
                               const CompilationUnit& unit,
                               const Array<te::Tensor>& outputs,
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
        const ValueInfo& value = GetValue(graph, unit.output_value_ids[i]);
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
    ValidateTEOutputContracts(graph, unit, outputs, spec.name);
    return relay::internal::LowerTensorGraphToTIR(
        abi_inputs, constants, outputs,
        relay::internal::PrimFuncIdentity{
            unit.symbol, unit.unit_id, String(spec.name), spec.schema_version,
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
    CapabilityVerifier::RequireEligible(CapabilityRequest{
        function, target, "graph", std::string(pipeline_fingerprint),
        CapabilityBoundary::kPrePartition, true});
    PreparedStaticGraph prepared;
    prepared.capability_boundary_checks = 1;
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

    Array<runtime::ValueSpec> value_specs;
    for (const ValueInfo& value : result.partitioned.value_graph.values) {
        const auto* type = value.checked_type.As<TensorTypeNode>();
        value_specs.push_back(runtime::ValueSpec(
            value.value_id, value.value_id, type->shape,
            runtime::DataTypeFromString(type->dtype), prepared.device,
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
            primitive.semantic_key != unit.semantic_key) {
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
              String(unit.semantic_key.digest()))) {
            throw std::invalid_argument("PrimFunc structural hash metadata mismatch");
        }
    }
}

}  // namespace kxc::api::internal
