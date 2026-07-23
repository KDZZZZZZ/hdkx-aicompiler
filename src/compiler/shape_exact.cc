#include "kxc/compiler/shape_exact.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "internal/execution_contract.h"
#include "internal/primitive_cache.h"
#include "../runtime/internal/memory_plan.h"
#include "kxc/profiling/profiling.h"
#include "kxc/runtime/device_api.h"

#ifndef KXC_ENABLE_SHAPE_PRODUCTION_EXACT
#define KXC_ENABLE_SHAPE_PRODUCTION_EXACT 0
#endif
#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif
#ifndef KXC_USE_CUDA
#define KXC_USE_CUDA 0
#endif

namespace kxc::api::experimental::shape_exact::v1 {
namespace {
namespace shape = kxc::shape::experimental::v1;

[[noreturn]] void Reject(const std::string& message) {
    throw std::invalid_argument("ProductionExactShapeAdapter: " + message);
}

void RequireEnabled() {
#if !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    throw std::runtime_error(
        "ProductionExactShapeAdapter is disabled; configure with "
        "-DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON");
#endif
}

void RequireBackendAvailable(const Target& target) {
#if !KXC_USE_LLVM
    if (target->kind == "llvm" && target->device_type == kCPU) {
        Reject("LLVM exact assembly requires KXC_ENABLE_LLVM=ON");
    }
#endif
#if !KXC_USE_CUDA
    if (target->kind == "cuda" && target->device_type == kCUDA) {
        Reject("CUDA exact assembly requires KXC_ENABLE_CUDA=ON");
    }
#endif
}

Array<int64_t> CloneIntArray(const Array<int64_t>& source) {
    Array<int64_t> result;
    for (int64_t value : source) result.push_back(value);
    return result;
}

/*! \brief Deep snapshot used before any compiler pass can mutate caller IR. */
class RelaySnapshotCloner final {
public:
    Function Clone(const Function& function) {
        const Expr snapshot = CloneExpr(Expr(ObjectRef(function)));
        if (!snapshot.As<FunctionNode>()) {
            Reject("Relay snapshot root is not a Function");
        }
        return Function(snapshot);
    }

    Expr CloneOperator(const Expr& source) {
        const auto* node = source.As<relay::OpNode>();
        if (!node) Reject("prepared Call does not reference an OpNode");
        const auto found = operators_.find(source.get());
        if (found != operators_.end()) return found->second;

        auto* copy = new relay::OpNode();
        copy->name = node->name;
        copy->description = node->description;
        for (const relay::ArgumentInfo& argument : node->arguments) {
            copy->arguments.push_back(argument);
        }
        copy->num_inputs = node->num_inputs;
        copy->attrs = node->attrs;
        copy->spec = node->spec;
        copy->spec.arguments = Array<relay::ArgumentInfo>();
        for (const relay::ArgumentInfo& argument : node->spec.arguments) {
            copy->spec.arguments.push_back(argument);
        }
        copy->has_spec = node->has_spec;
        const Expr result{ObjectRef(copy)};
        operators_.emplace(source.get(), result);
        CopyMetadata(source, result);
        return result;
    }

private:
    Type CloneType(const Type& source) {
        if (!source.defined()) return Type();
        const auto found = types_.find(source.get());
        if (found != types_.end()) return found->second;
        Type result;
        if (const auto* tensor = source.As<TensorTypeNode>()) {
            result = TensorType(CloneIntArray(tensor->shape), tensor->dtype);
        } else if (const auto* tuple = source.As<TupleTypeNode>()) {
            Array<Type> fields;
            for (const Type& field : tuple->fields) {
                fields.push_back(CloneType(field));
            }
            result = TupleType(std::move(fields));
        } else {
            Reject("unsupported Relay type in preparation snapshot");
        }
        types_.emplace(source.get(), result);
        return result;
    }

    Target CloneTarget(const Target& source) {
        if (!source.defined()) return Target();
        const auto* node = source.As<TargetNode>();
        if (!node) Reject("Relay snapshot contains an invalid Target");
        auto* copy = new TargetNode();
        copy->kind = node->kind;
        copy->device_type = node->device_type;
        copy->device_id = node->device_id;
        copy->attrs = node->attrs;
        return Target(ObjectRef(copy));
    }

    VirtualDevice CloneVirtualDevice(const VirtualDevice& source) {
        if (!source.defined()) return VirtualDevice();
        const auto* node = source.As<VirtualDeviceNode>();
        if (!node) Reject("Relay snapshot contains an invalid VirtualDevice");
        const Target target = CloneTarget(node->target);
        if (node->device.defined()) {
            return VirtualDevice(node->device, target, node->memory_scope,
                                 node->virtual_device_id);
        }
        return VirtualDevice(target, node->memory_scope,
                             node->virtual_device_id);
    }

    ObjectRef CloneAttrs(const ObjectRef& source) {
        if (!source.defined()) return ObjectRef();
        const auto found = attrs_.find(source.get());
        if (found != attrs_.end()) return found->second;
        ObjectRef result;
        if (const auto* node = source.As<relay::Conv2DAttrsNode>()) {
            result = relay::Conv2DAttrs::Create(
                CloneIntArray(node->strides), CloneIntArray(node->padding),
                CloneIntArray(node->dilation), node->groups, node->channels,
                CloneIntArray(node->kernel_size), node->data_layout,
                node->kernel_layout, node->out_layout, node->out_dtype);
        } else if (const auto* node = source.As<relay::DenseAttrsNode>()) {
            result = relay::DenseAttrs::Create(node->units, node->out_dtype);
        } else if (const auto* node = source.As<relay::MaxPool2DAttrsNode>()) {
            result = relay::MaxPool2DAttrs::Create(
                CloneIntArray(node->strides), CloneIntArray(node->padding),
                CloneIntArray(node->dilation), CloneIntArray(node->pool_size),
                node->layout, node->ceil_mode);
        } else if (const auto* node = source.As<relay::SoftmaxAttrsNode>()) {
            result = relay::SoftmaxAttrs::Create(node->axis);
        } else if (source.As<relay::AddAttrsNode>()) {
            result = relay::AddAttrs::Create();
        } else if (const auto* node = source.As<relay::CastAttrsNode>()) {
            result = relay::CastAttrs::Create(node->to);
        } else if (const auto* node = source.As<relay::ReduceMeanAttrsNode>()) {
            result = relay::ReduceMeanAttrs::Create(
                CloneIntArray(node->axes), node->keepdims);
        } else if (const auto* node = source.As<relay::ReshapeAttrsNode>()) {
            result = relay::ReshapeAttrs::Create(
                CloneIntArray(node->newshape), node->allowzero);
        } else if (const auto* node = source.As<relay::TransposeAttrsNode>()) {
            result = relay::TransposeAttrs::Create(CloneIntArray(node->perm));
        } else if (source.As<relay::ReluAttrsNode>()) {
            result = relay::ReluAttrs::Create();
        } else if (source.As<relay::GlobalAvgPool2DAttrsNode>()) {
            result = relay::GlobalAvgPool2DAttrs::Create();
        } else if (const auto* node = source.As<relay::FlattenAttrsNode>()) {
            result = relay::FlattenAttrs::Create(node->axis);
        } else if (const auto* node = source.As<relay::GemmAttrsNode>()) {
            result = relay::GemmAttrs::Create(
                node->alpha, node->beta, node->transA, node->transB);
        } else if (const auto* node = source.As<relay::DeviceCopyAttrsNode>()) {
            result = relay::DeviceCopyAttrs::Create(
                CloneVirtualDevice(node->src_virtual_device),
                CloneVirtualDevice(node->dst_virtual_device), node->async,
                node->in_group);
        } else if (const auto* node = source.As<relay::CollectiveAttrsNode>()) {
            result = relay::CollectiveAttrs::Create(
                node->kind, node->reduce_kind, node->in_group, node->group_id,
                node->root_worker);
        } else {
            Reject("unsupported Relay attrs in preparation snapshot: " +
                   std::string(source->GetTypeKey()));
        }
        attrs_.emplace(source.get(), result);
        return result;
    }

    runtime::NDArray CloneConstant(const runtime::NDArray& source) {
        if (!source.defined() || !source.storage().defined()) {
            Reject("Relay snapshot constant has no storage");
        }
        try {
            // The preparation API has no producer completion parameter.  Drain
            // nonblocking CUDA work before synchronously reading the payload.
            if (source.device().device_type() == kCUDA && source.NBytes() != 0) {
                DeviceSynchronize(source.device());
            }
            runtime::NDArray result = runtime::NDArray::Empty(
                source.shape(), source.dtype(), source.device(),
                source.storage()->alignment);
            result.CopyFrom(source);
            return result;
        } catch (const std::exception& error) {
            Reject(std::string("failed to snapshot Relay constant: ") +
                   error.what());
        }
    }

    void CopyMetadata(const Expr& source, const Expr& result) {
        const auto* source_node = dynamic_cast<const RelayNode*>(source.get());
        auto* result_node = const_cast<RelayNode*>(
            dynamic_cast<const RelayNode*>(result.get()));
        if (!source_node || !result_node) {
            Reject("Relay snapshot metadata requires Relay nodes");
        }
        result_node->checked_type_ = CloneType(source_node->checked_type_);
        result_node->virtual_device_ =
            CloneVirtualDevice(source_node->virtual_device_);
        if (source_node->span.defined()) {
            const auto* span = source_node->span.As<SpanNode>();
            if (!span) Reject("Relay snapshot contains an invalid Span");
            result_node->span = Span(span->source_name, span->line, span->column);
        }
    }

    Expr CloneExpr(const Expr& source) {
        if (!source.defined()) Reject("Relay snapshot contains an undefined Expr");
        const auto found = expressions_.find(source.get());
        if (found != expressions_.end()) return found->second;

        Expr result;
        if (const auto* node = source.As<VarNode>()) {
            result = Var(node->vid->name_hint, CloneType(node->type_annotation));
            expressions_.emplace(source.get(), result);
        } else if (const auto* node = source.As<ConstantNode>()) {
            result = Constant(CloneConstant(node->data));
            expressions_.emplace(source.get(), result);
        } else if (const auto* node = source.As<relay::OpNode>()) {
            // Keep canonical registry identity while compiler passes validate
            // the graph; prepared Calls detach their descriptors afterward.
            result = source;
            expressions_.emplace(source.get(), result);
            return result;
        } else if (const auto* node = source.As<CallNode>()) {
            Array<Expr> arguments;
            for (const Expr& argument : node->args) {
                arguments.push_back(CloneExpr(argument));
            }
            result = Call(CloneExpr(node->op), std::move(arguments),
                          CloneAttrs(node->attrs));
            expressions_.emplace(source.get(), result);
        } else if (const auto* node = source.As<FunctionNode>()) {
            Array<Var> parameters;
            for (const Var& parameter : node->params) {
                parameters.push_back(
                    Var(CloneExpr(Expr(ObjectRef(parameter)))));
            }
            result = Function(std::move(parameters), CloneExpr(node->body));
            expressions_.emplace(source.get(), result);
        } else if (const auto* node = source.As<TupleNode>()) {
            Array<Expr> fields;
            for (const Expr& field : node->fields) {
                fields.push_back(CloneExpr(field));
            }
            result = Tuple(std::move(fields));
            expressions_.emplace(source.get(), result);
        } else if (const auto* node = source.As<TupleGetItemNode>()) {
            result = TupleGetItem(CloneExpr(node->tuple), node->index);
            expressions_.emplace(source.get(), result);
        } else if (const auto* node = source.As<LetNode>()) {
            const Var variable(CloneExpr(Expr(ObjectRef(node->var))));
            result = Let(variable, CloneExpr(node->value), CloneExpr(node->body));
            expressions_.emplace(source.get(), result);
        } else if (const auto* node = source.As<IfNode>()) {
            result = If(CloneExpr(node->cond), CloneExpr(node->true_branch),
                        CloneExpr(node->false_branch));
            expressions_.emplace(source.get(), result);
        } else if (source.As<WhileNode>()) {
            Reject("While is rejected by the static preparation snapshot; use CompileControlFlowExact");
        } else {
            Reject("unsupported Relay node in preparation snapshot: " +
                   std::string(source->GetTypeKey()));
        }
        CopyMetadata(source, result);
        return result;
    }

    std::unordered_map<const Object*, Expr> expressions_;
    std::unordered_map<const Object*, Expr> operators_;
    std::unordered_map<const Object*, Type> types_;
    std::unordered_map<const Object*, ObjectRef> attrs_;
};

void FreezePreparedOperators(internal::PreparedCompilerGraph* prepared) {
    RelaySnapshotCloner cloner;
    for (const internal::CallInfo& record :
         prepared->graph.partitioned.value_graph.calls) {
        auto* call = const_cast<CallNode*>(record.call.As<CallNode>());
        if (!call) Reject("prepared value graph contains a non-Call record");
        call->op = cloner.CloneOperator(call->op);
    }
}

shape::TargetBackendAbiDescriptor TargetAbi(const Target& target) {
    if (!target.defined()) Reject("target is undefined");
    if (target->kind == "llvm" && target->device_type == kCPU) {
        const std::string& arch = target->attrs.arch;
        if (arch == "x86_64") {
            return {shape::TargetKind::kX86_64, shape::BackendKind::kLlvm,
                    shape::kShapeAbiVersion};
        }
        if (arch == "aarch64") {
            return {shape::TargetKind::kAArch64, shape::BackendKind::kLlvm,
                    shape::kShapeAbiVersion};
        }
        Reject("unsupported LLVM CPU architecture '" + arch + "'");
    }
    if (target->kind == "cuda" && target->device_type == kCUDA) {
        return {shape::TargetKind::kNvptx64, shape::BackendKind::kCuda,
                shape::kShapeAbiVersion};
    }
    Reject("target/device/backend mismatch");
}

shape::TensorAbiDescriptor TensorAbi(const TensorTypeNode* type,
                                     const Target& target) {
    if (!type) Reject("value is not a tensor");
    const DLDataType dtype = runtime::DataTypeFromString(type->dtype);
    shape::DataType shape_dtype;
    if (dtype.code == kDLFloat && dtype.bits == 16 && dtype.lanes == 1) {
        shape_dtype = shape::DataType::kFloat16;
    } else if (dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1) {
        shape_dtype = shape::DataType::kFloat32;
    } else {
        Reject("shape v1 supports only scalar-lane float16/float32 values");
    }
    const shape::DeviceKind device = target->device_type == kCPU
        ? shape::DeviceKind::kCpu : shape::DeviceKind::kCuda;
    if ((device == shape::DeviceKind::kCpu && target->kind != "llvm") ||
        (device == shape::DeviceKind::kCuda && target->kind != "cuda") ||
        target->device_id < 0) {
        Reject("value device/target/backend mismatch");
    }
    return {shape_dtype, shape::DeviceDescriptor(
            device, static_cast<uint32_t>(target->device_id)), TargetAbi(target)};
}

std::vector<int64_t> Strides(const TensorTypeNode* type) {
    std::vector<int64_t> result(type->shape.size(), 1);
    int64_t stride = 1;
    for (size_t i = type->shape.size(); i > 0; --i) {
        const int64_t extent = type->shape[i - 1];
        if (extent < 0) Reject("negative or legacy -1 dimension is unsupported");
        result[i - 1] = stride;
        if (extent != 0 && stride > std::numeric_limits<int64_t>::max() / extent) {
            Reject("row-major stride overflow");
        }
        stride *= extent;
    }
    return result;
}

shape::TensorShapeContract ValueContract(const internal::ValueInfo& value,
                                         const Target& target) {
    const auto* type = value.checked_type.As<TensorTypeNode>();
    if (!type) Reject("prepared value has no TensorType");
    std::vector<shape::DimExpr> dimensions;
    dimensions.reserve(type->shape.size());
    for (int64_t extent : type->shape) {
        if (extent < 0) Reject("negative or legacy -1 dimension is unsupported");
        dimensions.push_back(shape::DimExpr::Const(extent));
    }
    const shape::TensorAbiDescriptor abi = TensorAbi(type, target);
    std::vector<shape::DimExpr> strides;
    for (int64_t stride : Strides(type)) strides.push_back(shape::DimExpr::Const(stride));
    const int64_t alignment = static_cast<int64_t>(abi.element_bytes());
    return {shape::LogicalShape(dimensions),
            shape::PhysicalShape(dimensions, std::move(strides),
                "contiguous.row_major", alignment, "global"),
            shape::ValidExtent(dimensions), abi};
}

std::string ValueName(int64_t id) { return "value." + std::to_string(id); }

void AppendField(std::string* bytes, const std::string& name,
                 const std::string& value) {
    *bytes += std::to_string(name.size()) + ":" + name + "=" +
              std::to_string(value.size()) + ":" + value + ";";
}

std::string IdsCanonical(const Array<int64_t>& ids) {
    std::string result;
    AppendField(&result, "count", std::to_string(ids.size()));
    for (int64_t id : ids) AppendField(&result, "id", std::to_string(id));
    return result;
}

std::string PartitionCanonical(const internal::PartitionedGraph& partitioned) {
    std::string result;
    AppendField(&result, "kind", "partition.static-per-call.v2");
    AppendField(&result, "graph_inputs", IdsCanonical(partitioned.input_value_ids));
    AppendField(&result, "graph_constants", IdsCanonical(partitioned.constant_value_ids));
    AppendField(&result, "graph_outputs", IdsCanonical(partitioned.output_value_ids));
    AppendField(&result, "unit_count", std::to_string(partitioned.units.size()));
    for (const auto& unit : partitioned.units) {
        std::string encoded;
        AppendField(&encoded, "ordinal", std::to_string(unit.unit_id));
        AppendField(&encoded, "semantic", unit.semantic_key.canonical_bytes());
        AppendField(&encoded, "inputs", IdsCanonical(unit.input_value_ids));
        AppendField(&encoded, "outputs", IdsCanonical(unit.output_value_ids));
        AppendField(&result, "unit", encoded);
    }
    return result;
}

std::string ConstantCanonical(const internal::ValueInfo& value) {
    const auto* constant = value.source.As<ConstantNode>();
    if (!constant || !constant->data.defined()) {
        Reject("prepared constant value has no payload");
    }
    const runtime::NDArray data = constant->data;
    std::string result;
    AppendField(&result, "shape", IdsCanonical(data.shape()));
    const DLDataType dtype = data.dtype();
    AppendField(&result, "dtype_code", std::to_string(dtype.code));
    AppendField(&result, "dtype_bits", std::to_string(dtype.bits));
    AppendField(&result, "dtype_lanes", std::to_string(dtype.lanes));
    std::string payload(data.NBytes(), '\0');
    data.CopyToBytes(payload.empty() ? nullptr : payload.data(), payload.size());
    AppendField(&result, "payload", payload);
    return result;
}

std::string GraphCanonical(const internal::PartitionedGraph& partitioned) {
    std::string result;
    AppendField(&result, "kind", "prepared-static-graph-v1");
    AppendField(&result, "partition", PartitionCanonical(partitioned));
    AppendField(&result, "value_count",
                std::to_string(partitioned.value_graph.values.size()));
    for (const auto& value : partitioned.value_graph.values) {
        std::string encoded;
        AppendField(&encoded, "value_id", std::to_string(value.value_id));
        AppendField(&encoded, "origin",
                    std::to_string(static_cast<int>(value.origin)));
        AppendField(&encoded, "output_index",
                    std::to_string(value.output_index));
        AppendField(&encoded, "checked_type", TypeToString(value.checked_type));
        AppendField(&encoded, "is_graph_output",
                    value.is_graph_output ? "1" : "0");
        if (value.origin == internal::ValueOrigin::kConstant) {
            AppendField(&encoded, "constant", ConstantCanonical(value));
        }
        AppendField(&result, "value", encoded);
    }
    return result;
}

shape::GraphTemplate BuildTemplate(const internal::PreparedStaticGraph& prepared,
                                   const Target& target,
                                   const internal::CompilerExecutionContract& contract) {
    if (prepared.partitioned.units.empty()) {
        Reject("a production exact plan requires at least one ordinary compute unit");
    }
    const auto& graph = prepared.partitioned.value_graph;
    std::vector<shape::NamedTensorContract> inputs;
    std::vector<shape::NamedTensorContract> outputs;
    for (const auto& value : graph.values) {
        shape::NamedTensorContract named{ValueName(value.value_id),
                                         ValueContract(value, target)};
        if (value.origin == internal::ValueOrigin::kParameter ||
            value.origin == internal::ValueOrigin::kConstant) {
            inputs.push_back(std::move(named));
        } else {
            outputs.push_back(std::move(named));
        }
    }
    std::vector<shape::UnitSkeleton> units;
    units.reserve(prepared.partitioned.units.size());
    for (const auto& unit : prepared.partitioned.units) {
        shape::UnitSkeleton skeleton{shape::GraphLocalCallLocator(
            ValueName(unit.output_value_ids[0])),
            shape::UnitSemanticKey(shape::kShapeContractVersion,
                                   unit.semantic_key.canonical_bytes()), {}, {}};
        for (int64_t id : unit.input_value_ids) skeleton.input_value_names.push_back(ValueName(id));
        for (int64_t id : unit.output_value_ids) skeleton.output_value_names.push_back(ValueName(id));
        units.push_back(std::move(skeleton));
    }
    std::string capability;
    AppendField(&capability, "kind", "capability.static_exact.v1");
    AppendField(&capability, "boundaries",
                "compiler_entry|post_graph_pass|pre_partition");
    AppendField(&capability, "execution_contract", contract.canonical_bytes);
    AppendField(&capability, "target_snapshot",
                internal::CanonicalTargetSnapshot(target));
    shape::GraphTemplateKey key(
        shape::kShapeContractVersion, GraphCanonical(prepared.partitioned),
        contract.canonical_bytes, capability,
        PartitionCanonical(prepared.partitioned), TargetAbi(target));
    return shape::GraphTemplate(key, shape::ShapeProgram({}, std::move(inputs),
                                                         std::move(outputs)),
                                std::move(units));
}

bool ExactContract(const shape::ConcreteTensorShapeContract& contract,
                   const Target& target) {
    if (contract.logical != contract.physical || contract.logical != contract.valid ||
        contract.layout != "contiguous.row_major" || contract.memory_scope != "global" ||
        contract.alignment != static_cast<int64_t>(contract.abi.element_bytes()) ||
        !(contract.abi.target_backend_abi() == TargetAbi(target))) {
        return false;
    }
    std::vector<int64_t> strides(contract.logical.size(), 1);
    int64_t stride = 1;
    for (size_t i = contract.logical.size(); i > 0; --i) {
        if (contract.logical[i - 1] < 0 ||
            (contract.logical[i - 1] != 0 &&
             stride > std::numeric_limits<int64_t>::max() / contract.logical[i - 1])) return false;
        strides[i - 1] = stride;
        stride *= contract.logical[i - 1];
    }
    return contract.strides == strides &&
        ((contract.abi.dtype() == shape::DataType::kFloat16 && contract.abi.element_bytes() == 2) ||
         (contract.abi.dtype() == shape::DataType::kFloat32 && contract.abi.element_bytes() == 4));
}

bool SameContract(const shape::ConcreteTensorShapeContract& a,
                  const shape::ConcreteTensorShapeContract& b) {
    return a == b && a.logical == a.physical && a.logical == a.valid &&
           a.layout == "contiguous.row_major" && a.memory_scope == "global";
}

bool SameIds(const Array<int64_t>& actual, const Array<int64_t>& expected) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) return false;
    return true;
}

std::string FrozenPlanCanonical(const runtime::ExecutablePlan& plan) {
    plan.Validate();
    std::string result;
    AppendField(&result, "kind", "frozen-static-executable-plan-v1");
    AppendField(&result, "memory_plan_version",
                runtime::internal::kStaticMemoryPlanVersion);
    AppendField(&result, "inputs", IdsCanonical(plan.input_value_ids()));
    AppendField(&result, "constants", IdsCanonical(plan.constant_value_ids()));
    AppendField(&result, "outputs", IdsCanonical(plan.output_value_ids()));
    AppendField(&result, "value_count", std::to_string(plan.values().size()));
    for (const runtime::ValueSpec& value : plan.values()) {
        std::string encoded;
        AppendField(&encoded, "value_id", std::to_string(value->value_id));
        AppendField(&encoded, "storage_id", std::to_string(value->storage_id));
        AppendField(&encoded, "shape", IdsCanonical(value.shape()));
        AppendField(&encoded, "dtype_code", std::to_string(value->dtype.code));
        AppendField(&encoded, "dtype_bits", std::to_string(value->dtype.bits));
        AppendField(&encoded, "dtype_lanes", std::to_string(value->dtype.lanes));
        AppendField(&encoded, "device_type",
                    std::to_string(static_cast<int>(value->device.device_type())));
        AppendField(&encoded, "device_id",
                    std::to_string(value->device.device_id()));
        AppendField(&encoded, "is_input", value->is_input ? "1" : "0");
        AppendField(&encoded, "is_constant", value->is_constant ? "1" : "0");
        AppendField(&encoded, "is_output", value->is_output ? "1" : "0");
        AppendField(&encoded, "is_alias", value->is_alias ? "1" : "0");
        AppendField(&encoded, "is_async_live",
                    value->is_async_live ? "1" : "0");
        AppendField(&result, "value", encoded);
    }
    AppendField(&result, "call_count", std::to_string(plan.calls().size()));
    for (const runtime::KernelCall& call : plan.calls()) {
        std::string encoded;
        AppendField(&encoded, "symbol", std::string(call->symbol));
        AppendField(&encoded, "inputs", IdsCanonical(call.input_value_ids()));
        AppendField(&encoded, "outputs", IdsCanonical(call.output_value_ids()));
        AppendField(&result, "call", encoded);
    }
    return result;
}

const shape::ConcreteTensorShapeContract& ProfileValue(
    const shape::ExactOracle& oracle, const std::string& name) {
    return oracle.profile().Value(name).contract;
}

void VerifyOracle(const shape::GraphTemplate& graph, const shape::ExactOracle& oracle) {
    if (!(oracle.profile().key().graph_template() == graph.key()) ||
        !(oracle.profile().key().graph_template_content() == graph.content_key()) ||
        oracle.profile().key().policy_id() != "exact" ||
        oracle.profile().key().shape_abi_version() != shape::kShapeAbiVersion ||
        !oracle.profile().key().bindings().bindings().empty()) {
        Reject("oracle is not the empty exact profile of this prepared template");
    }
    const shape::ExactOracle rebuilt = shape::InstantiateExactProfile(
        graph, oracle.profile().key().bindings());
    if (!(rebuilt.profile().key() == oracle.profile().key()) ||
        rebuilt.profile().values().size() != oracle.profile().values().size()) {
        Reject("oracle content does not exactly match the prepared template");
    }
    for (size_t i = 0; i < rebuilt.profile().values().size(); ++i) {
        if (rebuilt.profile().values()[i].name != oracle.profile().values()[i].name ||
            !SameContract(rebuilt.profile().values()[i].contract,
                          oracle.profile().values()[i].contract)) {
            Reject("oracle value contract does not exactly match the prepared template");
        }
    }
}

void VerifyArg(const codegen::KernelArgSpec& arg, codegen::KernelArgRole role,
               const shape::ConcreteTensorShapeContract& contract,
               const Target& target) {
    const auto* node = arg.operator->();
    const DLDataType dtype = node->dtype;
    const bool f16 = contract.abi.dtype() == shape::DataType::kFloat16;
    const shape::DeviceKind expected_device = target->device_type == kCPU
        ? shape::DeviceKind::kCpu : shape::DeviceKind::kCuda;
    if (!ExactContract(contract, target) ||
        contract.abi.device().kind() != expected_device ||
        contract.abi.device().id() != static_cast<uint32_t>(target->device_id) ||
        node->role != role || dtype.code != kDLFloat || dtype.bits != (f16 ? 16 : 32) ||
        dtype.lanes != 1 || node->device.device_type() != target->device_type ||
        node->device.device_id() != target->device_id ||
        node->alignment != static_cast<uint64_t>(contract.alignment)) {
        Reject("compiled signature ABI does not match exact shape contract");
    }
    const Array<int64_t> shape = arg.shape();
    if (shape.size() != contract.logical.size()) Reject("compiled signature rank mismatch");
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != contract.logical[i]) Reject("compiled signature extent mismatch");
    }
}

void VerifyVariant(
    const shape::GraphTemplate& graph,
    const std::vector<shape::UnitSpecializationRequest>& requests,
    const internal::PreparedCompilerGraph& prepared,
    const CompileConfig& config,
    const internal::CompilerExecutionContract& contract,
    const shape::ExactOracle& oracle, const CompiledGraph& compiled) {
    const auto& partitioned = prepared.graph.partitioned;
    if (requests.size() != partitioned.units.size() ||
        compiled.plan.calls().size() != requests.size() ||
        compiled.artifact_pins.size() != requests.size() ||
        compiled.module.entry_count() != requests.size() ||
        compiled.plan.values().size() != partitioned.value_graph.values.size()) {
        Reject("compiled module/plan/pin cardinality does not match exact requests");
    }
    for (size_t i = 0; i < compiled.plan.values().size(); ++i) {
        const runtime::ValueSpec& value = compiled.plan.values()[i];
        const auto& source = partitioned.value_graph.values[i];
        const auto& exact = ProfileValue(oracle, ValueName(source.value_id));
        const bool f16 = exact.abi.dtype() == shape::DataType::kFloat16;
        if (value->value_id != source.value_id ||
            value->is_input !=
                (source.origin == internal::ValueOrigin::kParameter) ||
            value->is_constant !=
                (source.origin == internal::ValueOrigin::kConstant) ||
            value->is_output != source.is_graph_output ||
            !ExactContract(exact, config->target) ||
            value->dtype.code != kDLFloat ||
            value->dtype.bits != (f16 ? 16 : 32) ||
            value->dtype.lanes != 1 ||
            value->device.device_type() != config->target->device_type ||
            value->device.device_id() != config->target->device_id ||
            value.shape().size() != exact.logical.size()) {
            Reject("compiled plan value does not match the exact profile ABI");
        }
        const Array<int64_t> shape = value.shape();
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            if (shape[axis] != exact.logical[axis]) {
                Reject("compiled plan value extent does not match the exact profile");
            }
        }
    }
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        const auto& unit = partitioned.units[i];
        if (request.ordered_call_index != i ||
            request.call_locator.value() != ValueName(unit.output_value_ids[0]) ||
            !(request.shape_profile_key == oracle.profile().key()) ||
            request.artifact_key.unit_semantic_key().normalized_unit_fingerprint() !=
                unit.semantic_key.canonical_bytes() ||
            request.artifact_key.pipeline_fingerprint() !=
                contract.canonical_bytes ||
            request.artifact_key.capability_fingerprint() !=
                graph.key().capability_fingerprint() ||
            !(request.artifact_key.target_backend_abi() == TargetAbi(config->target)) ||
            request.artifact_key.ordered_inputs() != request.ordered_inputs ||
            request.artifact_key.ordered_outputs() != request.ordered_outputs ||
            !shape::MatchesExactSignatureDigest(
                request.signature_digest, request.ordered_inputs,
                request.ordered_outputs) ||
            request.ordered_inputs.size() != unit.input_value_ids.size() ||
            request.ordered_outputs.size() != unit.output_value_ids.size()) {
            Reject("exact request routing, profile, artifact, or semantic identity drifted");
        }
        for (size_t j = 0; j < request.ordered_inputs.size(); ++j) {
            const auto& value = request.ordered_inputs[j];
            if (!ExactContract(value, config->target) ||
                !SameContract(value, ProfileValue(
                    oracle, ValueName(unit.input_value_ids[j])))) {
                Reject("exact request input contract drifted");
            }
        }
        for (size_t j = 0; j < request.ordered_outputs.size(); ++j) {
            const auto& value = request.ordered_outputs[j];
            if (!ExactContract(value, config->target) ||
                !SameContract(value, ProfileValue(
                    oracle, ValueName(unit.output_value_ids[j])))) {
                Reject("exact request output contract drifted");
            }
        }
        const runtime::KernelCall& call = compiled.plan.calls()[i];
        if (!(call->symbol == unit.symbol) || !compiled.module.HasFunction(call->symbol) ||
            !SameIds(call.input_value_ids(), unit.input_value_ids) ||
            !SameIds(call.output_value_ids(), unit.output_value_ids)) {
            Reject("compiled plan routing does not match prepared partition");
        }
        const codegen::KernelSignature signature = compiled.module.signature(call->symbol);
        const Array<codegen::KernelArgSpec> args = signature.arguments();
        if (args.size() != unit.input_value_ids.size() + unit.output_value_ids.size()) {
            Reject("compiled signature arity does not match exact request");
        }
        for (size_t j = 0; j < unit.input_value_ids.size(); ++j) {
            const auto& value = partitioned.value_graph.values[static_cast<size_t>(unit.input_value_ids[j])];
            VerifyArg(args[j], value.origin == internal::ValueOrigin::kConstant
                          ? codegen::KernelArgRole::kConstant : codegen::KernelArgRole::kInput,
                      request.ordered_inputs[j], config->target);
        }
        for (size_t j = 0; j < unit.output_value_ids.size(); ++j) {
            VerifyArg(args[unit.input_value_ids.size() + j], codegen::KernelArgRole::kOutput,
                      request.ordered_outputs[j], config->target);
        }
        const codegen::KernelLaunchMetadata metadata = compiled.module.launch_metadata(call->symbol);
        if (metadata->device.device_type() != config->target->device_type ||
            metadata->device.device_id() != config->target->device_id ||
            (config->target->kind == "llvm" && metadata->backend != codegen::CodeGenBackend::kLLVM) ||
            (config->target->kind == "cuda" && metadata->backend != codegen::CodeGenBackend::kCUDA)) {
            Reject("compiled launch metadata does not match target/backend");
        }
        const ArtifactKey expected = internal::BuildPrimitiveArtifactKey(
            unit.semantic_key, config->target, contract.canonical_bytes,
            contract.schedule_version.c_str(), contract.backend_version.c_str());
        const ArtifactPin& public_pin = compiled.artifact_pins[i];
        if (!public_pin.defined()) {
            Reject("production artifact pin is undefined");
        }
        const internal::PrimitiveArtifactPin primitive_pin =
            internal::ProductionArtifactAccess::Pin(public_pin);
        const ArtifactRecord& record = public_pin.handle().record();
        if (!(record.artifact_key == expected) ||
            !(primitive_pin.key() == expected) ||
            record.signature_digest != profiling::HashText(
                primitive_pin.artifact().signature.ToString()) ||
            record.launch_metadata_digest != profiling::HashText(
                primitive_pin.artifact().launch_metadata.ToString()) ||
            metadata.ToString() !=
                primitive_pin.artifact().launch_metadata.ToString()) {
            Reject("production artifact pin does not retain the real full artifact key/signature/launch metadata");
        }
    }
}

}  // namespace

struct PreparedGraphTemplate::Impl final {
    Impl(CompileConfig config, internal::CompilerExecutionContract contract,
         internal::PreparedCompilerGraph prepared, shape::GraphTemplate graph,
         ShapeExactPreparationCounters counters)
        : config(std::move(config)), contract(std::move(contract)),
          prepared(std::move(prepared)), graph(std::move(graph)),
          counters(counters) {}
    CompileConfig config;
    internal::CompilerExecutionContract contract;
    internal::PreparedCompilerGraph prepared;
    shape::GraphTemplate graph;
    ShapeExactPreparationCounters counters;
};

struct ExactPlanVariant::Impl final {
    CompiledModule module;
    runtime::ExecutablePlan plan;
    shape::ShapeProfileKey profile;
    shape::PlanVariantKey key;
    std::vector<ArtifactPin> pins;
};

PreparedGraphTemplate::PreparedGraphTemplate() = default;
PreparedGraphTemplate::PreparedGraphTemplate(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
PreparedGraphTemplate::~PreparedGraphTemplate() = default;
PreparedGraphTemplate::PreparedGraphTemplate(const PreparedGraphTemplate&) = default;
PreparedGraphTemplate& PreparedGraphTemplate::operator=(const PreparedGraphTemplate&) = default;
PreparedGraphTemplate::PreparedGraphTemplate(PreparedGraphTemplate&&) noexcept = default;
PreparedGraphTemplate& PreparedGraphTemplate::operator=(PreparedGraphTemplate&&) noexcept = default;
const shape::GraphTemplate& PreparedGraphTemplate::graph_template() const {
    if (!impl_) Reject("prepared template is undefined");
    return impl_->graph;
}
const ShapeExactPreparationCounters& PreparedGraphTemplate::counters() const {
    if (!impl_) Reject("prepared template is undefined");
    return impl_->counters;
}
size_t PreparedGraphTemplate::unit_count() const { return graph_template().ordered_units().size(); }
bool PreparedGraphTemplate::multi_profile_supported() const noexcept { return false; }

ExactPlanVariant::ExactPlanVariant() = default;
ExactPlanVariant::ExactPlanVariant(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
ExactPlanVariant::~ExactPlanVariant() = default;
ExactPlanVariant::ExactPlanVariant(const ExactPlanVariant&) = default;
ExactPlanVariant& ExactPlanVariant::operator=(const ExactPlanVariant&) = default;
ExactPlanVariant::ExactPlanVariant(ExactPlanVariant&&) noexcept = default;
ExactPlanVariant& ExactPlanVariant::operator=(ExactPlanVariant&&) noexcept = default;
const CompiledModule& ExactPlanVariant::module() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->module; }
const runtime::ExecutablePlan& ExactPlanVariant::plan() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->plan; }
const shape::ShapeProfileKey& ExactPlanVariant::shape_profile_key() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->profile; }
const shape::PlanVariantKey& ExactPlanVariant::plan_variant_key() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->key; }
const std::vector<ArtifactPin>& ExactPlanVariant::artifact_pins() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->pins; }

bool ProductionExactShapeAdapter::IsEnabled() noexcept {
#if KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    return true;
#else
    return false;
#endif
}

PreparedGraphTemplate ProductionExactShapeAdapter::PrepareGraphTemplate(
    Function function, CompileConfig config) {
    RequireEnabled();
    const Function relay_snapshot = RelaySnapshotCloner().Clone(function);
    config.Validate();
    CompileConfig clone = CompileConfig::Create(
        internal::CloneTargetSnapshot(config->target), config->opt_level);
    clone->profile_options = config->profile_options;
    ShapeExactPreparationCounters counters;
    internal::CompilerExecutionContract contract =
        internal::ResolveCompilerExecutionContract(clone);
    ++counters.execution_contract_resolutions;
    internal::PreparedCompilerGraph prepared = internal::PrepareCompilerGraph(
        relay_snapshot, clone, contract);
    // Capability/registry identity checks have completed.  Detach every
    // lowering descriptor before the prepared graph becomes observable so
    // Assemble never rereads a caller-accessible registry OpNode.
    FreezePreparedOperators(&prepared);
    counters.relay_graph_pipelines = prepared.relay_graph_pipelines;
    counters.capability_boundary_checks =
        prepared.capability_boundary_checks;
    counters.value_graph_builds = prepared.value_graph_builds;
    counters.partitions = prepared.partitions;
    shape::GraphTemplate graph =
        BuildTemplate(prepared.graph, clone->target, contract);
    return PreparedGraphTemplate(
        std::make_shared<PreparedGraphTemplate::Impl>(
            std::move(clone), std::move(contract), std::move(prepared),
            std::move(graph), counters));
}

shape::ExactOracle ProductionExactShapeAdapter::InstantiateExactProfile(
    const PreparedGraphTemplate& prepared, const shape::BindingSet& bindings) {
    RequireEnabled();
    if (!bindings.bindings().empty()) {
        Reject("Relay is concrete-only; non-empty bindings and multi-profile exact requests are unsupported");
    }
    return shape::InstantiateExactProfile(prepared.graph_template(), bindings);
}

ExactPlanVariant ProductionExactShapeAdapter::AssembleExactPlan(
    const PreparedGraphTemplate& prepared, const shape::ExactOracle& oracle) {
    RequireEnabled();
    if (!prepared.impl_) Reject("prepared template is undefined");
    VerifyOracle(prepared.impl_->graph, oracle);
    const std::vector<shape::UnitSpecializationRequest> requests =
        shape::MakeExactSpecializationRequests(prepared.impl_->graph, oracle);
    RequireBackendAvailable(prepared.impl_->config->target);
    CompiledGraph compiled = internal::FinishCompilerGraph(
        prepared.impl_->prepared, prepared.impl_->config,
        prepared.impl_->contract);
    VerifyVariant(prepared.impl_->graph, requests,
                  prepared.impl_->prepared, prepared.impl_->config,
                  prepared.impl_->contract, oracle, compiled);
    const std::string frozen_plan = FrozenPlanCanonical(compiled.plan);
    std::vector<std::string> identities;
    identities.reserve(compiled.plan.calls().size());
    for (size_t i = 0; i < compiled.plan.calls().size(); ++i) {
        const ArtifactRecord& record =
            compiled.artifact_pins[i].handle().record();
        std::string identity;
        AppendField(&identity, "kind", "selected-static-exact-call-v1");
        AppendField(&identity, "call_locator",
                    prepared.impl_->graph.ordered_units()[i]
                        .call_locator.value());
        AppendField(&identity, "link_symbol",
                    std::string(compiled.plan.calls()[i]->symbol));
        AppendField(&identity, "generation", "0");
        AppendField(&identity, "shape_artifact",
                    requests[i].artifact_key.CanonicalBytes());
        AppendField(&identity, "shape_signature",
                    requests[i].signature_digest.value());
        AppendField(&identity, "production_artifact",
                    record.artifact_key.canonical_bytes());
        AppendField(&identity, "production_signature",
                    record.signature_digest);
        AppendField(&identity, "production_launch", record.launch_metadata_digest);
        if (i == 0) AppendField(&identity, "frozen_plan", frozen_plan);
        identities.push_back(std::move(identity));
    }
    auto impl = std::make_shared<ExactPlanVariant::Impl>(ExactPlanVariant::Impl{
        compiled.module, compiled.plan, oracle.profile().key(),
        shape::PlanVariantKey(prepared.impl_->graph.key(), oracle.profile().key(),
                              std::move(identities)),
        compiled.artifact_pins});
    return ExactPlanVariant(std::move(impl));
}

}  // namespace kxc::api::experimental::shape_exact::v1
