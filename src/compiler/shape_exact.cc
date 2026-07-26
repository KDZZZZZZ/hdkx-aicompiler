#include "kxc/compiler/shape_exact.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "internal/execution_contract.h"
#include "internal/lowered_graph.h"
#include "internal/primitive_cache.h"
#include "internal/primitive_compiler.h"
#include "../runtime/internal/memory_plan.h"
#include "support/canonical.h"
#include "support/hash.h"
#include "kxc/pass/context.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/visitor.h"
#include "kxc/tir/printer/print_ir.h"
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
namespace shape =
    kxc::api::experimental::shape_specialization::v1;

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

uint64_t PlannedStorageBytes(const runtime::ExecutablePlan& plan) {
    std::unordered_map<int64_t, uint64_t> bytes_by_storage;
    for (const auto& value : plan.values()) {
        uint64_t elements = 1;
        for (int64_t dimension : value.shape()) {
            if (dimension < 0) return std::numeric_limits<uint64_t>::max();
            if (dimension != 0 &&
                elements > std::numeric_limits<uint64_t>::max() /
                               static_cast<uint64_t>(dimension)) {
                return std::numeric_limits<uint64_t>::max();
            }
            elements *= static_cast<uint64_t>(dimension);
        }
        const uint64_t element_bytes =
            static_cast<uint64_t>(value->dtype.bits / 8) *
            static_cast<uint64_t>(value->dtype.lanes);
        if (element_bytes != 0 &&
            elements > std::numeric_limits<uint64_t>::max() / element_bytes) {
            return std::numeric_limits<uint64_t>::max();
        }
        bytes_by_storage[value->storage_id] = std::max(
            bytes_by_storage[value->storage_id], elements * element_bytes);
    }
    uint64_t total = 0;
    for (const auto& item : bytes_by_storage) {
        if (total > std::numeric_limits<uint64_t>::max() - item.second) {
            return std::numeric_limits<uint64_t>::max();
        }
        total += item.second;
    }
    return total;
}

void AddPrimitiveBatchFields(
    profiling::ScopedSpan* span,
    const internal::PreparedCompilerGraph& prepared,
    const internal::CompiledPrimitiveBatch& batch) {
    const runtime::ExecutablePlan plan =
        internal::BuildStaticExecutablePlan(prepared.graph);
    span->AddMetric("primitive_count",
                    static_cast<double>(batch.primitives.size()));
    span->AddMetric("value_count", static_cast<double>(plan.values().size()));
    std::unordered_set<int64_t> storage_ids;
    for (const auto& value : plan.values()) storage_ids.insert(value->storage_id);
    span->AddMetric("storage_slot_count",
                    static_cast<double>(storage_ids.size()));
    span->AddMetric("storage_reuse_count",
                    static_cast<double>(plan.values().size() - storage_ids.size()));
    span->AddMetric("planned_peak_storage_bytes",
                    static_cast<double>(PlannedStorageBytes(plan)));
    size_t cache_hits = 0;
    for (const internal::CompiledPrimitive& primitive : batch.primitives) {
        const internal::PrimitiveUnit& unit = prepared.graph.partitioned.units.at(
            static_cast<std::size_t>(primitive.unit_id));
        const std::string prefix = "unit." +
            std::to_string(primitive.unit_id) + ".";
        std::ostringstream stream;
        tir::printer::DumpPrimFunc(primitive.diagnostic_tir, stream);
        const std::string text = stream.str();
        const internal::CachedPrimitive& artifact = primitive.pin.artifact();
        span->AddField(prefix + "symbol", std::string(unit.symbol));
        span->AddField(
            prefix + "operator", std::string(unit.call.spec.name) + "@v" +
                std::to_string(unit.call.spec.schema_version));
        span->AddField(prefix + "ir_hash", support::HashText(text));
        span->AddMetric(prefix + "ir_bytes",
                        static_cast<double>(text.size()));
        span->AddField(prefix + "backend",
                       artifact.launch_metadata->backend ==
                               codegen::CodeGenBackend::kLLVM
                           ? "llvm"
                           : "cuda");
        span->AddMetric(prefix + "cache_hit", primitive.cache_hit ? 1.0 : 0.0);
        if (primitive.cache_hit) ++cache_hits;
    }
    span->AddMetric("cache_hits", static_cast<double>(cache_hits));
    span->AddMetric(
        "cache_hit_rate", batch.primitives.empty()
                              ? 0.0
                              : static_cast<double>(cache_hits) /
                                    static_cast<double>(batch.primitives.size()));
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

shape::TensorShapeContract ValueContract(const internal::ValueInfo& value) {
    const auto* type = value.checked_type.As<TensorTypeNode>();
    if (!type) Reject("prepared value has no TensorType");
    std::vector<shape::DimExpr> dimensions;
    dimensions.reserve(type->shape.size());
    for (int64_t extent : type->shape) {
        if (extent < 0) Reject("negative or legacy -1 dimension is unsupported");
        dimensions.push_back(shape::DimExpr::Const(extent));
    }
    return {shape::LogicalShape(dimensions),
            shape::PhysicalCapacity(dimensions),
            shape::ValidExtent(dimensions)};
}

std::string ValueName(int64_t id) { return "value." + std::to_string(id); }

void AppendField(std::string* bytes, const std::string& name,
                 const std::string& value) {
    support::CanonicalBytesEncoder field;
    field.Field(name, value);
    *bytes += std::move(field).Take();
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
        AppendField(&encoded, "ordinal", std::to_string(unit.id));
        AppendField(&encoded, "semantic", unit.semantic_key.canonical_bytes());
        AppendField(&encoded, "inputs",
                    IdsCanonical(unit.boundary_input_value_ids));
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
        AppendField(&encoded, "value_id", std::to_string(value.id));
        AppendField(&encoded, "origin",
                    std::to_string(static_cast<int>(value.origin)));
        AppendField(&encoded, "checked_type", TypeToString(value.checked_type));
        AppendField(&encoded, "device", value.device.ToString());
        AppendField(&encoded, "source_locator", value.source_locator);
        const bool is_graph_output =
            std::find(partitioned.output_value_ids.begin(),
                      partitioned.output_value_ids.end(),
                      value.id) != partitioned.output_value_ids.end();
        AppendField(&encoded, "is_graph_output",
                    is_graph_output ? "1" : "0");
        if (value.origin == internal::ValueOrigin::kConstant) {
            AppendField(&encoded, "constant", ConstantCanonical(value));
        }
        AppendField(&result, "value", encoded);
    }
    return result;
}

shape::GraphTemplate BuildTemplate(
    const internal::PreparedCompilerGraph& prepared) {
    if (prepared.graph.partitioned.units.empty()) {
        Reject("a production exact plan requires at least one ordinary compute unit");
    }
    const auto& partitioned = prepared.graph.partitioned;
    const auto& graph = partitioned.value_graph;
    std::vector<shape::NamedTensorContract> inputs;
    std::vector<shape::NamedTensorContract> outputs;
    for (const auto& value : graph.values) {
        shape::NamedTensorContract named{ValueName(value.id),
                                         ValueContract(value)};
        if (value.origin == internal::ValueOrigin::kParameter ||
            value.origin == internal::ValueOrigin::kConstant) {
            inputs.push_back(std::move(named));
        } else {
            outputs.push_back(std::move(named));
        }
    }
    std::vector<shape::UnitSkeleton> units;
    units.reserve(partitioned.units.size());
    for (const auto& unit : partitioned.units) {
        shape::UnitSkeleton skeleton{shape::GraphLocalCallLocator(
            ValueName(unit.output_value_ids[0])),
            unit.semantic_key, {}, {}};
        for (int64_t id : unit.boundary_input_value_ids) {
            skeleton.input_value_names.push_back(ValueName(id));
        }
        for (int64_t id : unit.output_value_ids) skeleton.output_value_names.push_back(ValueName(id));
        units.push_back(std::move(skeleton));
    }
    return shape::GraphTemplate(
        prepared.graph_semantic_key,
        shape::ShapeProgram({}, std::move(inputs), std::move(outputs)),
        std::move(units));
}

bool ExactContract(const shape::ConcreteTensorShapeContract& contract) {
    if (contract.logical != contract.physical ||
        contract.logical != contract.valid ||
        (!contract.axis_names.empty() &&
         contract.logical.size() != contract.axis_names.size())) {
        return false;
    }
    for (int64_t extent : contract.logical) {
        if (extent < 0) return false;
    }
    return true;
}

bool SameContract(const shape::ConcreteTensorShapeContract& a,
                  const shape::ConcreteTensorShapeContract& b) {
    return a == b && a.logical == a.physical && a.logical == a.valid &&
           ExactContract(a);
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
    if (oracle.profile().key().graph_semantic_key() != graph.key() ||
        oracle.profile().policy_id() != "exact" ||
        oracle.profile().shape_abi_version() !=
            shape::kShapeProfileAbiVersion ||
        !oracle.profile().bindings().bindings().empty()) {
        Reject("oracle is not the empty exact profile of this prepared template");
    }
    const shape::ExactOracle rebuilt = shape::InstantiateExactProfile(
        graph, oracle.profile().bindings());
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

bool SameDType(const DLDataType& left, const DLDataType& right) {
    return left.code == right.code && left.bits == right.bits &&
           left.lanes == right.lanes;
}

void VerifyArg(const codegen::KernelArgSpec& arg, codegen::KernelArgRole role,
                const shape::ConcreteTensorShapeContract& contract,
                const TensorTypeNode* expected_type,
                const Target& target) {
    if (!expected_type) Reject("compiled signature source is not a tensor");
    const auto* node = arg.operator->();
    const DLDataType expected_dtype =
        runtime::DataTypeFromString(expected_type->dtype);
    if (!ExactContract(contract) || node->role != role ||
        !SameDType(node->dtype, expected_dtype) ||
        node->device.device_type() != target->device_type ||
        node->device.device_id() != target->device_id) {
        Reject("compiled signature ABI does not match exact shape contract");
    }
    const Array<int64_t> shape = arg.shape();
    if (shape.size() != contract.logical.size() ||
        shape.size() != expected_type->shape.size()) {
        Reject("compiled signature rank mismatch");
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != contract.logical[i] ||
            shape[i] != expected_type->shape[i]) {
            Reject("compiled signature extent mismatch");
        }
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
        compiled.plan().calls().size() != requests.size() ||
        compiled.artifact_pins().size() != requests.size() ||
        compiled.module().entry_count() != requests.size() ||
        compiled.plan().values().size() != partitioned.value_graph.values.size()) {
        Reject("compiled module/plan/pin cardinality does not match exact requests");
    }
    for (size_t i = 0; i < compiled.plan().values().size(); ++i) {
        const runtime::ValueSpec& value = compiled.plan().values()[i];
        const auto& source = partitioned.value_graph.values[i];
        const auto& exact = ProfileValue(oracle, ValueName(source.id));
        const auto* expected_type = source.checked_type.As<TensorTypeNode>();
        if (!expected_type) Reject("compiled plan source is not a tensor");
        const DLDataType expected_dtype =
            runtime::DataTypeFromString(expected_type->dtype);
        const bool is_graph_output =
            std::find(partitioned.output_value_ids.begin(),
                      partitioned.output_value_ids.end(),
                      source.id) != partitioned.output_value_ids.end();
        if (value->value_id != source.id ||
            value->is_input !=
                (source.origin == internal::ValueOrigin::kParameter) ||
            value->is_constant !=
                (source.origin == internal::ValueOrigin::kConstant) ||
            value->is_output != is_graph_output ||
            !ExactContract(exact) ||
            !SameDType(value->dtype, expected_dtype) ||
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
            request.unit_semantic_key != unit.semantic_key ||
            !shape::MatchesExactSignatureDigest(
                request.signature_digest, request.ordered_inputs,
                request.ordered_outputs) ||
            request.ordered_inputs.size() !=
                unit.boundary_input_value_ids.size() ||
            request.ordered_outputs.size() != unit.output_value_ids.size()) {
            Reject("exact request routing, profile, artifact, or semantic identity drifted");
        }
        for (size_t j = 0; j < request.ordered_inputs.size(); ++j) {
            const auto& value = request.ordered_inputs[j];
            if (!ExactContract(value) ||
                !SameContract(value, ProfileValue(
                    oracle,
                    ValueName(unit.boundary_input_value_ids[j])))) {
                Reject("exact request input contract drifted");
            }
        }
        for (size_t j = 0; j < request.ordered_outputs.size(); ++j) {
            const auto& value = request.ordered_outputs[j];
            if (!ExactContract(value) ||
                !SameContract(value, ProfileValue(
                    oracle, ValueName(unit.output_value_ids[j])))) {
                Reject("exact request output contract drifted");
            }
        }
        const runtime::KernelCall& call = compiled.plan().calls()[i];
        if (!(call->symbol == unit.symbol) || !compiled.module().HasFunction(call->symbol) ||
            !SameIds(call.input_value_ids(),
                     unit.boundary_input_value_ids) ||
            !SameIds(call.output_value_ids(), unit.output_value_ids)) {
            Reject("compiled plan routing does not match prepared partition");
        }
        const codegen::KernelSignature signature = compiled.module().signature(call->symbol);
        const Array<codegen::KernelArgSpec> args = signature.arguments();
        if (args.size() != unit.boundary_input_value_ids.size() +
                               unit.output_value_ids.size()) {
            Reject("compiled signature arity does not match exact request");
        }
        for (size_t j = 0; j < unit.boundary_input_value_ids.size(); ++j) {
            const auto& value = partitioned.value_graph.values[
                static_cast<size_t>(unit.boundary_input_value_ids[j])];
            VerifyArg(args[j], value.origin == internal::ValueOrigin::kConstant
                          ? codegen::KernelArgRole::kConstant : codegen::KernelArgRole::kInput,
                      request.ordered_inputs[j],
                      value.checked_type.As<TensorTypeNode>(),
                      config->target);
        }
        for (size_t j = 0; j < unit.output_value_ids.size(); ++j) {
            const auto& value = partitioned.value_graph.values[
                static_cast<size_t>(unit.output_value_ids[j])];
            VerifyArg(args[unit.boundary_input_value_ids.size() + j],
                      codegen::KernelArgRole::kOutput,
                      request.ordered_outputs[j],
                      value.checked_type.As<TensorTypeNode>(),
                      config->target);
        }
        const codegen::KernelLaunchMetadata metadata = compiled.module().launch_metadata(call->symbol);
        if (metadata->device.device_type() != config->target->device_type ||
            metadata->device.device_id() != config->target->device_id ||
            (config->target->kind == "llvm" && metadata->backend != codegen::CodeGenBackend::kLLVM) ||
            (config->target->kind == "cuda" && metadata->backend != codegen::CodeGenBackend::kCUDA)) {
            Reject("compiled launch metadata does not match target/backend");
        }
        const PrimitiveArtifactKey expected =
            internal::BuildPrimitiveArtifactKey(
            unit.semantic_key, config->target, contract.canonical_bytes,
            contract.schedule_version.c_str(), contract.backend_version.c_str());
        const ArtifactPin& public_pin = compiled.artifact_pins()[i];
        if (!public_pin.defined()) {
            Reject("production artifact pin is undefined");
        }
        const internal::PrimitiveArtifactPin primitive_pin =
            internal::ArtifactPinAccess::Unwrap(public_pin);
        const ArtifactRecord& record = public_pin.record();
        if (!(record.artifact_key == expected) ||
            !(primitive_pin.key() == expected) ||
            record.signature_digest != support::HashText(
                primitive_pin.artifact().signature.CanonicalBytes()) ||
            record.launch_metadata_digest != support::HashText(
                primitive_pin.artifact().launch_metadata.CanonicalBytes()) ||
            metadata.CanonicalBytes() !=
                primitive_pin.artifact().launch_metadata.CanonicalBytes()) {
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
    CompiledGraph graph;
    ShapeProfileKey profile;
    PlanVariantKey key;
    DispatchKey dispatch;
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
const CompiledModule& ExactPlanVariant::module() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->graph.module(); }
const runtime::ExecutablePlan& ExactPlanVariant::plan() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->graph.plan(); }
const ShapeProfileKey& ExactPlanVariant::shape_profile_key() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->profile; }
const PlanVariantKey& ExactPlanVariant::plan_variant_key() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->key; }
const DispatchKey& ExactPlanVariant::dispatch_key() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->dispatch; }
const std::vector<ArtifactPin>& ExactPlanVariant::artifact_pins() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->graph.artifact_pins(); }

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
    ShapeExactPreparationCounters counters;
    internal::CompilerExecutionContract contract =
        internal::ResolveCompilerExecutionContract(config);
    ++counters.execution_contract_resolutions;
    internal::PreparedCompilerGraph prepared = internal::PrepareCompilerGraph(
        relay_snapshot, config, contract);
    // Capability/registry identity checks have completed.  Detach every
    // lowering descriptor before the prepared graph becomes observable so
    // Assemble never rereads a caller-accessible registry OpNode.
    FreezePreparedOperators(&prepared);
    counters.relay_graph_pipelines = prepared.relay_graph_pipelines;
    counters.capability_boundary_checks =
        prepared.capability_boundary_checks;
    counters.value_graph_builds = prepared.value_graph_builds;
    counters.partitions = prepared.partitions;
    shape::GraphTemplate graph = BuildTemplate(prepared);
    return PreparedGraphTemplate(
        std::make_shared<PreparedGraphTemplate::Impl>(
            std::move(config), std::move(contract), std::move(prepared),
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
    const internal::PreparedCompilerGraph& compiler_prepared =
        prepared.impl_->prepared;
    const CompileConfig& config = prepared.impl_->config;
    const internal::CompilerExecutionContract& contract =
        prepared.impl_->contract;
    RequireBackendAvailable(config->target);
    if (compiler_prepared.execution_contract_canonical !=
            contract.canonical_bytes ||
        internal::CanonicalTargetSnapshot(compiler_prepared.target) !=
            internal::CanonicalTargetSnapshot(config->target) ||
        internal::CanonicalTargetSnapshot(compiler_prepared.graph.target) !=
            internal::CanonicalTargetSnapshot(config->target) ||
        compiler_prepared.graph.device !=
            Device(config->target->device_type, config->target->device_id) ||
        std::string(compiler_prepared.graph.pipeline_fingerprint) !=
            contract.fingerprint) {
        Reject("prepared target or execution contract changed before assembly");
    }
    const auto stage_event = [&config](const char* stage) {
        profiling::EventSpec event;
        event.component = "compiler";
        event.event_type = "compile_stage";
        event.pass_name = stage;
        event.fields = profiling::MakeFields({
            {"stage", stage},
            {"target_kind", config->target->kind},
            {"device_type",
             std::to_string(static_cast<int>(config->target->device_type))},
            {"device_id", std::to_string(config->target->device_id)},
            {"opt_level", std::to_string(config->opt_level)},
        });
        return event;
    };
    const auto profile_context = compiler_prepared.profile_context;
    const std::string& run_id = compiler_prepared.profile_run_id;
    profiling::ActivationScope activation(profile_context, run_id);
    internal::CompiledPrimitiveBatch batch;
    {
        const PassContext pass_context = PassContext::MergeTarget(
            relay::PassContextFromRelay(
                compiler_prepared.graph.partitioned.value_graph.function),
            config->target);
        PassContext::Scope pass_scope(pass_context);
        profiling::ScopedSpan compile_span(
            profile_context, stage_event("compile_primitives"), run_id);
        try {
            batch = internal::CompilePrimitiveUnits(
                compiler_prepared.graph.partitioned.units,
                compiler_prepared.graph.partitioned.value_graph.values,
                config, contract);
            AddPrimitiveBatchFields(&compile_span, compiler_prepared, batch);
        } catch (const std::exception& error) {
            compile_span.SetStatus("error");
            compile_span.SetMessage(error.what());
            throw std::runtime_error(
                std::string("Compiler stage 'compile_primitives' failed: ") +
                error.what());
        }
    }
    std::vector<internal::PrimitiveArtifactPin> pins;
    pins.reserve(batch.primitives.size());
    for (const internal::CompiledPrimitive& primitive : batch.primitives) {
        pins.push_back(primitive.pin);
    }
    profiling::ScopedSpan assemble_span(
        profile_context, stage_event("assemble"), run_id);
    CompiledGraph compiled = internal::AssembleCompiledGraph(
        compiler_prepared, pins, batch.constants);
    AddPrimitiveBatchFields(&assemble_span, compiler_prepared, batch);
    if (profile_context) profile_context->Flush();
    VerifyVariant(prepared.impl_->graph, requests, compiler_prepared, config,
                  contract, oracle, compiled);
    std::vector<OrderedArtifactSelectionIdentity> selections;
    selections.reserve(compiled.plan().calls().size());
    for (size_t i = 0; i < compiled.plan().calls().size(); ++i) {
        const ArtifactRecord& record =
            compiled.artifact_pins()[i].record();
        selections.push_back(OrderedArtifactSelectionIdentity{
            i, std::string(compiled.plan().calls()[i]->symbol),
            record.artifact_key, 0});
    }
    auto impl = std::make_shared<ExactPlanVariant::Impl>(ExactPlanVariant::Impl{
        std::move(compiled), oracle.profile().key(),
        BuildPlanVariantKey(
            prepared.impl_->graph.key(), oracle.profile().key(),
            selections, runtime::internal::kStaticMemoryPlanVersion),
        BuildStaticExactDispatchKey(prepared.impl_->graph.key(),
                                    oracle.profile().key())});
    return ExactPlanVariant(std::move(impl));
}

}  // namespace kxc::api::experimental::shape_exact::v1
