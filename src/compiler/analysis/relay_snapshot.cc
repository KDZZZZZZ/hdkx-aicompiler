/*! \file src/compiler/analysis/relay_snapshot.cc
 * \brief Relay AST deep-cloning for prepared-graph snapshots.
 */

#include "../internal/relay_snapshot.h"

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "kxc/relay/visitor.h"
#include "kxc/runtime/device_api.h"

namespace kxc::api::internal {
namespace {

[[noreturn]] void Reject(const std::string& message) {
    throw std::invalid_argument("Relay snapshot: " + message);
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
        // M3 形状链：折叠前的 gather/concat 与折叠后的受限形状值算子都要
        // 能进入准备快照，否则受限入口无法接收未折叠的形状表达式。
        } else if (const auto* node = source.As<relay::GatherAttrsNode>()) {
            result = relay::GatherAttrs::Create(node->axis);
        } else if (const auto* node = source.As<relay::ConcatenateAttrsNode>()) {
            result = relay::ConcatenateAttrs::Create(node->axis);
        } else if (const auto* node = source.As<relay::SliceAttrsNode>()) {
            result = relay::SliceAttrs::Create(CloneIntArray(node->starts), CloneIntArray(node->ends),
                CloneIntArray(node->axes), CloneIntArray(node->steps), node->prefix_axis, node->extent_axis,
                node->window_size, node->window_extent_axis);
        } else if (const auto* node = source.As<relay::ShapeExprAttrsNode>()) {
            result = relay::ShapeExprAttrs::Create(
                CloneIntArray(node->expr_kinds), CloneIntArray(node->expr_values),
                CloneIntArray(node->expr_axes));
        } else if (const auto* node =
                       source.As<relay::ReshapeDynamicAttrsNode>()) {
            result = relay::ReshapeDynamicAttrs::Create(
                CloneIntArray(node->expr_kinds), CloneIntArray(node->expr_values),
                CloneIntArray(node->expr_axes));
        } else if (const auto* node = source.As<relay::ExpandDynamicAttrsNode>()) {
            result = relay::ExpandDynamicAttrs::Create(
                CloneIntArray(node->expr_kinds), CloneIntArray(node->expr_values),
                CloneIntArray(node->expr_axes));
        } else if (const auto* node =
                       source.As<relay::ConstantOfShapeAttrsNode>()) {
            result = relay::ConstantOfShapeAttrs::Create(
                CloneIntArray(node->target), node->dtype_code, node->value,
                CloneIntArray(node->expr_kinds), CloneIntArray(node->expr_values), CloneIntArray(node->expr_axes));
        } else if (const auto* node = source.As<relay::TriluAttrsNode>()) {
            result = relay::TriluAttrs::Create(node->upper, node->k);
        } else if (const auto* node = source.As<relay::SqueezeAttrsNode>()) {
            result = relay::SqueezeAttrs::Create(CloneIntArray(node->axes));
        } else if (const auto* node = source.As<relay::UnsqueezeAttrsNode>()) {
            result = relay::UnsqueezeAttrs::Create(CloneIntArray(node->axes));
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

}  // namespace

Function CloneRelaySnapshot(const Function& function) {
    return RelaySnapshotCloner().Clone(function);
}

void FreezePreparedOperators(PreparedCompilerGraph* prepared) {
    RelaySnapshotCloner cloner;
    for (const CallInfo& record :
         prepared->graph.partitioned.value_graph.calls) {
        auto* call = const_cast<CallNode*>(record.call.As<CallNode>());
        if (!call) Reject("prepared value graph contains a non-Call record");
        call->op = cloner.CloneOperator(call->op);
    }
}

}  // namespace kxc::api::internal
