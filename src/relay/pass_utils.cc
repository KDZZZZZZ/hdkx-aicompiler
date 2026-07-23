/*! \file src/relay/pass_utils.cc
 * \brief 实现 Relay 节点、算子元数据、pass 工具和公共注册。
 */

#include "kxc/relay/pass_utils.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>

#include "kxc/relay/visitor.h"
#include "kxc/relay/op.h"
#include "kxc/relay/pass/print_ir.h"

namespace kxc {
namespace relay {
namespace pass_utils {

namespace {

// 判断 NDArray 的逻辑元素数是否恰为一。
bool IsScalarShape(const runtime::NDArray& array) {
    if (!array.defined()) {
        return false;
    }
    int64_t elements = 1;
    for (const int64_t dim : array->shape_storage) {
        if (dim < 0) {
            return false;
        }
        elements *= dim;
    }
    return elements == 1;
}

// 将支持的 DLPack 标量类型转换为 NDArray dtype 名称。
std::string DTypeToString(const DLDataType& dtype) {
    if (dtype.lanes != 1) {
        return "float64";
    }
    if (dtype.code == kDLFloat) {
        if (dtype.bits == 32) return "float32";
        if (dtype.bits == 64) return "float64";
    }
    if (dtype.code == kDLInt) {
        if (dtype.bits == 8) return "int8";
        if (dtype.bits == 32) return "int32";
        if (dtype.bits == 64) return "int64";
    }
    if (dtype.code == kDLUInt) {
        if (dtype.bits == 8) return "uint8";
    }
    if (dtype.code == kDLBool && dtype.bits == 8) return "bool";
    return "float64";
}

template <typename T>
// 把 double 中间值按目标标量类型写入主机缓冲区。
void WriteScalarValue(void* raw, double value) {
    *static_cast<T*>(raw) = static_cast<T>(value);
}

// 通过 NDArray 复制接口写入一个标量值。
bool WriteScalarToArray(const runtime::NDArray& array, double value) {
    if (!array.defined() || array.NBytes() == 0) {
        return false;
    }
    // 通过主机暂存区写入，避免 Pass 直接解引用可能位于 CUDA 上的张量地址。
    std::vector<uint8_t> storage(array.NBytes());
    void* raw = storage.data();
    const auto finish = [&] {
        array.CopyFromBytes(storage.data(), storage.size());
        return true;
    };
    const DLDataType& dtype = array->dl_tensor.dtype;
    if (dtype.code == kDLFloat) {
        if (dtype.bits == 32) {
            WriteScalarValue<float>(raw, value);
            return finish();
        }
        if (dtype.bits == 64) {
            WriteScalarValue<double>(raw, value);
            return finish();
        }
    }
    if (dtype.code == kDLInt) {
        if (dtype.bits == 8) {
            WriteScalarValue<int8_t>(raw, value);
            return finish();
        }
        if (dtype.bits == 16) {
            WriteScalarValue<int16_t>(raw, value);
            return finish();
        }
        if (dtype.bits == 32) {
            WriteScalarValue<int32_t>(raw, value);
            return finish();
        }
        if (dtype.bits == 64) {
            WriteScalarValue<int64_t>(raw, value);
            return finish();
        }
    }
    if (dtype.code == kDLUInt || dtype.code == kDLBool) {
        if (dtype.bits == 8) {
            WriteScalarValue<uint8_t>(raw, value);
            return finish();
        }
        if (dtype.bits == 16) {
            WriteScalarValue<uint16_t>(raw, value);
            return finish();
        }
        if (dtype.bits == 32) {
            WriteScalarValue<uint32_t>(raw, value);
            return finish();
        }
        if (dtype.bits == 64) {
            WriteScalarValue<uint64_t>(raw, value);
            return finish();
        }
    }
    return false;
}

}  // namespace

// 尝试读取标量常量的数值，不要求调用方接收 dtype。
bool TryGetScalarConstantValue(const Expr& expr, double* out_value) {
    return TryGetScalarConstantValueWithDType(expr, out_value, nullptr);
}

// 通过主机副本读取标量常量，并可同时返回原始 DLPack dtype。
bool TryGetScalarConstantValueWithDType(const Expr& expr, double* out_value,
                                        DLDataType* out_dtype) {
    if (!out_value) {
        return false;
    }
    const auto* constant = expr.As<ConstantNode>();
    if (!constant || !constant->data.defined() || constant->data.NBytes() == 0) {
        return false;
    }
    if (!IsScalarShape(constant->data)) {
        return false;
    }

    const DLDataType& dtype = constant->data->dl_tensor.dtype;
    if (dtype.lanes != 1) {
        return false;
    }
    if (out_dtype) {
        *out_dtype = dtype;
    }

    // 常量读取统一走 NDArray 复制接口，CPU-only Pass 不接触设备裸指针。
    std::vector<uint8_t> storage(constant->data.NBytes());
    constant->data.CopyToBytes(storage.data(), storage.size());
    const void* raw = storage.data();
    if (dtype.code == kDLFloat) {
        if (dtype.bits == 32) {
            *out_value = static_cast<double>(*static_cast<const float*>(raw));
            return true;
        }
        if (dtype.bits == 64) {
            *out_value = *static_cast<const double*>(raw);
            return true;
        }
        return false;
    }

    if (dtype.code == kDLInt || dtype.code == kDLUInt || dtype.code == kDLBool) {
        if (dtype.bits == 8) {
            *out_value = static_cast<double>(*static_cast<const uint8_t*>(raw));
            return true;
        }
        if (dtype.bits == 16) {
            *out_value = static_cast<double>(*static_cast<const int16_t*>(raw));
            return true;
        }
        if (dtype.bits == 32) {
            *out_value = static_cast<double>(*static_cast<const int32_t*>(raw));
            return true;
        }
        if (dtype.bits == 64) {
            *out_value = static_cast<double>(*static_cast<const int64_t*>(raw));
            return true;
        }
        return false;
    }

    return false;
}

// 在 CPU Storage 上构造指定 dtype 的标量 Relay Constant。
Constant MakeScalarConstant(double value, const DLDataType& dtype) {
    runtime::NDArray array = runtime::NDArray::Empty(
        {}, runtime::DataTypeFromString(DTypeToString(dtype)), Device::CPU());
    if (!WriteScalarToArray(array, value)) {
        runtime::NDArray fallback = runtime::NDArray::Empty(
            {}, runtime::DataTypeFromString("float64"), Device::CPU());
        WriteScalarToArray(fallback, value);
        return Constant(fallback);
    }
    return Constant(array);
}

// 判断表达式是否为数值零标量常量。
bool IsConstZero(const Expr& expr) {
    double value = 0.0;
    return TryGetScalarConstantValue(expr, &value) && value == 0.0;
}

// 判断表达式是否为数值一标量常量。
bool IsConstOne(const Expr& expr) {
    double value = 0.0;
    return TryGetScalarConstantValue(expr, &value) && value == 1.0;
}

// 递归判断表达式是否包含设备操作或未知调用等潜在副作用。
bool HasSideEffect(const Expr& expr) {
    std::unordered_map<const Object*, bool> memo;

    std::function<bool(const Expr&)> visit = [&](const Expr& current) -> bool {
        if (!current.defined()) {
            return false;
        }
        auto it = memo.find(current.get());
        if (it != memo.end()) {
            return it->second;
        }

        bool result = false;
        if (current.As<VarNode>() || current.As<ConstantNode>()) {
            result = false;
        } else if (const auto* tuple = current.As<TupleNode>()) {
            for (const auto& field : tuple->fields) {
                if (visit(field)) {
                    result = true;
                    break;
                }
            }
        } else if (const auto* tuple_get = current.As<TupleGetItemNode>()) {
            result = visit(tuple_get->tuple);
        } else if (const auto* if_node = current.As<IfNode>()) {
            result = visit(if_node->cond) || visit(if_node->true_branch) ||
                     visit(if_node->false_branch);
        } else if (const auto* while_node = current.As<WhileNode>()) {
            result = visit(while_node->initial_state) || visit(while_node->condition) ||
                     visit(while_node->body);
        } else if (const auto* let_node = current.As<LetNode>()) {
            result = visit(let_node->value) || visit(let_node->body);
        } else if (const auto* fn = current.As<FunctionNode>()) {
            result = visit(fn->body);
        } else if (const auto* call = current.As<CallNode>()) {
            const std::string op_name = GetCallOpName(call);
            if (op_name.empty()) {
                result = true;
            } else if (op_name.rfind("device.", 0) == 0) {
                result = true;
            } else {
                result = false;
                for (const auto& arg : call->args) {
                    if (visit(arg)) {
                        result = true;
                        break;
                    }
                }
            }
        } else {
            result = true;
        }

        memo[current.get()] = result;
        return result;
    };

    return visit(expr);
}

// 按对象身份统计目标变量在表达式中的引用次数。
size_t CountVarUses(const Expr& expr, const Var& var) {
    if (!expr.defined() || !var.defined()) {
        return 0;
    }
    const Object* target = var.get();
    size_t count = 0;

    // 递归遍历所有 Relay 子表达式，并按目标 Var 的对象身份累计引用。
    std::function<void(const Expr&)> visit = [&](const Expr& current) {
        if (!current.defined()) {
            return;
        }

        if (current.get() == target && current.As<VarNode>()) {
            ++count;
            return;
        }

        if (const auto* call = current.As<CallNode>()) {
            visit(call->op);
            for (const auto& arg : call->args) {
                visit(arg);
            }
            return;
        }
        if (const auto* fn = current.As<FunctionNode>()) {
            visit(fn->body);
            return;
        }
        if (const auto* if_node = current.As<IfNode>()) {
            visit(if_node->cond);
            visit(if_node->true_branch);
            visit(if_node->false_branch);
            return;
        }
        if (const auto* while_node = current.As<WhileNode>()) {
            visit(while_node->initial_state);
            visit(while_node->condition);
            visit(while_node->body);
            return;
        }
        if (const auto* let_node = current.As<LetNode>()) {
            visit(let_node->value);
            visit(let_node->body);
            return;
        }
        if (const auto* tuple = current.As<TupleNode>()) {
            for (const auto& field : tuple->fields) {
                visit(field);
            }
            return;
        }
        if (const auto* tuple_get = current.As<TupleGetItemNode>()) {
            visit(tuple_get->tuple);
            return;
        }
    };

    visit(expr);
    return count;
}

// 把源表达式的 VirtualDevice 与 checked_type 元数据复制到新表达式。
Expr CopyVirtualDevice(const Expr& source, const Expr& dest) {
    if (!source.defined() || !dest.defined()) {
        return dest;
    }
    const RelayNode* source_node = dynamic_cast<const RelayNode*>(source.get());
    RelayNode* dest_node = const_cast<RelayNode*>(dynamic_cast<const RelayNode*>(dest.get()));
    if (!source_node || !dest_node) {
        return dest;
    }
    dest_node->virtual_device_ = source_node->virtual_device_;
    dest_node->checked_type_ = source_node->checked_type_;
    return dest;
}

// 取得 Call 直接引用的注册算子名称，非 Op 调用返回空串。
std::string GetCallOpName(const CallNode* call) {
    if (!call) {
        return "";
    }
    const auto* op_node = call->op.As<OpNode>();
    if (!op_node) {
        return "";
    }
    return op_node->name;
}

// 生成用于 pass 去重和比较的稳定结构文本键。
std::string ExprStructuralKey(const Expr& expr) {
    return relay::pass::ToText(expr);
}

// 在遵守函数参数与 let 绑定遮蔽规则的前提下替换自由变量。
Expr SubstituteVar(const Expr& expr, const Var& target, const Expr& replacement) {
    if (!expr.defined() || !target.defined() || !replacement.defined()) {
        return expr;
    }

    // 执行具备词法作用域感知的变量替换。
    class VarSubstituter : public RelayPass {
    public:
        // 保存待替换变量和替代表达式。
        VarSubstituter(Var target, Expr replacement)
            : target_(std::move(target)), replacement_(std::move(replacement)) {}

    protected:
        // 仅替换对象身份匹配的变量。
        Expr VisitVar(const VarNode* op, const Expr& ref) override {
            (void)op;
            if (ref.get() == target_.get()) {
                return replacement_;
            }
            return ref;
        }

        // 参数遮蔽目标变量时停止进入函数体。
        Expr VisitFunction(const FunctionNode* op, const Expr& ref) override {
            for (const auto& param : op->params) {
                if (param.get() == target_.get()) {
                    return ref;
                }
            }
            return RelayPass::VisitFunction(op, ref);
        }

        // While state binding only scopes condition and body, not initial_state.
        Expr VisitWhile(const WhileNode* op, const Expr& ref) override {
            Expr initial = Mutate(op->initial_state);
            if (op->loop_var.get() == target_.get()) {
                if (initial.get() == op->initial_state.get()) return ref;
                return CopyVirtualDevice(ref, While(initial, op->loop_var, op->condition,
                                                    op->body, op->max_trip_count));
            }
            Expr condition = Mutate(op->condition);
            Expr body = Mutate(op->body);
            if (initial.get() == op->initial_state.get() &&
                condition.get() == op->condition.get() && body.get() == op->body.get()) return ref;
            return CopyVirtualDevice(ref, While(initial, op->loop_var, condition, body,
                                                op->max_trip_count));
        }

        // let 绑定只遮蔽 body，不遮蔽 value。
        Expr VisitLet(const LetNode* op, const Expr& ref) override {
            Expr new_value = Mutate(op->value);
            if (op->var.get() == target_.get()) {
                if (new_value.get() == op->value.get()) {
                    return ref;
                }
                return CopyVirtualDevice(ref, Let(op->var, new_value, op->body));
            }
            Expr new_body = Mutate(op->body);
            if (new_value.get() == op->value.get() && new_body.get() == op->body.get()) {
                return ref;
            }
            return CopyVirtualDevice(ref, Let(op->var, new_value, new_body));
        }

    private:
        Var target_;
        Expr replacement_;
    };

    VarSubstituter substituter(target, replacement);
    return substituter.Mutate(expr);
}

// 克隆 VirtualDevice 并只替换 memory_scope，保留物理与逻辑身份。
VirtualDevice WithMemoryScope(const VirtualDevice& virtual_device,
                              const std::string& memory_scope) {
    if (!virtual_device.defined()) {
        return virtual_device;
    }
    auto* node = new VirtualDeviceNode();
    node->device = virtual_device->device;
    node->target = virtual_device->target;
    node->memory_scope = memory_scope;
    node->virtual_device_id = virtual_device->virtual_device_id;
    return VirtualDevice(ObjectRef(node));
}

}  // namespace pass_utils
}  // namespace relay
}  // namespace kxc
