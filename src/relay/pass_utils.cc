#include "relay/pass_utils.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>

#include "base/pass.h"
#include "relay/op.h"
#include "relay/pass/print_ir.h"

namespace kxc {
namespace relay {
namespace pass_utils {

namespace {

bool IsScalarShape(const runtime::NDArray& array) {
    if (!array.defined()) {
        return false;
    }
    int64_t elements = 1;
    for (const int64_t dim : array->shape) {
        if (dim < 0) {
            return false;
        }
        elements *= dim;
    }
    return elements == 1;
}

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
    if (dtype.code == kDLUint) {
        if (dtype.bits == 1) return "bool";
        if (dtype.bits == 8) return "uint8";
    }
    return "float64";
}

template <typename T>
void WriteScalarValue(void* raw, double value) {
    *static_cast<T*>(raw) = static_cast<T>(value);
}

bool WriteScalarToArray(const runtime::NDArray& array, double value) {
    if (!array.defined() || !array->dl_tensor.data) {
        return false;
    }
    void* raw = array->dl_tensor.data;
    const DLDataType& dtype = array->dl_tensor.dtype;
    if (dtype.code == kDLFloat) {
        if (dtype.bits == 32) {
            WriteScalarValue<float>(raw, value);
            return true;
        }
        if (dtype.bits == 64) {
            WriteScalarValue<double>(raw, value);
            return true;
        }
    }
    if (dtype.code == kDLInt) {
        if (dtype.bits == 8) {
            WriteScalarValue<int8_t>(raw, value);
            return true;
        }
        if (dtype.bits == 16) {
            WriteScalarValue<int16_t>(raw, value);
            return true;
        }
        if (dtype.bits == 32) {
            WriteScalarValue<int32_t>(raw, value);
            return true;
        }
        if (dtype.bits == 64) {
            WriteScalarValue<int64_t>(raw, value);
            return true;
        }
    }
    if (dtype.code == kDLUint) {
        if (dtype.bits == 1 || dtype.bits == 8) {
            WriteScalarValue<uint8_t>(raw, value);
            return true;
        }
        if (dtype.bits == 16) {
            WriteScalarValue<uint16_t>(raw, value);
            return true;
        }
        if (dtype.bits == 32) {
            WriteScalarValue<uint32_t>(raw, value);
            return true;
        }
        if (dtype.bits == 64) {
            WriteScalarValue<uint64_t>(raw, value);
            return true;
        }
    }
    return false;
}

}  // namespace

bool TryGetScalarConstantValue(const Expr& expr, double* out_value) {
    return TryGetScalarConstantValueWithDType(expr, out_value, nullptr);
}

bool TryGetScalarConstantValueWithDType(const Expr& expr, double* out_value,
                                        DLDataType* out_dtype) {
    if (!out_value) {
        return false;
    }
    const auto* constant = expr.As<ConstantNode>();
    if (!constant || !constant->data.defined() || !constant->data->dl_tensor.data) {
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

    const void* raw = constant->data->dl_tensor.data;
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

    if (dtype.code == kDLInt || dtype.code == kDLUint) {
        if (dtype.bits == 1 || dtype.bits == 8) {
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

Constant MakeScalarConstant(double value, const DLDataType& dtype) {
    runtime::NDArray array(Array<int64_t>{}, DTypeToString(dtype));
    if (!WriteScalarToArray(array, value)) {
        runtime::NDArray fallback(Array<int64_t>{}, "float64");
        WriteScalarToArray(fallback, value);
        return Constant(fallback);
    }
    return Constant(array);
}

bool IsConstZero(const Expr& expr) {
    double value = 0.0;
    return TryGetScalarConstantValue(expr, &value) && value == 0.0;
}

bool IsConstOne(const Expr& expr) {
    double value = 0.0;
    return TryGetScalarConstantValue(expr, &value) && value == 1.0;
}

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

size_t CountVarUses(const Expr& expr, const Var& var) {
    if (!expr.defined() || !var.defined()) {
        return 0;
    }
    const Object* target = var.get();
    size_t count = 0;

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
    return dest;
}

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

std::string ExprStructuralKey(const Expr& expr) {
    return relay::pass::ToText(expr);
}

Expr SubstituteVar(const Expr& expr, const Var& target, const Expr& replacement) {
    if (!expr.defined() || !target.defined() || !replacement.defined()) {
        return expr;
    }

    class VarSubstituter : public RelayPass {
    public:
        VarSubstituter(Var target, Expr replacement)
            : target_(std::move(target)), replacement_(std::move(replacement)) {}

    protected:
        Expr VisitVar(const VarNode* op, const Expr& ref) override {
            (void)op;
            if (ref.get() == target_.get()) {
                return replacement_;
            }
            return ref;
        }

        Expr VisitFunction(const FunctionNode* op, const Expr& ref) override {
            for (const auto& param : op->params) {
                if (param.get() == target_.get()) {
                    return ref;
                }
            }
            return RelayPass::VisitFunction(op, ref);
        }

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

VirtualDevice WithMemoryScope(const VirtualDevice& virtual_device,
                              const std::string& memory_scope) {
    if (!virtual_device.defined()) {
        return virtual_device;
    }
    auto* node = new VirtualDeviceNode();
    node->device_obj = virtual_device->device_obj;
    node->target = virtual_device->target;
    node->memory_scope = memory_scope;
    node->virtual_device_id = virtual_device->virtual_device_id;
    return VirtualDevice(ObjectRef(node));
}

}  // namespace pass_utils
}  // namespace relay
}  // namespace kxc
