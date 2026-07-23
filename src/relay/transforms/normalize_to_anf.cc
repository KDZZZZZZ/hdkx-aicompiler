/*! \file src/relay/transforms/normalize_to_anf.cc
 * \brief Deterministic Relay administrative-normal-form normalization.
 */

#include "kxc/relay/transforms/normalize_to_anf.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kxc/relay/op.h"
#include "kxc/relay/pass_utils.h"

namespace kxc {
namespace relay {
namespace {

Expr CopyMetadata(const Expr& source, const Expr& destination) {
    Expr result = pass_utils::CopyVirtualDevice(source, destination);
    const auto* source_node = dynamic_cast<const ExprNode*>(source.get());
    auto* destination_node =
        const_cast<ExprNode*>(dynamic_cast<const ExprNode*>(result.get()));
    if (source_node && destination_node) {
        destination_node->span = source_node->span;
    }
    return result;
}

bool IsAtomic(const Expr& expr) {
    return expr.As<VarNode>() || expr.As<ConstantNode>();
}

std::string NodeKind(const Expr& expr) {
    if (!expr.defined()) return "undefined";
    if (expr.As<VarNode>()) return "Var";
    if (expr.As<ConstantNode>()) return "Constant";
    if (expr.As<CallNode>()) return "Call";
    if (expr.As<FunctionNode>()) return "Function";
    if (expr.As<TupleNode>()) return "Tuple";
    if (expr.As<TupleGetItemNode>()) return "TupleGetItem";
    if (expr.As<IfNode>()) return "If";
    if (expr.As<LetNode>()) return "Let";
    if (expr.As<OpNode>()) return "Op";
    return "unknown";
}

class ANFChecker {
public:
    bool Check(const Function& function, std::string* diagnostic) {
        diagnostic_ = diagnostic;
        if (diagnostic_) diagnostic_->clear();
        if (!function.defined()) return Fail("function", "Function is undefined");
        if (!function.checked_type().defined()) {
            return Fail("function", "Function has no checked_type");
        }
        for (size_t i = 0; i < function->params.size(); ++i) {
            const Var& parameter = function->params[i];
            const std::string path = "function.params[" + std::to_string(i) + "]";
            if (!parameter.defined() || !parameter->type_annotation.defined() ||
                !parameter.checked_type().defined()) {
                return Fail(path, "parameter must be typed");
            }
            bindings_[parameter.get()] += 1;
        }
        const bool result = CheckTerminal(function->body, "function.body");
        for (const Var& parameter : function->params) bindings_.erase(parameter.get());
        return result;
    }

private:
    std::unordered_map<const Object*, size_t> bindings_;
    std::string* diagnostic_{nullptr};

    bool Fail(const std::string& path, const std::string& message) {
        if (diagnostic_) {
            *diagnostic_ = "ANF violation at path=" + path + "; node=" +
                           NodeKindAt(path) + "; " + message;
        }
        return false;
    }

    // The path itself is enough to identify the offending location; callers put
    // the actual node kind in the explanatory text below.
    std::string NodeKindAt(const std::string&) const { return "Relay"; }

    bool CheckTyped(const Expr& expr, const std::string& path) {
        if (!expr.defined()) return Fail(path, "node is undefined");
        if (!expr.checked_type().defined()) {
            return Fail(path, "node " + NodeKind(expr) + " has no checked_type");
        }
        return true;
    }

    bool CheckAtomic(const Expr& expr, const std::string& path) {
        if (!CheckTyped(expr, path)) return false;
        if (const auto* var = expr.As<VarNode>()) {
            if (bindings_.count(expr.get()) == 0) {
                return Fail(path, "Var '" + var->vid->name_hint + "' is free");
            }
            return true;
        }
        if (expr.As<ConstantNode>()) return true;
        return Fail(path, "node " + NodeKind(expr) + " must be atomic");
    }

    bool CheckValue(const Expr& expr, const std::string& path) {
        if (!CheckTyped(expr, path)) return false;
        if (IsAtomic(expr)) return CheckAtomic(expr, path);
        if (const auto* call = expr.As<CallNode>()) {
            for (size_t i = 0; i < call->args.size(); ++i) {
                if (!CheckAtomic(call->args[i], path + ".args[" +
                                                   std::to_string(i) + "]")) {
                    return false;
                }
            }
            return true;
        }
        if (const auto* if_node = expr.As<IfNode>()) {
            return CheckAtomic(if_node->cond, path + ".cond") &&
                   CheckTerminal(if_node->true_branch, path + ".true_branch") &&
                   CheckTerminal(if_node->false_branch, path + ".false_branch");
        }
        if (const auto* get_item = expr.As<TupleGetItemNode>()) {
            return CheckAtomic(get_item->tuple, path + ".tuple");
        }
        if (const auto* tuple = expr.As<TupleNode>()) {
            for (size_t i = 0; i < tuple->fields.size(); ++i) {
                if (!CheckAtomic(tuple->fields[i], path + ".fields[" +
                                                     std::to_string(i) + "]")) {
                    return false;
                }
            }
            return true;
        }
        return Fail(path, "node " + NodeKind(expr) + " is not an ANF value");
    }

    bool CheckTerminal(const Expr& expr, const std::string& path) {
        if (!CheckTyped(expr, path)) return false;
        if (IsAtomic(expr)) return CheckAtomic(expr, path);
        if (const auto* tuple = expr.As<TupleNode>()) {
            for (size_t i = 0; i < tuple->fields.size(); ++i) {
                if (!CheckAtomic(tuple->fields[i], path + ".fields[" +
                                                     std::to_string(i) + "]")) {
                    return false;
                }
            }
            return true;
        }
        if (const auto* let = expr.As<LetNode>()) {
            if (!let->var.defined() || !let->var.checked_type().defined()) {
                return Fail(path + ".var", "Let binder must be typed");
            }
            if (!CheckValue(let->value, path + ".value")) return false;
            if (!TypeEqual(let->var.checked_type(), let->value.checked_type())) {
                return Fail(path + ".var", "Let binder type does not match value type");
            }
            if (let->var->type_annotation.defined() &&
                !TypeEqual(let->var->type_annotation, let->value.checked_type())) {
                return Fail(path + ".var", "Let annotation does not match value type");
            }
            bindings_[let->var.get()] += 1;
            const bool result = CheckTerminal(let->body, path + ".body");
            auto binding = bindings_.find(let->var.get());
            if (--binding->second == 0) bindings_.erase(binding);
            return result;
        }
        return Fail(path, "executable " + NodeKind(expr) + " must be let-bound");
    }
};

class ANFNormalizer {
public:
    explicit ANFNormalizer(const Function& function) { CollectNames(Expr(ObjectRef(function))); }

    Function Normalize(const Function& function) {
        Expr body = NormalizeTerminal(function->body);
        if (body.get() == function->body.get()) return function;
        return Function(CopyMetadata(Expr(ObjectRef(function)),
                                     Function(function->params, body)));
    }

private:
    using Continuation = std::function<Expr(const Expr&)>;
    std::unordered_set<std::string> names_;
    // A source DAG edge denotes one shared Relay computation.  The cache is
    // lexical: branch normalization snapshots it so branch-local work cannot
    // escape or be hoisted into another branch.
    std::unordered_map<const Object*, Var> atom_cache_;
    size_t next_name_{0};

    void CollectNames(const Expr& expr) {
        if (!expr.defined()) return;
        if (const auto* var = expr.As<VarNode>()) {
            names_.insert(var->vid->name_hint);
        } else if (const auto* call = expr.As<CallNode>()) {
            for (const Expr& argument : call->args) CollectNames(argument);
        } else if (const auto* function = expr.As<FunctionNode>()) {
            for (const Var& parameter : function->params) CollectNames(Expr(ObjectRef(parameter)));
            CollectNames(function->body);
        } else if (const auto* if_node = expr.As<IfNode>()) {
            CollectNames(if_node->cond);
            CollectNames(if_node->true_branch);
            CollectNames(if_node->false_branch);
        } else if (const auto* let = expr.As<LetNode>()) {
            CollectNames(Expr(ObjectRef(let->var)));
            CollectNames(let->value);
            CollectNames(let->body);
        } else if (const auto* tuple = expr.As<TupleNode>()) {
            for (const Expr& field : tuple->fields) CollectNames(field);
        } else if (const auto* get_item = expr.As<TupleGetItemNode>()) {
            CollectNames(get_item->tuple);
        }
    }

    Var FreshVar(const Expr& value) {
        const Type type = value.checked_type();
        if (!type.defined()) {
            throw std::invalid_argument("NormalizeToANF requires checked_type on " +
                                        NodeKind(value));
        }
        std::string name;
        do {
            name = "anf" + std::to_string(next_name_++);
        } while (!names_.insert(name).second);
        Var variable(name, type);
        CopyMetadata(value, Expr(ObjectRef(variable)));
        return variable;
    }

    Expr GeneratedLet(const Expr& source, const Var& variable, const Expr& value,
                      const Expr& body) {
        Expr result = CopyMetadata(source, Let(variable, value, body));
        SetCheckedType(result, body.checked_type());
        return result;
    }

    Expr OriginalLet(const Expr& source, const Var& variable, const Expr& value,
                     const Expr& body) {
        return CopyMetadata(source, Let(variable, value, body));
    }

    Expr NormalizeAtoms(const Array<Expr>& values, size_t index, Array<Expr> atoms,
                        const std::function<Expr(const Array<Expr>&)>& continuation) {
        if (index == values.size()) return continuation(atoms);
        return NormalizeToAtom(values[index],
            [this, &values, index, atoms = std::move(atoms), continuation](const Expr& atom) mutable {
                atoms.push_back(atom);
                return NormalizeAtoms(values, index + 1, std::move(atoms), continuation);
            });
    }

    Expr NormalizeToAtom(const Expr& expr, const Continuation& continuation) {
        if (IsAtomic(expr)) return continuation(expr);
        const auto cached = atom_cache_.find(expr.get());
        if (cached != atom_cache_.end()) return continuation(Expr(ObjectRef(cached->second)));
        if (const auto* let = expr.As<LetNode>()) {
            return NormalizeValue(let->value,
                [this, let, expr, continuation](const Expr& value) {
                    const auto saved = atom_cache_.find(let->value.get());
                    const bool had_saved = saved != atom_cache_.end();
                    const Var saved_var = had_saved ? saved->second : Var();
                    atom_cache_[let->value.get()] = let->var;
                    const Expr body = NormalizeToAtom(let->body, continuation);
                    if (had_saved) {
                        atom_cache_[let->value.get()] = saved_var;
                    } else {
                        atom_cache_.erase(let->value.get());
                    }
                    return OriginalLet(expr, let->var, value, body);
                });
        }
        return NormalizeValue(expr, [this, expr, continuation](const Expr& value) {
            const Var variable = FreshVar(value);
            atom_cache_[expr.get()] = variable;
            return GeneratedLet(expr, variable, value,
                                continuation(Expr(ObjectRef(variable))));
        });
    }

    Expr NormalizeValue(const Expr& expr, const Continuation& continuation) {
        if (IsAtomic(expr)) return continuation(expr);
        if (const auto* call = expr.As<CallNode>()) {
            return NormalizeAtoms(call->args, 0, {}, [expr, call, continuation](const Array<Expr>& args) {
                return continuation(CopyMetadata(expr, Call(call->op, args, call->attrs)));
            });
        }
        if (const auto* if_node = expr.As<IfNode>()) {
            return NormalizeToAtom(if_node->cond,
                [this, expr, if_node, continuation](const Expr& cond) {
                    const Expr normalized = CopyMetadata(
                        expr, If(cond, NormalizeBranch(if_node->true_branch),
                                 NormalizeBranch(if_node->false_branch)));
                    return continuation(normalized);
                });
        }
        if (const auto* get_item = expr.As<TupleGetItemNode>()) {
            return NormalizeToAtom(get_item->tuple,
                [expr, get_item, continuation](const Expr& tuple) {
                    return continuation(CopyMetadata(expr, TupleGetItem(tuple, get_item->index)));
                });
        }
        if (const auto* tuple = expr.As<TupleNode>()) {
            return NormalizeAtoms(tuple->fields, 0, {}, [expr, continuation](const Array<Expr>& fields) {
                return continuation(CopyMetadata(expr, Tuple(fields)));
            });
        }
        if (expr.As<FunctionNode>()) {
            throw std::invalid_argument("NormalizeToANF does not accept nested Function values");
        }
        throw std::invalid_argument("NormalizeToANF encountered unsupported Relay node: " +
                                    NodeKind(expr));
    }

    Expr NormalizeBranch(const Expr& expr) {
        const auto saved = atom_cache_;
        Expr normalized = NormalizeTerminal(expr);
        atom_cache_ = saved;
        return normalized;
    }

    Expr NormalizeTerminal(const Expr& expr) {
        if (IsAtomic(expr)) return expr;
        if (const auto* let = expr.As<LetNode>()) {
            return NormalizeValue(let->value, [this, expr, let](const Expr& value) {
                const auto saved = atom_cache_.find(let->value.get());
                const bool had_saved = saved != atom_cache_.end();
                const Var saved_var = had_saved ? saved->second : Var();
                atom_cache_[let->value.get()] = let->var;
                const Expr body = NormalizeTerminal(let->body);
                if (had_saved) {
                    atom_cache_[let->value.get()] = saved_var;
                } else {
                    atom_cache_.erase(let->value.get());
                }
                return OriginalLet(expr, let->var, value, body);
            });
        }
        if (const auto* tuple = expr.As<TupleNode>()) {
            return NormalizeAtoms(tuple->fields, 0, {}, [expr](const Array<Expr>& fields) {
                return CopyMetadata(expr, Tuple(fields));
            });
        }
        return NormalizeToAtom(expr, [](const Expr& atom) { return atom; });
    }
};

void RequireTyped(const Expr& expr, const std::string& path) {
    if (!expr.defined()) {
        throw std::invalid_argument("NormalizeToANF requires a defined node at " + path);
    }
    if (!expr.As<OpNode>() && !expr.checked_type().defined()) {
        throw std::invalid_argument("NormalizeToANF requires checked_type at " + path +
                                    " (" + NodeKind(expr) + ")");
    }
    if (const auto* call = expr.As<CallNode>()) {
        for (size_t i = 0; i < call->args.size(); ++i) {
            RequireTyped(call->args[i], path + ".args[" + std::to_string(i) + "]");
        }
    } else if (const auto* function = expr.As<FunctionNode>()) {
        for (size_t i = 0; i < function->params.size(); ++i) {
            if (!function->params[i]->type_annotation.defined()) {
                throw std::invalid_argument("NormalizeToANF requires a typed parameter at " +
                                            path + ".params[" + std::to_string(i) + "]");
            }
            RequireTyped(Expr(ObjectRef(function->params[i])),
                         path + ".params[" + std::to_string(i) + "]");
        }
        RequireTyped(function->body, path + ".body");
    } else if (const auto* if_node = expr.As<IfNode>()) {
        RequireTyped(if_node->cond, path + ".cond");
        RequireTyped(if_node->true_branch, path + ".true_branch");
        RequireTyped(if_node->false_branch, path + ".false_branch");
    } else if (const auto* let = expr.As<LetNode>()) {
        RequireTyped(Expr(ObjectRef(let->var)), path + ".var");
        RequireTyped(let->value, path + ".value");
        RequireTyped(let->body, path + ".body");
    } else if (const auto* tuple = expr.As<TupleNode>()) {
        for (size_t i = 0; i < tuple->fields.size(); ++i) {
            RequireTyped(tuple->fields[i], path + ".fields[" + std::to_string(i) + "]");
        }
    } else if (const auto* get_item = expr.As<TupleGetItemNode>()) {
        RequireTyped(get_item->tuple, path + ".tuple");
    }
}

}  // namespace

bool IsANF(const Function& function, std::string* diagnostic) {
    ANFChecker checker;
    return checker.Check(function, diagnostic);
}

void VerifyANF(const Function& function) {
    std::string diagnostic;
    if (!IsANF(function, &diagnostic)) throw std::invalid_argument(diagnostic);
}

Function NormalizeToANF(const Function& function) {
    if (!function.defined()) {
        throw std::invalid_argument("NormalizeToANF requires a defined Function");
    }
    RequireTyped(Expr(ObjectRef(function)), "function");
    if (IsANF(function)) return function;
    Function normalized = ANFNormalizer(function).Normalize(function);
    VerifyANF(normalized);
    return normalized;
}

}  // namespace relay
}  // namespace kxc
