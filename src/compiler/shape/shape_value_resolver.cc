/*! \file src/compiler/shape/shape_value_resolver.cc
 * \brief M3 受限形状值解析：链式证明、折叠与目标表达式投影。
 */

#include "shape_value_resolver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "kxc/relay/relay.h"
#include "kxc/relay/type_infer.h"
#include "kxc/runtime/ndarray.h"

namespace kxc::api::experimental::restricted_symbolic_shape::v1 {
namespace shape_resolution {

std::optional<int64_t> ProveNonnegativeConstantOffset(
    const kxc::shape::experimental::v1::DimExpr& expression,
    const kxc::shape::experimental::v1::DimExpr& base) {
    using namespace kxc::shape::experimental::v1;
    std::set<std::string> symbols;
    for (const auto& name : expression.Symbols()) symbols.insert(name);
    for (const auto& name : base.Symbols()) symbols.insert(name);
    std::vector<Binding> zeroes;
    for (const auto& name : symbols) zeroes.push_back({name,0});
    const BindingSet zero(std::move(zeroes));
    const int64_t left = expression.Evaluate(zero), right = base.Evaluate(zero);
    if (left < right) return std::nullopt;
    const int64_t offset = left - right;
    if (expression != DimExpr::Add({base,DimExpr::Const(offset)})) return std::nullopt;
    return offset;
}

namespace {

using DimExpr = kxc::shape::experimental::v1::DimExpr;
using BindingSet = kxc::shape::experimental::v1::BindingSet;

// 受限形状表达式的最大元素数（声明子集；Select 链与编码按此封顶）。
constexpr size_t kMaxShapeExprElements = 16;

[[noreturn]] void Reject(const std::string& message) {
    throw std::invalid_argument("RestrictedShapeValueResolver: " + message);
}

enum class ValueKind { kData, kShapeValue };

struct NodeInfo final {
    ValueKind kind{ValueKind::kData};
    bool scalar_shape_value{false};  // Internal Gather scalar; consumed by Unsqueeze.
    bool constant_control{false};   // Signed control literal, never a dimension expression.
    std::vector<DimExpr> dims;      // kData：符号维表达式
    std::vector<DimExpr> elements;  // kShapeValue：元素表达式（相对根源）
    const Object* source{nullptr};  // kShapeValue：根数据节点（原树）
};

/*! \brief One tensor leaf: its resolved proof plus the rewritten reference. */
using Leaf = std::pair<NodeInfo, Expr>;

// Read scalar/rank-1 integer control payloads without changing their rank or
// reading an int32 buffer as int64. Constants remain data in other contexts.
std::vector<int64_t> ReadConstantVector(const kxc::ConstantNode* constant,
                                        const std::string& context,
                                        bool allow_scalar = false) {
    if (!constant || !constant->data.defined()) {
        Reject(context + " requires a defined constant tensor payload");
    }
    const auto& shape = constant->data->shape_storage;
    if (shape.size() != 1 && !(allow_scalar && shape.empty())) {
        Reject(context + " constant has an unsupported rank");
    }
    const DLDataType dtype = constant->data->dl_tensor.dtype;
    if (dtype.lanes != 1 ||
        !(dtype.code == kDLInt && (dtype.bits == 32 || dtype.bits == 64))) {
        Reject(context + " constant dtype must be int32 or int64");
    }
    const size_t count = shape.empty() ? 1 : static_cast<size_t>(shape[0]);
    std::vector<int64_t> values(count);
    if (count) {
        if (dtype.bits == 32) {
            std::vector<int32_t> raw(count);
            constant->data.CopyToBytes(raw.data(), count * sizeof(int32_t));
            std::copy(raw.begin(), raw.end(), values.begin());
        } else {
            constant->data.CopyToBytes(values.data(), count * sizeof(int64_t));
        }
    }
    return values;
}

std::vector<DimExpr> BroadcastDims(const std::vector<DimExpr>& left,
                                 const std::vector<DimExpr>& right,
                                 const std::string& context) {
    const size_t rank = std::max(left.size(), right.size());
    const DimExpr one = DimExpr::Const(1);
    std::vector<DimExpr> result;
    for (size_t axis = 0; axis < rank; ++axis) {
        const DimExpr& a = axis < rank - left.size() ? one : left[axis - (rank - left.size())];
        const DimExpr& b = axis < rank - right.size() ? one : right[axis - (rank - right.size())];
        if (a == b || b == one) result.push_back(a);
        else if (a == one) result.push_back(b);
        else Reject(context + " requires equal dimension expressions or a constant singleton axis");
    }
    return result;
}

DimExpr ProductOf(const std::vector<DimExpr>& terms) {
    return DimExpr::Mul(terms);
}

int64_t EvaluateAtBounds(const DimExpr& expression, const std::map<std::string,int64_t>& bounds) {
    std::vector<kxc::shape::experimental::v1::Binding> bindings;
    for (const auto& symbol : expression.Symbols()) bindings.push_back({symbol,bounds.at(symbol)});
    return expression.Evaluate(BindingSet(std::move(bindings)));
}

// 把元素表达式编码为 (kinds/values/axes)：符号 → (1,0,axis)，常量 → (0,v,0)。
// reference_dims 是编码基准（shape_expr 源维 / reshape / expand 数据维）。
std::optional<EncodedExpr> TryEncodeElements(const std::vector<DimExpr>& elements,
                                             const std::vector<DimExpr>& reference_dims) {
    EncodedExpr encoded;
    for (const auto& element : elements) {
        if (element.kind() == DimExpr::Kind::kConst) {
            encoded.kinds.push_back(kExprKindConst);
            encoded.values.push_back(element.Evaluate(BindingSet()));
            encoded.axes.push_back(0);
            continue;
        }
        int matched_axis = -1;
        int64_t best_offset = std::numeric_limits<int64_t>::max();
        bool ambiguous = false;
        for (size_t axis = 0; axis < reference_dims.size(); ++axis) {
            const auto offset = ProveNonnegativeConstantOffset(element,reference_dims[axis]);
            if (!offset || *offset > best_offset) continue;
            if (matched_axis >= 0 && *offset == best_offset) { ambiguous = true; continue; }
            matched_axis = static_cast<int>(axis);
            best_offset = *offset;
            ambiguous = false;
        }
        if (matched_axis < 0 || ambiguous) return std::nullopt;
        encoded.kinds.push_back(best_offset == 0 ? kExprKindInputAxis : kExprKindInputAxisOffset);
        encoded.values.push_back(best_offset);
        encoded.axes.push_back(matched_axis);
    }
    return encoded;
}

EncodedExpr EncodeElements(const std::vector<DimExpr>& elements,
                           const std::vector<DimExpr>& reference_dims,
                           const std::string& context) {
    const auto encoded = TryEncodeElements(elements,reference_dims);
    if (!encoded) Reject(context + " symbolic elements require an unambiguous reference tensor axis");
    return *encoded;
}

int NormalizeAxis(const std::string& context, int64_t axis, int64_t rank) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
        Reject(context + " axis out of range");
    }
    return static_cast<int>(axis);
}

class Resolver final {
public:
    Resolver(const Function& snapshot,
             const std::vector<InputAxisSymbol>& symbols)
        : snapshot_(snapshot) {
        if (!snapshot.defined() || snapshot_->params.empty() ||
            !snapshot_->body.defined()) {
            Reject("the representative snapshot must have params and a body");
        }
        for (size_t index = 0; index < snapshot_->params.size(); ++index) {
            param_index_.emplace(snapshot_->params[index].get(), index);
        }
        for (const InputAxisSymbol& symbol : symbols) {
            axis_symbols_.emplace(
                std::make_pair(symbol.parameter_index, symbol.axis),
                symbol.symbol);
            symbol_lowers_.emplace(symbol.symbol, symbol.lower);
            symbol_uppers_.emplace(symbol.symbol, symbol.upper);
        }
    }

    Resolution Run() {
        Resolution result;
        result.rewritten =
            Function(snapshot_->params, RewriteResult(snapshot_->body));
        std::set<const Object*> visited;
        WalkRewritten(result.rewritten->body, &visited, &result);
        result.node_leaf_dims = node_leaf_dims_;
        return result;
    }

private:
    const Function& snapshot_;
    std::map<const Object*, size_t> param_index_;
    std::map<std::pair<size_t, size_t>, std::string> axis_symbols_;
    std::map<std::string, int64_t> symbol_lowers_;
    std::map<std::string, int64_t> symbol_uppers_;
    std::map<const Object*, NodeInfo> info_;
    std::map<const Object*, NodeInfo> rewritten_info_;
    std::map<const Object*, Expr> rewritten_;
    std::map<const Object*, EncodedExpr> value_expr_overrides_;
    std::set<const Object*> result_tensors_;
    /*! \brief Per rewritten node, the symbolic dims of each tensor leaf. Keyed
     *  by the rewritten Expr pointer so a control-plan value built from the
     *  same node finds its proof without a second traversal. */
    std::map<const Object*, std::vector<std::vector<DimExpr>>> node_leaf_dims_;
    /*! \brief Structural leaf proofs for tuples and control nodes. */
    std::map<const Object*, std::vector<Leaf>> leaf_memo_;
    /*! \brief Bound tuple/loop variables (leaf proofs plus rewritten refs). */
    std::map<const Object*, std::vector<Leaf>> leaf_bindings_;

    static size_t LeafCount(const Type& type) {
        if (type.As<TensorTypeNode>()) return 1;
        if (const auto* tuple = type.As<TupleTypeNode>()) {
            size_t count = 0;
            for (const Type& field : tuple->fields) count += LeafCount(field);
            return count;
        }
        Reject("control leaves must be tensors or tuples of tensors");
    }

    static Type CheckedType(const Expr& expr, const char* context) {
        const Type type = expr.checked_type();
        if (!type.defined()) {
            Reject(std::string(context) + " has no checked type");
        }
        return type;
    }

    /*! \brief Resolve an expression into its ordered tensor leaves.
     *
     *  Tuple/TupleGetItem/Let/If/While traverse structurally so a proof exists
     *  for every leaf that becomes a PrimitiveUnit boundary. If branches must
     *  agree leaf-wise on kind and dims; While binds its loop variable to the
     *  initial leaf shapes (loop-invariant tensor shapes in this step). */
    std::vector<Leaf> ResolveLeaves(const Expr& expr) {
        if (!expr.defined()) Reject("expression is undefined");
        const Object* node = expr.get();
        const auto memo = leaf_memo_.find(node);
        if (memo != leaf_memo_.end()) return memo->second;
        if (const auto* var = expr.As<kxc::VarNode>()) {
            const auto bound = leaf_bindings_.find(node);
            if (bound != leaf_bindings_.end()) {
                std::vector<std::vector<DimExpr>> dims;
                dims.reserve(bound->second.size());
                for (const Leaf& leaf : bound->second) {
                    dims.push_back(ValueTensorDims(leaf.first));
                }
                node_leaf_dims_[node] = std::move(dims);
                return leaf_memo_[node] = bound->second;
            }
        }

        std::vector<Leaf> result;
        if (const auto* tuple = expr.As<kxc::TupleNode>()) {
            for (const Expr& field : tuple->fields) {
                std::vector<Leaf> leaves = ResolveLeaves(field);
                result.insert(result.end(), leaves.begin(), leaves.end());
            }
        } else if (const auto* get_item = expr.As<kxc::TupleGetItemNode>()) {
            const std::vector<Leaf> tuple_leaves = ResolveLeaves(get_item->tuple);
            const auto* tuple_type =
                CheckedType(get_item->tuple, "TupleGetItem tuple")
                    .As<TupleTypeNode>();
            if (!tuple_type || get_item->index < 0 ||
                static_cast<size_t>(get_item->index) >= tuple_type->fields.size()) {
                Reject("control TupleGetItem index is outside its checked type");
            }
            size_t begin = 0;
            for (int index = 0; index < get_item->index; ++index) {
                begin += LeafCount(tuple_type->fields[static_cast<size_t>(index)]);
            }
            const size_t count =
                LeafCount(tuple_type->fields[static_cast<size_t>(get_item->index)]);
            if (begin + count > tuple_leaves.size()) {
                Reject("control TupleGetItem exceeds its tuple leaves");
            }
            result.assign(tuple_leaves.begin() + static_cast<std::ptrdiff_t>(begin),
                          tuple_leaves.begin() +
                              static_cast<std::ptrdiff_t>(begin + count));
            if (result.size() == 1) {
                rewritten_[node] = result.front().second;
                rewritten_info_[result.front().second.get()] = result.front().first;
            }
        } else if (const auto* conditional = expr.As<kxc::IfNode>()) {
            const std::vector<Leaf> predicate = ResolveLeaves(conditional->cond);
            if (predicate.size() != 1 ||
                predicate.front().first.kind != ValueKind::kData) {
                Reject("If requires a single data predicate");
            }
            const std::vector<Leaf> then_leaves =
                ResolveLeaves(conditional->true_branch);
            const std::vector<Leaf> else_leaves =
                ResolveLeaves(conditional->false_branch);
            if (then_leaves.size() != else_leaves.size() || then_leaves.empty()) {
                Reject("If branches must produce the same nonempty leaf arity");
            }
            for (size_t index = 0; index < then_leaves.size(); ++index) {
                if (then_leaves[index].first.kind != else_leaves[index].first.kind ||
                    then_leaves[index].first.dims != else_leaves[index].first.dims ||
                    then_leaves[index].first.elements !=
                        else_leaves[index].first.elements) {
                    Reject("If branch leaves must agree on kind, dims and elements");
                }
            }
            const Expr rewritten = kxc::If(
                predicate.front().second, then_leaves.front().second,
                else_leaves.front().second);
            rewritten_[node] = rewritten;
            if (then_leaves.size() == 1) {
                rewritten_info_[rewritten.get()] = then_leaves.front().first;
            }
            result = then_leaves;
        } else if (const auto* loop = expr.As<kxc::WhileNode>()) {
            const std::vector<Leaf> initial = ResolveLeaves(loop->initial_state);
            const auto* initial_type =
                CheckedType(loop->initial_state, "While initial state")
                    .As<TupleTypeNode>();
            if (!initial_type || initial_type->fields.size() != initial.size() ||
                initial.empty()) {
                Reject("While initial state must bind one leaf per tuple field");
            }
            Array<Expr> initial_rewrites;
            for (const Leaf& leaf : initial) initial_rewrites.push_back(leaf.second);
            std::vector<Leaf> loop_binding;
            loop_binding.reserve(initial.size());
            for (size_t index = 0; index < initial.size(); ++index) {
                loop_binding.push_back(
                    {initial[index].first,
                     kxc::TupleGetItem(loop->loop_var, static_cast<int>(index))});
            }
            const auto saved = leaf_bindings_.find(loop->loop_var.get());
            const auto previous = saved == leaf_bindings_.end()
                                      ? std::nullopt
                                      : std::optional<std::vector<Leaf>>(saved->second);
            leaf_bindings_[loop->loop_var.get()] = loop_binding;
            std::vector<Leaf> condition;
            std::vector<Leaf> body;
            try {
                condition = ResolveLeaves(loop->condition);
                body = ResolveLeaves(loop->body);
            } catch (...) {
                if (previous) {
                    leaf_bindings_[loop->loop_var.get()] = *previous;
                } else {
                    leaf_bindings_.erase(loop->loop_var.get());
                }
                throw;
            }
            if (previous) {
                leaf_bindings_[loop->loop_var.get()] = *previous;
            } else {
                leaf_bindings_.erase(loop->loop_var.get());
            }
            if (body.size() != initial.size()) {
                Reject("While body must return one leaf per initial field");
            }
            for (size_t index = 0; index < body.size(); ++index) {
                if (body[index].first.kind != initial[index].first.kind ||
                    body[index].first.dims != initial[index].first.dims ||
                    body[index].first.elements != initial[index].first.elements) {
                    Reject("While carried leaves must preserve their shape invariant");
                }
            }
            Array<Expr> condition_rewrites;
            for (const Leaf& leaf : condition) condition_rewrites.push_back(leaf.second);
            Array<Expr> body_rewrites;
            for (const Leaf& leaf : body) body_rewrites.push_back(leaf.second);
            // A While condition is a single scalar, not a tuple of values.
            if (condition_rewrites.size() != 1) {
                Reject("While condition must be exactly one scalar tensor");
            }
            const Expr rewritten = kxc::While(
                kxc::Tuple(std::move(initial_rewrites)), loop->loop_var,
                condition_rewrites[0],
                kxc::Tuple(std::move(body_rewrites)), loop->max_trip_count);
            rewritten_[node] = rewritten;
            // The loop variable carries the initial leaf proofs.
            std::vector<Leaf> loop_leaves;
            loop_leaves.reserve(initial.size());
            for (size_t index = 0; index < initial.size(); ++index) {
                loop_leaves.push_back(
                    {initial[index].first,
                     kxc::TupleGetItem(loop->loop_var, static_cast<int>(index))});
            }
            leaf_memo_[loop->loop_var.get()] = loop_leaves;
            result = initial;
        } else if (const auto* let = expr.As<kxc::LetNode>()) {
            const std::vector<Leaf> value_leaves = ResolveLeaves(let->value);
            const auto saved = leaf_bindings_.find(let->var.get());
            const auto previous = saved == leaf_bindings_.end()
                                      ? std::nullopt
                                      : std::optional<std::vector<Leaf>>(saved->second);
            leaf_bindings_[let->var.get()] = value_leaves;
            std::vector<Leaf> body;
            try {
                body = ResolveLeaves(let->body);
            } catch (...) {
                if (previous) {
                    leaf_bindings_[let->var.get()] = *previous;
                } else {
                    leaf_bindings_.erase(let->var.get());
                }
                throw;
            }
            if (previous) {
                leaf_bindings_[let->var.get()] = *previous;
            } else {
                leaf_bindings_.erase(let->var.get());
            }
            result = body;
        } else {
            const NodeInfo info = Resolve(expr);
            result.push_back({info, rewritten_.at(node)});
        }
        std::vector<std::vector<DimExpr>> dims;
        dims.reserve(result.size());
        for (const Leaf& leaf : result) dims.push_back(ValueTensorDims(leaf.first));
        node_leaf_dims_[node] = std::move(dims);
        return leaf_memo_[node] = result;
    }

    // Tuple is the existing Relay result structure. It introduces no tensor
    // value or compute unit; field order and nesting survive materialization.
    Expr RewriteResult(const Expr& expr) {
        if (expr.As<ConstantNode>()) {
            Reject("constant results must be materialized by a compute call into fresh storage");
        }
        if (const auto* tuple = expr.As<TupleNode>()) {
            if (tuple->fields.empty()) Reject("empty result tuples are unsupported");
            Array<Expr> fields;
            for (const Expr& field : tuple->fields) {
                fields.push_back(RewriteResult(field));
            }
            return Tuple(std::move(fields));
        }
        if (expr.As<IfNode>() || expr.As<WhileNode>()) {
            (void)ResolveLeaves(expr);
            const auto found = rewritten_.find(expr.get());
            if (found == rewritten_.end()) Reject("control result has no rewrite");
            return found->second;
        }
        if (const auto* let = expr.As<LetNode>()) {
            // ANF wraps non-atomic values in Let bindings; thread them through
            // so a Let-chained control body still has a rewritten result.
            const Expr value = RewriteResult(let->value);
            const Expr body = RewriteResult(let->body);
            return Let(let->var, value, body);
        }
        if (!result_tensors_.insert(expr.get()).second) {
            Reject("duplicate result tensors are unsupported by ExecutablePlan");
        }
        const NodeInfo result = Resolve(expr);
        if (result.scalar_shape_value) {
            Reject("scalar shape results must be made into a vector by Unsqueeze");
        }
        const auto rewritten = rewritten_.find(expr.get());
        if (rewritten == rewritten_.end()) {
            Reject("bounded result has no rewrite");
        }
        if (rewritten->second.As<ConstantNode>()) {
            Reject("constant results must be materialized by a compute call into fresh storage");
        }
        return rewritten->second;
    }

    std::vector<DimExpr> ParameterDims(const kxc::VarNode* parameter) const {
        const auto* type = parameter->type_annotation.As<kxc::TensorTypeNode>();
        if (!type) Reject("every parameter must carry a TensorType annotation");
        const size_t index = param_index_.at(parameter);
        std::vector<DimExpr> dims;
        dims.reserve(type->shape.size());
        for (size_t axis = 0; axis < type->shape.size(); ++axis) {
            const auto found = axis_symbols_.find(std::make_pair(index, axis));
            if (found != axis_symbols_.end()) {
                dims.push_back(DimExpr::Symbol(found->second));
            } else {
                dims.push_back(DimExpr::Const(type->shape[axis]));
            }
        }
        return dims;
    }

    static std::string OpName(const kxc::CallNode* call) {
        const auto* op = call->op.As<kxc::relay::OpNode>();
        if (!op) Reject("only calls to registered Relay operators are supported");
        return op->name;
    }

    static size_t ExpectedArity(const std::string& name) {
        if (name == "where") return 3;
        if (name == "add" || name == "subtract" || name == "mul" || name == "divide" || name == "pow" ||
            name == "matmul" || name == "gather" || name == "equal" || name == "masked_softmax" ||
            name == "concatenate" || name == "reshape_dynamic" || name == "expand_dynamic") {
            return 2;
        }
        return 1;
    }

    static void RequireData(const NodeInfo& info, const std::string& context) {
        if (info.kind != ValueKind::kData) {
            Reject(context + " must be a data tensor, not a shape value");
        }
    }

    static void RequireShapeValue(const NodeInfo& info, const std::string& context) {
        if (info.kind != ValueKind::kShapeValue) {
            Reject(context +
                   " must be a restricted shape value (shape_of / folded chain)");
        }
    }

    // 属性策略：attr-free 算子不得携带属性；受控算子（reshape_dynamic/
    // expand）必须无调用方属性——目标表达式由本解析器唯一写入。
    static void RequireNoCallerAttrs(const kxc::CallNode* call,
                                     const std::string& name) {
        if (call->attrs.defined()) {
            Reject(name + " must arrive without caller attrs; the restricted "
                           "preparation is the only attrs authority");
        }
    }

    NodeInfo Resolve(const Expr& expr) {
        if (!expr.defined()) Reject("expression is undefined");
        const Object* node = expr.get();
        const auto memo = info_.find(node);
        if (memo != info_.end()) return memo->second;

        // Aggregate and control nodes carry no single tensor leaf of their own;
        // they resolve through the leaf model. A caller that reaches here means
        // the node is used as one tensor, which the leaf resolver rejects.
        if (expr.As<kxc::TupleNode>() || expr.As<kxc::TupleGetItemNode>() ||
            expr.As<kxc::IfNode>() || expr.As<kxc::WhileNode>() ||
            expr.As<kxc::LetNode>()) {
            const std::vector<Leaf> leaves = ResolveLeaves(expr);
            if (leaves.size() != 1) {
                Reject("a multi-leaf control expression cannot be one tensor");
            }
            info_[node] = leaves.front().first;
            return leaves.front().first;
        }

        if (const auto* var = expr.As<kxc::VarNode>()) {
            if (param_index_.find(node) == param_index_.end()) {
                Reject("graph references a non-parameter variable");
            }
            NodeInfo info;
            info.kind = ValueKind::kData;
            info.dims = ParameterDims(var);
            info_[node] = info;
            rewritten_[node] = expr;
            rewritten_info_[node] = info;
            return info;
        }
        if (const auto* constant = expr.As<kxc::ConstantNode>()) {
            if (!constant->data.defined() || constant->data.device() != Device::CPU()) {
                Reject("bounded constants require a defined CPU:0 tensor");
            }
            NodeInfo info;
            for (int64_t extent : constant->data.shape()) {
                if (extent < 0) Reject("constant tensor extents must be static");
                info.dims.push_back(DimExpr::Const(extent));
            }
            info_[node] = info;
            rewritten_[node] = expr;
            rewritten_info_[node] = info;
            return info;
        }

        const auto* call = expr.As<kxc::CallNode>();
        if (!call) {
            Reject("data expressions must be parameters or restricted calls");
        }
        const std::string name = OpName(call);
        if (call->args.size() != ExpectedArity(name) &&
            !(name == "slice" && call->args.size() == 5)) {
            Reject("operator " + name + " arity mismatch");
        }

        NodeInfo info;
        if (name == "add" || name == "subtract" || name == "mul" || name == "divide" || name == "pow") {
            const NodeInfo& lhs = Resolve(call->args[0]);
            const NodeInfo& rhs = Resolve(call->args[1]);
            if (lhs.kind == ValueKind::kShapeValue || rhs.kind == ValueKind::kShapeValue ||
                lhs.constant_control || rhs.constant_control) {
                info = ResolveControlArithmetic(name, call, node);
            } else {
                if (name == "subtract") Reject("subtract is restricted to proved shape controls");
                RequireData(lhs, name + " lhs");
                RequireData(rhs, name + " rhs");
                info.kind = ValueKind::kData;
                info.dims = BroadcastDims(lhs.dims, rhs.dims, name);
                rewritten_[node] =
                    RebuildCall(name, call, kxc::relay::Attrs(),
                                {rewritten_.at(call->args[0].get()),
                                 rewritten_.at(call->args[1].get())});
            }
        } else if (name == "relu" || name == "nn_relu" || name == "sqrt" || name == "sigmoid" || name == "neg") {
            if (name != "relu" && name != "nn_relu") RequireNoCallerAttrs(call, name);
            const NodeInfo& input = Resolve(call->args[0]);
            RequireData(input, name + " input");
            info.kind = ValueKind::kData;
            info.dims = input.dims;
            rewritten_[node] = RebuildCall(
                name, call, kxc::relay::Attrs(),
            {rewritten_.at(call->args[0].get())});
        } else if (name == "trilu") {
            const NodeInfo input = Resolve(call->args[0]);
            RequireData(input, "trilu input");
            const auto* attrs = call->attrs.As<relay::TriluAttrsNode>();
            if (!attrs || (attrs->upper != 0 && attrs->upper != 1) || input.dims.size() < 2) {
                Reject("trilu requires rank >= 2 and upper 0 or 1");
            }
            info.dims = input.dims;
            rewritten_[node] = RebuildCall(name, call, call->attrs,
                {rewritten_.at(call->args[0].get())});
        } else if (name == "matmul") {
            const NodeInfo lhs = Resolve(call->args[0]);
            const NodeInfo rhs = Resolve(call->args[1]);
            RequireData(lhs, "matmul lhs");
            RequireData(rhs, "matmul rhs");
            if (lhs.dims.size() < 2 || rhs.dims.size() < 2) {
                Reject("bounded matmul requires ranks >= 2");
            }
            if (lhs.dims.back() != rhs.dims[rhs.dims.size() - 2]) {
                Reject("bounded matmul reduction dimensions must be identical expressions");
            }
            info.dims = BroadcastDims(
                {lhs.dims.begin(), lhs.dims.end() - 2},
                {rhs.dims.begin(), rhs.dims.end() - 2}, "matmul batch");
            info.dims.push_back(lhs.dims[lhs.dims.size() - 2]);
            info.dims.push_back(rhs.dims.back());
            rewritten_[node] = RebuildCall(
                name, call, kxc::relay::Attrs(),
                {rewritten_.at(call->args[0].get()),
                 rewritten_.at(call->args[1].get())});
        } else if (name == "cast") {
            const NodeInfo input = Resolve(call->args[0]);
            if (call->attrs.defined() && !call->attrs.As<kxc::relay::CastAttrsNode>()) {
                Reject("cast requires CastAttrs");
            }
            if (input.kind == ValueKind::kShapeValue || input.constant_control) {
                const auto* attrs = call->attrs.As<kxc::relay::CastAttrsNode>();
                if (!attrs || attrs->to != 2) Reject("shape control cast must preserve int64");
                info = input;
                rewritten_[node] = rewritten_.at(call->args[0].get());
            } else {
                info.dims = input.dims;
                rewritten_[node] = RebuildCall(name, call, call->attrs,
                                              {rewritten_.at(call->args[0].get())});
            }
        } else if (name == "reduce_mean") {
            const NodeInfo input = Resolve(call->args[0]);
            RequireData(input, "reduce_mean input");
            const auto* attrs = call->attrs.As<kxc::relay::ReduceMeanAttrsNode>();
            if (call->attrs.defined() && !attrs) Reject("reduce_mean requires ReduceMeanAttrs");
            if (attrs && attrs->keepdims != 0 && attrs->keepdims != 1) {
                Reject("reduce_mean keepdims must be 0 or 1");
            }
            std::set<int64_t> axes;
            if (!attrs || attrs->axes.empty()) {
                for (size_t axis = 0; axis < input.dims.size(); ++axis) axes.insert(axis);
            } else {
                for (int64_t raw : attrs->axes) {
                    if (!axes.insert(NormalizeAxis(name, raw, input.dims.size())).second) {
                        Reject("reduce_mean axes must be unique");
                    }
                }
            }
            int64_t count = 1;
            for (size_t axis = 0; axis < input.dims.size(); ++axis) {
                if (!axes.count(axis)) {
                    info.dims.push_back(input.dims[axis]);
                    continue;
                }
                // Existing ReduceMean TE divides by a constant reduction size.
                // Batch/sequence may vary, but reduced axes must remain fixed.
                if (input.dims[axis].kind() != DimExpr::Kind::kConst) {
                    Reject("bounded reduce_mean requires static reduced axes");
                }
                const int64_t extent = input.dims[axis].Evaluate(BindingSet());
                if (extent < 1 || count > std::numeric_limits<int64_t>::max() / extent) {
                    Reject("reduce_mean reduction size must be positive and fit int64");
                }
                count *= extent;
                if (!attrs || attrs->keepdims) info.dims.push_back(DimExpr::Const(1));
            }
            Array<int64_t> normalized_axes;
            for (int64_t axis : axes) normalized_axes.push_back(axis);
            rewritten_[node] = RebuildCall(name, call,
                kxc::relay::ReduceMeanAttrs::Create(normalized_axes, attrs ? attrs->keepdims : 1),
                {rewritten_.at(call->args[0].get())});
        } else if (name == "transpose") {
            const NodeInfo input = Resolve(call->args[0]);
            RequireData(input, "transpose input");
            const auto* attrs = call->attrs.As<kxc::relay::TransposeAttrsNode>();
            if (call->attrs.defined() && !attrs) {
                Reject("transpose requires TransposeAttrs");
            }
            kxc::Array<int64_t> axes;
            if (!attrs || attrs->perm.empty()) {
                for (size_t i = input.dims.size(); i > 0; --i) {
                    axes.push_back(static_cast<int64_t>(i - 1));
                }
            } else {
                for (int64_t axis : attrs->perm) {
                    axes.push_back(NormalizeAxis(name, axis, input.dims.size()));
                }
            }
            if (axes.size() != input.dims.size() ||
                std::set<int64_t>(axes.begin(), axes.end()).size() != axes.size()) {
                Reject("transpose axes must be a fixed-rank permutation");
            }
            for (int64_t axis : axes) info.dims.push_back(input.dims[axis]);
            rewritten_[node] = RebuildCall(
                name, call, kxc::relay::TransposeAttrs::Create(axes),
                {rewritten_.at(call->args[0].get())});
        } else if (name == "softmax" || name == "masked_softmax") {
            const NodeInfo input = Resolve(call->args[0]);
            RequireData(input, name + " input");
            const auto* attrs = call->attrs.As<kxc::relay::SoftmaxAttrsNode>();
            if (call->attrs.defined() && !attrs) {
                Reject(name + " requires SoftmaxAttrs");
            }
            const int axis = NormalizeAxis(name, attrs ? attrs->axis : -1,
                                           input.dims.size());
            const DimExpr& reduction = input.dims[axis];
            const int64_t lower = EvaluateAtBounds(reduction,symbol_lowers_);
            if (lower < 1) {
                Reject("bounded " + name + " reduction extent must be provably positive");
            }
            info.dims = input.dims;
            Array<Expr> args{rewritten_.at(call->args[0].get())};
            if (name == "masked_softmax") {
                const NodeInfo mask = Resolve(call->args[1]);
                RequireData(mask, "masked_softmax mask");
                if (BroadcastDims(input.dims, mask.dims, name) != input.dims) {
                    Reject("masked_softmax mask cannot expand the data shape");
                }
                args.push_back(rewritten_.at(call->args[1].get()));
            }
            rewritten_[node] = RebuildCall(
                name, call, kxc::relay::SoftmaxAttrs::Create(axis),
                args);
        } else if (name == "equal") {
            info = ResolveControlEqual(call, node);
        } else if (name == "where") {
            info = ResolveControlWhere(call, node);
        } else if (name == "constant_of_shape") {
            info = ResolveControlFill(call, node);
        } else if (name == "shape_of") {
            const NodeInfo& source = Resolve(call->args[0]);
            RequireNoCallerAttrs(call, name);
            if (source.kind == ValueKind::kShapeValue) {
                if (source.scalar_shape_value) Reject("shape_of scalar controls are outside the subset");
                info = MakeControl(call, node, nullptr,
                    {DimExpr::Const(static_cast<int64_t>(source.elements.size()))});
            } else {
                RequireData(source, "shape_of source");
                if (source.dims.empty()) {
                    Reject("shape_of requires a rank >= 1 source");
                }
                info.kind = ValueKind::kShapeValue;
                info.elements = source.dims;
                info.source = call->args[0].get();
                rewritten_[node] = RebuildCall(
                    name, call, kxc::relay::Attrs(),
                    {rewritten_.at(call->args[0].get())});
            }
        } else if (name == "gather") {
            info = ResolveGather(call, node);
        } else if (name == "concatenate") {
            info = ResolveConcatenate(call, node);
        } else if (name == "slice") {
            info = ResolveSlice(call, node);
        } else if (name == "reshape_dynamic") {
            info = ResolveReshapeDynamic(call, node);
        } else if (name == "expand_dynamic") {
            info = ResolveExpand(call, node);
        } else if (name == "squeeze") {
            info = ResolveSqueeze(call, node);
        } else if (name == "unsqueeze") {
            info = ResolveUnsqueeze(call, node);
        } else {
            Reject("unsupported Relay operation in the restricted shape-value "
                   "walk: " + name);
        }

        info_[node] = info;
        rewritten_info_[rewritten_[node].get()] = info;
        return info;
    }

    Expr RebuildCall(const std::string& name, const kxc::CallNode* call,
                     kxc::relay::Attrs attrs, kxc::Array<kxc::Expr> args) {
        if (!attrs.defined()) {
            RequireNoCallerAttrs(call, name);
            return kxc::Call(kxc::relay::Op::Get(name), std::move(args));
        }
        return kxc::Call(kxc::relay::Op::Get(name), std::move(args),
                         std::move(attrs));
    }

    // 链折叠：以根源为唯一输入生成 shape_expr，记录 value 表达式覆盖。
    NodeInfo FoldShapeExpr(const kxc::CallNode* call, const Object* node,
                           const Object* source,
                           std::vector<DimExpr> elements) {
        const NodeInfo& source_info = info_.at(source);
        if (elements.size() > kMaxShapeExprElements) {
            Reject("folded shape value exceeds the declared element subset");
        }
        const EncodedExpr encoded = EncodeElements(
            elements, source_info.dims, "shape_expr fold");
        Expr folded = kxc::Call(
            kxc::relay::Op::Get("shape_expr"), {rewritten_.at(source)},
            kxc::relay::ShapeExprAttrs::Create(encoded.kinds, encoded.values,
                                               encoded.axes));
        rewritten_[node] = folded;
        value_expr_overrides_[folded.get()] = encoded;
        NodeInfo info;
        info.kind = ValueKind::kShapeValue;
        info.elements = std::move(elements);
        info.source = source;
        return info;
    }

    // Data Gather requires a static table and keeps index shape expressions.
    // Shape-value Gather uses constant axis-0 indices and folds to shape_expr.
    NodeInfo ResolveGather(const kxc::CallNode* call, const Object* node) {
        const NodeInfo& data = Resolve(call->args[0]);
        if (data.kind == ValueKind::kData) {
            const auto* attrs = call->attrs.As<relay::GatherAttrsNode>();
            if (!attrs || data.dims.empty()) Reject("data gather requires attrs and nonempty data rank");
            const int64_t axis=NormalizeAxis("data gather",attrs->axis,data.dims.size());
            for (const auto& dim:data.dims) {
                if (dim.kind()!=DimExpr::Kind::kConst) Reject("bounded data gather requires a static table");
            }
            const NodeInfo indices=Resolve(call->args[1]);
            RequireData(indices,"data gather indices");
            NodeInfo result;
            result.dims.insert(result.dims.end(),data.dims.begin(),data.dims.begin()+axis);
            result.dims.insert(result.dims.end(),indices.dims.begin(),indices.dims.end());
            result.dims.insert(result.dims.end(),data.dims.begin()+axis+1,data.dims.end());
            rewritten_[node]=RebuildCall("gather",call,call->attrs,
                {rewritten_.at(call->args[0].get()),rewritten_.at(call->args[1].get())});
            return result;
        }
        // 形状值恒为 rank-1 int64 向量，其长度即元素数；这里只需确认来源
        // 是形状值，长度由下面的索引边界检查裁决。
        RequireShapeValue(data, "gather data");
        if (data.scalar_shape_value) Reject("gather shape data must be a vector");
        const auto* gather_attrs = call->attrs.As<kxc::relay::GatherAttrsNode>();
        if (!gather_attrs) {
            Reject("gather requires GatherAttrs");
        }
        if (NormalizeAxis("gather", gather_attrs->axis, 1) != 0) {
            Reject("bounded gather is restricted to axis 0");
        }
        const std::vector<int64_t> raw =
            ReadConstantVector(call->args[1].As<kxc::ConstantNode>(),
                               "gather indices", true);
        const int64_t length = static_cast<int64_t>(data.elements.size());
        std::vector<DimExpr> elements;
        elements.reserve(raw.size());
        for (int64_t index : raw) {
            if (index < 0) index += length;
            if (index < 0 || index >= length) {
                Reject("gather index is out of bounds for the shape value");
            }
            elements.push_back(data.elements[static_cast<size_t>(index)]);
        }
        return MakeControl(call, node, data.source, std::move(elements),
                           call->args[1].As<ConstantNode>()->data.shape().empty());
    }

    NodeInfo ShapeArgument(const Expr& expression, const std::string& context,
                           bool allow_scalar = false) {
        NodeInfo result = Resolve(expression);
        if (const auto* constant = expression.As<ConstantNode>()) {
            const DLDataType dtype = constant->data.dtype();
            if (dtype.code != kDLInt || dtype.bits != 64 || dtype.lanes != 1) {
                Reject(context + " literal shape vectors must use int64");
            }
            result.kind = ValueKind::kShapeValue;
            result.scalar_shape_value = constant->data.shape().empty();
            for (int64_t value : ReadConstantVector(constant, context, allow_scalar)) {
                result.elements.push_back(DimExpr::Const(value));
            }
        }
        RequireShapeValue(result, context);
        if (!allow_scalar && result.scalar_shape_value) Reject(context + " must be a shape vector");
        return result;
    }

    // Materialize only proved constant control values. Symbolic controls keep
    // the existing ShapeExpr encoding and single source; ordinary data is not evaluated.
    NodeInfo MakeControl(const kxc::CallNode* call, const Object* node,
                         const Object* source, std::vector<DimExpr> elements,
                         bool scalar = false) {
        if (elements.size() > kMaxShapeExprElements || (scalar && elements.size() != 1)) {
            Reject("control value exceeds the scalar/vector subset");
        }
        if (std::all_of(elements.begin(), elements.end(), [](const DimExpr& value) {
                return value.kind() == DimExpr::Kind::kConst;
            })) {
            std::vector<int64_t> values;
            for (const DimExpr& value : elements) values.push_back(value.Evaluate(BindingSet()));
            Array<int64_t> shape;
            if (!scalar) shape.push_back(static_cast<int64_t>(values.size()));
            auto data = runtime::NDArray::Empty(shape, runtime::DataTypeFromString("int64"), Device::CPU());
            if (!values.empty()) data.CopyFromBytes(values.data(), values.size() * sizeof(int64_t));
            rewritten_[node] = Constant(std::move(data));
            NodeInfo result;
            result.kind = ValueKind::kShapeValue;
            result.scalar_shape_value = scalar;
            result.elements = std::move(elements);
            return result;
        }
        for (const auto& element : elements) (void)EvaluateAtBounds(element,symbol_uppers_);
        // Prefer the original source. Cancellation can expose an earlier graph
        // input dimension (e.g. total-1 = past); re-anchor only after proof.
        if (!source || !TryEncodeElements(elements,info_.at(source).dims)) {
            source = nullptr;
            for (const auto& parameter : snapshot_->params) {
                const NodeInfo candidate = Resolve(parameter);
                if (TryEncodeElements(elements,candidate.dims)) { source = parameter.get(); break; }
            }
        }
        if (!source) Reject("symbolic control requires a proved tensor source");
        NodeInfo result = FoldShapeExpr(call, node, source, std::move(elements));
        result.scalar_shape_value = scalar;
        return result;
    }

    // A binary shape add can be needed only as an intermediate slice endpoint
    // (for example P+C, where P and C come from different graph inputs).  It
    // is deliberately kept as a transient proof; consumers must lower it to
    // a prepared operator with explicit shape anchors instead of materializing
    // an unrestricted runtime shape tensor.
    NodeInfo MakeDeferredControl(const std::string& name, const kxc::CallNode* call,
                                 const Object* node, const NodeInfo& lhs,
                                 const NodeInfo& rhs, std::vector<DimExpr> elements,
                                 bool scalar) {
        for (const auto& element : elements) (void)EvaluateAtBounds(element, symbol_uppers_);
        rewritten_[node] = RebuildCall(
            name, call, kxc::relay::Attrs(),
            {rewritten_.at(call->args[0].get()), rewritten_.at(call->args[1].get())});
        NodeInfo result;
        result.kind = ValueKind::kShapeValue;
        result.scalar_shape_value = scalar;
        result.elements = std::move(elements);
        return result;
    }

    static size_t ControlBroadcastLength(size_t left, size_t right) {
        if (!left || !right || (left != right && left != 1 && right != 1)) {
            Reject("control scalar/vector lengths cannot broadcast");
        }
        return std::max(left, right);
    }

    struct ControlOperand {
        NodeInfo proof;
        std::optional<std::vector<int64_t>> literals;
        bool scalar{false};
        size_t size() const { return literals ? literals->size() : proof.elements.size(); }
        std::optional<int64_t> Known(size_t index) const {
            index = size() == 1 ? 0 : index;
            if (literals) return literals->at(index);
            const DimExpr& expression = proof.elements.at(index);
            if (expression.kind() == DimExpr::Kind::kConst) return expression.Evaluate(BindingSet());
            return std::nullopt;
        }
        const DimExpr& Expression(size_t index) const {
            const size_t selected = size() == 1 ? 0 : index;
            if (literals && literals->at(selected) < 0) {
                Reject("slice control values must be nonnegative before symbolic proof");
            }
            return proof.elements.at(selected);
        }
    };

    ControlOperand ReadControl(const Expr& expression, const std::string& context) {
        ControlOperand result;
        result.proof = Resolve(expression);
        result.scalar = result.proof.scalar_shape_value;
        if (result.proof.kind != ValueKind::kShapeValue) {
            const auto* constant = rewritten_.at(expression.get()).As<ConstantNode>();
            if (!constant || constant->data.dtype().code != kDLInt ||
                constant->data.dtype().bits != 64 || constant->data.dtype().lanes != 1 ||
                constant->data.shape().size() > 1 || constant->data.NBytes() > 8 * kMaxShapeExprElements) {
                Reject(context + " requires proved int64 scalar/vector controls");
            }
            result.scalar = constant->data.shape().empty();
            result.literals = ReadConstantVector(constant, context, true);
            // Keep a parallel DimExpr view for callers that need the
            // expression form of a literal (for example slice start/end
            // proofs).  Previously literals had a valid payload but an empty
            // proof.elements vector, so Expression(0) reached vector::at()
            // and surfaced as the platform-specific "invalid vector
            // subscript" exception instead of a controlled proof rejection.
            result.proof.elements.reserve(result.literals->size());
            for (const int64_t value : *result.literals) {
                // DimExpr intentionally admits only nonnegative dimensions;
                // retain negative controls as literals so consumers can issue
                // a precise control-domain rejection.
                if (value >= 0) result.proof.elements.push_back(DimExpr::Const(value));
            }
            result.proof.scalar_shape_value = result.scalar;
        }
        return result;
    }

    const Object* SourceForDirectSymbol(const DimExpr& expression) const {
        if (expression.kind() != DimExpr::Kind::kSymbol) return nullptr;
        const auto symbols = expression.Symbols();
        if (symbols.size() != 1) return nullptr;
        for (const auto& entry : axis_symbols_) {
            if (entry.second == symbols[0] && entry.first.first < snapshot_->params.size()) {
                return snapshot_->params[entry.first.first].get();
            }
        }
        return nullptr;
    }

    std::pair<const Object*, int> SourceForStaticExtent(int64_t extent) const {
        for (std::size_t index = 0; index < snapshot_->params.size(); ++index) {
            const auto* parameter = snapshot_->params[index].operator->();
            const auto dims = ParameterDims(parameter);
            for (std::size_t axis = 0; axis < dims.size(); ++axis) {
                if (dims[axis].kind() == DimExpr::Kind::kConst &&
                    dims[axis].Evaluate(BindingSet()) == extent) {
                    return {parameter, static_cast<int>(axis)};
                }
            }
        }
        return {nullptr, -1};
    }

    NodeInfo MakeIntegerControl(const kxc::CallNode* call, const Object* node,
                                std::vector<int64_t> values, bool scalar = false) {
        if (std::all_of(values.begin(), values.end(), [](int64_t value) { return value >= 0; })) {
            std::vector<DimExpr> elements;
            for (int64_t value : values) elements.push_back(DimExpr::Const(value));
            return MakeControl(call, node, nullptr, std::move(elements), scalar);
        }
        if (values.size() > kMaxShapeExprElements || (scalar && values.size() != 1)) {
            Reject("signed control exceeds the scalar/vector subset");
        }
        NodeInfo result;
        result.constant_control = true;
        Array<int64_t> shape;
        if (!scalar) {
            shape.push_back(static_cast<int64_t>(values.size()));
            result.dims.push_back(DimExpr::Const(static_cast<int64_t>(values.size())));
        }
        auto data = runtime::NDArray::Empty(shape, runtime::DataTypeFromString("int64"), Device::CPU());
        if (!values.empty()) data.CopyFromBytes(values.data(), values.size() * sizeof(int64_t));
        rewritten_[node] = Constant(std::move(data));
        return result;
    }

    NodeInfo ResolveControlArithmetic(const std::string& name, const kxc::CallNode* call,
                                       const Object* node) {
        RequireNoCallerAttrs(call, name);
        const auto lhs = ReadControl(call->args[0], name + " control lhs");
        const auto rhs = ReadControl(call->args[1], name + " control rhs");
        const size_t count = ControlBroadcastLength(lhs.size(), rhs.size());
        bool symbolic = false;
        for (size_t i = 0; i < count; ++i) symbolic = symbolic || !lhs.Known(i) || !rhs.Known(i);
        if (symbolic) {
            if (name != "add" && name != "subtract") Reject("control arithmetic requires statically proved integer values");
            std::vector<DimExpr> elements;
            const Object* source = nullptr;
            bool deferred_binary_add = false;
            for (size_t i = 0; i < count; ++i) {
                const auto left = lhs.Known(i), right = rhs.Known(i);
                if (left && right) {
                    if (name == "add" && *left >= 0 && *right >= 0) {
                        elements.push_back(DimExpr::Add({DimExpr::Const(*left),DimExpr::Const(*right)}));
                    } else if (name == "subtract" && *left >= *right && *right >= 0) {
                        elements.push_back(DimExpr::Const(*left-*right));
                    } else Reject("symbolic control arithmetic requires nonnegative Add/Sub");
                    continue;
                }
                if (name == "add" && (left || right) && (left ? *left : *right) >= 0) {
                    const auto& dynamic = left ? rhs : lhs;
                    elements.push_back(DimExpr::Add({dynamic.Expression(i),DimExpr::Const(left ? *left : *right)}));
                    source = dynamic.proof.source;
                } else if (name == "add" && !left && !right) {
                    const DimExpr& left_expression = lhs.Expression(i);
                    const DimExpr& right_expression = rhs.Expression(i);
                    if (left_expression.kind() != DimExpr::Kind::kSymbol ||
                        right_expression.kind() != DimExpr::Kind::kSymbol ||
                        left_expression == right_expression) {
                        Reject("symbolic control arithmetic requires an input-axis expression and a nonnegative constant");
                    }
                    elements.push_back(DimExpr::Add({left_expression, right_expression}));
                    deferred_binary_add = true;
                } else if (name == "subtract" && !left && !right) {
                    const DimExpr& left_expression = lhs.Expression(i);
                    const DimExpr& right_expression = rhs.Expression(i);
                    const auto constant_residual =
                        ProveNonnegativeConstantOffset(left_expression, right_expression);
                    if (constant_residual) {
                        elements.push_back(DimExpr::Const(*constant_residual));
                        continue;
                    }
                    std::optional<DimExpr> residual;
                    for (const auto& symbol : left_expression.Symbols()) {
                        const DimExpr candidate = DimExpr::Symbol(symbol);
                        if (left_expression == DimExpr::Add({right_expression, candidate})) {
                            if (residual) Reject("shape subtraction has an ambiguous residual axis");
                            residual = candidate;
                        }
                    }
                    if (!residual) {
                        Reject("symbolic control arithmetic requires an input-axis expression and a nonnegative constant (subtract residual lhs=" +
                               left_expression.CanonicalString() + ", rhs=" +
                               right_expression.CanonicalString() + ")");
                    }
                    elements.push_back(*residual);
                    source = SourceForDirectSymbol(*residual);
                } else if (name == "subtract" && !left && right && *right >= 0) {
                    const auto& expression = lhs.Expression(i);
                    const auto symbols = expression.Symbols();
                    if (symbols.size() != 1) Reject("shape subtraction requires a single proved symbol");
                    const auto base = DimExpr::Symbol(symbols[0]);
                    const auto offset = ProveNonnegativeConstantOffset(expression,base);
                    if (!offset || *offset < *right) Reject("shape subtraction cannot prove a nonnegative constant offset");
                    elements.push_back(DimExpr::Add({base,DimExpr::Const(*offset-*right)}));
                    source = lhs.proof.source;
                } else Reject("symbolic control arithmetic requires an input-axis expression and a nonnegative constant");
            }
            if (deferred_binary_add) {
                return MakeDeferredControl(name, call, node, lhs.proof, rhs.proof,
                                           std::move(elements), lhs.scalar && rhs.scalar);
            }
            return MakeControl(call,node,source,std::move(elements),lhs.scalar && rhs.scalar);
        }
        std::vector<int64_t> values;
        for (size_t i = 0; i < count; ++i) {
            const auto left = lhs.Known(i), right = rhs.Known(i);
            if (!left || !right) {
                Reject("control arithmetic requires statically proved integer values");
            }
            const int64_t a = *left, b = *right;
            if (a == std::numeric_limits<int64_t>::min() || b == std::numeric_limits<int64_t>::min()) {
                Reject("control magnitude exceeds nonnegative DimExpr arithmetic");
            }
            if (name == "mul") {
                // DimExpr owns checked magnitude arithmetic; signs belong to
                // literal control payloads and never enter dimension contracts.
                const int64_t magnitude = DimExpr::Mul(
                    {DimExpr::Const(a < 0 ? -a : a), DimExpr::Const(b < 0 ? -b : b)}).Evaluate(BindingSet());
                values.push_back((a < 0) != (b < 0) ? -magnitude : magnitude);
            } else if (name == "add" && a >= 0 && b >= 0) {
                values.push_back(DimExpr::Add({DimExpr::Const(a), DimExpr::Const(b)}).Evaluate(BindingSet()));
            } else if (name == "subtract" && a >= b && b >= 0) {
                values.push_back(a-b);
            } else if (name == "divide" && a >= 0 && b > 0) {
                values.push_back(DimExpr::FloorDiv(DimExpr::Const(a), b).Evaluate(BindingSet()));
            } else Reject("control arithmetic admits Mul, nonnegative Add and nonnegative Div by a positive integer");
        }
        return MakeIntegerControl(call, node, std::move(values), lhs.scalar && rhs.scalar);
    }

    NodeInfo ResolveControlFill(const kxc::CallNode* call, const Object* node) {
        const NodeInfo control = ShapeArgument(call->args[0], "constant_of_shape control");
        const auto* attrs = call->attrs.As<relay::ConstantOfShapeAttrsNode>();
        if (!attrs || !attrs->target.empty() || !attrs->expr_kinds.empty() ||
            !attrs->expr_values.empty() || !attrs->expr_axes.empty()) {
            Reject("constant_of_shape source cannot carry caller target expressions");
        }
        if (attrs->dtype_code == 0) {
            if (std::isnan(attrs->value) || control.elements.empty() || control.elements.size() > 8) {
                Reject("constant_of_shape data fill requires a non-NaN float32 and rank in [1,8]");
            }
            std::vector<kxc::shape::experimental::v1::Binding> bounds;
            for (const auto& bound : symbol_uppers_) bounds.push_back({bound.first, bound.second});
            // Reuse checked DimExpr arithmetic for the entire declared domain.
            const int64_t elements = ProductOf(control.elements).Evaluate(BindingSet(std::move(bounds)));
            if (elements > relay::kMaxConstantOfShapeBytes / 4) {
                Reject("constant_of_shape output exceeds the declared byte cap at symbol upper bounds");
            }
            Array<int64_t> target;
            bool dynamic = false;
            for (const auto& dim : control.elements) {
                dynamic = dynamic || dim.kind() != DimExpr::Kind::kConst;
                if (dim.kind() == DimExpr::Kind::kConst) target.push_back(dim.Evaluate(BindingSet()));
            }
            if (dynamic) {
                if (!control.source) Reject("constant_of_shape requires a proved data anchor");
                const auto encoded = EncodeElements(control.elements, info_.at(control.source).dims,
                    "constant_of_shape target");
                rewritten_[node] = RebuildCall("constant_of_shape", call,
                    relay::ConstantOfShapeAttrs::Create({}, 0, attrs->value,
                        encoded.kinds, encoded.values, encoded.axes),
                    {rewritten_.at(control.source), rewritten_.at(call->args[0].get())});
            } else {
                rewritten_[node] = RebuildCall("constant_of_shape", call,
                    relay::ConstantOfShapeAttrs::Create(target, 0, attrs->value),
                    {rewritten_.at(call->args[0].get())});
            }
            NodeInfo info;
            info.dims = control.elements;
            return info;
        }
        if (attrs->dtype_code != 2 ||
            !std::isfinite(attrs->value) || std::trunc(attrs->value) != attrs->value ||
            std::abs(attrs->value) > static_cast<double>(int64_t{1} << 53)) {
            Reject("constant_of_shape control fill requires int64, an exact integer and no caller target");
        }
        if (control.elements.size() != 1 || control.elements[0].kind() != DimExpr::Kind::kConst) {
            Reject("constant_of_shape control length must be statically proved");
        }
        const int64_t count = control.elements[0].Evaluate(BindingSet());
        if (count < 0 || count > static_cast<int64_t>(kMaxShapeExprElements)) {
            Reject("constant_of_shape control exceeds the vector length cap");
        }
        return MakeIntegerControl(call, node,
            std::vector<int64_t>(static_cast<size_t>(count), static_cast<int64_t>(attrs->value)));
    }

    NodeInfo ResolveControlEqual(const kxc::CallNode* call, const Object* node) {
        RequireNoCallerAttrs(call, "equal");
        const auto lhs = ReadControl(call->args[0], "equal lhs");
        const auto rhs = ReadControl(call->args[1], "equal rhs");
        const size_t count = ControlBroadcastLength(lhs.size(), rhs.size());
        std::vector<uint8_t> values(count);
        for (size_t i = 0; i < count; ++i) {
            const auto a = lhs.Known(i), b = rhs.Known(i);
            if (a && b) values[i] = *a == *b;
            else if (!a && !b && lhs.Expression(i) == rhs.Expression(i)) values[i] = true;
            else if (a.has_value() != b.has_value()) {
                const DimExpr& expression = a ? rhs.Expression(i) : lhs.Expression(i);
                const int64_t literal = a ? *a : *b;
                if (literal >= EvaluateAtBounds(expression,symbol_lowers_) &&
                    literal <= EvaluateAtBounds(expression,symbol_uppers_)) {
                    Reject("shape control equality cannot be proved for every admitted input");
                }
                values[i] = false;
            } else Reject("shape control equality cannot be proved for every admitted input");
        }
        Array<int64_t> shape;
        NodeInfo result;
        if (!(lhs.scalar && rhs.scalar)) {
            shape.push_back(static_cast<int64_t>(count));
            result.dims.push_back(DimExpr::Const(static_cast<int64_t>(count)));
        }
        auto data = runtime::NDArray::Empty(shape, runtime::DataTypeFromString("bool"), Device::CPU());
        data.CopyFromBytes(values.data(), values.size());
        rewritten_[node] = Constant(std::move(data));
        return result;
    }

    NodeInfo ResolveControlWhere(const kxc::CallNode* call, const Object* node) {
        RequireNoCallerAttrs(call, "where");
        (void)Resolve(call->args[0]);
        const auto* condition = rewritten_.at(call->args[0].get()).As<ConstantNode>();
        const DLDataType boolean = runtime::DataTypeFromString("bool");
        if (!condition || condition->data.dtype().code != boolean.code ||
            condition->data.dtype().bits != boolean.bits || condition->data.dtype().lanes != 1 ||
            condition->data.shape().size() > 1 || condition->data.NBytes() > kMaxShapeExprElements) {
            Reject("shape Where requires a statically proved scalar/vector bool condition");
        }
        const NodeInfo lhs = ShapeArgument(call->args[1], "where true branch", true);
        const NodeInfo rhs = ShapeArgument(call->args[2], "where false branch", true);
        const size_t count = ControlBroadcastLength(
            ControlBroadcastLength(lhs.elements.size(), rhs.elements.size()), condition->data.NBytes());
        std::vector<uint8_t> predicate(condition->data.NBytes());
        condition->data.CopyToBytes(predicate.data(), predicate.size());
        std::vector<DimExpr> elements;
        const Object* source = nullptr;
        for (size_t i = 0; i < count; ++i) {
            const NodeInfo& branch = predicate[predicate.size() == 1 ? 0 : i] ? lhs : rhs;
            const DimExpr& element = branch.elements[branch.elements.size() == 1 ? 0 : i];
            if (element.kind() != DimExpr::Kind::kConst) {
                if (source && source != branch.source) Reject("shape Where mixes independent tensor sources");
                source = branch.source;
            }
            elements.push_back(element);
        }
        return MakeControl(call, node, source, std::move(elements),
            condition->data.shape().empty() && lhs.scalar_shape_value && rhs.scalar_shape_value);
    }

    NodeInfo ResolveConcatenate(const kxc::CallNode* call, const Object* node) {
        const NodeInfo left = Resolve(call->args[0]), right = Resolve(call->args[1]);
        if (left.kind == ValueKind::kData && right.kind == ValueKind::kData &&
            !(rewritten_.at(call->args[0].get()).As<ConstantNode>() &&
              rewritten_.at(call->args[1].get()).As<ConstantNode>())) {
            const auto* attrs = call->attrs.As<kxc::relay::ConcatenateAttrsNode>();
            if (!attrs || left.dims.empty() || left.dims.size() != right.dims.size()) {
                Reject("data concatenate requires attrs and equal nonzero rank");
            }
            const size_t axis = NormalizeAxis("concatenate", attrs->axis, left.dims.size());
            if (left.dims[axis].kind() != DimExpr::Kind::kConst &&
                right.dims[axis].kind() != DimExpr::Kind::kConst) {
                Reject("data concatenate requires at least one static concatenation axis");
            }
            NodeInfo result;
            result.dims = left.dims;
            for (size_t i = 0; i < left.dims.size(); ++i) {
                if (i != axis && left.dims[i] != right.dims[i]) {
                    Reject("data concatenate non-axis dimensions must be identical expressions");
                }
            }
            result.dims[axis] = DimExpr::Add({left.dims[axis], right.dims[axis]});
            rewritten_[node] = RebuildCall("concatenate", call, call->attrs,
                {rewritten_.at(call->args[0].get()), rewritten_.at(call->args[1].get())});
            return result;
        }
        const NodeInfo lhs = ShapeArgument(call->args[0], "concatenate lhs");
        const NodeInfo rhs = ShapeArgument(call->args[1], "concatenate rhs");
        // 两侧都是 rank-1 形状值；长度可任意，拼接结果长度是二者之和，
        // 上限由 FoldShapeExpr 的元素子集检查裁决。
        RequireShapeValue(lhs, "concatenate lhs");
        RequireShapeValue(rhs, "concatenate rhs");
        const auto* concatenate_attrs =
            call->attrs.As<kxc::relay::ConcatenateAttrsNode>();
        if (!concatenate_attrs) {
            Reject("concatenate requires ConcatenateAttrs");
        }
        if (NormalizeAxis("concatenate", concatenate_attrs->axis, 1) != 0) {
            Reject("bounded concatenate is restricted to axis 0");
        }
        if (lhs.source && rhs.source && lhs.source != rhs.source) {
            Reject("shape chains must stay rooted at a single source tensor");
        }
        const Object* source = lhs.source ? lhs.source : rhs.source;
        std::vector<DimExpr> elements = lhs.elements;
        elements.insert(elements.end(), rhs.elements.begin(), rhs.elements.end());
        return MakeControl(call, node, source, std::move(elements));
    }

    NodeInfo ResolveSlice(const kxc::CallNode* call, const Object* node) {
        const NodeInfo data = Resolve(call->args[0]);
        RequireData(data, "slice data");
        kxc::relay::Attrs attrs = call->attrs;
        if (call->args.size() == 5) {
            if (attrs.defined()) Reject("slice source controls must arrive without attrs");
            std::vector<ControlOperand> operands;
            for (size_t argument = 1; argument < 5; ++argument) {
                operands.push_back(ReadControl(call->args[argument], "slice"));
                if (operands.back().scalar || operands.back().size() == 0) {
                    Reject("slice controls must be nonempty vectors");
                }
            }
            bool dynamic_end = false;
            for (size_t i = 0; i < operands[1].size(); ++i) {
                dynamic_end = dynamic_end || !operands[1].Known(i);
            }
            if (dynamic_end) {
                for (const auto& operand : operands) {
                    if (operand.size() != 1) Reject("slice dynamic prefix requires one selected axis");
                }
                if (operands[3].Known(0) != std::optional<int64_t>(1) || !operands[2].Known(0)) {
                    Reject("slice dynamic prefix/window requires step 1 and a static axis");
                }
                for (const auto& dim : data.dims) {
                    if (dim.kind() != DimExpr::Kind::kConst) Reject("slice dynamic prefix requires a static table");
                }
                const int axis = NormalizeAxis("slice prefix", *operands[2].Known(0), data.dims.size());
                const DimExpr end = operands[1].Expression(0);
                const bool prefix = operands[0].Known(0) == std::optional<int64_t>(0);
                if (const auto static_start = operands[0].Known(0);
                    static_start && *static_start != 0) {
                    Reject("slice dynamic prefix requires start 0");
                }
                int64_t count = -1;
                const Object* source = operands[1].proof.source;
                const Object* window_source = source;
                DimExpr anchor = end;
                if (!prefix) {
                    anchor = operands[0].Expression(0);
                    source = operands[0].proof.source;
                    const auto static_start = operands[0].Known(0);
                    if (static_start) {
                        const auto static_source = SourceForStaticExtent(*static_start);
                        source = static_source.first;
                        anchor = DimExpr::Const(*static_start);
                    }
                    const auto proved_count = ProveNonnegativeConstantOffset(end,anchor);
                    if (proved_count) {
                        count = *proved_count;
                    } else {
                        // Dynamic window: end must be anchor + one distinct
                        // direct input axis (the current-token length).
                        const auto start_symbols = anchor.Symbols();
                        if ((!static_start && (start_symbols.size() != 1 ||
                                               anchor.kind() != DimExpr::Kind::kSymbol)) ||
                            !source) {
                            Reject("slice dynamic window requires a direct start input-axis symbol or fixed extent anchor");
                        }
                        std::optional<DimExpr> dynamic_count;
                        for (const auto& symbol : end.Symbols()) {
                            const DimExpr candidate = DimExpr::Symbol(symbol);
                            if (candidate != anchor &&
                                end == DimExpr::Add({anchor, candidate})) {
                                if (dynamic_count) Reject("slice dynamic window has an ambiguous length axis");
                                dynamic_count = candidate;
                            }
                        }
                        if (!dynamic_count) {
                            Reject("slice window requires a proved constant or direct dynamic length");
                        }
                        window_source = SourceForDirectSymbol(*dynamic_count);
                        if (!source || !window_source || source == window_source) {
                            Reject("slice dynamic window requires distinct start and length tensor sources");
                        }
                        const int start_axis = static_start ?
                            SourceForStaticExtent(*static_start).second :
                            EncodeElements({anchor}, info_.at(source).dims,
                                            "slice dynamic window start").axes[0];
                        const auto encoded_count = EncodeElements({*dynamic_count},
                                                                  info_.at(window_source).dims,
                                                                  "slice dynamic window length");
                        const int64_t capacity = data.dims[axis].Evaluate(BindingSet());
                        const int64_t start_upper = static_start ? *static_start
                            : symbol_uppers_.at(start_symbols[0]);
                        const int64_t count_upper = symbol_uppers_.at(dynamic_count->Symbols()[0]);
                        if (start_upper > capacity || count_upper > capacity - start_upper) {
                            Reject("slice dynamic window upper bound exceeds table capacity");
                        }
                        NodeInfo result;
                        result.dims = data.dims;
                        result.dims[axis] = *dynamic_count;
                        rewritten_[node] = RebuildCall("slice", call,
                            relay::SliceAttrs::Create({}, {}, {}, {}, axis,
                                start_axis, -2, encoded_count.axes[0]),
                            {rewritten_.at(call->args[0].get()), rewritten_.at(source),
                             rewritten_.at(window_source)});
                        return result;
                    }
                }
                if (!source || anchor.kind() != DimExpr::Kind::kSymbol) {
                    Reject("slice dynamic prefix/window requires a direct input-axis symbol");
                }
                const int64_t capacity = data.dims[axis].Evaluate(BindingSet());
                if (count > capacity || symbol_uppers_.at(anchor.Symbols()[0]) > capacity - (prefix ? 0 : count)) {
                    Reject("slice prefix/window upper bound exceeds table capacity");
                }
                const auto encoded = EncodeElements({anchor}, info_.at(source).dims, "slice prefix/window");
                if (encoded.kinds[0] != kExprKindInputAxis) Reject("slice window anchor must be a direct axis");
                NodeInfo result;
                result.dims = data.dims;
                result.dims[axis] = prefix ? end : DimExpr::Const(count);
                rewritten_[node] = RebuildCall("slice", call,
                    relay::SliceAttrs::Create({}, {}, {}, {}, axis, encoded.axes[0],count),
                    {rewritten_.at(call->args[0].get()), rewritten_.at(source)});
                return result;
            }
            Array<Array<int64_t>> controls;
            for (const auto& control : operands) {
                Array<int64_t> values;
                for (size_t i = 0; i < control.size(); ++i) {
                    const auto value = control.Known(i);
                    if (!value) Reject("slice controls must be provably constant");
                    values.push_back(*value);
                }
                controls.push_back(std::move(values));
            }
            attrs = kxc::relay::SliceAttrs::Create(controls[0], controls[1], controls[2], controls[3]);
        }
        // InferType owns endpoint normalization and static sliced-axis checks.
        // Untouched symbolic axes retain their original DimExpr proof.
        Array<int64_t> shape;
        for (const auto& dim : data.dims) {
            shape.push_back(dim.kind() == DimExpr::Kind::kConst ? dim.Evaluate(BindingSet()) : -1);
        }
        const Type output = kxc::relay::SliceInferType(attrs, {TensorType(shape, "float32")});
        const auto* type = output.As<TensorTypeNode>();
        NodeInfo result;
        result.dims = data.dims;
        for (size_t axis = 0; axis < result.dims.size(); ++axis) {
            if (type->shape[axis] >= 0) result.dims[axis] = DimExpr::Const(type->shape[axis]);
        }
        rewritten_[node] = RebuildCall("slice", call, attrs, {rewritten_.at(call->args[0].get())});
        return result;
    }

    // 控制目标投影：opset-17 0-copy 解析；-1 与符号歧义拒绝；元素总数以
    // -1 is a Reshape instruction, never a negative DimExpr. Read the source
    // target before ordinary shape concatenation rejects negative dimensions.
    void ReadReshapeTarget(const Expr& expr, std::vector<std::optional<DimExpr>>* dims,
                           const Object** source) {
        if (const auto* call = expr.As<CallNode>(); call && OpName(call) == "concatenate") {
            const auto* attrs = call->attrs.As<relay::ConcatenateAttrsNode>();
            if (call->args.size() != 2 || !attrs || NormalizeAxis("reshape control concat",attrs->axis,1) != 0) {
                Reject("reshape control concatenation requires two vectors and axis 0");
            }
            ReadReshapeTarget(call->args[0],dims,source);
            ReadReshapeTarget(call->args[1],dims,source);
        } else {
            const auto control = ReadControl(expr,"reshape target");
            if (control.scalar) Reject("reshape target requires a vector");
            for (size_t i=0; i<control.size(); ++i) {
                const auto literal = control.Known(i);
                if (literal) {
                    if (*literal < -1) Reject("reshape target constants must be >= -1");
                    dims->push_back(*literal == -1 ? std::nullopt : std::optional<DimExpr>(DimExpr::Const(*literal)));
                } else {
                    if (*source && *source != control.proof.source) Reject("reshape target mixes independent shape sources");
                    *source = control.proof.source;
                    dims->push_back(control.Expression(i));
                }
            }
        }
        if (dims->size() > kMaxShapeExprElements) Reject("reshape target exceeds the vector length cap");
    }

    // Only an entirely static quotient is admitted. Symbol multiplicities
    // must match, and cancellation requires strictly positive symbol bounds.
    DimExpr InferFixedReshapeDim(const std::vector<DimExpr>& data, const std::vector<DimExpr>& target) {
        std::multiset<std::string> numerator_symbols, denominator_symbols;
        std::vector<DimExpr> numerator, denominator;
        const auto factors = [&](const std::vector<DimExpr>& dims, std::multiset<std::string>* symbols,
                                 std::vector<DimExpr>* constants) {
            for (const auto& dim : dims) {
                if (dim.kind() == DimExpr::Kind::kConst) constants->push_back(dim);
                else if (EvaluateAtBounds(dim,symbol_lowers_) > 0) {
                    symbols->insert(dim.CanonicalString());
                } else Reject("inferred reshape dimension requires positive direct symbols");
            }
        };
        factors(data,&numerator_symbols,&numerator);
        factors(target,&denominator_symbols,&denominator);
        if (numerator_symbols != denominator_symbols) Reject("inferred reshape dimension must be static after cancellation");
        const int64_t n=ProductOf(numerator).Evaluate(BindingSet()), d=ProductOf(denominator).Evaluate(BindingSet());
        if (d <= 0 || n <= 0 || n%d != 0) Reject("inferred reshape dimension must be a positive integral quotient");
        return DimExpr::FloorDiv(DimExpr::Const(n),d);
    }

    // DimExpr canonical 乘积等值证明。
    NodeInfo ResolveReshapeDynamic(const kxc::CallNode* call, const Object* node) {
        const NodeInfo& data = Resolve(call->args[0]);
        RequireNoCallerAttrs(call, "reshape_dynamic");
        if (data.kind == ValueKind::kShapeValue) {
            const auto control = ReadControl(call->args[1], "shape-vector reshape control");
            if (data.scalar_shape_value || control.scalar || control.size() != 1 || !control.Known(0)) {
                Reject("shape-vector reshape only admits a statically proved rank-one identity");
            }
            const int64_t target = *control.Known(0);
            if (target != -1 && target != 0 && target != static_cast<int64_t>(data.elements.size())) {
                Reject("shape-vector reshape changes the known element count");
            }
            return MakeControl(call, node, data.source, data.elements);
        }
        RequireData(data, "reshape_dynamic data");
        std::vector<std::optional<DimExpr>> control;
        const Object* control_source = nullptr;
        ReadReshapeTarget(call->args[1], &control, &control_source);
        std::vector<DimExpr> target;
        std::optional<size_t> inferred_axis;
        target.reserve(control.size());
        for (size_t i = 0; i < control.size(); ++i) {
            if (!control[i]) {
                if (inferred_axis) Reject("reshape target admits at most one inferred dimension");
                inferred_axis=i; target.push_back(DimExpr::Const(1)); continue;
            }
            const DimExpr& element = *control[i];
            if (element.kind() == DimExpr::Kind::kConst) {
                const int64_t value = element.Evaluate(BindingSet());
                if (value == 0) {
                    // opset 17（allowzero=0）：0 → 复制数据同位置维。
                    if (i >= data.dims.size()) {
                        Reject("reshape 0-dim copy index exceeds the data rank");
                    }
                    target.push_back(data.dims[i]);
                    continue;
                }
                target.push_back(element);
                continue;
            }
            int matched = -1;
            for (size_t axis = 0; axis < data.dims.size(); ++axis) {
                if (element == data.dims[axis]) {
                    if (matched >= 0) {
                        Reject("reshape target symbol matches multiple data axes");
                    }
                    matched = static_cast<int>(axis);
                }
            }
            if (matched < 0) {
                Reject("reshape control must stay rooted at the reshape data");
            }
            target.push_back(element);
        }
        if (target.empty()) {
            Reject("reshape_dynamic requires a rank >= 1 target");
        }
        if (inferred_axis) target[*inferred_axis]=InferFixedReshapeDim(data.dims,target);
        if (!(ProductOf(data.dims) == ProductOf(target))) {
            Reject("reshape element count cannot be proven equal");
        }
        const EncodedExpr encoded =
            EncodeElements(target, data.dims, "reshape_dynamic target");
        Expr resolved_control;
        if (inferred_axis) {
            resolved_control = kxc::Call(relay::Op::Get("shape_expr"),
                {rewritten_.at(call->args[0].get())},
                relay::ShapeExprAttrs::Create(encoded.kinds,encoded.values,encoded.axes));
            NodeInfo control_info;
            control_info.kind=ValueKind::kShapeValue; control_info.elements=target;
            control_info.source=call->args[0].get();
            rewritten_info_[resolved_control.get()]=control_info;
            value_expr_overrides_[resolved_control.get()]=encoded;
        } else {
            (void)ShapeArgument(call->args[1],"reshape_dynamic control");
            resolved_control=rewritten_.at(call->args[1].get());
        }
        rewritten_[node] = kxc::Call(
            kxc::relay::Op::Get("reshape_dynamic"),
            {rewritten_.at(call->args[0].get()),
             resolved_control},
            kxc::relay::ReshapeDynamicAttrs::Create(encoded.kinds, encoded.values,
                                                    encoded.axes));
        NodeInfo info;
        info.kind = ValueKind::kData;
        info.dims = std::move(target);
        return info;
    }

    NodeInfo ResolveExpand(const kxc::CallNode* call, const Object* node) {
        const NodeInfo& data = Resolve(call->args[0]);
        const NodeInfo control = ShapeArgument(call->args[1], "expand control");
        RequireData(data, "expand data");
        RequireShapeValue(control, "expand control");
        RequireNoCallerAttrs(call, "expand_dynamic");
        if (control.elements.size() != data.dims.size()) {
            Reject("expand target length must equal the data rank");
        }
        for (size_t axis = 0; axis < control.elements.size(); ++axis) {
            const DimExpr& data_dim = data.dims[axis];
            const DimExpr& target_dim = control.elements[axis];
            if (data_dim == target_dim || data_dim == DimExpr::Const(1)) {
                continue;
            }
            Reject("expand cannot broadcast axis " + std::to_string(axis) +
                   " under the restricted proof rules");
        }
        const EncodedExpr encoded =
            EncodeElements(control.elements, data.dims, "expand target");
        rewritten_[node] = kxc::Call(
            kxc::relay::Op::Get("expand_dynamic"),
            {rewritten_.at(call->args[0].get()),
             rewritten_.at(call->args[1].get())},
            kxc::relay::ExpandDynamicAttrs::Create(encoded.kinds, encoded.values,
                                            encoded.axes));
        NodeInfo info;
        info.kind = ValueKind::kData;
        info.dims = control.elements;
        return info;
    }

    NodeInfo ResolveSqueeze(const kxc::CallNode* call, const Object* node) {
        const NodeInfo& input = Resolve(call->args[0]);
        RequireData(input, "squeeze input");
        const auto* squeeze_attrs = call->attrs.As<kxc::relay::SqueezeAttrsNode>();
        if (!squeeze_attrs || squeeze_attrs->axes.empty()) {
            Reject("squeeze requires explicit axes");
        }
        const int64_t rank = static_cast<int64_t>(input.dims.size());
        if (rank < 1) {
            Reject("squeeze requires data rank >= 1");
        }
        std::vector<bool> removed(static_cast<size_t>(rank), false);
        for (int64_t raw_axis : squeeze_attrs->axes) {
            const int axis = NormalizeAxis("squeeze", raw_axis, rank);
            if (removed[static_cast<size_t>(axis)]) {
                Reject("squeeze duplicate axis");
            }
            removed[static_cast<size_t>(axis)] = true;
            const DimExpr& dim = input.dims[static_cast<size_t>(axis)];
            if (!(dim == DimExpr::Const(1))) {
                Reject("squeeze requires each removed axis to be provably 1");
            }
        }
        std::vector<DimExpr> dims;
        for (int64_t axis = 0; axis < rank; ++axis) {
            if (!removed[static_cast<size_t>(axis)]) {
                dims.push_back(input.dims[static_cast<size_t>(axis)]);
            }
        }
        if (dims.empty()) {
            Reject("squeeze must keep at least one axis");
        }
        rewritten_[node] = RebuildCall(
            "squeeze", call, call->attrs,
            {rewritten_.at(call->args[0].get())});
        NodeInfo info;
        info.kind = ValueKind::kData;
        info.dims = std::move(dims);
        return info;
    }

    NodeInfo ResolveUnsqueeze(const kxc::CallNode* call, const Object* node) {
        const NodeInfo& input = Resolve(call->args[0]);
        const auto* unsqueeze_attrs =
            call->attrs.As<kxc::relay::UnsqueezeAttrsNode>();
        if (!unsqueeze_attrs || unsqueeze_attrs->axes.empty()) {
            Reject("unsqueeze requires explicit axes");
        }
        if (input.kind == ValueKind::kShapeValue) {
            // The scalar never escapes as a mis-ranked tensor: it is consumed
            // here and the whole chain becomes the existing int64 shape_expr.
            if (!input.scalar_shape_value || unsqueeze_attrs->axes.size() != 1 ||
                NormalizeAxis("shape scalar unsqueeze", unsqueeze_attrs->axes[0], 1) != 0) {
                Reject("shape-value Unsqueeze only admits a scalar to a length-one vector");
            }
            return MakeControl(call, node, input.source, input.elements);
        }
        RequireData(input, "unsqueeze input");
        const int64_t input_rank = static_cast<int64_t>(input.dims.size());
        const int64_t new_rank =
            input_rank + static_cast<int64_t>(unsqueeze_attrs->axes.size());
        if (new_rank > 8) {
            Reject("unsqueeze output rank exceeds the declared subset");
        }
        std::vector<bool> placed(static_cast<size_t>(new_rank), false);
        for (int64_t raw_axis : unsqueeze_attrs->axes) {
            const int axis = NormalizeAxis("unsqueeze", raw_axis, new_rank);
            if (placed[static_cast<size_t>(axis)]) {
                Reject("unsqueeze duplicate axis");
            }
            placed[static_cast<size_t>(axis)] = true;
        }
        std::vector<DimExpr> dims(static_cast<size_t>(new_rank), DimExpr::Const(1));
        size_t next = 0;
        for (size_t axis = 0; axis < dims.size(); ++axis) {
            if (placed[axis]) continue;
            dims[axis] = input.dims[next++];
        }
        rewritten_[node] = RebuildCall(
            "unsqueeze", call, call->attrs,
            {rewritten_.at(call->args[0].get())});
        NodeInfo info;
        info.kind = ValueKind::kData;
        info.dims = std::move(dims);
        return info;
    }

    // 阶段二：post-order 走折叠后 DAG，每个调用只收集一次。value id
    // 留给 ValueGraph；tuple 仅保持遍历顺序，不占一个计算单元。
    void WalkRewritten(const Expr& expr, std::set<const Object*>* visited,
                       Resolution* result) {
        if (!expr.defined()) Reject("rewritten expression is undefined");
        const Object* node = expr.get();
        if (!visited->insert(node).second || expr.As<kxc::VarNode>()) return;
        if (const auto* tuple = expr.As<TupleNode>()) {
            for (const Expr& field : tuple->fields) {
                WalkRewritten(field, visited, result);
            }
            return;
        }
        if (const auto* get_item = expr.As<kxc::TupleGetItemNode>()) {
            WalkRewritten(get_item->tuple, visited, result);
            return;
        }
        if (const auto* conditional = expr.As<IfNode>()) {
            // Order matches control lowering: predicate, then, else.
            WalkRewritten(conditional->cond, visited, result);
            WalkRewritten(conditional->true_branch, visited, result);
            WalkRewritten(conditional->false_branch, visited, result);
            return;
        }
        if (const auto* loop = expr.As<WhileNode>()) {
            // Order matches control lowering: initial, condition, body.
            WalkRewritten(loop->initial_state, visited, result);
            WalkRewritten(loop->condition, visited, result);
            WalkRewritten(loop->body, visited, result);
            return;
        }
        if (const auto* let = expr.As<LetNode>()) {
            WalkRewritten(let->value, visited, result);
            WalkRewritten(let->body, visited, result);
            return;
        }
        if (expr.As<kxc::ConstantNode>()) return;
        const auto* call = expr.As<kxc::CallNode>();
        if (!call) {
            Reject("the rewritten restricted graph has an unsupported expression");
        }
        for (const Expr& argument : call->args) {
            WalkRewritten(argument, visited, result);
        }
        const std::string name = OpName(call);
        result->registry_operations.push_back(name);
        const auto override_found = value_expr_overrides_.find(node);
        if (override_found != value_expr_overrides_.end()) {
            result->unit_value_expressions.push_back(override_found->second);
        } else {
            result->unit_value_expressions.push_back(std::nullopt);
        }
        result->unit_output_dimensions.push_back(
            ValueTensorDims(rewritten_info_.at(node)));
    }

    // 单元产出的张量维度：数据单元取其符号维；形状值单元是
    // int64[元素数] 行向量，长度静态已知。
    static std::vector<DimExpr> ValueTensorDims(const NodeInfo& info) {
        if (info.kind == ValueKind::kShapeValue) {
            return {DimExpr::Const(
                static_cast<int64_t>(info.elements.size()))};
        }
        return info.dims;
    }
};

}  // namespace

Resolution ResolveShapeValues(
    const Function& snapshot,
    const std::vector<InputAxisSymbol>& parameter_symbols) {
    Resolver resolver(snapshot, parameter_symbols);
    return resolver.Run();
}

}  // namespace shape_resolution
}  // namespace kxc::api::experimental::restricted_symbolic_shape::v1
