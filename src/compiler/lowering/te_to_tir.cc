/*! \file src/compiler/lowering/te_to_tir.cc
 * \brief Private TE DAG to TIR primitive lowering.
 */

#include "../internal/te_to_tir.h"
#include "tir/internal/static_integer.h"
#include "te/internal/program_access.h"
#include "kxc/pass/context.h"
#include "kxc/runtime/kernel_abi.h"
#include "kxc/te/te.h"
#include "kxc/tir/expr.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"
#include "kxc/tir/visitor.h"
#include "support/canonical.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kxc {
namespace relay {

namespace {

tir::PrimExpr FlattenIndex(const Array<tir::PrimExpr>& indices,
                           const Array<tir::PrimExpr>& shape) {
    const tir::DataType index_dtype = tir::DataType::Int(64);
    if (shape.empty()) return tir::IntImm(0, index_dtype);
    if (indices.size() != shape.size()) {
        throw std::runtime_error("Index rank mismatch during flattening");
    }
    const auto widen = [&](const tir::PrimExpr& value) {
        return value.dtype() == index_dtype
                   ? value
                   : tir::PrimExpr(tir::Call(index_dtype, "cast", {value}));
    };
    tir::PrimExpr linear = widen(indices[0]);
    for (size_t index = 1; index < indices.size(); ++index) {
        linear = linear * widen(shape[index]) + widen(indices[index]);
    }
    return linear;
}

tir::PrimExpr MakeIdentityForReduce(te::ReduceType type,
                                    tir::DataType dtype) {
    if (type == te::ReduceType::kSum) {
        if (dtype.code == 2) {
            return dtype.bits == 64 ? tir::FloatImm(0.0, dtype)
                                    : tir::FloatImm(0.0f, dtype);
        }
        return tir::IntImm(0, dtype);
    }
    if (type == te::ReduceType::kMax) {
        if (dtype.code == 2) {
            return dtype.bits == 64
                       ? tir::FloatImm(
                             -std::numeric_limits<double>::infinity(), dtype)
                       : tir::FloatImm(
                             -std::numeric_limits<float>::infinity(), dtype);
        }
        return tir::IntImm(std::numeric_limits<int64_t>::min(), dtype);
    }
    if (type == te::ReduceType::kMin) {
        if (dtype.code == 2) {
            return dtype.bits == 64
                       ? tir::FloatImm(
                             std::numeric_limits<double>::infinity(), dtype)
                       : tir::FloatImm(
                             std::numeric_limits<float>::infinity(), dtype);
        }
        return tir::IntImm(std::numeric_limits<int64_t>::max(), dtype);
    }
    throw std::runtime_error("Unsupported reduce type");
}

// 递归收集 TE 表达式中的 ProducerLoad 依赖。
void FindProducerLoads(const tir::PrimExpr& expr, std::vector<te::Tensor>* deps) {
    if (!expr.defined()) return;
    if (auto* pl = expr.As<te::ProducerLoadNode>()) {
        deps->push_back(pl->tensor);
        for (const auto& idx : pl->indices) FindProducerLoads(idx, deps);
        return;
    }
    if (auto* red = expr.As<te::ReduceNode>()) {
        for (const auto& src : red->source) FindProducerLoads(src, deps);
        return;
    }
    if (auto* n = expr.As<tir::BinaryOpNode>()) {
        FindProducerLoads(n->a, deps);
        FindProducerLoads(n->b, deps);
        return;
    }
    if (auto* n = expr.As<tir::CallNode>()) {
        for (const auto& arg : n->args) FindProducerLoads(arg, deps);
        return;
    }
    if (auto* n = expr.As<tir::SelectNode>()) {
        FindProducerLoads(n->condition, deps);
        FindProducerLoads(n->true_value, deps);
        FindProducerLoads(n->false_value, deps);
        return;
    }
    if (auto* n = expr.As<tir::NotNode>()) {
        FindProducerLoads(n->value, deps);
        return;
    }
    if (auto* n = expr.As<tir::LoadNode>()) {
        FindProducerLoads(n->index, deps);
        if (n->predicate.defined()) FindProducerLoads(n->predicate, deps);
    }
}

// 对 TE 张量依赖图执行 DFS，生成生产者在前的拓扑顺序。
void CollectOpsDFS(const te::Tensor& t,
                   std::unordered_set<const Object*>* visited,
                   std::unordered_map<const Object*, std::vector<te::Tensor>>* op_outputs,
                   std::vector<te::Operation>* topo) {
    if (!t.defined() || !t->op.defined()) return;
    const Object* op_ptr = t->op.get();
    auto& outputs = (*op_outputs)[op_ptr];
    bool recorded = false;
    for (const auto& existing : outputs) {
        if (existing.get() == t.get()) {
            recorded = true;
            break;
        }
    }
    if (!recorded) outputs.push_back(t);
    if (visited->count(op_ptr)) return;

    if (auto* cop = t->op.As<te::ComputeOpNode>()) {
        for (const auto& body_expr : cop->body) {
            std::vector<te::Tensor> deps;
            FindProducerLoads(body_expr, &deps);
            for (const auto& dep : deps) {
                CollectOpsDFS(dep, visited, op_outputs, topo);
            }
        }
    }

    visited->insert(op_ptr);
    topo->push_back(t->op);
}

using tir::internal::EvaluateStaticInt64;

tir::PrimExpr CastIndex(const tir::PrimExpr& value,
                        tir::DataType dtype) {
    return value.dtype() == dtype
               ? value
               : tir::PrimExpr(tir::Call(dtype, "cast", {value}));
}

class VarSubstituter final : public TIRPass {
public:
    VarSubstituter(tir::Var variable, tir::PrimExpr replacement)
        : variable_(std::move(variable)), replacement_(std::move(replacement)) {}

protected:
    tir::PrimExpr VisitVar(const tir::VarNode* op,
                           const tir::PrimExpr& ref) override {
        (void)op;
        return ref.get() == variable_.get() ? replacement_ : ref;
    }

private:
    tir::Var variable_;
    tir::PrimExpr replacement_;
};

struct StageAxisPlan final {
    std::unordered_map<const Object*, tir::PrimExpr> root_values;
    Array<te::IterVar> data_leaves;
    Array<te::IterVar> reduction_leaves;
    tir::PrimExpr data_predicate;
    tir::PrimExpr full_predicate;
};

StageAxisPlan BuildStageAxisPlan(const te::Stage& stage,
                                 const te::ComputeOpNode* op) {
    if (!stage.defined() || stage->op.get() != op ||
        stage->root_iter_vars.size() !=
            op->axis.size() + op->reduce_axis.size()) {
        throw std::invalid_argument(
            "TE schedule stage does not match its ComputeOp axes");
    }

    StageAxisPlan plan;
    std::unordered_set<const Object*> all_axes;
    for (size_t index = 0; index < stage->root_iter_vars.size(); ++index) {
        const te::IterVar& root = stage->root_iter_vars[index];
        const bool reduction = index >= op->axis.size();
        const tir::Var expected = reduction
                                      ? op->reduce_axis[index - op->axis.size()]->var
                                      : op->axis[index];
        if (!root.defined() || root->var.get() != expected.get() ||
            root->is_reduction != reduction ||
            !all_axes.insert(root.get()).second) {
            throw std::invalid_argument(
                "TE schedule root-axis contract was mutated");
        }
        plan.root_values.emplace(root->var.get(), root->var);
    }

    Array<te::IterVar> relation_leaves;
    std::vector<te::IterVar> expected_all_axes;
    for (const te::IterVar& root : stage->root_iter_vars) {
        relation_leaves.push_back(root);
        expected_all_axes.push_back(root);
    }
    for (const te::SplitRelation& relation : stage->split_relations) {
        const auto parent = std::find(relation_leaves.begin(),
                                      relation_leaves.end(), relation.parent);
        int64_t factor = 0;
        int64_t parent_extent = 0;
        int64_t outer_min = 0;
        int64_t outer_extent = 0;
        int64_t inner_min = 0;
        int64_t inner_extent = 0;
        const bool valid_domains =
            relation.parent.defined() && relation.outer.defined() &&
            relation.inner.defined() &&
            EvaluateStaticInt64(relation.factor, &factor) && factor > 0 &&
            EvaluateStaticInt64(relation.parent->dom_extent,
                                &parent_extent) &&
            parent_extent >= 0 &&
            EvaluateStaticInt64(relation.outer->dom_min, &outer_min) &&
            EvaluateStaticInt64(relation.outer->dom_extent,
                                &outer_extent) &&
            EvaluateStaticInt64(relation.inner->dom_min, &inner_min) &&
            EvaluateStaticInt64(relation.inner->dom_extent,
                                &inner_extent) &&
            outer_min == 0 && inner_min == 0 && inner_extent == factor &&
            outer_extent ==
                (parent_extent == 0 ? 0 : 1 + (parent_extent - 1) / factor);
        if (parent == relation_leaves.end() || !relation.parent.defined() ||
            !relation.outer.defined() || !relation.inner.defined() ||
            !valid_domains ||
            relation.outer->var->dtype != relation.parent->var->dtype ||
            relation.inner->var->dtype != relation.parent->var->dtype ||
            relation.outer->is_reduction != relation.parent->is_reduction ||
            relation.inner->is_reduction != relation.parent->is_reduction ||
            !all_axes.insert(relation.outer.get()).second ||
            !all_axes.insert(relation.inner.get()).second) {
            throw std::invalid_argument(
                "TE schedule contains an invalid split relation");
        }
        const size_t position = static_cast<size_t>(
            std::distance(relation_leaves.begin(), parent));
        relation_leaves[position] = relation.outer;
        relation_leaves.insert(relation_leaves.begin() + position + 1,
                               relation.inner);
        expected_all_axes.push_back(relation.outer);
        expected_all_axes.push_back(relation.inner);

        const tir::DataType dtype = relation.parent->var->dtype;
        const tir::PrimExpr replacement =
            CastIndex(relation.parent->dom_min, dtype) +
            relation.outer->var * CastIndex(relation.factor, dtype) +
            relation.inner->var;
        for (auto& [root, value] : plan.root_values) {
            (void)root;
            value = VarSubstituter(relation.parent->var, replacement)
                        .Mutate(value);
        }
    }

    if (stage->all_iter_vars.size() != expected_all_axes.size()) {
        throw std::invalid_argument(
            "TE schedule all-axis list does not match split relations");
    }
    for (size_t index = 0; index < expected_all_axes.size(); ++index) {
        if (!stage->all_iter_vars[index].defined() ||
            stage->all_iter_vars[index].get() != expected_all_axes[index].get()) {
            throw std::invalid_argument(
                "TE schedule all-axis list was mutated");
        }
    }
    if (relation_leaves.size() != stage->leaf_iter_vars.size()) {
        throw std::invalid_argument(
            "TE schedule leaf count does not match split relations");
    }
    std::unordered_set<const Object*> expected_leaves;
    for (const te::IterVar& axis : relation_leaves) {
        expected_leaves.insert(axis.get());
    }
    std::unordered_set<const Object*> seen_leaves;
    bool saw_reduction = false;
    for (const te::IterVar& axis : stage->leaf_iter_vars) {
        if (!axis.defined() || expected_leaves.count(axis.get()) == 0 ||
            !seen_leaves.insert(axis.get()).second) {
            throw std::invalid_argument(
                "TE schedule leaf order is not a split-leaf permutation");
        }
        if (axis->is_reduction) {
            saw_reduction = true;
            plan.reduction_leaves.push_back(axis);
        } else {
            if (saw_reduction) {
                throw std::invalid_argument(
                    "TE schedule requires data leaves before reduction leaves");
            }
            plan.data_leaves.push_back(axis);
        }
    }

    for (const te::IterVar& root : stage->root_iter_vars) {
        const tir::PrimExpr value = plan.root_values.at(root->var.get());
        if (value.get() == root->var.get()) continue;
        const tir::DataType dtype = root->var->dtype;
        const tir::PrimExpr predicate =
            value < (CastIndex(root->dom_min, dtype) +
                     CastIndex(root->dom_extent, dtype));
        plan.full_predicate = plan.full_predicate.defined()
                                  ? plan.full_predicate && predicate
                                  : predicate;
        if (!root->is_reduction) {
            plan.data_predicate = plan.data_predicate.defined()
                                      ? plan.data_predicate && predicate
                                      : predicate;
        }
    }
    return plan;
}

tir::ForType LowerForType(const te::IterVar& axis) {
    switch (axis->iter_type) {
        case te::IterVarType::kDataPar:
            if (axis->is_reduction) break;
            return tir::ForType::Serial;
        case te::IterVarType::kCommReduce:
            if (!axis->is_reduction) break;
            return tir::ForType::Serial;
        case te::IterVarType::kVectorized:
            if (!axis->is_reduction) return tir::ForType::Vectorized;
            break;
        case te::IterVarType::kParallel:
            if (!axis->is_reduction) return tir::ForType::Parallel;
            break;
        case te::IterVarType::kUnrolled:
            return tir::ForType::Unrolled;
    }
    throw std::invalid_argument(
        "TE schedule axis has an incompatible execution annotation");
}

tir::Stmt Guarded(const tir::PrimExpr& predicate, tir::Stmt body) {
    return predicate.defined()
               ? tir::Stmt(tir::IfThenElse(predicate, std::move(body)))
               : body;
}

tir::Stmt WrapLoops(const Array<te::IterVar>& axes, tir::Stmt body) {
    for (size_t index = axes.size(); index > 0; --index) {
        const te::IterVar& axis = axes[index - 1];
        body = tir::For(axis->var, axis->dom_min, axis->dom_extent,
                        LowerForType(axis), body);
    }
    return body;
}

// 把 TE 表达式中的轴和张量访问改写为 scheduled TIR 表达式。
class ExprLowerer {
public:
    ExprLowerer(
        const std::unordered_map<const Object*, tir::Var>& buffer_var_by_tensor,
        const std::unordered_map<const Object*, tir::PrimExpr>& axis_values)
        : buffer_var_by_tensor_(buffer_var_by_tensor),
          axis_values_(axis_values) {}

    tir::PrimExpr Lower(const tir::PrimExpr& expr) const {
        if (!expr.defined()) return expr;

        if (expr.As<tir::VarNode>()) {
            const auto axis = axis_values_.find(expr.get());
            return axis == axis_values_.end() ? expr : axis->second;
        }
        if (auto* n = expr.As<te::ProducerLoadNode>()) {
            auto it = buffer_var_by_tensor_.find(n->tensor.get());
            if (it == buffer_var_by_tensor_.end()) {
                throw std::runtime_error("Missing buffer var for tensor in ProducerLoad");
            }
            Array<tir::PrimExpr> lowered_indices;
            for (const auto& idx : n->indices) lowered_indices.push_back(Lower(idx));
            tir::PrimExpr linear = FlattenIndex(lowered_indices, n->tensor->shape);
            return tir::Load(it->second, linear);
        }

        if (auto* n = expr.As<tir::AddNode>()) return tir::Add(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::SubNode>()) return tir::Sub(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::MulNode>()) return tir::Mul(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::DivNode>()) return tir::Div(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::ModNode>()) return tir::Mod(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::MinNode>()) return tir::Min(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::MaxNode>()) return tir::Max(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::EQNode>()) return tir::EQ(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::LTNode>()) return tir::LT(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::AndNode>()) return tir::And(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::OrNode>()) return tir::Or(Lower(n->a), Lower(n->b));

        if (auto* n = expr.As<tir::NotNode>()) return tir::Not(Lower(n->value));
        if (auto* n = expr.As<tir::SelectNode>()) return tir::Select(Lower(n->condition), Lower(n->true_value), Lower(n->false_value));

        if (auto* n = expr.As<tir::CallNode>()) {
            Array<tir::PrimExpr> args;
            for (const auto& arg : n->args) args.push_back(Lower(arg));
            return tir::Call(n->dtype, n->name, args);
        }

        if (auto* n = expr.As<tir::LoadNode>()) {
            tir::PrimExpr pred = n->predicate.defined() ? Lower(n->predicate) : tir::PrimExpr();
            return tir::Load(n->buffer_var, Lower(n->index), pred);
        }

        if (expr.As<te::ReduceNode>()) {
            throw std::runtime_error("Reduce must be lowered at statement level");
        }
        return expr;
    }

private:
    const std::unordered_map<const Object*, tir::Var>& buffer_var_by_tensor_;
    const std::unordered_map<const Object*, tir::PrimExpr>& axis_values_;
};

// 将 TE ComputeOp 的指定 value_index 按 Stage leaf 轴降为 TIR。
tir::Stmt LowerComputeStmt(
    const te::Tensor& out_tensor, const te::Stage& stage,
    const std::unordered_map<const Object*, tir::Var>& buffer_var_by_tensor) {
    const auto* op = out_tensor->op.As<te::ComputeOpNode>();
    if (!op) {
        throw std::runtime_error("LowerComputeStmt expects ComputeOpNode");
    }
    if (op->body.empty()) {
        throw std::runtime_error("ComputeOp body is empty");
    }
    if (out_tensor->value_index < 0 ||
        static_cast<size_t>(out_tensor->value_index) >= op->body.size()) {
        throw std::runtime_error(
            "LowerComputeStmt tensor value_index is outside ComputeOp body");
    }

    const auto out_it = buffer_var_by_tensor.find(out_tensor.get());
    if (out_it == buffer_var_by_tensor.end()) {
        throw std::runtime_error("Missing output buffer var for compute tensor");
    }
    const tir::Var out_var = out_it->second;
    const StageAxisPlan axes = BuildStageAxisPlan(stage, op);
    const ExprLowerer expr_lowerer(buffer_var_by_tensor, axes.root_values);

    Array<tir::PrimExpr> data_indices;
    for (const auto& axis : op->axis) {
        data_indices.push_back(expr_lowerer.Lower(axis));
    }
    const tir::PrimExpr out_index = FlattenIndex(data_indices, op->shape);

    const tir::PrimExpr body_expr =
        op->body[static_cast<size_t>(out_tensor->value_index)];
    if (const auto* reduction = body_expr.As<te::ReduceNode>()) {
        if (reduction->source.size() != 1) {
            throw std::runtime_error("Only single-source reduce is supported");
        }
        const tir::PrimExpr init_value =
            MakeIdentityForReduce(reduction->reduce_type, out_tensor->dtype);
        tir::Stmt init_store = Guarded(
            axes.data_predicate, tir::Store(out_var, init_value, out_index));

        const tir::PrimExpr source = expr_lowerer.Lower(reduction->source[0]);
        const tir::PrimExpr old = tir::Load(out_var, out_index);
        tir::PrimExpr update_value;
        if (reduction->reduce_type == te::ReduceType::kSum) {
            update_value = old + source;
        } else if (reduction->reduce_type == te::ReduceType::kMax) {
            update_value = tir::Max(old, source);
        } else if (reduction->reduce_type == te::ReduceType::kMin) {
            update_value = tir::Min(old, source);
        } else {
            throw std::runtime_error("Unsupported reduce type in update");
        }
        tir::Stmt update_store = Guarded(
            axes.full_predicate,
            tir::Store(out_var, update_value, out_index));
        update_store = WrapLoops(axes.reduction_leaves, update_store);
        return WrapLoops(
            axes.data_leaves,
            tir::SeqStmt({std::move(init_store), std::move(update_store)}));
    }

    tir::Stmt store = Guarded(
        axes.data_predicate,
        tir::Store(out_var, expr_lowerer.Lower(body_expr), out_index));
    return WrapLoops(axes.data_leaves, store);
}


bool IsRuntimeExtentBufferVar(const tir::Var& buffer) {
    return buffer.defined() && buffer.As<tir::VarNode>() &&
           buffer->dtype == tir::DataType::UInt(64);
}

void ValidateRuntimeExtentBuffers(const Array<tir::Var>& buffers) {
    std::unordered_set<const Object*> seen;
    for (const tir::Var& buffer : buffers) {
        if (!IsRuntimeExtentBufferVar(buffer) || buffer->name_hint.empty() ||
            !seen.insert(buffer.get()).second) {
            throw std::invalid_argument(
                "Runtime extent buffers must be unique named uint64 Vars");
        }
    }
}

bool MatchRuntimeExtentLoadImpl(const tir::PrimExpr& expression,
                                const Array<tir::Var>& buffers,
                                size_t* buffer_index) {
    const auto* cast = expression.As<tir::CallNode>();
    if (!cast || cast->name != "cast" || cast->args.size() != 1 ||
        cast->dtype != tir::DataType::Int(64)) {
        return false;
    }
    const auto* load = cast->args[0].As<tir::LoadNode>();
    int64_t index = -1;
    if (!load || load->predicate.defined() ||
        load->dtype != tir::DataType::UInt(64) ||
        !EvaluateStaticInt64(load->index, &index) || index != 0) {
        return false;
    }
    for (size_t slot = 0; slot < buffers.size(); ++slot) {
        if (load->buffer_var.get() == buffers[slot].get()) {
            if (buffer_index) *buffer_index = slot;
            return true;
        }
    }
    return false;
}

bool MatchRuntimeExtentOffsetImpl(const tir::PrimExpr& expression,
                                  const Array<tir::Var>& buffers,
                                  size_t* buffer_index, int64_t* offset = nullptr) {
    if (MatchRuntimeExtentLoadImpl(expression, buffers, buffer_index)) {
        if (offset) *offset = 0;
        return true;
    }
    const auto* add = expression.As<tir::AddNode>();
    if (!add || add->dtype != tir::DataType::Int(64)) return false;
    for (bool reverse : {false, true}) {
        const auto* constant = (reverse ? add->a : add->b).As<tir::IntImmNode>();
        if (constant && constant->dtype == tir::DataType::Int(64) && constant->value >= 0 &&
            MatchRuntimeExtentLoadImpl(reverse ? add->b : add->a, buffers, buffer_index)) {
            if (offset) *offset = constant->value;
            return true;
        }
    }
    return false;
}

// 递归收集 TE compute 体中出现的 runtime extent buffer，供 schedule 合同
// 区分“驱动循环轴的 extent”与“作为存储值被消费的 extent”。
void CollectRuntimeExtentBufferUses(const tir::PrimExpr& expr,
                                    const Array<tir::Var>& buffers,
                                    std::unordered_set<const Object*>* used) {
    if (!expr.defined()) return;
    if (expr.As<tir::VarNode>() || expr.As<tir::IntImmNode>()) return;
    if (auto* n = expr.As<tir::LoadNode>()) {
        for (const tir::Var& buffer : buffers) {
            if (buffer.get() == n->buffer_var.get()) {
                used->insert(buffer.get());
            }
        }
        CollectRuntimeExtentBufferUses(n->index, buffers, used);
        if (n->predicate.defined()) {
            CollectRuntimeExtentBufferUses(n->predicate, buffers, used);
        }
        return;
    }
    if (auto* n = expr.As<tir::BinaryOpNode>()) {
        CollectRuntimeExtentBufferUses(n->a, buffers, used);
        CollectRuntimeExtentBufferUses(n->b, buffers, used);
        return;
    }
    if (auto* n = expr.As<tir::CallNode>()) {
        for (const auto& arg : n->args) {
            CollectRuntimeExtentBufferUses(arg, buffers, used);
        }
        return;
    }
    if (auto* n = expr.As<tir::SelectNode>()) {
        CollectRuntimeExtentBufferUses(n->condition, buffers, used);
        CollectRuntimeExtentBufferUses(n->true_value, buffers, used);
        CollectRuntimeExtentBufferUses(n->false_value, buffers, used);
        return;
    }
    if (auto* n = expr.As<tir::NotNode>()) {
        CollectRuntimeExtentBufferUses(n->value, buffers, used);
        return;
    }
}

void ValidateLoweringTensor(
    const Array<tir::PrimExpr>& shape, tir::DataType dtype,
    const std::string& context, const Array<tir::Var>& runtime_buffers,
    const std::vector<int64_t>& runtime_uppers) {
    bool all_static = true;
    for (const tir::PrimExpr& expression : shape) {
        int64_t ignored = 0;
        all_static = all_static && EvaluateStaticInt64(expression, &ignored);
    }
    if (all_static) {
        internal::ValidateStaticLoweringTensor(shape, dtype, context);
        return;
    }
    if (runtime_buffers.empty()) {
        throw std::invalid_argument(
            context + " requires static extents outside bounded dynamic lowering");
    }
    if (dtype.bits == 0 || dtype.lanes == 0) {
        throw std::invalid_argument(context +
                                    " has an invalid zero-width tensor dtype");
    }
    const size_t scalar_bytes =
        (static_cast<size_t>(dtype.bits) + 7U) / 8U;
    if (scalar_bytes > std::numeric_limits<size_t>::max() / dtype.lanes) {
        throw std::overflow_error(context +
                                  " dtype byte width overflows size_t");
    }
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        int64_t extent = 0;
        if (EvaluateStaticInt64(shape[axis], &extent)) {
            if (extent < 0 || extent > std::numeric_limits<int32_t>::max()) {
                throw std::invalid_argument(
                    context + " has an invalid static extent at axis " +
                    std::to_string(axis));
            }
            continue;
        }
        size_t slot = 0;
        int64_t offset = 0;
        if (!MatchRuntimeExtentOffsetImpl(shape[axis], runtime_buffers, &slot, &offset)) {
            throw std::invalid_argument(
                context + " axis " + std::to_string(axis) +
                " must be a generated runtime extent load with a nonnegative constant offset");
        }
        if (offset != 0 && (runtime_uppers.size() != runtime_buffers.size() ||
            offset > std::numeric_limits<int32_t>::max() - runtime_uppers.at(slot))) {
            throw std::invalid_argument(context + " derived extent exceeds the proved int32 loop domain");
        }
    }
}

Array<tir::PrimExpr> CanonicalBufferShape(
    const Array<tir::PrimExpr>& shape,
    const Array<tir::Var>& runtime_buffers) {
    Array<tir::PrimExpr> canonical;
    for (const tir::PrimExpr& expression : shape) {
        int64_t extent = 0;
        if (EvaluateStaticInt64(expression, &extent)) {
            if (extent < 0) {
                throw std::invalid_argument(
                    "TE-to-TIR Buffer shape contains a negative extent");
            }
            canonical.push_back(
                tir::IntImm(extent, tir::DataType::Int(64)));
        } else if (MatchRuntimeExtentOffsetImpl(expression, runtime_buffers,
                                              nullptr)) {
            canonical.push_back(expression);
        } else {
            throw std::invalid_argument(
                "TE-to-TIR Buffer shape is neither static nor a runtime extent load");
        }
    }
    return canonical;
}

// 为公开输出生成唯一且可读的 TIR 参数名。
std::string MakeOutputVarName(const te::Tensor& tensor,
                              size_t output_index,
                              std::unordered_set<std::string>* used_names) {
    std::string base = tensor->name.empty() ? "output" : tensor->name;
    std::string name = base + "_out";
    if (output_index != 0) {
        name += "_" + std::to_string(output_index);
    }
    while (used_names && used_names->count(name)) {
        name += "_";
    }
    if (used_names) {
        used_names->insert(name);
    }
    return name;
}

}  // namespace

namespace internal {

bool EvaluateStaticLoweringInt64(const tir::PrimExpr& expression,
                                 int64_t* result) {
    return EvaluateStaticInt64(expression, result);
}

void ValidateStaticLoweringTensor(const Array<tir::PrimExpr>& shape,
                                  tir::DataType dtype,
                                  const std::string& context) {
    if (dtype.bits == 0 || dtype.lanes == 0) {
        throw std::invalid_argument(context +
                                    " has an invalid zero-width tensor dtype");
    }
    const size_t scalar_bytes =
        (static_cast<size_t>(dtype.bits) + 7U) / 8U;
    if (scalar_bytes > std::numeric_limits<size_t>::max() / dtype.lanes) {
        throw std::overflow_error(context + " dtype byte width overflows size_t");
    }
    const size_t element_bytes = scalar_bytes * dtype.lanes;

    bool has_zero_extent = false;
    std::vector<int64_t> extents;
    extents.reserve(shape.size());
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        int64_t extent = 0;
        if (!EvaluateStaticInt64(shape[axis], &extent)) {
            throw std::invalid_argument(
                context + " requires an int64-representable static integer extent at axis " +
                std::to_string(axis));
        }
        if (extent < 0) {
            throw std::invalid_argument(
                context + " requires non-negative extent at axis " +
                std::to_string(axis));
        }
        if (extent > std::numeric_limits<int32_t>::max()) {
            throw std::overflow_error(
                context + " extent exceeds the int32 iteration domain at axis " +
                std::to_string(axis));
        }
        has_zero_extent = has_zero_extent || extent == 0;
        extents.push_back(extent);
    }

    int64_t elements = has_zero_extent ? 0 : 1;
    if (!has_zero_extent) {
        for (int64_t extent : extents) {
            if (elements > std::numeric_limits<int64_t>::max() / extent) {
                throw std::overflow_error(
                    context + " row-major element count/flatten index overflows int64");
            }
            elements *= extent;
        }
    }
    if (elements != 0 &&
        elements > std::numeric_limits<int64_t>::max() /
                       static_cast<int64_t>(element_bytes)) {
        throw std::overflow_error(context + " tensor byte count overflows int64");
    }
    if (static_cast<uint64_t>(elements) >
        std::numeric_limits<size_t>::max() / element_bytes) {
        throw std::overflow_error(context + " tensor byte count overflows size_t");
    }
}

namespace {

void ValidateScheduleTarget(const Target& target) {
    const auto* node = target.As<TargetNode>();
    if (!node || node->device_id < 0 ||
        !((node->kind == "llvm" && node->device_type == kCPU) ||
          (node->kind == "cuda" && node->device_type == kCUDA))) {
        throw std::invalid_argument(
            "TE scheduling requires a complete supported Target");
    }
}

int64_t StaticAxisExtent(const te::IterVar& axis,
                         const std::string& context) {
    int64_t extent = 0;
    if (!axis.defined() ||
        !EvaluateStaticInt64(axis->dom_extent, &extent) || extent < 0) {
        throw std::invalid_argument(context +
                                    " requires a static non-negative extent");
    }
    return extent;
}

const char* OperationKind(const te::Operation& operation) {
    if (operation.As<te::PlaceholderOpNode>()) return "placeholder";
    if (operation.As<te::ComputeOpNode>()) return "compute";
    throw std::invalid_argument(
        "TE schedule contains an unsupported operation kind");
}

void ValidateSerialTESchedule(const te::Schedule& schedule,
                              const char* context) {
    for (const te::Stage& stage : schedule->stages) {
        if (!stage.defined() || !stage->op.defined()) {
            throw std::invalid_argument(
                std::string(context) + " contains an undefined stage");
        }
        if (!stage->split_relations.empty() ||
            stage->leaf_iter_vars.size() != stage->root_iter_vars.size()) {
            throw std::invalid_argument(
                std::string(context) + " must stay unsplit");
        }
        for (size_t index = 0; index < stage->leaf_iter_vars.size(); ++index) {
            const te::IterVar& leaf = stage->leaf_iter_vars[index];
            const te::IterVar& root = stage->root_iter_vars[index];
            const te::IterVarType expected =
                leaf->is_reduction ? te::IterVarType::kCommReduce
                                   : te::IterVarType::kDataPar;
            if (leaf.get() != root.get() || leaf->iter_type != expected) {
                throw std::invalid_argument(
                    std::string(context) + " must stay serial");
            }
        }
    }
}

void ValidateCudaTESchedule(const te::Schedule& schedule) {
    try {
        ValidateSerialTESchedule(schedule, "CUDA TE schedule");
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(
            std::string(error.what()) +
            "; BindCudaThreads is the sole launch mapping authority");
    }
}

}  // namespace

te::Schedule BuildDefaultTESchedule(const Array<te::Tensor>& outputs,
                                    const Target& target) {
    ValidateScheduleTarget(target);
    if (outputs.empty()) {
        throw std::invalid_argument(
            "BuildDefaultTESchedule requires output tensors");
    }
    Array<te::Operation> output_operations;
    std::unordered_set<const Object*> seen;
    for (const te::Tensor& output : outputs) {
        if (!output.defined() || !output->op.defined()) {
            throw std::invalid_argument(
                "BuildDefaultTESchedule outputs must be defined");
        }
        if (seen.insert(output->op.get()).second) {
            output_operations.push_back(output->op);
        }
    }
    te::Schedule schedule = te::create_schedule(output_operations);
    schedule.operator->()->policy = kDefaultTESchedulePolicy;

    if (target->kind == "cuda") {
        return schedule;
    }

    for (const te::Stage& stage_ref : schedule->stages) {
        te::Stage stage(stage_ref);
        const auto* compute = stage->op.As<te::ComputeOpNode>();
        if (!compute || stage->root_iter_vars.empty()) continue;

        Array<te::IterVar> data_roots;
        Array<te::IterVar> reduction_roots;
        for (const te::IterVar& root : stage->root_iter_vars) {
            (root->is_reduction ? reduction_roots : data_roots).push_back(root);
        }
        if (data_roots.empty()) continue;

        if (!reduction_roots.empty()) {
            const int64_t data_extent =
                StaticAxisExtent(data_roots[0], "CPU default schedule");
            if (data_extent > 1) stage.parallel(data_roots[0]);
            const te::IterVar reduction =
                reduction_roots[reduction_roots.size() - 1];
            const int64_t reduction_extent =
                StaticAxisExtent(reduction, "CPU default reduction schedule");
            if (reduction_extent > 0 && reduction_extent <= 8) {
                stage.unroll(reduction);
            }
            continue;
        }

        const te::IterVar innermost = data_roots[data_roots.size() - 1];
        const int64_t inner_extent =
            StaticAxisExtent(innermost, "CPU default elementwise schedule");
        if (inner_extent >= 4 && inner_extent % 4 == 0) {
            te::IterVar outer;
            te::IterVar inner;
            stage.split(innermost, tir::IntImm(4), &outer, &inner);
            stage.vectorize(inner);
            const int64_t outer_extent =
                StaticAxisExtent(outer, "CPU default split schedule");
            const te::IterVar first = stage->leaf_iter_vars[0];
            if (outer_extent > 1 || first.get() != outer.get()) {
                stage.parallel(first);
            }
        } else if (inner_extent > 0 && inner_extent <= 8) {
            stage.unroll(innermost);
        } else if (inner_extent > 1) {
            stage.parallel(stage->leaf_iter_vars[0]);
        }
    }
    return schedule;
}

te::Schedule BuildStatefulKvTESchedule(const Array<te::Tensor>& outputs,
                                       const Target& target) {
    ValidateScheduleTarget(target);
    if (target->kind != "llvm" || target->device_type != kCPU) {
        throw std::invalid_argument(
            "Stateful KV TE scheduling is LLVM/CPU-only");
    }
    if (outputs.empty()) {
        throw std::invalid_argument(
            "BuildStatefulKvTESchedule requires output tensors");
    }
    Array<te::Operation> output_operations;
    std::unordered_set<const Object*> seen;
    for (const te::Tensor& output : outputs) {
        if (!output.defined() || !output->op.defined()) {
            throw std::invalid_argument(
                "BuildStatefulKvTESchedule outputs must be defined");
        }
        if (seen.insert(output->op.get()).second) {
            output_operations.push_back(output->op);
        }
    }
    te::Schedule schedule = te::create_schedule(output_operations);
    schedule.operator->()->policy = kStatefulKvTESchedulePolicy;
    ValidateSerialTESchedule(schedule, "Stateful KV TE schedule");
    return schedule;
}

te::Schedule BuildBoundedDynamicTESchedule(
    const Array<te::Tensor>& outputs, const Target& target) {
    ValidateScheduleTarget(target);
    if (outputs.empty()) {
        throw std::invalid_argument(
            "BuildBoundedDynamicTESchedule requires output tensors");
    }
    bool has_symbolic_extent = false;
    Array<te::Operation> output_operations;
    std::unordered_set<const Object*> seen;
    for (const te::Tensor& output : outputs) {
        if (!output.defined() || !output->op.defined()) {
            throw std::invalid_argument(
                "BuildBoundedDynamicTESchedule outputs must be defined");
        }
        for (const tir::PrimExpr& extent : output->shape) {
            int64_t ignored = 0;
            has_symbolic_extent =
                has_symbolic_extent || !EvaluateStaticInt64(extent, &ignored);
        }
        if (seen.insert(output->op.get()).second) {
            output_operations.push_back(output->op);
        }
    }
    if (!has_symbolic_extent) {
        throw std::invalid_argument(
            "Bounded dynamic TE scheduling requires a symbolic output extent");
    }
    te::Schedule schedule = te::create_schedule(output_operations);
    schedule.operator->()->policy = kBoundedDynamicTESchedulePolicy;
    ValidateSerialTESchedule(schedule, "Bounded dynamic TE schedule");
    return schedule;
}

te::Schedule BuildBoundedDynamicTESchedule(
    const Array<te::Tensor>& outputs, const Target& target,
    const Array<tir::Var>& runtime_extent_buffers) {
    ValidateRuntimeExtentBuffers(runtime_extent_buffers);
    if (runtime_extent_buffers.empty()) {
        return BuildBoundedDynamicTESchedule(outputs, target);
    }
    ValidateScheduleTarget(target);
    if (outputs.empty()) {
        throw std::invalid_argument(
            "BuildBoundedDynamicTESchedule requires output tensors");
    }
    Array<te::Operation> output_operations;
    std::unordered_set<const Object*> seen;
    for (const te::Tensor& output : outputs) {
        if (!output.defined() || !output->op.defined()) {
            throw std::invalid_argument(
                "BuildBoundedDynamicTESchedule outputs must be defined");
        }
        if (seen.insert(output->op.get()).second) {
            output_operations.push_back(output->op);
        }
    }
    // M3 形状值单元（如 shape_of）的输出形状完全静态，但 kernel 体要消费
    // runtime extent 作为存储值；此类单元仍属 bounded serial 策略。
    te::Schedule schedule = te::create_schedule(output_operations);
    schedule.operator->()->policy = kBoundedDynamicTESchedulePolicy;
    ValidateSerialTESchedule(schedule, "Bounded dynamic TE schedule");
    return schedule;
}

tir::PrimExpr LoadRuntimeExtent(const tir::Var& buffer) {
    if (!IsRuntimeExtentBufferVar(buffer) || buffer->name_hint.empty()) {
        throw std::invalid_argument(
            "LoadRuntimeExtent requires a named uint64 buffer Var");
    }
    return tir::Call(
        tir::DataType::Int(64), "cast",
        {tir::Load(buffer, tir::IntImm(0, tir::DataType::Int(64)))});
}

bool MatchRuntimeExtentLoad(const tir::PrimExpr& expression,
                            const Array<tir::Var>& buffers,
                            size_t* buffer_index) {
    ValidateRuntimeExtentBuffers(buffers);
    return MatchRuntimeExtentLoadImpl(expression, buffers, buffer_index);
}

bool MatchRuntimeExtentOffset(const tir::PrimExpr& expression,
                              const Array<tir::Var>& buffers,
                              size_t* buffer_index, int64_t* offset) {
    ValidateRuntimeExtentBuffers(buffers);
    return MatchRuntimeExtentOffsetImpl(expression, buffers, buffer_index, offset);
}

std::string CanonicalTEScheduleContract(
    const te::Schedule& schedule, const Target& target,
    const Array<tir::Var>& runtime_extent_buffers,
    const std::vector<size_t>& body_only_runtime_extents,
    const std::unordered_set<const Object*>* body_consumed_extents,
    const std::unordered_set<const Object*>* boundary_shape_extents,
    const std::vector<int64_t>& runtime_extent_upper_bounds) {
    ValidateScheduleTarget(target);
    ValidateRuntimeExtentBuffers(runtime_extent_buffers);
    if (!schedule.defined() || schedule->policy.empty() ||
        schedule->outputs.empty() || schedule->stages.empty() ||
        schedule->op_map.size() != schedule->stages.size()) {
        throw std::invalid_argument(
            "CanonicalTEScheduleContract requires a complete Schedule");
    }
    const bool stateful_policy = schedule->policy == kStatefulKvTESchedulePolicy;
    const bool bounded_dynamic =
        !runtime_extent_buffers.empty() && !stateful_policy;
    if (!runtime_extent_buffers.empty()) {
        if (stateful_policy && (target->kind != "llvm" || target->device_type != kCPU)) {
            throw std::invalid_argument(
                "Stateful runtime extent schedules require the LLVM/CPU policy");
        }
        if (stateful_policy) {
            ValidateSerialTESchedule(schedule, "Stateful KV TE schedule");
        } else {
            if (!body_only_runtime_extents.empty()) {
                throw std::invalid_argument(
                    "Bounded dynamic TE schedule extents must drive loop axes");
            }
            if (schedule->policy != kBoundedDynamicTESchedulePolicy) {
                throw std::invalid_argument(
                    "Runtime extent schedules require the bounded serial policy");
            }
            ValidateSerialTESchedule(schedule, "Bounded dynamic TE schedule");
        }
    } else {
        if (schedule->policy == kBoundedDynamicTESchedulePolicy ||
            stateful_policy) {
            throw std::invalid_argument(
                "Stateful or bounded dynamic TE schedule requires runtime extent buffers");
        }
        if (target->kind == "cuda") ValidateCudaTESchedule(schedule);
    }
    std::unordered_set<size_t> body_only_extents;
    for (size_t index : body_only_runtime_extents) {
        if (index >= runtime_extent_buffers.size() ||
            !body_only_extents.insert(index).second) {
            throw std::invalid_argument(
                "Body-only runtime extent indices must be unique and in range");
        }
    }

    support::CanonicalBytesEncoder encoder(
        runtime_extent_buffers.empty()
            ? "kxc.te.schedule.v1"
            : (stateful_policy ? "kxc.te.schedule.v3" : "kxc.te.schedule.v2"));
    encoder.Field("policy", schedule->policy);
    encoder.Field("target_kind", target->kind);
    encoder.IntegerField("target_device_type",
                         static_cast<int>(target->device_type));
    std::vector<bool> used_runtime_extents(runtime_extent_buffers.size(),
                                           false);
    if (!runtime_extent_buffers.empty()) {
        encoder.IntegerField("runtime_extent_count",
                             runtime_extent_buffers.size());
        if (bounded_dynamic && target->kind == "cuda") {
            if (runtime_extent_upper_bounds.size() != runtime_extent_buffers.size()) {
                throw std::invalid_argument("CUDA schedule identity requires one upper bound per extent");
            }
            for (int64_t upper : runtime_extent_upper_bounds) {
                if (upper < 0 || upper > std::numeric_limits<int32_t>::max()) {
                    throw std::invalid_argument("CUDA schedule upper bound exceeds the int32 extent ABI");
                }
                encoder.IntegerField("cuda_runtime_extent_upper_bound", upper);
            }
        }
        for (size_t index : body_only_runtime_extents) {
            encoder.IntegerField("body_only_extent", index);
        }
    }
    std::unordered_map<const Object*, size_t> stage_indices;
    for (size_t index = 0; index < schedule->stages.size(); ++index) {
        const te::Stage& stage = schedule->stages[index];
        if (!stage.defined() || !stage->op.defined() ||
            !stage_indices.emplace(stage->op.get(), index).second) {
            throw std::invalid_argument(
                "TE schedule stages must have unique defined operations");
        }
    }
    for (const te::Operation& output : schedule->outputs) {
        const auto stage = output.defined()
                               ? stage_indices.find(output.get())
                               : stage_indices.end();
        if (stage == stage_indices.end()) {
            throw std::invalid_argument(
                "TE schedule output operation has no stage");
        }
        encoder.IntegerField("output_stage", stage->second);
    }

    for (size_t stage_index = 0; stage_index < schedule->stages.size();
         ++stage_index) {
        const te::Stage& stage = schedule->stages[stage_index];
        if (!stage.defined() || !stage->op.defined() ||
            !schedule->op_map.count(stage->op) ||
            schedule->op_map.at(stage->op).get() != stage.get()) {
            throw std::invalid_argument(
                "TE schedule contains an undefined or unindexed stage");
        }
        encoder.IntegerField("stage", stage_index);
        encoder.Field("operation_kind", OperationKind(stage->op));

        std::unordered_map<const Object*, size_t> axis_ids;
        for (size_t axis_index = 0;
             axis_index < stage->all_iter_vars.size(); ++axis_index) {
            const te::IterVar& axis = stage->all_iter_vars[axis_index];
            if (!axis.defined() ||
                !axis_ids.emplace(axis.get(), axis_index).second) {
                throw std::invalid_argument(
                    "TE schedule all_iter_vars must be defined and unique");
            }
            int64_t minimum = 0;
            if (!EvaluateStaticInt64(axis->dom_min, &minimum)) {
                throw std::invalid_argument(
                    "TE schedule canonical identity requires a static axis minimum");
            }
            encoder.IntegerField("axis", axis_index);
            encoder.IntegerField("axis_min", minimum);
            int64_t extent = 0;
            if (EvaluateStaticInt64(axis->dom_extent, &extent)) {
                encoder.IntegerField("axis_extent", extent);
            } else {
                size_t runtime_extent = 0;
                int64_t offset = 0;
                if (!bounded_dynamic ||
                    !MatchRuntimeExtentOffsetImpl(
                        axis->dom_extent, runtime_extent_buffers,
                        &runtime_extent, &offset)) {
                    throw std::invalid_argument(
                        "TE schedule axis extent is not canonicalizable");
                }
                used_runtime_extents[runtime_extent] = true;
                encoder.IntegerField("axis_runtime_extent", runtime_extent);
                encoder.IntegerField("axis_runtime_offset", offset);
            }
            encoder.BoolField("axis_reduction", axis->is_reduction);
        }
        for (const te::IterVar& root : stage->root_iter_vars) {
            if (axis_ids.count(root.get()) == 0) {
                throw std::invalid_argument(
                    "TE schedule root is absent from all_iter_vars");
            }
            encoder.IntegerField("root_axis", axis_ids.at(root.get()));
        }
        for (const te::SplitRelation& split : stage->split_relations) {
            int64_t factor = 0;
            if (axis_ids.count(split.parent.get()) == 0 ||
                axis_ids.count(split.outer.get()) == 0 ||
                axis_ids.count(split.inner.get()) == 0 ||
                !EvaluateStaticInt64(split.factor, &factor) || factor <= 0) {
                throw std::invalid_argument(
                    "TE schedule split cannot be canonicalized");
            }
            encoder.IntegerField("split_parent",
                                 axis_ids.at(split.parent.get()));
            encoder.IntegerField("split_outer",
                                 axis_ids.at(split.outer.get()));
            encoder.IntegerField("split_inner",
                                 axis_ids.at(split.inner.get()));
            encoder.IntegerField("split_factor", factor);
        }
        for (const te::IterVar& leaf : stage->leaf_iter_vars) {
            if (axis_ids.count(leaf.get()) == 0) {
                throw std::invalid_argument(
                    "TE schedule leaf is absent from all_iter_vars");
            }
            encoder.IntegerField("leaf_axis", axis_ids.at(leaf.get()));
            encoder.IntegerField("leaf_type",
                                 static_cast<int>(leaf->iter_type));
        }
        if (const auto* compute = stage->op.As<te::ComputeOpNode>()) {
            (void)BuildStageAxisPlan(stage, compute);
        }
    }
    if (bounded_dynamic) {
        for (size_t index = 0; index < used_runtime_extents.size(); ++index) {
            if (used_runtime_extents[index]) continue;
            if (body_consumed_extents &&
                body_consumed_extents->count(
                    runtime_extent_buffers[index].get()) != 0) {
                // 形状值单元把 extent 作为存储值消费，而不是循环轴。
                used_runtime_extents[index] = true;
                continue;
            }
            if (boundary_shape_extents && boundary_shape_extents->count(
                    runtime_extent_buffers[index].get()) != 0) {
                encoder.IntegerField("boundary_shape_extent", index);
                used_runtime_extents[index] = true;
                continue;
            }
            throw std::invalid_argument(
                "Every runtime extent buffer must drive a scheduled loop axis "
                "or be consumed as a lowered compute value or shape-only input boundary");
        }
    }
    if (!runtime_extent_buffers.empty() && !bounded_dynamic) {
        for (size_t index = 0; index < used_runtime_extents.size(); ++index) {
            if (!used_runtime_extents[index] &&
                body_only_extents.count(index) == 0) {
                throw std::invalid_argument(
                    "Every stateful KV runtime extent buffer must drive a "
                    "scheduled loop axis or be declared body-only");
            }
        }
    }
    return std::move(encoder).Take();
}

std::string GetTEScheduleContract(const tir::PrimFunc& function) {
    if (!function.defined() ||
        !function->attrs.count(String(kTEScheduleContractAttr))) {
        throw std::invalid_argument(
            "PrimFunc is missing its TE schedule contract");
    }
    const auto* contract =
        function->attrs.at(String(kTEScheduleContractAttr)).As<StringObj>();
    if (!contract || contract->data.empty()) {
        throw std::invalid_argument(
            "PrimFunc TE schedule contract is malformed");
    }
    return contract->data;
}

LoweredFunction LowerProgramToTIR(
    const te::Program& program, const std::vector<ConstantTensor>& constants,
    const Target& target, const std::string& tir_pipeline_canonical,
    const PrimFuncIdentity& identity) {
    const auto& saved = te::internal::ProgramAccess::Borrow(program);
    if (!target.defined() || saved.target_canonical != target.CanonicalBytes() ||
        saved.tir_pipeline_canonical != tir_pipeline_canonical ||
        saved.constants.size() != constants.size()) {
        throw std::invalid_argument("TE Program lowering Target, pipeline or constant ABI mismatch");
    }
    // TIR is mutable. Give the materializer a private working copy so returned
    // buffers/expressions cannot expose the retained Program's snapshot.
    const te::Program working(saved.inputs, saved.constants, saved.outputs,
        saved.schedule, target, saved.tir_pipeline_canonical, saved.metadata_only_inputs);
    if (working.canonical_bytes() != program.canonical_bytes()) {
        throw std::logic_error("TE Program immutable snapshot identity drifted");
    }
    const auto& snapshot = te::internal::ProgramAccess::Borrow(working);
    std::vector<ConstantTensor> bindings;
    for (size_t i = 0; i < constants.size(); ++i) {
        bindings.push_back({snapshot.constants[i], constants[i].key, constants[i].value});
    }
    LoweredFunction lowered = LowerTensorGraphToTIR(
        snapshot.inputs, bindings, snapshot.outputs, snapshot.schedule, target,
        identity, {}, {}, {}, snapshot.metadata_only_inputs);
    auto attrs = lowered->prim_func->attrs;
    // One candidate identity slot: static Program replaces the legacy schedule
    // bytes; bounded/stateful contracts keep their explicit existing schemas.
    attrs.Set(kTEScheduleContractAttr, String(program.canonical_bytes()));
    const auto& function = lowered->prim_func;
    return LoweredFunction(tir::PrimFunc(function->params, function->body,
                                         function->buffer_map, attrs), lowered.constants());
}

LoweredFunction LowerTensorGraphToTIR(
    const Array<te::Tensor>& inputs,
    const std::vector<ConstantTensor>& constants,
    const Array<te::Tensor>& outputs,
    const te::Schedule& schedule,
    const Target& target,
    const PrimFuncIdentity& identity,
    const Array<tir::Var>& runtime_extent_buffers,
    const std::vector<size_t>& body_only_runtime_extents,
    const std::vector<int64_t>& runtime_extent_upper_bounds,
    const Array<te::Tensor>& metadata_only_inputs) {
    ValidateScheduleTarget(target);
    ValidateRuntimeExtentBuffers(runtime_extent_buffers);
    if (!runtime_extent_upper_bounds.empty() &&
        (runtime_extent_upper_bounds.size() != runtime_extent_buffers.size() ||
         std::any_of(runtime_extent_upper_bounds.begin(),
                     runtime_extent_upper_bounds.end(),
                     [](int64_t upper) {
                         return upper < 0 ||
                                upper > std::numeric_limits<int32_t>::max();
                     }))) {
        throw std::invalid_argument(
            "Bounded runtime extent upper bounds must match the int32 extent ABI");
    }
    if (!runtime_extent_buffers.empty() && target->kind == "cuda" &&
        runtime_extent_upper_bounds.size() != runtime_extent_buffers.size()) {
        throw std::invalid_argument(
            "CUDA runtime extent lowering requires one finite upper bound per extent");
    }
    if (!schedule.defined()) {
        throw std::invalid_argument(
            "TE-to-TIR lowering requires an explicit Schedule");
    }
    if (std::string(identity.symbol).empty()) {
        throw std::invalid_argument("PrimFunc identity requires a non-empty symbol");
    }
    if (identity.unit_id >= 0 &&
        (std::string(identity.operator_name).empty() ||
         identity.operator_schema_version <= 0 ||
         std::string(identity.structural_hash).empty())) {
        throw std::invalid_argument(
            "Per-unit PrimFunc identity requires operator and structural metadata");
    }
    if (outputs.empty()) {
        throw std::invalid_argument("TE-to-TIR lowering requires output tensors");
    }

    std::unordered_map<const Object*, size_t> output_index_by_tensor;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const te::Tensor& output = outputs[i];
        if (!output.defined()) {
            throw std::invalid_argument("TE-to-TIR output tensor is undefined");
        }
        if (!output_index_by_tensor.emplace(output.get(), i).second) {
            throw std::invalid_argument(
                "TE-to-TIR output tensors must be distinct logical values");
        }
        ValidateLoweringTensor(
            output->shape, output->dtype,
            "TE-to-TIR output tensor '" + output->name + "'",
            runtime_extent_buffers, runtime_extent_upper_bounds);
        if (!output->op.As<te::ComputeOpNode>()) {
            throw std::invalid_argument(
                "TE-to-TIR public outputs must be produced by ComputeOp");
        }
    }

    std::unordered_set<const Object*> visited_ops;
    std::unordered_map<const Object*, std::vector<te::Tensor>> op_output_tensors;
    std::vector<te::Operation> topo_ops;
    for (const auto& output : outputs) {
        CollectOpsDFS(output, &visited_ops, &op_output_tensors, &topo_ops);
    }
    for (auto& entry : op_output_tensors) {
        std::sort(entry.second.begin(), entry.second.end(),
                  [](const te::Tensor& lhs, const te::Tensor& rhs) {
                      return lhs->value_index < rhs->value_index;
                  });
        for (const auto& tensor : entry.second) {
            ValidateLoweringTensor(
                tensor->shape, tensor->dtype,
                "TE-to-TIR tensor '" + tensor->name + "'",
                runtime_extent_buffers, runtime_extent_upper_bounds);
        }
    }
    for (const auto& operation : topo_ops) {
        const auto* compute = operation.As<te::ComputeOpNode>();
        if (!compute || compute->reduce_axis.empty()) continue;
        Array<tir::PrimExpr> reduction_shape;
        for (const auto& axis : compute->reduce_axis) {
            reduction_shape.push_back(axis->dom_extent);
        }
        ValidateLoweringTensor(
            reduction_shape, tir::DataType::UInt(8),
            "TE-to-TIR reduction iteration domain for '" + operation->name + "'",
            runtime_extent_buffers, runtime_extent_upper_bounds);
    }

    Array<te::Operation> unique_output_operations;
    std::unordered_set<const Object*> seen_output_operations;
    for (const te::Tensor& output : outputs) {
        if (seen_output_operations.insert(output->op.get()).second) {
            unique_output_operations.push_back(output->op);
        }
    }
    if (schedule->outputs.size() != unique_output_operations.size() ||
        schedule->stages.size() != topo_ops.size()) {
        throw std::invalid_argument(
            "TE schedule does not cover the lowering graph");
    }
    for (size_t index = 0; index < unique_output_operations.size(); ++index) {
        if (schedule->outputs[index].get() !=
            unique_output_operations[index].get()) {
            throw std::invalid_argument(
                "TE schedule outputs do not match lowering outputs");
        }
    }
    for (size_t index = 0; index < topo_ops.size(); ++index) {
        const te::Stage& stage = schedule->stages[index];
        if (!stage.defined() || stage->op.get() != topo_ops[index].get() ||
            !schedule->op_map.count(topo_ops[index]) ||
            schedule->op_map.at(topo_ops[index]).get() != stage.get()) {
            throw std::invalid_argument(
                "TE schedule stages do not match producer-first graph order");
        }
    }
    std::unordered_set<const Object*> body_consumed_extents, boundary_shape_extents;
    if (!runtime_extent_buffers.empty()) {
        for (const auto& operation : topo_ops) {
            const auto* compute = operation.As<te::ComputeOpNode>();
            if (!compute) continue;
            for (const tir::PrimExpr& body_expr : compute->body) {
                CollectRuntimeExtentBufferUses(body_expr,
                                               runtime_extent_buffers,
                                               &body_consumed_extents);
            }
        }
    }
    for (const auto& input : metadata_only_inputs) {
        const bool is_input = std::any_of(inputs.begin(), inputs.end(),
            [&](const te::Tensor& candidate) { return candidate.get() == input.get(); });
        if (!input.defined() || !is_input || !input->op.As<te::PlaceholderOpNode>() ||
            visited_ops.count(input->op.get())) {
            throw std::invalid_argument("Shape-only metadata must name an ABI input with no tensor payload reads");
        }
        for (const auto& extent : input->shape) {
            CollectRuntimeExtentBufferUses(extent, runtime_extent_buffers, &boundary_shape_extents);
        }
    }
    const std::string schedule_contract =
        CanonicalTEScheduleContract(schedule, target,
                                    runtime_extent_buffers,
                                    body_only_runtime_extents,
                                    &body_consumed_extents, &boundary_shape_extents,
                                    runtime_extent_upper_bounds);

    Array<tir::Var> params;
    Map<tir::Var, tir::Buffer> buffer_map;
    std::unordered_map<const Object*, tir::Var> buffer_var_by_tensor;
    for (const auto& tensor : inputs) {
        if (!tensor.defined()) {
            throw std::invalid_argument("TE-to-TIR input tensor is undefined");
        }
        ValidateLoweringTensor(
            tensor->shape, tensor->dtype,
            "TE-to-TIR input tensor '" + tensor->name + "'",
            runtime_extent_buffers, runtime_extent_upper_bounds);
        tir::Var data_var(tensor->name, tensor->dtype);
        tir::Buffer buffer(data_var, tensor->dtype,
                           CanonicalBufferShape(tensor->shape,
                                                runtime_extent_buffers), {},
                           tir::IntImm(0), tensor->name, 0, 0);
        params.push_back(data_var);
        buffer_map.Set(data_var, buffer);
        buffer_var_by_tensor[tensor.get()] = data_var;
    }
    const int64_t input_count = static_cast<int64_t>(inputs.size());
    const int64_t runtime_extent_param_start = input_count;
    for (const tir::Var& data_var : runtime_extent_buffers) {
        const std::string name = data_var->name_hint;
        params.push_back(data_var);
        buffer_map.Set(
            data_var,
            tir::Buffer(data_var, tir::DataType::UInt(64),
                        {tir::IntImm(1, tir::DataType::Int(64))}, {},
                        tir::IntImm(0), name, 8, 0));
    }
    const int64_t runtime_extent_count =
        static_cast<int64_t>(runtime_extent_buffers.size());

    Array<ConstantBinding> constant_bindings;
    Array<String> constant_keys;
    for (const auto& record : constants) {
        if (!record.tensor.defined() || !record.value.defined() ||
            std::string(record.key).empty()) {
            throw std::invalid_argument(
                "TE-to-TIR constant tensor record is incomplete");
        }
        const te::Tensor& tensor = record.tensor;
        ValidateStaticLoweringTensor(
            tensor->shape, tensor->dtype,
            "TE-to-TIR constant tensor '" + tensor->name + "'");
        tir::Var data_var(tensor->name, tensor->dtype);
        tir::Buffer buffer(data_var, tensor->dtype,
                           CanonicalBufferShape(tensor->shape, {}), {},
                           tir::IntImm(0), tensor->name, 0, 0);
        params.push_back(data_var);
        buffer_map.Set(data_var, buffer);
        buffer_var_by_tensor[tensor.get()] = data_var;
        const int64_t param_index = static_cast<int64_t>(params.size() - 1);
        constant_bindings.push_back(
            ConstantBinding(record.key, record.value, param_index));
        constant_keys.push_back(record.key);
    }
    const int64_t constant_count = static_cast<int64_t>(constants.size());
    const int64_t output_param_start =
        input_count + runtime_extent_count + constant_count;

    std::unordered_set<const Object*> public_outputs;
    std::unordered_set<std::string> used_output_names;
    for (size_t index = 0; index < outputs.size(); ++index) {
        const te::Tensor& tensor = outputs[index];
        public_outputs.insert(tensor.get());
        const std::string name =
            MakeOutputVarName(tensor, index, &used_output_names);
        tir::Var data_var(name, tensor->dtype);
        tir::Buffer buffer(data_var, tensor->dtype,
                           CanonicalBufferShape(tensor->shape,
                                                runtime_extent_buffers), {},
                           tir::IntImm(0), name, 0, 0);
        params.push_back(data_var);
        buffer_map.Set(data_var, buffer);
        buffer_var_by_tensor[tensor.get()] = data_var;
    }

    std::vector<te::Tensor> intermediates;
    for (const auto& operation : topo_ops) {
        if (!operation.As<te::ComputeOpNode>()) continue;
        const auto tensors_it = op_output_tensors.find(operation.get());
        if (tensors_it == op_output_tensors.end()) continue;
        for (const auto& tensor : tensors_it->second) {
            if (public_outputs.count(tensor.get()) != 0) continue;
            // Prove the maximum allocation using the invocation's bounds.
            // The generated Allocate still uses actual extents and existing
            // LLVM scoped heap storage, so smaller runs use smaller scratch.
            if (!runtime_extent_buffers.empty() &&
                schedule->policy != kStatefulKvTESchedulePolicy) {
                if (runtime_extent_upper_bounds.size() !=
                    runtime_extent_buffers.size()) {
                    throw std::invalid_argument(
                        "Bounded intermediate allocation requires invocation extent upper bounds");
                }
                Array<tir::PrimExpr> maximum_shape;
                for (const tir::PrimExpr& extent : tensor->shape) {
                    int64_t maximum = 0;
                    if (!EvaluateStaticInt64(extent, &maximum)) {
                        size_t index = 0;
                        int64_t offset = 0;
                        if (!MatchRuntimeExtentOffsetImpl(
                                extent, runtime_extent_buffers, &index, &offset)) {
                            throw std::invalid_argument(
                                "Bounded scratch extent has no invocation source");
                        }
                        maximum = runtime_extent_upper_bounds.at(index) + offset;
                    }
                    maximum_shape.push_back(
                        tir::IntImm(maximum, tir::DataType::Int(64)));
                }
                ValidateStaticLoweringTensor(
                    maximum_shape, tensor->dtype,
                    "TE-to-TIR bounded intermediate maximum allocation");
            }
            tir::Var local_var(tensor->name, tensor->dtype);
            buffer_var_by_tensor[tensor.get()] = local_var;
            intermediates.push_back(tensor);
        }
    }

    Array<tir::Stmt> compute_sequence;
    for (const auto& operation : topo_ops) {
        if (operation.As<te::PlaceholderOpNode>()) continue;
        const auto tensors_it = op_output_tensors.find(operation.get());
        if (tensors_it == op_output_tensors.end() || tensors_it->second.empty()) {
            throw std::runtime_error(
                "Missing tensor outputs during TE statement lowering");
        }
        for (const auto& tensor : tensors_it->second) {
            if (!tensor->op.As<te::ComputeOpNode>()) {
                throw std::runtime_error(
                    "Unsupported non-compute operation in TIR lowering");
            }
            try {
                compute_sequence.push_back(LowerComputeStmt(
                    tensor, schedule->op_map.at(operation),
                    buffer_var_by_tensor));
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    "LowerComputeStmt failed for tensor '" + tensor->name +
                    "': " + error.what());
            }
        }
    }

    tir::Stmt body;
    if (compute_sequence.size() == 1) {
        body = compute_sequence[0];
    } else if (!compute_sequence.empty()) {
        body = tir::SeqStmt(compute_sequence);
    }
    for (auto it = intermediates.rbegin(); it != intermediates.rend(); ++it) {
        const te::Tensor& tensor = *it;
        body = tir::Allocate(buffer_var_by_tensor.at(tensor.get()), tensor->dtype,
                             tensor->shape,
                             tir::IntImm(1, tir::DataType::Bool()), body);
    }

    Map<String, ObjectRef> attrs;
    attrs.Set("global_symbol", identity.symbol);
    attrs.Set("tir.noalias", tir::IntImm(1, tir::DataType::Bool()));
    attrs.Set("kxc.kernel_abi_version",
              tir::IntImm(codegen::kKernelAbiVersion,
                          tir::DataType::Int(64)));
    attrs.Set("kxc.input_count",
              tir::IntImm(input_count, tir::DataType::Int(64)));
    attrs.Set("kxc.runtime_extent_count",
              tir::IntImm(runtime_extent_count,
                          tir::DataType::Int(64)));
    attrs.Set("kxc.runtime_extent_param_start",
              tir::IntImm(runtime_extent_param_start,
                          tir::DataType::Int(64)));
    if (runtime_extent_count != 0 && target->kind == "cuda") {
        Array<int64_t> bounds;
        for (int64_t upper : runtime_extent_upper_bounds) bounds.push_back(upper);
        attrs.Set(tir::kCudaRuntimeExtentBoundsAttr, bounds);
    }
    attrs.Set("kxc.constant_count",
              tir::IntImm(constant_count, tir::DataType::Int(64)));
    attrs.Set("kxc.output_count",
              tir::IntImm(static_cast<int64_t>(outputs.size()),
                          tir::DataType::Int(64)));
    attrs.Set("kxc.output_param_start",
              tir::IntImm(output_param_start, tir::DataType::Int(64)));
    attrs.Set("kxc.constant_keys", codegen::KernelConstantKeys(constant_keys));
    attrs.Set(kTEScheduleContractAttr, String(schedule_contract));
    if (identity.unit_id >= 0) {
        const String operator_identity(
            std::string(identity.operator_name) + "@v" +
            std::to_string(identity.operator_schema_version));
        attrs.Set("kxc.unit_id",
                  tir::IntImm(identity.unit_id, tir::DataType::Int(64)));
        attrs.Set("kxc.operator_name", identity.operator_name);
        attrs.Set("kxc.operator_schema_version",
                  tir::IntImm(identity.operator_schema_version,
                              tir::DataType::Int(64)));
        attrs.Set("kxc.operator_identity", operator_identity);
        attrs.Set("kxc.structural_hash", identity.structural_hash);
    }
    const PassContext lowering_context = PassContext::MergeTarget(
        PassContext::Current(), target);
    attrs = tir::AttachPassContextAttrs(attrs, lowering_context);
    return LoweredFunction(tir::PrimFunc(params, body, buffer_map, attrs),
                           constant_bindings);
}

}  // namespace internal

}  // namespace relay
}  // namespace kxc
