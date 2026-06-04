/*! \file src/te/te.cc
 * \brief 实现 TE tensor、compute、reduce 和 schedule helper。
 */

#include "te/te.h"

#include <algorithm>
#include <functional>

namespace kxc {
namespace te {

Tensor::Tensor(Array<tir::PrimExpr> shape, tir::DataType dtype, Operation op, int value_index) {
    auto* node = new TensorNode();
    node->shape = std::move(shape);
    node->dtype = dtype;
    node->op = std::move(op);
    node->value_index = value_index;
    if (node->op.defined()) {
        node->name = node->op->name;
        if (node->op->num_outputs() > 1) {
            node->name += ".v" + std::to_string(value_index);
        }
    }
    SetData(node);
}

tir::PrimExpr Tensor::operator()(const Array<tir::PrimExpr>& indices) const {
    return ProducerLoad(*this, indices);
}

tir::PrimExpr Tensor::operator()(const Array<tir::Var>& indices) const {
    Array<tir::PrimExpr> prim_indices;
    for (const auto& v : indices) {
        prim_indices.push_back(v);
    }
    return ProducerLoad(*this, prim_indices);
}

ProducerLoad::ProducerLoad(Tensor tensor, Array<tir::PrimExpr> indices) {
    auto* node = new ProducerLoadNode();
    node->tensor = std::move(tensor);
    node->indices = std::move(indices);
    node->dtype = node->tensor->dtype;
    SetData(node);
}

IterVar::IterVar(tir::PrimExpr min, tir::PrimExpr extent, IterVarType type,
                 std::string thread_tag, std::string name) {
    auto* node = new IterVarNode();
    node->var = tir::Var(std::move(name));
    node->dom_min = std::move(min);
    node->dom_extent = std::move(extent);
    node->iter_type = type;
    node->thread_tag = std::move(thread_tag);
    SetData(node);
}

IterVar reduce_axis(tir::PrimExpr min, tir::PrimExpr extent, std::string name) {
    return IterVar(std::move(min), std::move(extent), IterVarType::kCommReduce, "", std::move(name));
}

Reduce::Reduce(Array<IterVar> axis, Array<tir::PrimExpr> source, ReduceType type) {
    auto* node = new ReduceNode();
    node->axis = std::move(axis);
    node->source = std::move(source);
    node->reduce_type = type;
    if (!node->source.empty()) node->dtype = node->source[0].dtype();
    SetData(node);
}

tir::PrimExpr sum(tir::PrimExpr expr, Array<IterVar> axis) {
    return Reduce(std::move(axis), {std::move(expr)}, ReduceType::kSum);
}

tir::PrimExpr max(tir::PrimExpr expr, Array<IterVar> axis) {
    return Reduce(std::move(axis), {std::move(expr)}, ReduceType::kMax);
}

PlaceholderOp::PlaceholderOp(std::string name, Array<tir::PrimExpr> shape, tir::DataType dtype) {
    auto* node = new PlaceholderOpNode();
    node->name = std::move(name);
    node->shape = std::move(shape);
    node->dtype = dtype;
    SetData(node);
}

ComputeOp::ComputeOp(std::string name, std::string tag, Map<String, ObjectRef> attrs,
                     Array<tir::Var> axis, Array<tir::PrimExpr> body,
                     Array<tir::PrimExpr> shape) {
    auto* node = new ComputeOpNode();
    node->name = std::move(name);
    node->tag = std::move(tag);
    node->attrs = std::move(attrs);
    node->axis = std::move(axis);
    node->body = std::move(body);
    node->shape = std::move(shape);

    std::function<void(const tir::PrimExpr&)> find_reduce = [&](const tir::PrimExpr& expr) {
        if (auto* reduce = expr.As<ReduceNode>()) {
            for (auto& ax : reduce->axis) {
                bool exists = false;
                for (auto& exist_ax : node->reduce_axis) {
                    if (exist_ax == ax) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) node->reduce_axis.push_back(ax);
            }
            for (auto& src : reduce->source) find_reduce(src);
        } else if (auto* bin = expr.As<tir::BinaryOpNode>()) {
            find_reduce(bin->a);
            find_reduce(bin->b);
        } else if (auto* call = expr.As<tir::CallNode>()) {
            for (auto& arg : call->args) find_reduce(arg);
        } else if (auto* sel = expr.As<tir::SelectNode>()) {
            find_reduce(sel->condition);
            find_reduce(sel->true_value);
            find_reduce(sel->false_value);
        } else if (auto* not_node = expr.As<tir::NotNode>()) {
            find_reduce(not_node->value);
        } else if (auto* load = expr.As<ProducerLoadNode>()) {
            for (auto& idx : load->indices) find_reduce(idx);
        }
    };

    for (auto& expr : node->body) {
        find_reduce(expr);
    }

    SetData(node);
}

Tensor placeholder(Array<tir::PrimExpr> shape, tir::DataType dtype, std::string name) {
    PlaceholderOp op(std::move(name), shape, dtype);
    return Tensor(std::move(shape), dtype, std::move(op), 0);
}

Tensor compute(Array<tir::PrimExpr> shape, FCompute fcompute, std::string name, std::string tag,
               Map<String, ObjectRef> attrs) {
    Array<tir::Var> axis;
    for (size_t i = 0; i < shape.size(); ++i) {
        axis.push_back(tir::Var("ax" + std::to_string(i)));
    }

    tir::PrimExpr body = fcompute(axis);
    ComputeOp op(std::move(name), std::move(tag), std::move(attrs), std::move(axis), {body},
                 shape);
    return Tensor(std::move(shape), body.dtype(), std::move(op), 0);
}

Stage::Stage(Operation op) {
    auto* node = new StageNode();
    node->op = std::move(op);

    if (auto* compute_op = node->op.As<ComputeOpNode>()) {
        for (const auto& var : compute_op->axis) {
            IterVar iv(0, 0, IterVarType::kDataPar, "", var->name_hint);
            node->leaf_iter_vars.push_back(iv);
            node->all_iter_vars.push_back(iv);
        }
        for (const auto& iv : compute_op->reduce_axis) {
            node->leaf_iter_vars.push_back(iv);
            node->all_iter_vars.push_back(iv);
        }
    }

    SetData(node);
}

IterVar Stage::split(IterVar parent, tir::PrimExpr factor, IterVar* p_outer, IterVar* p_inner) {
    IterVar outer(0, 0, IterVarType::kDataPar, "", parent->var->name_hint + ".outer");
    IterVar inner(0, factor, IterVarType::kDataPar, "", parent->var->name_hint + ".inner");

    auto* node = this->operator->();
    auto it = std::find(node->leaf_iter_vars.begin(), node->leaf_iter_vars.end(), parent);
    if (it != node->leaf_iter_vars.end()) {
        size_t idx = std::distance(node->leaf_iter_vars.begin(), it);
        node->leaf_iter_vars[idx] = outer;
        node->leaf_iter_vars.insert(node->leaf_iter_vars.begin() + idx + 1, inner);
    }

    node->all_iter_vars.push_back(outer);
    node->all_iter_vars.push_back(inner);

    if (p_outer) *p_outer = outer;
    if (p_inner) *p_inner = inner;

    return outer;
}

IterVar Stage::fuse(IterVar outer, IterVar inner) {
    IterVar fused(0, 0, IterVarType::kDataPar, "",
                  outer->var->name_hint + "." + inner->var->name_hint + ".fused");

    auto* node = this->operator->();
    auto it_outer = std::find(node->leaf_iter_vars.begin(), node->leaf_iter_vars.end(), outer);
    auto it_inner = std::find(node->leaf_iter_vars.begin(), node->leaf_iter_vars.end(), inner);

    if (it_outer != node->leaf_iter_vars.end() && it_inner != node->leaf_iter_vars.end()) {
        node->leaf_iter_vars.erase(it_inner);
        it_outer = std::find(node->leaf_iter_vars.begin(), node->leaf_iter_vars.end(), outer);
        if (it_outer != node->leaf_iter_vars.end()) {
            *it_outer = fused;
        }
    }

    node->all_iter_vars.push_back(fused);
    return fused;
}

void Stage::reorder(const Array<IterVar>& order) {
    auto* node = this->operator->();
    node->leaf_iter_vars = order;
}

void Stage::tile(IterVar x_parent, IterVar y_parent, tir::PrimExpr x_factor,
                 tir::PrimExpr y_factor, IterVar* x_outer, IterVar* y_outer,
                 IterVar* x_inner, IterVar* y_inner) {
    split(x_parent, std::move(x_factor), x_outer, x_inner);
    split(y_parent, std::move(y_factor), y_outer, y_inner);

    auto* node = this->operator->();
    Array<IterVar> current_leaves = node->leaf_iter_vars;
    reorder({*x_outer, *y_outer, *x_inner, *y_inner});

    Array<IterVar> new_leaves = node->leaf_iter_vars;
    for (auto& iv : current_leaves) {
        if (iv == x_parent || iv == y_parent) continue;
        bool found = false;
        for (auto& new_iv : new_leaves) {
            if (iv == new_iv) {
                found = true;
                break;
            }
        }
        if (!found) new_leaves.push_back(iv);
    }
    node->leaf_iter_vars = new_leaves;
}

void Stage::vectorize(IterVar var) {
    const_cast<IterVarNode*>(var.operator->())->iter_type = IterVarType::kVectorized;
}

void Stage::unroll(IterVar var) {
    const_cast<IterVarNode*>(var.operator->())->iter_type = IterVarType::kUnrolled;
}

void Stage::parallel(IterVar var) {
    const_cast<IterVarNode*>(var.operator->())->iter_type = IterVarType::kParallel;
}

void Stage::bind(IterVar var, IterVar thread_axis) {
    const_cast<IterVarNode*>(var.operator->())->iter_type = IterVarType::kThreadIndex;
    const_cast<IterVarNode*>(var.operator->())->thread_tag = thread_axis->thread_tag;
}

IterVar thread_axis(tir::PrimExpr dom, std::string tag) {
    return IterVar(0, std::move(dom), IterVarType::kThreadIndex, tag, tag);
}

Schedule create_schedule(const Array<Operation>& ops) {
    auto* node = new ScheduleNode();
    node->outputs = ops;
    for (auto& op : ops) {
        Stage stage(op);
        node->stages.push_back(stage);
        node->op_map.Set(op, stage);
    }
    return Schedule(node);
}

}  // namespace te
}  // namespace kxc

