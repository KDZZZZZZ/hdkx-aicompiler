/*! \file src/te/te.cc
 * \brief 实现 TE tensor、compute、reduce 和 schedule helper。
 */

#include "kxc/te/te.h"
#include "kxc/support/object_registration.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace kxc {
namespace te {

KXC_OBJECT_DEFINE(OperationNode)
KXC_OBJECT_DEFINE(TensorNode)
KXC_OBJECT_DEFINE(ProducerLoadNode)
KXC_OBJECT_DEFINE_WITH_KEY(IterVarNode, "kxc.te.IterVarNode")
KXC_OBJECT_DEFINE(StageNode)
KXC_OBJECT_DEFINE(ReduceNode)
KXC_OBJECT_DEFINE(PlaceholderOpNode)
KXC_OBJECT_DEFINE(ComputeOpNode)
KXC_OBJECT_DEFINE(ScheduleNode)

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

namespace {

IterVar MakeIterVar(tir::Var var, tir::PrimExpr min, tir::PrimExpr extent,
                    IterVarType type, bool is_reduction) {
    auto* node = new IterVarNode();
    node->var = std::move(var);
    node->dom_min = std::move(min);
    node->dom_extent = std::move(extent);
    node->iter_type = type;
    node->is_reduction = is_reduction;
    return IterVar(node);
}

int64_t RequirePositiveSplitFactor(const tir::PrimExpr& factor) {
    const auto* value = factor.As<tir::IntImmNode>();
    if (!value || (value->dtype.code != 0 && value->dtype.code != 1) ||
        value->dtype.lanes != 1 || value->value <= 0 ||
        value->value > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(
            "Stage::split requires a positive static int32 factor");
    }
    return value->value;
}

bool IsBaseIterType(const IterVar& var) {
    return var->iter_type == IterVarType::kDataPar ||
           var->iter_type == IterVarType::kCommReduce;
}

void RequireCurrentLeaf(const StageNode* stage, const IterVar& var,
                        const char* primitive) {
    if (!var.defined() ||
        std::find(stage->leaf_iter_vars.begin(), stage->leaf_iter_vars.end(),
                  var) == stage->leaf_iter_vars.end()) {
        throw std::invalid_argument(std::string("Stage::") + primitive +
                                    " requires a current leaf axis");
    }
}

void MarkLeaf(StageNode* stage, const IterVar& var, IterVarType type,
              bool allow_reduction, const char* primitive) {
    RequireCurrentLeaf(stage, var, primitive);
    if (!IsBaseIterType(var)) {
        throw std::invalid_argument(std::string("Stage::") + primitive +
                                    " axis already has an execution annotation");
    }
    if (var->is_reduction && !allow_reduction) {
        throw std::invalid_argument(std::string("Stage::") + primitive +
                                    " requires a data-parallel axis");
    }
    const_cast<IterVarNode*>(var.operator->())->iter_type = type;
}

void FindProducerLoads(const tir::PrimExpr& expr,
                       std::vector<Tensor>* dependencies) {
    if (!expr.defined()) return;
    if (const auto* load = expr.As<ProducerLoadNode>()) {
        dependencies->push_back(load->tensor);
        for (const auto& index : load->indices) {
            FindProducerLoads(index, dependencies);
        }
        return;
    }
    if (const auto* reduce = expr.As<ReduceNode>()) {
        for (const auto& source : reduce->source) {
            FindProducerLoads(source, dependencies);
        }
        return;
    }
    if (const auto* binary = expr.As<tir::BinaryOpNode>()) {
        FindProducerLoads(binary->a, dependencies);
        FindProducerLoads(binary->b, dependencies);
        return;
    }
    if (const auto* call = expr.As<tir::CallNode>()) {
        for (const auto& argument : call->args) {
            FindProducerLoads(argument, dependencies);
        }
        return;
    }
    if (const auto* select = expr.As<tir::SelectNode>()) {
        FindProducerLoads(select->condition, dependencies);
        FindProducerLoads(select->true_value, dependencies);
        FindProducerLoads(select->false_value, dependencies);
        return;
    }
    if (const auto* logical_not = expr.As<tir::NotNode>()) {
        FindProducerLoads(logical_not->value, dependencies);
        return;
    }
    if (const auto* load = expr.As<tir::LoadNode>()) {
        FindProducerLoads(load->index, dependencies);
        FindProducerLoads(load->predicate, dependencies);
    }
}

void CollectOperation(const Operation& operation,
                      std::unordered_set<const Object*>* visiting,
                      std::unordered_set<const Object*>* visited,
                      std::vector<Operation>* producer_first) {
    if (!operation.defined()) {
        throw std::invalid_argument("create_schedule received an undefined operation");
    }
    if (visited->count(operation.get()) != 0) return;
    if (!visiting->insert(operation.get()).second) {
        throw std::invalid_argument("create_schedule requires an acyclic TE graph");
    }
    if (const auto* compute = operation.As<ComputeOpNode>()) {
        for (const auto& expression : compute->body) {
            std::vector<Tensor> dependencies;
            FindProducerLoads(expression, &dependencies);
            for (const Tensor& dependency : dependencies) {
                if (!dependency.defined() || !dependency->op.defined()) {
                    throw std::invalid_argument(
                        "create_schedule found an undefined producer dependency");
                }
                CollectOperation(dependency->op, visiting, visited,
                                 producer_first);
            }
        }
    }
    visiting->erase(operation.get());
    visited->insert(operation.get());
    producer_first->push_back(operation);
}

}  // namespace

IterVar::IterVar(tir::PrimExpr min, tir::PrimExpr extent, IterVarType type,
                 std::string name) {
    const bool is_reduction = type == IterVarType::kCommReduce;
    *this = MakeIterVar(tir::Var(std::move(name)), std::move(min),
                        std::move(extent), type, is_reduction);
}

IterVar reduce_axis(tir::PrimExpr min, tir::PrimExpr extent, std::string name) {
    return IterVar(std::move(min), std::move(extent),
                   IterVarType::kCommReduce, std::move(name));
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

tir::PrimExpr min(tir::PrimExpr expr, Array<IterVar> axis) {
    return Reduce(std::move(axis), {std::move(expr)}, ReduceType::kMin);
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
    if (!op.defined()) {
        throw std::invalid_argument("Stage requires a defined operation");
    }
    auto* node = new StageNode();
    node->op = std::move(op);

    if (const auto* compute_op = node->op.As<ComputeOpNode>()) {
        for (size_t index = 0; index < compute_op->axis.size(); ++index) {
            const tir::Var& var = compute_op->axis[index];
            IterVar axis = MakeIterVar(
                var, tir::IntImm(0, var->dtype), compute_op->shape[index],
                IterVarType::kDataPar, false);
            node->root_iter_vars.push_back(axis);
            node->leaf_iter_vars.push_back(axis);
            node->all_iter_vars.push_back(axis);
        }
        for (const IterVar& reduction : compute_op->reduce_axis) {
            IterVar axis = MakeIterVar(
                reduction->var, reduction->dom_min, reduction->dom_extent,
                IterVarType::kCommReduce, true);
            node->root_iter_vars.push_back(axis);
            node->leaf_iter_vars.push_back(axis);
            node->all_iter_vars.push_back(axis);
        }
    }

    SetData(node);
}

IterVar Stage::split(IterVar parent, tir::PrimExpr factor,
                     IterVar* p_outer, IterVar* p_inner) {
    auto* node = operator->();
    RequireCurrentLeaf(node, parent, "split");
    if (!IsBaseIterType(parent)) {
        throw std::invalid_argument(
            "Stage::split cannot split an annotated leaf axis");
    }
    const int64_t factor_value = RequirePositiveSplitFactor(factor);
    const tir::DataType dtype = parent->var->dtype;
    const tir::PrimExpr normalized_factor = tir::IntImm(factor_value, dtype);
    const tir::PrimExpr outer_extent =
        (parent->dom_extent + tir::IntImm(factor_value - 1, dtype)) /
        normalized_factor;
    const IterVarType base_type = parent->is_reduction
                                      ? IterVarType::kCommReduce
                                      : IterVarType::kDataPar;
    IterVar outer = MakeIterVar(
        tir::Var(parent->var->name_hint + ".outer", dtype),
        tir::IntImm(0, dtype), outer_extent, base_type,
        parent->is_reduction);
    IterVar inner = MakeIterVar(
        tir::Var(parent->var->name_hint + ".inner", dtype),
        tir::IntImm(0, dtype), normalized_factor, base_type,
        parent->is_reduction);

    const auto found = std::find(node->leaf_iter_vars.begin(),
                                 node->leaf_iter_vars.end(), parent);
    const size_t index = static_cast<size_t>(
        std::distance(node->leaf_iter_vars.begin(), found));
    node->leaf_iter_vars[index] = outer;
    node->leaf_iter_vars.insert(node->leaf_iter_vars.begin() + index + 1,
                                inner);
    node->all_iter_vars.push_back(outer);
    node->all_iter_vars.push_back(inner);
    node->split_relations.push_back(
        SplitRelation{parent, outer, inner, normalized_factor});

    if (p_outer != nullptr) *p_outer = outer;
    if (p_inner != nullptr) *p_inner = inner;
    return outer;
}

void Stage::reorder(const Array<IterVar>& order) {
    auto* node = operator->();
    if (order.size() != node->leaf_iter_vars.size()) {
        throw std::invalid_argument(
            "Stage::reorder requires every current leaf exactly once");
    }
    std::unordered_set<const Object*> seen;
    bool saw_reduction = false;
    for (const IterVar& axis : order) {
        RequireCurrentLeaf(node, axis, "reorder");
        if (!seen.insert(axis.get()).second) {
            throw std::invalid_argument(
                "Stage::reorder does not accept duplicate axes");
        }
        if (axis->is_reduction) {
            saw_reduction = true;
        } else if (saw_reduction) {
            throw std::invalid_argument(
                "Stage::reorder requires data axes before reduction axes");
        }
    }
    Array<IterVar> reordered;
    for (const IterVar& axis : order) reordered.push_back(axis);
    node->leaf_iter_vars = std::move(reordered);
}

void Stage::vectorize(IterVar var) {
    auto* node = operator->();
    RequireCurrentLeaf(node, var, "vectorize");
    if (node->leaf_iter_vars.empty() ||
        node->leaf_iter_vars[node->leaf_iter_vars.size() - 1] != var) {
        throw std::invalid_argument(
            "Stage::vectorize requires the innermost leaf axis");
    }
    MarkLeaf(node, var, IterVarType::kVectorized, false, "vectorize");
}

void Stage::unroll(IterVar var) {
    MarkLeaf(operator->(), var, IterVarType::kUnrolled, true, "unroll");
}

void Stage::parallel(IterVar var) {
    MarkLeaf(operator->(), var, IterVarType::kParallel, false, "parallel");
}

Schedule create_schedule(const Array<Operation>& ops) {
    if (ops.empty()) {
        throw std::invalid_argument(
            "create_schedule requires at least one output operation");
    }
    std::unordered_set<const Object*> outputs;
    std::unordered_set<const Object*> visiting;
    std::unordered_set<const Object*> visited;
    std::vector<Operation> producer_first;
    for (const Operation& operation : ops) {
        if (!operation.defined() || !outputs.insert(operation.get()).second) {
            throw std::invalid_argument(
                "create_schedule outputs must be defined and unique");
        }
        CollectOperation(operation, &visiting, &visited, &producer_first);
    }

    auto* node = new ScheduleNode();
    for (const Operation& operation : ops) {
        node->outputs.push_back(operation);
    }
    for (const Operation& operation : producer_first) {
        Stage stage(operation);
        node->stages.push_back(stage);
        node->op_map.Set(operation, stage);
    }
    return Schedule(node);
}

}  // namespace te
}  // namespace kxc

