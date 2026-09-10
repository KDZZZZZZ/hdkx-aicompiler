#include "kxc/te/program.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "internal/program_access.h"
#include "support/canonical.h"
#include "support/hash.h"
#include "tir/internal/static_integer.h"

namespace kxc::te {
namespace {

template<class Node> const Node* Exact(const ObjectRef& object) {
    return object.defined() && typeid(*object.get()) == typeid(Node) ? object.As<Node>() : nullptr;
}

void Require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(std::string("TE Program: ") + message);
}

int64_t StaticInteger(const tir::PrimExpr& expr) {
    int64_t value = 0;
    Require(expr.defined() && expr.dtype().lanes == 1 &&
                tir::internal::EvaluateStaticInt64(expr, &value),
            "requires checked static scalar integer domains");
    return value;
}

Array<tir::PrimExpr> CopyShape(const Array<tir::PrimExpr>& shape) {
    Array<tir::PrimExpr> result;
    for (const auto& dim : shape) {
        const int64_t extent = StaticInteger(dim);
        Require(extent >= 0 && extent <= std::numeric_limits<int32_t>::max(),
                "shape exceeds the existing static int32 iteration domain");
        result.push_back(tir::IntImm(extent, dim.dtype()));
    }
    return result;
}

bool SameShape(const Array<tir::PrimExpr>& a, const Array<tir::PrimExpr>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (StaticInteger(a[i]) != StaticInteger(b[i])) return false;
    }
    return true;
}

void EncodeDType(support::CanonicalBytesEncoder* bytes, tir::DataType dtype) {
    bytes->IntegerField("code", dtype.code);
    bytes->IntegerField("bits", dtype.bits);
    bytes->IntegerField("lanes", dtype.lanes);
}

void EncodeShape(support::CanonicalBytesEncoder* bytes,
                 const Array<tir::PrimExpr>& shape) {
    bytes->IntegerField("rank", shape.size());
    for (const auto& dim : shape) bytes->IntegerField("extent", StaticInteger(dim));
}

// Copies the existing TE vocabulary. This is a snapshot/encoder, not another
// evaluator or schedule materializer: stage decisions are replayed through
// Stage's existing checked APIs, and TIR lowering still proves the candidate.
class Capture final {
public:
    std::shared_ptr<internal::ProgramState> Run(
        const Array<Tensor>& inputs, const Array<Tensor>& constants,
        const Array<Tensor>& outputs, const Schedule& schedule,
        const Target& target, std::string pipeline,
        const Array<Tensor>& metadata) {
        Require(target.defined() && !pipeline.empty(), "requires Target and explicit TIR pipeline");
        Require(target->device_id >= 0 &&
                    ((target->kind == "llvm" && target->device_type == kCPU) ||
                     (target->kind == "cuda" && target->device_type == kCUDA)),
                "Target kind and device identity are inconsistent");
        Require(schedule.defined() && !schedule->policy.empty() && !outputs.empty(),
                "requires a schedule, policy and ordered outputs");
        auto state = std::make_shared<internal::ProgramState>();
        state->target_canonical = target.CanonicalBytes();
        state->tir_pipeline_canonical = std::move(pipeline);
        bytes_.Field("target", state->target_canonical);
        bytes_.Field("tir_pipeline", state->tir_pipeline_canonical);
        bytes_.Field("storage", "contiguous;readonly-inputs-constants;fresh-outputs;root-stages-v1");
        bytes_.Field("policy", schedule->policy);
        Boundary(inputs, "inputs", &state->inputs);
        Boundary(constants, "constants", &state->constants);
        std::unordered_set<const Object*> metadata_ops;
        bytes_.IntegerField("metadata_count", metadata.size());
        size_t previous = 0;
        for (size_t i = 0; i < metadata.size(); ++i) {
            auto found = std::find(inputs.begin(), inputs.end(), metadata[i]);
            Require(found != inputs.end(), "metadata-only boundary must be an input");
            const size_t ordinal = std::distance(inputs.begin(), found);
            Require(i == 0 || ordinal > previous, "metadata-only input order is not canonical");
            previous = ordinal;
            metadata_ops.insert(metadata[i]->op.get());
            bytes_.IntegerField("metadata_input", ordinal);
            state->metadata_only_inputs.push_back(state->inputs[ordinal]);
        }

        Array<Operation> output_ops;
        std::unordered_set<const Object*> seen_outputs;
        for (const Tensor& output : outputs) {
            CheckTensor(output);
            Require(Exact<ComputeOpNode>(output->op) != nullptr, "outputs must be fresh compute results");
            if (seen_outputs.insert(output->op.get()).second) output_ops.push_back(output->op);
        }
        Require(schedule->outputs.size() == output_ops.size() &&
                    !schedule->stages.empty() &&
                    schedule->op_map.size() == schedule->stages.size(),
                "schedule does not cover the exact output DAG");
        for (size_t i = 0; i < output_ops.size(); ++i) {
            Require(schedule->outputs[i] == output_ops[i], "schedule output order drifted");
        }
        bytes_.IntegerField("stages", schedule->stages.size());
        std::unordered_set<const Object*> seen_stages;
        for (size_t i = 0; i < schedule->stages.size(); ++i) {
            Require(schedule->stages[i].defined() && schedule->stages[i]->op.defined(), "undefined stage");
            const Operation& op = schedule->stages[i]->op;
            Require(seen_stages.insert(op.get()).second &&
                        schedule->op_map.count(op) && schedule->op_map.at(op) == schedule->stages[i],
                    "stage or producer order is not canonical");
            if (op.As<PlaceholderOpNode>()) {
                Require(operations_.count(op.get()), "undeclared placeholder dependency");
                Require(!metadata_ops.count(op.get()), "metadata-only input is read by the payload DAG");
                used_inputs_.insert(op.get());
            } else {
                CopyCompute(op);
            }
            bytes_.IntegerField("stage_op", operation_ids_.at(op.get()));
            Stage copied(operations_.at(op.get()));
            CopyStage(schedule->stages[i], &copied);
            stages_.push_back(copied);
        }
        for (const auto& entry : boundary_ops_) {
            Require(used_inputs_.count(entry) || metadata_ops.count(entry), "unused ABI boundary is not declared metadata-only");
        }
        std::set<std::pair<size_t, int>> output_values;
        bytes_.IntegerField("outputs", outputs.size());
        for (const Tensor& output : outputs) {
            const auto identity = std::make_pair(operation_ids_.at(output->op.get()), output->value_index);
            Require(output_values.insert(identity).second, "duplicate output boundary");
            state->outputs.push_back(CopyTensor(output));
            bytes_.IntegerField("output_op", identity.first);
            bytes_.IntegerField("output_index", identity.second);
        }
        Array<Operation> copied_output_ops;
        for (const Operation& op : output_ops) copied_output_ops.push_back(operations_.at(op.get()));
        // The copy is now known acyclic. Reuse TE's DAG traversal to prove that
        // no extra, missing or reordered stage was smuggled into the candidate.
        const Schedule expected = create_schedule(copied_output_ops);
        Require(expected->stages.size() == stages_.size(), "schedule does not cover the exact output DAG");
        for (size_t i = 0; i < stages_.size(); ++i) {
            Require(expected->stages[i]->op == stages_[i]->op, "stage or producer order is not canonical");
        }
        auto* node = new ScheduleNode();
        for (const Operation& op : output_ops) node->outputs.push_back(operations_.at(op.get()));
        node->policy = schedule->policy;
        node->stages = std::move(stages_);
        for (const Stage& stage : node->stages) node->op_map.Set(stage->op, stage);
        state->schedule = Schedule(node);
        state->canonical = std::move(bytes_).Take();
        state->digest = support::HashText(state->canonical);
        return state;
    }

private:
    void CheckTensor(const Tensor& tensor) {
        Require(tensor.defined() && tensor->op.defined(), "undefined tensor or producer");
        Require(Exact<PlaceholderOpNode>(tensor->op) || Exact<ComputeOpNode>(tensor->op), "unsupported producer kind");
        Require(tensor->value_index >= 0 && tensor->value_index < tensor->op.num_outputs(),
                "invalid producer output index");
        Require(tensor->dtype == tensor->op.output_dtype(tensor->value_index) &&
                    SameShape(tensor->shape, tensor->op.output_shape(tensor->value_index)),
                "tensor shape/dtype drifted from its producer");
    }

    void Boundary(const Array<Tensor>& tensors, const char* role, Array<Tensor>* copied) {
        bytes_.IntegerField(role, tensors.size());
        for (const Tensor& tensor : tensors) {
            CheckTensor(tensor);
            const auto* op = tensor->op.As<PlaceholderOpNode>();
            Require(op && op->attrs.size() == 0, "input/constant boundary requires a plain placeholder");
            Require(boundary_ops_.insert(op).second, "input and constant boundaries must be disjoint and unique");
            const size_t ordinal = operations_.size();
            operations_.emplace(op, PlaceholderOp(op->name, CopyShape(op->shape), op->dtype));
            operation_ids_.emplace(op, ordinal);
            bytes_.Field("role", role);
            bytes_.IntegerField("boundary_op", ordinal);
            EncodeDType(&bytes_, tensor->dtype);
            EncodeShape(&bytes_, tensor->shape);
            copied->push_back(CopyTensor(tensor));
        }
    }

    Tensor CopyTensor(const Tensor& tensor) {
        CheckTensor(tensor);
        auto producer = operations_.find(tensor->op.get());
        Require(producer != operations_.end(), "producer is not available in topological order");
        const auto key = std::make_pair(operation_ids_.at(tensor->op.get()), tensor->value_index);
        auto found = tensors_.find(key);
        if (found != tensors_.end()) return found->second;
        Tensor result(CopyShape(tensor->shape), tensor->dtype, producer->second, tensor->value_index);
        tensors_.emplace(key, result);
        return result;
    }

    tir::Var BindVar(const tir::Var& var) {
        Require(var.defined() && var.dtype().code == 0 && var.dtype().lanes == 1 &&
                    var.dtype().bits == 32 && !variables_.count(var.get()),
                "iteration variables must be distinct int32 binders");
        const size_t ordinal = variables_.size();
        tir::Var result(var->name_hint, var.dtype());
        variables_.emplace(var.get(), std::make_pair(result, ordinal));
        return result;
    }

    void CopyCompute(const Operation& op) {
        const auto* compute = Exact<ComputeOpNode>(op);
        Require(compute && compute->attrs.size() == 0 && !compute->body.empty(),
                "unsupported operation or TE attrs");
        Require(compute->axis.size() == compute->shape.size(), "compute axis rank mismatch");
        variables_.clear();
        reductions_.clear();
        reduction_vars_.clear();
        Array<tir::Var> axes;
        for (const auto& axis : compute->axis) axes.push_back(BindVar(axis));
        bytes_.Field("op_kind", "compute");
        bytes_.Field("tag", compute->tag);
        EncodeShape(&bytes_, compute->shape);
        bytes_.IntegerField("reduce_axes", compute->reduce_axis.size());
        for (const IterVar& axis : compute->reduce_axis) {
            Require(axis.defined() && axis->is_reduction &&
                        axis->iter_type == IterVarType::kCommReduce,
                    "invalid compute reduction binder");
            tir::Var var = BindVar(axis->var);
            reduction_vars_.insert(axis->var.get());
            const int64_t minimum = StaticInteger(axis->dom_min);
            const int64_t extent = StaticInteger(axis->dom_extent);
            Require(extent >= 0 && extent <= std::numeric_limits<int32_t>::max(), "invalid reduction extent");
            IterVar copy(tir::IntImm(minimum), tir::IntImm(extent), IterVarType::kCommReduce, var->name_hint);
            const_cast<IterVarNode*>(copy.operator->())->var = var;
            reductions_.emplace(axis.get(), copy);
            bytes_.IntegerField("reduce_min", minimum);
            bytes_.IntegerField("reduce_extent", extent);
        }
        Array<tir::PrimExpr> body;
        bytes_.IntegerField("body_count", compute->body.size());
        for (const auto& expr : compute->body) body.push_back(CopyExpr(expr));
        ComputeOp copy(compute->name, compute->tag, {}, std::move(axes), std::move(body), CopyShape(compute->shape));
        Require(copy.As<ComputeOpNode>()->reduce_axis.size() == compute->reduce_axis.size(), "unused or duplicate reduction axes");
        for (size_t i = 0; i < compute->reduce_axis.size(); ++i) {
            Require(copy.As<ComputeOpNode>()->reduce_axis[i] == reductions_.at(compute->reduce_axis[i].get()), "noncanonical reduction axis order");
        }
        operation_ids_.emplace(op.get(), operations_.size());
        operations_.emplace(op.get(), std::move(copy));
    }

    tir::PrimExpr CopyExpr(const tir::PrimExpr& expr) {
        Require(expr.defined(), "undefined expression");
        Require(active_exprs_.size() < 256, "expression nesting exceeds the snapshot limit");
        Require(active_exprs_.insert(expr.get()).second, "cyclic expression");
        tir::PrimExpr copy = CopyExprNode(expr);
        active_exprs_.erase(expr.get());
        Require(copy.dtype() == expr.dtype(), "expression dtype drifted");
        return copy;
    }

    tir::PrimExpr CopyExprNode(const tir::PrimExpr& expr) {
        EncodeDType(&bytes_, expr.dtype());
        if (const auto* imm = Exact<tir::IntImmNode>(expr)) {
            bytes_.Field("expr", "integer");
            bytes_.IntegerField("value", imm->value);
            return tir::IntImm(imm->value, imm->dtype);
        }
        if (const auto* imm = Exact<tir::FloatImmNode>(expr)) {
            bytes_.Field("expr", "float");
            uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(imm->value));
            std::memcpy(&bits, &imm->value, sizeof(bits));
            bytes_.IntegerField("bits", bits);
            return tir::FloatImm(imm->value, imm->dtype);
        }
        if (Exact<tir::VarNode>(expr)) {
            auto found = variables_.find(expr.get());
            Require(found != variables_.end(), "free variable in compute body");
            Require(!reduction_vars_.count(expr.get()) || active_reductions_.count(expr.get()),
                    "reduction variable used outside its binder");
            bytes_.Field("expr", "var");
            bytes_.IntegerField("binder", found->second.second);
            return found->second.first;
        }
        if (const auto* load = Exact<ProducerLoadNode>(expr)) {
            Tensor tensor = CopyTensor(load->tensor);
            Require(load->indices.size() == tensor->shape.size(), "producer load rank mismatch");
            bytes_.Field("expr", "producer_load");
            bytes_.IntegerField("producer", operation_ids_.at(load->tensor->op.get()));
            bytes_.IntegerField("output", load->tensor->value_index);
            bytes_.IntegerField("indices", load->indices.size());
            Array<tir::PrimExpr> indices;
            for (const auto& index : load->indices) {
                Require(index.defined() && index.dtype().code <= 1 && index.dtype().lanes == 1 &&
                            index.dtype().bits > 0 && index.dtype().bits <= 64, "producer index must be a scalar integer");
                indices.push_back(CopyExpr(index));
            }
            return ProducerLoad(tensor, std::move(indices));
        }
        if (const auto* reduce = Exact<ReduceNode>(expr)) {
            Require(reduce->source.size() == 1 && !reduce->axis.empty() &&
                        (reduce->reduce_type == ReduceType::kSum || reduce->reduce_type == ReduceType::kMin ||
                         reduce->reduce_type == ReduceType::kMax), "unsupported reduction");
            bytes_.Field("expr", "reduce");
            bytes_.IntegerField("kind", static_cast<int>(reduce->reduce_type));
            bytes_.IntegerField("axes", reduce->axis.size());
            Array<IterVar> axes;
            std::unordered_set<const Object*> seen;
            for (const auto& axis : reduce->axis) {
                Require(axis.defined() && reductions_.count(axis.get()) && seen.insert(axis.get()).second,
                        "unknown or duplicate reduction axis");
                axes.push_back(reductions_.at(axis.get()));
                bytes_.IntegerField("binder", variables_.at(axis->var.get()).second);
                Require(active_reductions_.insert(axis->var.get()).second, "reduction variable rebound in nested scope");
            }
            auto value = CopyExpr(reduce->source[0]);
            for (const auto& axis : reduce->axis) active_reductions_.erase(axis->var.get());
            return Reduce(std::move(axes), {value}, reduce->reduce_type);
        }
#define KXC_COPY_BINARY(Name) \
        if (const auto* binary = Exact<tir::Name##Node>(expr)) { \
            bytes_.Field("expr", #Name); \
            auto a = CopyExpr(binary->a); \
            auto b = CopyExpr(binary->b); \
            return tir::Name(std::move(a), std::move(b)); \
        }
        KXC_COPY_BINARY(Add) KXC_COPY_BINARY(Sub) KXC_COPY_BINARY(Mul)
        KXC_COPY_BINARY(Div) KXC_COPY_BINARY(Mod) KXC_COPY_BINARY(Min)
        KXC_COPY_BINARY(Max) KXC_COPY_BINARY(EQ) KXC_COPY_BINARY(LT)
        KXC_COPY_BINARY(And) KXC_COPY_BINARY(Or)
#undef KXC_COPY_BINARY
        if (const auto* value = Exact<tir::NotNode>(expr)) {
            bytes_.Field("expr", "not");
            return tir::Not(CopyExpr(value->value));
        }
        if (const auto* call = Exact<tir::CallNode>(expr)) {
            bytes_.Field("expr", "call");
            bytes_.Field("callee", call->name);
            bytes_.IntegerField("arguments", call->args.size());
            Array<tir::PrimExpr> args;
            for (const auto& arg : call->args) args.push_back(CopyExpr(arg));
            return tir::Call(call->dtype, call->name, std::move(args));
        }
        if (const auto* select = Exact<tir::SelectNode>(expr)) {
            bytes_.Field("expr", "select");
            auto condition = CopyExpr(select->condition);
            auto yes = CopyExpr(select->true_value);
            auto no = CopyExpr(select->false_value);
            return tir::Select(std::move(condition), std::move(yes), std::move(no));
        }
        throw std::invalid_argument("TE Program: unsupported static expression kind");
    }

    void CopyStage(const Stage& source, Stage* result) {
        Require(source->root_iter_vars.size() == (*result)->root_iter_vars.size(), "stage root count drifted");
        std::unordered_map<const Object*, IterVar> axes;
        const auto* compute = Exact<ComputeOpNode>(source->op);
        for (size_t i = 0; i < source->root_iter_vars.size(); ++i) {
            const IterVar& axis = source->root_iter_vars[i];
            Require(axis.defined() && axes.emplace(axis.get(), (*result)->root_iter_vars[i]).second,
                    "duplicate or undefined stage root");
            const tir::Var& binder = i < compute->axis.size() ? compute->axis[i] :
                compute->reduce_axis[i - compute->axis.size()]->var;
            Require(axis->var == binder, "stage root is not its compute binder");
        }
        for (const SplitRelation& relation : source->split_relations) {
            Require(relation.parent.defined() && relation.outer.defined() && relation.inner.defined() &&
                        axes.count(relation.parent.get()), "split parent is not available");
            IterVar outer, inner;
            result->split(axes.at(relation.parent.get()), tir::IntImm(StaticInteger(relation.factor)), &outer, &inner);
            Require(axes.emplace(relation.outer.get(), outer).second && axes.emplace(relation.inner.get(), inner).second,
                    "split axes must be fresh and distinct");
        }
        Require(source->all_iter_vars.size() == (*result)->all_iter_vars.size(), "stage all-axis count drifted");
        Array<IterVar> order;
        for (const IterVar& axis : source->leaf_iter_vars) {
            Require(axis.defined() && axes.count(axis.get()), "unknown leaf axis");
            order.push_back(axes.at(axis.get()));
        }
        result->reorder(order);
        for (size_t i = 0; i < order.size(); ++i) {
            switch (source->leaf_iter_vars[i]->iter_type) {
                case IterVarType::kDataPar: case IterVarType::kCommReduce: break;
                case IterVarType::kVectorized: result->vectorize(order[i]); break;
                case IterVarType::kUnrolled: result->unroll(order[i]); break;
                case IterVarType::kParallel: result->parallel(order[i]); break;
                default: Require(false, "unsupported leaf execution annotation");
            }
        }
        std::unordered_map<const Object*, size_t> ordinals;
        std::unordered_set<const Object*> axis_variables;
        bytes_.IntegerField("roots", source->root_iter_vars.size());
        bytes_.IntegerField("all_axes", source->all_iter_vars.size());
        for (size_t i = 0; i < source->all_iter_vars.size(); ++i) {
            const IterVar& original = source->all_iter_vars[i];
            const IterVar& copied = (*result)->all_iter_vars[i];
            Require(original.defined() && original->var.defined() && axes.count(original.get()) &&
                        axes.at(original.get()) == copied && ordinals.emplace(original.get(), i).second &&
                        axis_variables.insert(original->var.get()).second,
                    "stage all-axis order is not canonical");
            Require(original->var.dtype() == copied->var.dtype() &&
                        original->is_reduction == copied->is_reduction && original->iter_type == copied->iter_type &&
                        StaticInteger(original->dom_min) == StaticInteger(copied->dom_min) &&
                        StaticInteger(original->dom_extent) == StaticInteger(copied->dom_extent),
                    "stage axis facts drifted from checked schedule decisions");
            bytes_.IntegerField("min", StaticInteger(copied->dom_min));
            bytes_.IntegerField("extent", StaticInteger(copied->dom_extent));
            bytes_.IntegerField("execution", static_cast<int>(copied->iter_type));
            bytes_.BoolField("reduction", copied->is_reduction);
        }
        bytes_.IntegerField("splits", source->split_relations.size());
        for (const auto& relation : source->split_relations) {
            bytes_.IntegerField("parent", ordinals.at(relation.parent.get()));
            bytes_.IntegerField("outer", ordinals.at(relation.outer.get()));
            bytes_.IntegerField("inner", ordinals.at(relation.inner.get()));
            bytes_.IntegerField("factor", StaticInteger(relation.factor));
        }
        bytes_.IntegerField("leaves", source->leaf_iter_vars.size());
        for (const auto& leaf : source->leaf_iter_vars) bytes_.IntegerField("leaf", ordinals.at(leaf.get()));
    }

    support::CanonicalBytesEncoder bytes_{"kxc.te.program.v1"};
    std::unordered_map<const Object*, Operation> operations_;
    std::unordered_map<const Object*, size_t> operation_ids_;
    std::map<std::pair<size_t, int>, Tensor> tensors_;
    std::unordered_map<const Object*, std::pair<tir::Var, size_t>> variables_;
    std::unordered_map<const Object*, IterVar> reductions_;
    std::unordered_set<const Object*> boundary_ops_, used_inputs_, active_exprs_;
    std::unordered_set<const Object*> reduction_vars_, active_reductions_;
    Array<Stage> stages_;
};

}  // namespace

Program::Program(const Array<Tensor>& inputs, const Array<Tensor>& constants,
                 const Array<Tensor>& outputs, const Schedule& schedule,
                 const Target& target, std::string pipeline,
                 const Array<Tensor>& metadata_only_inputs)
    : state_(Capture().Run(inputs, constants, outputs, schedule, target,
                           std::move(pipeline), metadata_only_inputs)) {}

bool Program::defined() const noexcept { return static_cast<bool>(state_); }
const std::string& Program::canonical_bytes() const { return internal::ProgramAccess::Borrow(*this).canonical; }
const std::string& Program::digest() const { return internal::ProgramAccess::Borrow(*this).digest; }

const internal::ProgramState& internal::ProgramAccess::Borrow(const Program& program) {
    Require(program.defined(), "undefined candidate");
    return *program.state_;
}

}  // namespace kxc::te
