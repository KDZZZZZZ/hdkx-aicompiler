/*! \file src/tir/transforms/bind_cuda_threads.cc
 * \brief 将独立输出映射到 CUDA 线程，并保留线程内串行归约。
 */

#include "kxc/tir/transforms/bind_cuda_threads.h"
#include "kxc/support/object_registration.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kxc/tir/visitor.h"
#include "kxc/tir/pass_utils.h"
#include "tir/internal/static_integer.h"

namespace kxc::tir {

KXC_OBJECT_DEFINE_WITH_KEY(CudaScheduleResultNode, "kxc.tir.CudaScheduleResultNode")

namespace {

struct DataAxis {
    Var var;
    int64_t minimum;
    int64_t extent;
    int64_t stride;
};

constexpr int64_t kMaxPrivateScratchBytes = 64 * 1024;

int64_t CheckedAdd(int64_t a, int64_t b) {
    if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b) {
        throw std::invalid_argument("BindCudaThreads integer domain overflows int64");
    }
    return a + b;
}

int64_t CheckedMultiply(int64_t a, int64_t b) {
    if (a < 0 || b < 0 || (b != 0 && a > std::numeric_limits<int64_t>::max() / b)) {
        throw std::invalid_argument("BindCudaThreads integer domain overflows int64");
    }
    return a * b;
}

bool ReadOutputIndex(const PrimExpr& expr, const std::vector<DataAxis>& axes,
                     std::vector<int64_t>* coefficients, int64_t* constant);

std::pair<int64_t, int64_t> StaticLoopDomain(const ForNode* loop,
                                           bool data_axis) {
    int64_t minimum = 0, extent = 0;
    std::vector<int64_t> unused;
    const DataType dtype = loop->loop_var->dtype;
    if (loop->for_type != ForType::Serial || dtype.lanes != 1 ||
        (dtype.code != 0 && dtype.code != 1) ||
        (dtype.bits != 32 && dtype.bits != 64) ||
        !ReadOutputIndex(loop->min, {}, &unused, &minimum) ||
        !ReadOutputIndex(loop->extent, {}, &unused, &extent) || minimum < 0 ||
        extent < (data_axis ? 1 : 0)) {
        throw std::invalid_argument(
            "BindCudaThreads requires static nonnegative serial integer loops");
    }
    const int64_t limit = dtype.bits == 64
        ? std::numeric_limits<int64_t>::max()
        : (dtype.code == 0 ? std::numeric_limits<int32_t>::max()
                          : static_cast<int64_t>(std::numeric_limits<uint32_t>::max()));
    // A retained serial loop must also represent its exclusive end and final ++.
    const int64_t end = CheckedAdd(minimum, extent - (data_axis ? 1 : 0));
    if (end > limit) {
        throw std::invalid_argument("BindCudaThreads loop domain exceeds its integer dtype");
    }
    return {minimum, extent};
}

// A small affine proof for row-major output addresses, not an index evaluator.
// Only nonnegative sums/products and lossless integer widening are admitted.
bool ReadOutputIndex(const PrimExpr& expr, const std::vector<DataAxis>& axes,
                     std::vector<int64_t>* coefficients, int64_t* constant) {
    if (!expr.defined()) return false;
    coefficients->assign(axes.size(), 0);
    *constant = 0;
    const auto finish = [&] {
        const DataType dtype = expr.dtype();
        if ((dtype.code != 0 && dtype.code != 1) || dtype.lanes != 1 ||
            (dtype.bits != 32 && dtype.bits != 64)) return false;
        int64_t maximum = *constant;
        for (size_t i = 0; i < axes.size(); ++i) {
            maximum = CheckedAdd(maximum, CheckedMultiply((*coefficients)[i],
                CheckedAdd(axes[i].minimum, axes[i].extent - 1)));
        }
        const int64_t limit = dtype.bits == 64
            ? std::numeric_limits<int64_t>::max()
            : (dtype.code == 0 ? std::numeric_limits<int32_t>::max()
                              : static_cast<int64_t>(std::numeric_limits<uint32_t>::max()));
        return maximum <= limit;
    };
    if (const auto* value = expr.As<IntImmNode>()) {
        if (value->value < 0) return false;
        *constant = value->value;
        return finish();
    }
    for (size_t i = 0; i < axes.size(); ++i) {
        if (expr.get() == axes[i].var.get()) {
            if (axes[i].extent == 1) *constant = axes[i].minimum;
            else (*coefficients)[i] = 1;
            return finish();
        }
    }
    if (const auto* add = expr.As<AddNode>()) {
        std::vector<int64_t> rhs;
        int64_t rhs_constant = 0;
        if (!ReadOutputIndex(add->a, axes, coefficients, constant) ||
            !ReadOutputIndex(add->b, axes, &rhs, &rhs_constant)) return false;
        for (size_t i = 0; i < axes.size(); ++i) {
            (*coefficients)[i] = CheckedAdd((*coefficients)[i], rhs[i]);
        }
        *constant = CheckedAdd(*constant, rhs_constant);
        return finish();
    }
    if (const auto* mul = expr.As<MulNode>()) {
        PrimExpr variable = mul->b;
        std::vector<int64_t> unused;
        int64_t factor = 0;
        if (!ReadOutputIndex(mul->a, {}, &unused, &factor)) {
            if (!ReadOutputIndex(mul->b, {}, &unused, &factor)) return false;
            variable = mul->a;
        }
        if (!ReadOutputIndex(variable, axes, coefficients, constant)) return false;
        for (int64_t& coefficient : *coefficients) {
            coefficient = CheckedMultiply(coefficient, factor);
        }
        *constant = CheckedMultiply(*constant, factor);
        return finish();
    }
    if (const auto* call = expr.As<CallNode>()) {
        if (call->name == "cast" && call->args.size() == 1 &&
            call->dtype.code == call->args[0].dtype().code &&
            call->dtype.lanes == 1 && call->args[0].dtype().lanes == 1 &&
            call->dtype.bits >= call->args[0].dtype().bits &&
            ReadOutputIndex(call->args[0], axes, coefficients, constant)) return finish();
    }
    return false;
}

bool IsOutputIndex(const PrimExpr& expr, const std::vector<DataAxis>& axes) {
    std::vector<int64_t> coefficients(axes.size(), 0);
    int64_t constant = 0, expected_constant = 0;
    if (!ReadOutputIndex(expr, axes, &coefficients, &constant)) return false;
    for (size_t i = 0; i < axes.size(); ++i) {
        const int64_t coefficient = axes[i].extent == 1 ? 0 : axes[i].stride;
        if (coefficients[i] != coefficient) return false;
        if (axes[i].extent == 1) {
            expected_constant = CheckedAdd(expected_constant,
                CheckedMultiply(axes[i].minimum, axes[i].stride));
        }
    }
    return constant == expected_constant;
}

// Every parallel iteration owns one distinct element in each written buffer.
// Reduction loops may revisit that element but must not alter its index.
void CollectIndependentWrites(const Stmt& stmt, const std::vector<DataAxis>& axes,
                              std::unordered_set<const Object*>* writes,
                              std::unordered_set<const Object*>* bound_vars,
                              size_t* store_count) {
    if (const auto* store = stmt.As<StoreNode>()) {
        if (!IsOutputIndex(store->index, axes)) {
            throw std::invalid_argument(
                "BindCudaThreads cannot prove Store index is iteration-independent");
        }
        writes->insert(store->buffer_var.get());
        ++(*store_count);
        return;
    }
    if (const auto* let_stmt = stmt.As<LetStmtNode>()) {
        if (!bound_vars->insert(let_stmt->var.get()).second) {
            throw std::invalid_argument("BindCudaThreads rejects rebound variables");
        }
        CollectIndependentWrites(let_stmt->body, axes, writes, bound_vars, store_count);
        bound_vars->erase(let_stmt->var.get());
        return;
    }
    if (const auto* loop = stmt.As<ForNode>()) {
        StaticLoopDomain(loop, false);
        if (!bound_vars->insert(loop->loop_var.get()).second) {
            throw std::invalid_argument("BindCudaThreads rejects rebound variables");
        }
        CollectIndependentWrites(loop->body, axes, writes, bound_vars, store_count);
        bound_vars->erase(loop->loop_var.get());
        return;
    }
    if (const auto* sequence = stmt.As<SeqStmtNode>()) {
        for (const auto& child : sequence->seq) {
            CollectIndependentWrites(child, axes, writes, bound_vars, store_count);
        }
        return;
    }
    // Allocate and multi-stage/conditional bodies need a separate storage proof.
    throw std::invalid_argument(
        "BindCudaThreads requires output-owned Store/Let/Seq/serial reduction bodies");
}

// True when an expression contains any TIR Load, including through arithmetic or Select.
bool ContainsLoad(const PrimExpr& expr) {
    if (!expr.defined()) return false;
    if (expr.As<LoadNode>()) return true;
    if (const auto* binary = expr.As<BinaryOpNode>()) {
        return ContainsLoad(binary->a) || ContainsLoad(binary->b);
    }
    if (const auto* select = expr.As<SelectNode>()) {
        return ContainsLoad(select->condition) || ContainsLoad(select->true_value) ||
               ContainsLoad(select->false_value);
    }
    if (const auto* call = expr.As<CallNode>()) {
        for (const auto& argument : call->args) {
            if (ContainsLoad(argument)) return true;
        }
        return false;
    }
    if (const auto* not_expr = expr.As<NotNode>()) return ContainsLoad(not_expr->value);
    return false;
}

// Detect indirect reads before choosing the additional address proof.
bool HasIndirectLoad(const PrimExpr& expr) {
    if (!expr.defined()) return false;
    if (const auto* load = expr.As<LoadNode>()) {
        return ContainsLoad(load->index) || HasIndirectLoad(load->index) ||
               HasIndirectLoad(load->predicate);
    }
    if (const auto* binary = expr.As<BinaryOpNode>()) {
        return HasIndirectLoad(binary->a) || HasIndirectLoad(binary->b);
    }
    if (const auto* select = expr.As<SelectNode>()) {
        return HasIndirectLoad(select->condition) || HasIndirectLoad(select->true_value) ||
               HasIndirectLoad(select->false_value);
    }
    if (const auto* call = expr.As<CallNode>()) {
        for (const auto& argument : call->args) {
            if (HasIndirectLoad(argument)) return true;
        }
        return false;
    }
    if (const auto* not_expr = expr.As<NotNode>()) return HasIndirectLoad(not_expr->value);
    return false;
}

bool HasIndirectLoad(const Stmt& stmt) {
    if (const auto* store = stmt.As<StoreNode>()) {
        return HasIndirectLoad(store->value) || HasIndirectLoad(store->predicate);
    }
    if (const auto* let_stmt = stmt.As<LetStmtNode>()) {
        return HasIndirectLoad(let_stmt->value) || HasIndirectLoad(let_stmt->body);
    }
    if (const auto* sequence = stmt.As<SeqStmtNode>()) {
        for (const auto& child : sequence->seq) {
            if (HasIndirectLoad(child)) return true;
        }
    }
    if (const auto* loop = stmt.As<ForNode>()) return HasIndirectLoad(loop->body);
    return false;
}

void RejectIndirectLoads(const Stmt& stmt) {
    if (HasIndirectLoad(stmt)) {
        throw std::invalid_argument("BindCudaThreads rejects indirect Load index expressions");
    }
}

// 检查表达式是否读取某个被并行写入的 buffer，保守拒绝潜在跨线程 RAW 竞争。
class WrittenBufferReadDetector : public TIRExprFunctor<bool> {
public:
    WrittenBufferReadDetector(const std::unordered_set<const Object*>& writes,
                             const std::unordered_set<const Object*>& initialized,
                             const std::vector<DataAxis>& axes)
        : writes_(writes), initialized_(initialized), axes_(axes) {}

protected:
    bool VisitLoad(const LoadNode* op, const PrimExpr& ref) override {
        (void)ref;
        return (writes_.count(op->buffer_var.get()) != 0 &&
                (initialized_.count(op->buffer_var.get()) == 0 ||
                 !IsOutputIndex(op->index, axes_))) || VisitExpr(op->index) ||
               VisitExpr(op->predicate);
    }

#define KXC_VISIT_BINARY(NodeType)                                      \
    bool Visit##NodeType(const NodeType##Node* op,                       \
                         const PrimExpr& ref) override {                 \
        (void)ref;                                                       \
        return VisitExpr(op->a) || VisitExpr(op->b);                    \
    }
    KXC_VISIT_BINARY(Add)
    KXC_VISIT_BINARY(Sub)
    KXC_VISIT_BINARY(Mul)
    KXC_VISIT_BINARY(Div)
    KXC_VISIT_BINARY(Mod)
    KXC_VISIT_BINARY(Min)
    KXC_VISIT_BINARY(Max)
    KXC_VISIT_BINARY(EQ)
    KXC_VISIT_BINARY(LT)
    KXC_VISIT_BINARY(And)
    KXC_VISIT_BINARY(Or)
#undef KXC_VISIT_BINARY

    bool VisitNot(const NotNode* op, const PrimExpr& ref) override {
        (void)ref;
        return VisitExpr(op->value);
    }

    bool VisitCall(const CallNode* op, const PrimExpr& ref) override {
        (void)ref;
        for (const auto& argument : op->args) {
            if (VisitExpr(argument)) return true;
        }
        return false;
    }

    bool VisitSelect(const SelectNode* op, const PrimExpr& ref) override {
        (void)ref;
        return VisitExpr(op->condition) || VisitExpr(op->true_value) ||
               VisitExpr(op->false_value);
    }

private:
    const std::unordered_set<const Object*>& writes_;
    const std::unordered_set<const Object*>& initialized_;
    const std::vector<DataAxis>& axes_;
};

// Only a dominating unconditional store permits later reads of the owned cell.
void ValidateOwnedReads(const Stmt& stmt,
                        const std::unordered_set<const Object*>& writes,
                        const std::vector<DataAxis>& axes,
                        std::unordered_set<const Object*>* initialized) {
    WrittenBufferReadDetector detector(writes, *initialized, axes);
    if (const auto* store = stmt.As<StoreNode>()) {
        if (detector.VisitExpr(store->value) || detector.VisitExpr(store->predicate)) {
            throw std::invalid_argument(
                "BindCudaThreads detected an uninitialized or cross-thread output read");
        }
        if (!store->predicate.defined() || pass_utils::IsConstOne(store->predicate)) {
            initialized->insert(store->buffer_var.get());
        }
        return;
    }
    if (const auto* let_stmt = stmt.As<LetStmtNode>()) {
        if (detector.VisitExpr(let_stmt->value)) {
            throw std::invalid_argument(
                "BindCudaThreads detected a loop-carried value dependency");
        }
        ValidateOwnedReads(let_stmt->body, writes, axes, initialized);
        return;
    }
    if (const auto* loop = stmt.As<ForNode>()) {
        auto after_loop = *initialized;
        ValidateOwnedReads(loop->body, writes, axes, &after_loop);
        if (StaticLoopDomain(loop, false).second > 0) *initialized = std::move(after_loop);
        return;
    }
    const auto* sequence = stmt.As<SeqStmtNode>();
    if (!sequence) return;
    for (const auto& child : sequence->seq) ValidateOwnedReads(child, writes, axes, initialized);
}

// 用 CUDA 线性线程号替换原循环变量，保持 TIR body 不依赖隐式命名规则。
class CudaBodyRewriter final : public TIRPass {
public:
    explicit CudaBodyRewriter(std::vector<std::pair<Var, PrimExpr>> replacements,
                              bool bounded = false)
        : replacements_(std::move(replacements)), bounded_(bounded) {}

protected:
    PrimExpr VisitVar(const VarNode* op, const PrimExpr& ref) override {
        for (const auto& replacement : replacements_) {
            if (ref.get() == replacement.first.get()) return replacement.second;
        }
        return ref;
    }

    Stmt VisitFor(const ForNode* op, const Stmt& ref) override {
        (void)ref;
        if (bounded_) {
            return For(op->loop_var, Mutate(op->min), Mutate(op->extent),
                       op->for_type, Mutate(op->body));
        }
        const auto domain = StaticLoopDomain(op, false);
        // Emit the proven bounds in the induction dtype so min + extent cannot
        // overflow in narrower C++ literal arithmetic before comparison.
        return For(op->loop_var, IntImm(domain.first, op->loop_var->dtype),
                   IntImm(domain.second, op->loop_var->dtype), op->for_type,
                   Mutate(op->body));
    }

private:
    std::vector<std::pair<Var, PrimExpr>> replacements_;
    bool bounded_;
};

struct ProvenIteration {
    Stmt body;
    std::vector<DataAxis> axes;
    int64_t work_size;
    PrimExpr actual_work;
    std::vector<PrimExpr> actual_extents;
    std::vector<PrimExpr> actual_strides;
};

// Polynomial equality proves compact row-major ownership for every admitted
// shape, not just its maximum. Atoms are ordered extent parameters and loop
// variables; this is an address proof, not a second runtime shape evaluator.
// The guarded integer proof reuses the bounded owner's actual-shape checks.
// These hooks contain proof facts only; they do not rewrite runtime addresses.
struct CudaReadProofHooks {
    std::function<bool(const LoadNode*)> direct_read;
    std::function<bool(const PrimExpr&, const PrimExpr&)> same_index;
    std::function<std::optional<std::pair<int64_t, int64_t>>(const PrimExpr&)> integer_bounds;
};
void CheckBoundedCudaReadValue(const PrimFunc& function, const PrimExpr& value,
    const std::unordered_set<const Object*>& writes, const CudaReadProofHooks& hooks);

class BoundedCudaIterationProof final {
    using Monomial = std::vector<size_t>;
    using Polynomial = std::map<Monomial, int64_t>;
public:
    explicit BoundedCudaIterationProof(const PrimFunc& function) : function_(function) {
        const int64_t count = Attribute("kxc.runtime_extent_count");
        const int64_t start = Attribute("kxc.runtime_extent_param_start");
        output_start_ = Attribute("kxc.output_param_start");
        const String bounds_key(kCudaRuntimeExtentBoundsAttr);
        const auto* bounds = function_->attrs.count(bounds_key)
            ? function_->attrs.at(bounds_key).As<ArrayNode<int64_t>>() : nullptr;
        if (count <= 0 || start < 0 || start > output_start_ || count > output_start_ - start ||
            output_start_ >= static_cast<int64_t>(function_->params.size()) || !bounds ||
            bounds->data.size() != static_cast<size_t>(count) ||
            start != Attribute("kxc.input_count") ||
            Attribute("kxc.constant_count") != output_start_ - start - count ||
            Attribute("kxc.output_count") != static_cast<int64_t>(function_->params.size()) - output_start_) Reject();
        scalar_count_ = static_cast<size_t>(count);
        for (size_t i = 0; i < scalar_count_; ++i) {
            const Var& var = function_->params[static_cast<size_t>(start) + i];
            const int64_t upper = bounds->data[i];
            if (upper < 0 || upper > INT32_MAX || !function_->buffer_map.count(var)) Reject();
            const auto& buffer = function_->buffer_map.at(var);
            if (var->dtype != DataType::UInt(64) || buffer->dtype != DataType::UInt(64) ||
                buffer->shape.size() != 1 || !pass_utils::IsConstOne(buffer->shape[0]) ||
                !scalars_.emplace(var.get(), i).second) Reject();
            maxima_.push_back(upper);
            minima_.push_back(0);
            upper_.push_back(Atom(i));
        }
        for (size_t i = 0; i < function_->params.size(); ++i) {
            const Var& var = function_->params[i];
            if (!bound_.insert(var.get()).second || !function_->buffer_map.count(var)) Reject();
            const auto& buffer = function_->buffer_map.at(var);
            if (buffer->data.get() != var.get() || buffer->dtype != var->dtype ||
                !buffer->strides.empty() || !pass_utils::IsConstZero(buffer->elem_offset)) Reject();
            Polynomial elements = Constant(1);
            for (const auto& dim : buffer->shape) elements = Multiply(elements, Shape(dim));
            (void)Maximum(elements);
            elements_.emplace(var.get(), std::move(elements));
            shapes_.emplace(var.get(), buffer->shape);
            if (static_cast<int64_t>(i) >= output_start_) outputs_.insert(var.get());
        }
    }

    ProvenIteration Check() {
        if (function_->body.As<AllocateNode>()) return CheckMultiStage();
        ProvenIteration result{function_->body, {}, 1, IntImm(1, DataType::Int(64)), {}, {}};
        while (const auto* loop = result.body.As<ForNode>()) {
            const auto extent = Loop(loop);
            const int64_t maximum = Maximum(extent);
            BindAxis(loop->loop_var, extent);
            result.axes.push_back({loop->loop_var, 0, maximum, 1});
            result.actual_extents.push_back(Widen(loop->extent));
            result.actual_work = result.actual_work * Widen(loop->extent);
            result.work_size = CheckedMultiply(result.work_size, maximum);
            result.body = loop->body;
        }
        if (result.work_size == 0) {
            result.body = Evaluate(IntImm(0));
            result.axes.clear();
            result.actual_extents.clear();
            result.actual_work = IntImm(0, DataType::Int(64));
            return result;
        }
        result.actual_strides.resize(result.axes.size());
        PrimExpr stride = IntImm(1, DataType::Int(64));
        Polynomial symbolic_stride = Constant(1);
        for (size_t i = result.axes.size(); i != 0; --i) {
            const size_t axis = i - 1;
            result.actual_strides[axis] = stride;
            owned_ = Add(owned_, Multiply(Variable(result.axes[axis].var), symbolic_stride));
            symbolic_stride = Multiply(symbolic_stride, Shape(result.actual_extents[axis]));
            stride = stride * result.actual_extents[axis];
        }
        for (const Object* output : outputs_) {
            if (elements_.at(output) != symbolic_stride) Reject();
        }
        std::unordered_set<const Object*> initialized;
        Statement(result.body, &initialized);
        if (initialized != outputs_) Reject();
        return result;
    }

private:
    struct Stage {
        Stmt body;
        Stmt iteration;
        Var output;
        std::vector<Var> axes;
    };

    // Only addresses already proved within one owner's row are privatized.
    // All surviving loops and shape expressions retain their actual extents.
    class StageProjection final : public TIRPass {
    public:
        StageProjection(const std::vector<std::pair<Var, PrimExpr>>& rows,
                        const std::unordered_map<const Object*, PrimExpr>& addresses,
                        const std::unordered_set<const Object*>& empty_loops)
            : rows_(rows), addresses_(addresses), empty_loops_(empty_loops) {}
    protected:
        PrimExpr VisitVar(const VarNode* op, const PrimExpr& ref) override {
            for (const auto& row : rows_) if (row.first.get() == op) return row.second;
            return ref;
        }
        Stmt VisitFor(const ForNode* op, const Stmt& ref) override {
            if (empty_loops_.count(op)) return Evaluate(IntImm(0));
            for (const auto& row : rows_) if (row.first.get() == op->loop_var.get()) return Mutate(op->body);
            return TIRPass::VisitFor(op, ref);
        }
        PrimExpr VisitLoad(const LoadNode* op, const PrimExpr& ref) override {
            const auto found = addresses_.find(op);
            return found == addresses_.end() ? TIRPass::VisitLoad(op, ref)
                : PrimExpr(Load(op->buffer_var, Mutate(found->second), Mutate(op->predicate)));
        }
        Stmt VisitStore(const StoreNode* op, const Stmt& ref) override {
            const auto found = addresses_.find(op);
            return found == addresses_.end() ? TIRPass::VisitStore(op, ref)
                : Stmt(Store(op->buffer_var, Mutate(op->value), Mutate(found->second), Mutate(op->predicate)));
        }
    private:
        const std::vector<std::pair<Var, PrimExpr>>& rows_;
        const std::unordered_map<const Object*, PrimExpr>& addresses_;
        const std::unordered_set<const Object*>& empty_loops_;
    };

    Var Writer(const Stmt& stmt) const {
        if (const auto* store = stmt.As<StoreNode>()) {
            if (!outputs_.count(store->buffer_var.get()) && !scratch_.count(store->buffer_var.get())) Reject();
            return store->buffer_var;
        }
        if (const auto* loop = stmt.As<ForNode>()) return Writer(loop->body);
        if (const auto* let = stmt.As<LetStmtNode>()) return Writer(let->body);
        if (const auto* sequence = stmt.As<SeqStmtNode>()) {
            Var writer;
            for (const auto& child : sequence->seq) {
                const Var current = Writer(child);
                if (writer.defined() && writer.get() != current.get()) Reject();
                writer = current;
            }
            if (writer.defined()) return writer;
        }
        Reject();
    }

    ProvenIteration CheckMultiStage() {
        Stmt body = function_->body;
        std::vector<Var> allocations;
        while (const auto* allocation = body.As<AllocateNode>()) {
            const Var var = allocation->buffer_var;
            if (!pass_utils::IsConstOne(allocation->condition) || var->dtype != allocation->dtype ||
                !bound_.insert(var.get()).second) Reject();
            Polynomial elements = Constant(1);
            for (const auto& dim : allocation->extents) elements = Multiply(elements, Shape(dim));
            (void)Maximum(elements);
            elements_.emplace(var.get(), std::move(elements));
            shapes_.emplace(var.get(), allocation->extents);
            scratch_.insert(var.get());
            allocations.push_back(var);
            body = allocation->body;
        }
        const auto* sequence = body.As<SeqStmtNode>();
        if (!sequence || sequence->seq.size() < 2) Reject();
        std::vector<Stage> stages;
        for (size_t i = 0; i < sequence->seq.size(); ++i) {
            Stmt statement = sequence->seq[i];
            const Var writer = Writer(statement);
            if (shapes_.at(writer.get()).empty()) {
                Array<Stmt> group{statement};
                while (i + 1 < sequence->seq.size() && Writer(sequence->seq[i + 1]).get() == writer.get())
                    group.push_back(sequence->seq[++i]);
                if (group.size() > 1) statement = SeqStmt(group);
            }
            Stage stage{statement, statement, writer, {}};
            const auto& shape = shapes_.at(writer.get());
            while (const auto* loop = stage.iteration.As<ForNode>()) {
                const size_t axis = stage.axes.size();
                if (axis >= shape.size() || Loop(loop) != Shape(shape[axis])) Reject();
                stage.axes.push_back(loop->loop_var);
                stage.iteration = loop->body;
            }
            if (stage.axes.size() != shape.size() || !writers_.insert(writer.get()).second) Reject();
            stages.push_back(std::move(stage));
        }
        for (const Object* output : outputs_) if (!writers_.count(output)) Reject();
        for (const Object* scratch : scratch_) if (!writers_.count(scratch)) Reject();

        // Common leading coordinates identify an independent compact row.
        // Non-prefix coordinates remain serial; no cross-thread scratch is used.
        const auto& common = shapes_.at(stages.back().output.get());
        size_t prefix = 0;
        for (; prefix < common.size(); ++prefix) {
            bool shared = true;
            for (const auto& stage : stages) {
                const auto& shape = shapes_.at(stage.output.get());
                shared &= prefix < shape.size() && Shape(shape[prefix]) == Shape(common[prefix]);
            }
            if (!shared) break;
        }
        ProvenIteration result{{}, {}, 1, IntImm(1, DataType::Int(64)), {}, {}};
        for (size_t i = 0; i < prefix; ++i) {
            const int64_t maximum = Maximum(Shape(common[i]));
            result.axes.push_back({Var("cuda_row_" + std::to_string(i), DataType::Int(64)), 0, maximum, 1});
            result.actual_extents.push_back(Widen(common[i]));
            result.actual_work = result.actual_work * Widen(common[i]);
            result.work_size = CheckedMultiply(result.work_size, maximum);
        }
        PrimExpr stride = IntImm(1, DataType::Int(64));
        result.actual_strides.resize(prefix);
        for (size_t i = prefix; i != 0; --i) {
            result.actual_strides[i - 1] = stride;
            stride = stride * result.actual_extents[i - 1];
        }
        if (result.work_size == 0) {
            result.body = Evaluate(IntImm(0));
            result.axes.clear();
            result.actual_extents.clear();
            result.actual_strides.clear();
            result.actual_work = IntImm(0, DataType::Int(64));
            return result;
        }
        Array<Stmt> projected;
        for (const auto& stage : stages) {
            outputs_ = {stage.output.get()};
            owned_.clear();
            addresses_.clear();
            row_axes_.assign(stage.axes.begin(), stage.axes.begin() + prefix);
            const auto& shape = shapes_.at(stage.output.get());
            for (size_t i = 0; i < stage.axes.size(); ++i) BindAxis(stage.axes[i], Shape(shape[i]));
            Polynomial stride = Constant(1);
            for (size_t i = stage.axes.size(); i != 0; --i) {
                owned_ = Add(owned_, Multiply(Variable(stage.axes[i - 1]), stride));
                stride = Multiply(stride, Shape(shape[i - 1]));
            }
            std::unordered_set<const Object*> initialized;
            if (Maximum(elements_.at(stage.output.get())) != 0) {
                Statement(stage.iteration, &initialized);
                if (initialized != outputs_) Reject();
                std::vector<std::pair<Var, PrimExpr>> replacements;
                for (size_t i = 0; i < prefix; ++i) {
                    const Var& row = result.axes[i].var;
                    PrimExpr coordinate = row;
                    if (row->dtype != stage.axes[i]->dtype) coordinate = Call(stage.axes[i]->dtype, "cast", {row});
                    replacements.emplace_back(stage.axes[i], coordinate);
                }
                StageProjection projection(replacements, addresses_, empty_loops_);
                projected.push_back(projection.Mutate(stage.body));
            }
            for (const Var& var : stage.axes) { variables_.erase(var.get()); bound_.erase(var.get()); }
            completed_.insert(stage.output.get());
        }
        body = projected.empty() ? Stmt(Evaluate(IntImm(0))) : Stmt(SeqStmt(projected));
        int64_t scratch_bytes = 0;
        for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
            Array<PrimExpr> private_shape;
            int64_t elements = 1;
            const auto& shape = shapes_.at(it->get());
            for (size_t i = 0; i < shape.size(); ++i) {
                const int64_t upper = i < prefix ? 1 : std::max<int64_t>(1, Maximum(Shape(shape[i])));
                private_shape.push_back(IntImm(upper, DataType::Int(64)));
                elements = CheckedMultiply(elements, upper);
            }
            scratch_bytes = CheckedAdd(scratch_bytes,
                CheckedMultiply(elements, (static_cast<int64_t>((*it)->dtype.bits) + 7) / 8));
            if (scratch_bytes > kMaxPrivateScratchBytes) Reject();
            body = Allocate(*it, (*it)->dtype, private_shape, IntImm(1, DataType::Bool()), body);
        }
        result.body = body;
        return result;
    }

    [[noreturn]] static void Reject() {
        throw std::invalid_argument("BindCudaThreads cannot prove bounded compact output ownership or address safety");
    }
    int64_t Attribute(const char* key) const {
        if (!function_->attrs.count(String(key))) Reject();
        const auto* value = function_->attrs.at(String(key)).As<IntImmNode>();
        if (!value || value->value < 0) Reject();
        return value->value;
    }
    static int64_t SignedAdd(int64_t a, int64_t b) {
        if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) Reject();
        return a + b;
    }
    static int64_t SignedMultiply(int64_t a, int64_t b) {
        if (!a || !b) return 0;
        if (a > 0 ? (b > 0 ? a > INT64_MAX / b : b < INT64_MIN / a)
                  : (b > 0 ? a < INT64_MIN / b : a < INT64_MAX / b)) Reject();
        return a * b;
    }
    static Polynomial Constant(int64_t value) { return value ? Polynomial{{{}, value}} : Polynomial{}; }
    static Polynomial Atom(size_t index) { return {{{index}, 1}}; }
    static Polynomial Add(Polynomial a, const Polynomial& b, int64_t sign = 1) {
        for (const auto& term : b) {
            const auto value = SignedAdd(a[term.first], SignedMultiply(sign, term.second));
            if (value) a[term.first] = value;
            else a.erase(term.first);
        }
        if (a.size() > 256) Reject();
        return a;
    }
    static Polynomial Multiply(const Polynomial& a, const Polynomial& b) {
        Polynomial result;
        for (const auto& left : a) for (const auto& right : b) {
            Monomial key = left.first;
            key.insert(key.end(), right.first.begin(), right.first.end());
            if (key.size() > 32) Reject();
            std::sort(key.begin(), key.end());
            result = Add(std::move(result), {{std::move(key), SignedMultiply(left.second, right.second)}});
        }
        return result;
    }
    int64_t Maximum(const Polynomial& value) const {
        int64_t result = 0;
        for (const auto& term : value) {
            int64_t product = 1;
            for (size_t index : term.first) product = CheckedMultiply(product,
                term.second < 0 ? minima_.at(index) : maxima_.at(index));
            result = SignedAdd(result, SignedMultiply(term.second, product));
        }
        return result;
    }
    int64_t Minimum(const Polynomial& value) const {
        int64_t result = 0;
        for (const auto& term : value) {
            int64_t product = 1;
            for (size_t index : term.first) product = CheckedMultiply(product,
                term.second < 0 ? maxima_.at(index) : minima_.at(index));
            result = SignedAdd(result, SignedMultiply(term.second, product));
        }
        return result;
    }
    void Fit(const Polynomial& value, DataType dtype) const {
        if (dtype.lanes != 1 || (dtype.code != 0 && dtype.code != 1) ||
            (dtype.bits != 32 && dtype.bits != 64 && !(dtype.code == 1 && dtype.bits == 1))) Reject();
        const int64_t limit = dtype.bits == 64 ? INT64_MAX : dtype.bits == 1 ? 1 :
            dtype.code == 0 ? INT32_MAX : static_cast<int64_t>(UINT32_MAX);
        if (Minimum(value) < 0 || Maximum(value) > limit) Reject();
    }
    Polynomial Variable(const Var& var) const {
        const auto found = variables_.find(var.get());
        if (found == variables_.end()) Reject();
        const auto branch = branch_values_.find(var.get());
        if (branch != branch_values_.end()) return branch->second;
        return maxima_[found->second] == 0 ? Polynomial{} : Atom(found->second);
    }
    bool Positive(const Polynomial& value) const {
        if (Minimum(value) > 0) return true;
        if (value.size() != 1 || value.begin()->second <= 0) return false;
        std::unordered_set<size_t> positive;
        // Reaching a loop body proves every factor of a monomial extent is
        // positive. A sum such as P+1 gives no such fact about P.
        for (const auto& variable : variables_) {
            const auto extent = Add(upper_.at(variable.second), Constant(1));
            if (extent.size() == 1 && extent.begin()->second > 0)
                positive.insert(extent.begin()->first.begin(), extent.begin()->first.end());
        }
        for (size_t atom : value.begin()->first) if (!positive.count(atom)) return false;
        return true;
    }
    bool Within(const Polynomial& index, const Polynomial& elements) const {
        Polynomial maximum;
        for (const auto& term : index) {
            Polynomial expanded = Constant(term.second);
            for (size_t atom : term.first) expanded = Multiply(expanded,
                term.second < 0 ? Constant(minima_.at(atom)) : upper_.at(atom));
            maximum = Add(std::move(maximum), expanded);
        }
        const auto remaining = Add(elements, Add(maximum, Constant(1)), -1);
        return Minimum(remaining) >= 0;
    }
    Polynomial Divide(const Polynomial& numerator, const Polynomial& divisor, bool modulo) {
        if (divisor.size() != 1 || !Positive(divisor)) Reject();
        const auto& factor = *divisor.begin();
        for (size_t atom : factor.first) if (atom >= scalar_count_) Reject();
        Polynomial quotient, remainder;
        for (const auto& term : numerator) {
            if (term.second < 0) Reject();
            Monomial residual = term.first;
            bool divides = true;
            for (size_t atom : factor.first) {
                const auto found = std::find(residual.begin(), residual.end(), atom);
                if (found == residual.end()) { divides = false; break; }
                residual.erase(found);
            }
            if (divides) {
                quotient = Add(std::move(quotient), {{residual, term.second / factor.second}});
                if (term.second % factor.second)
                    remainder = Add(std::move(remainder), {{term.first, term.second % factor.second}});
            } else remainder = Add(std::move(remainder), {{term.first, term.second}});
        }
        if (remainder.empty() || Within(remainder, divisor)) return modulo ? remainder : quotient;
        const auto key = std::make_tuple(remainder, divisor, modulo);
        auto found = index_atoms_.find(key);
        size_t atom = 0;
        if (found != index_atoms_.end()) atom = found->second;
        else {
            atom = maxima_.size();
            const int64_t bound = modulo ? Maximum(divisor) - 1
                : Maximum(remainder) / std::max<int64_t>(1, Minimum(divisor));
            minima_.push_back(0);
            maxima_.push_back(bound);
            upper_.push_back(modulo ? Add(divisor, Constant(-1)) : Constant(bound));
            index_atoms_.emplace(key, atom);
        }
        return modulo ? Atom(atom) : Add(std::move(quotient), Atom(atom));
    }
    Polynomial Parse(const PrimExpr& expr) {
        Polynomial result;
        if (const auto* value = expr.As<IntImmNode>()) {
            if (value->value < 0) Reject();
            result = Constant(value->value);
        } else if (expr.As<VarNode>()) {
            result = Variable(Var(expr));
        } else if (const auto* load = expr.As<LoadNode>()) {
            const auto found = scalars_.find(load->buffer_var.get());
            if (found == scalars_.end() || load->dtype != DataType::UInt(64) ||
                !pass_utils::IsConstZero(load->index) ||
                (load->predicate.defined() && !pass_utils::IsConstOne(load->predicate))) Reject();
            result = Atom(found->second);
        } else if (const auto* cast = expr.As<CallNode>()) {
            if (cast->name != "cast" || cast->args.size() != 1) Reject();
            result = Parse(cast->args[0]);
        } else if (const auto* select = expr.As<SelectNode>()) {
            if (pass_utils::IsConstOne(select->condition)) result = Parse(select->true_value);
            else if (pass_utils::IsConstZero(select->condition)) result = Parse(select->false_value);
            else {
                const auto* equal = select->condition.As<EQNode>();
                PrimExpr coordinate = select->false_value;
                while (const auto* cast = coordinate.As<CallNode>()) {
                    if (cast->name != "cast" || cast->args.size() != 1) Reject();
                    coordinate = cast->args[0];
                }
                const auto variable = variables_.find(coordinate.get());
                if (!equal || !pass_utils::IsConstZero(select->true_value) || variable == variables_.end()) Reject();
                PrimExpr extent = equal->a;
                if (!pass_utils::IsConstOne(equal->b)) {
                    if (!pass_utils::IsConstOne(equal->a)) Reject();
                    extent = equal->b;
                }
                if (Shape(extent) != Add(upper_.at(variable->second), Constant(1))) Reject();
                // In select(S==1,0,i), the i<S domain makes i zero on the
                // singleton branch. The original runtime expression is kept.
                result = Parse(select->false_value);
            }
        } else if (const auto* binary = expr.As<BinaryOpNode>()) {
            const auto a = Parse(binary->a), b = Parse(binary->b);
            if (expr.As<AddNode>()) result = Add(a, b);
            else if (expr.As<SubNode>()) result = Add(a, b, -1);
            else if (expr.As<MulNode>()) result = Multiply(a, b);
            else if (expr.As<DivNode>() || expr.As<ModNode>()) result = Divide(a, b, bool(expr.As<ModNode>()));
            else Reject();
            // CUDA C++ also has to represent the intermediate arithmetic.
            Fit(result, DataType::Int(std::max(binary->a.dtype().bits, binary->b.dtype().bits)));
        } else Reject();
        Fit(result, expr.dtype());
        return result;
    }
    Polynomial Shape(const PrimExpr& expr) {
        const auto result = Parse(expr);
        for (const auto& term : result) for (size_t index : term.first) if (index >= scalar_count_) Reject();
        return result;
    }
    std::optional<Polynomial> TryParse(const PrimExpr& expr) {
        try { return Parse(expr); }
        catch (const std::invalid_argument&) { return std::nullopt; }
    }
    Polynomial Loop(const ForNode* loop) {
        if (loop->for_type != ForType::Serial || !pass_utils::IsConstZero(loop->min) ||
            loop->loop_var->dtype.lanes != 1 || loop->loop_var->dtype.code != 0 ||
            (loop->loop_var->dtype.bits != 32 && loop->loop_var->dtype.bits != 64)) Reject();
        const auto extent = Shape(loop->extent);
        Fit(extent, loop->loop_var->dtype);  // includes the final serial ++
        return extent;
    }
    size_t BindAxis(const Var& var, const Polynomial& extent) {
        if (!bound_.insert(var.get()).second) Reject();
        const size_t atom = maxima_.size();
        variables_.emplace(var.get(), atom);
        maxima_.push_back(std::max<int64_t>(0, Maximum(extent) - 1));
        minima_.push_back(0);
        upper_.push_back(Add(extent, Constant(-1)));
        return atom;
    }
    static PrimExpr Widen(const PrimExpr& value) {
        return value.dtype() == DataType::Int(64) ? value : PrimExpr(Call(DataType::Int(64), "cast", {value}));
    }
    void CheckRange(const Polynomial& index, const Polynomial& elements) const {
        if (!Within(index, elements)) Reject();
    }
    void ProjectAddress(const Object* access, const Var& buffer, const PrimExpr& expr) {
        const auto& shape = shapes_.at(buffer.get());
        Polynomial row_offset, stride = Constant(1);
        for (size_t i = shape.size(); i != 0; --i) {
            if (i - 1 < row_axes_.size()) row_offset = Add(row_offset, Multiply(Variable(row_axes_[i - 1]), stride));
            stride = Multiply(stride, Shape(shape[i - 1]));
        }
        const auto local = Add(Parse(expr), row_offset, -1);
        for (const auto& term : local) {
            if (term.second < 0) Reject();
            for (const Var& row : row_axes_) if (std::find(term.first.begin(), term.first.end(), variables_.at(row.get())) != term.first.end()) Reject();
        }
        Polynomial private_elements = Constant(1);
        for (size_t i = row_axes_.size(); i < shape.size(); ++i) private_elements = Multiply(private_elements, Shape(shape[i]));
        CheckRange(local, private_elements);
        if (scratch_.count(buffer.get())) {
            std::vector<std::pair<Var, PrimExpr>> zeros;
            for (const Var& row : row_axes_) zeros.emplace_back(row, IntImm(0, row->dtype));
            CudaBodyRewriter projection(std::move(zeros), true);
            addresses_.emplace(access, projection.Mutate(expr));
        }
    }
    bool Read(const LoadNode* load, const std::unordered_set<const Object*>& initialized) {
        if (!elements_.count(load->buffer_var.get()) || load->dtype != load->buffer_var->dtype ||
            (load->predicate.defined() && !pass_utils::IsConstOne(load->predicate))) Reject();
        const auto parsed = TryParse(load->index);
        if (!parsed) {
            // The complementary guarded proof may address only a static,
            // read-only table. Every nested index load is still checked here
            // against its actual compact shape, not its allocation upper bound.
            if (writers_.count(load->buffer_var.get()) || outputs_.count(load->buffer_var.get())) Reject();
            for (const PrimExpr& dim : shapes_.at(load->buffer_var.get())) {
                int64_t extent = 0;
                if (!internal::EvaluateStaticInt64(dim, &extent) || extent < 0) Reject();
            }
            needs_read_range_ = true;
            Expression(load->index, initialized);
            return false;
        }
        const auto& index = *parsed;
        if (writers_.count(load->buffer_var.get())) {
            if (!outputs_.count(load->buffer_var.get()) && !completed_.count(load->buffer_var.get())) Reject();
            ProjectAddress(load, load->buffer_var, load->index);
        }
        if (outputs_.count(load->buffer_var.get())) {
            if (!initialized.count(load->buffer_var.get()) || index != owned_) Reject();
            return true;
        }
        CheckRange(index, elements_.at(load->buffer_var.get()));
        return true;
    }
    void Expression(const PrimExpr& expr, const std::unordered_set<const Object*>& initialized) {
        if (!expr.defined()) return;
        if (const auto* load = expr.As<LoadNode>()) { Read(load, initialized); return; }
        if (const auto* binary = expr.As<BinaryOpNode>()) {
            Expression(binary->a, initialized); Expression(binary->b, initialized);
        } else if (const auto* select = expr.As<SelectNode>()) {
            Expression(select->condition, initialized);
            Branch(select->true_value, select->condition, true, initialized);
            Branch(select->false_value, select->condition, false, initialized);
        } else if (const auto* call = expr.As<CallNode>()) {
            for (const auto& arg : call->args) Expression(arg, initialized);
        } else if (const auto* logical = expr.As<NotNode>()) Expression(logical->value, initialized);
        else if (expr.As<VarNode>()) {
            if (!bound_.count(expr.get()) || elements_.count(expr.get())) Reject();
        } else if (!expr.As<IntImmNode>() && !expr.As<FloatImmNode>()) Reject();
    }
    static PrimExpr WidenedCoordinate(PrimExpr coordinate) {
        while (const auto* cast = coordinate.As<CallNode>()) {
            if (cast->name != "cast" || cast->args.size() != 1 || cast->dtype.code != 0 ||
                cast->args[0].dtype().code != 0 || cast->dtype.bits < cast->args[0].dtype().bits)
                return PrimExpr();
            coordinate = cast->args[0];
        }
        return coordinate;
    }
    bool PartitionedBranch(const PrimExpr& value, const LTNode* comparison, bool truth,
                           const std::unordered_set<const Object*>& initialized) {
        const PrimExpr coordinate = WidenedCoordinate(comparison->a);
        const auto variable = variables_.find(coordinate.get());
        if (variable == variables_.end() || branch_values_.count(coordinate.get())) return false;
        const auto boundary = TryParse(comparison->b);
        if (!boundary) return false;
        bool symbolic = false;
        for (const auto& term : *boundary) for (size_t atom : term.first) {
            if (atom >= scalar_count_) return false;
            symbolic = true;
        }
        if (!symbolic) return false;
        const size_t atom = variable->second;
        // This is the existing concatenate contract: [0,P+C) partitioned at
        // P, with a fixed nonnegative suffix C. Other partitions stay closed.
        const auto suffix = Add(Add(upper_[atom], Constant(1)), *boundary, -1);
        if (minima_[atom] != 0 || suffix.size() > 1 ||
            (!suffix.empty() && (!suffix.begin()->first.empty() || suffix.begin()->second < 0))) return false;
        const int64_t count = suffix.empty() ? 0 : suffix.begin()->second;
        auto outer_index_atoms = index_atoms_;
        if (truth) {
            if (Maximum(*boundary) == 0) return true;
            const auto outer_upper = upper_[atom];
            const int64_t outer_maximum = maxima_[atom];
            upper_[atom] = Add(*boundary, Constant(-1));
            maxima_[atom] = Maximum(*boundary) - 1;
            Expression(value, initialized);
            upper_[atom] = outer_upper;
            maxima_[atom] = outer_maximum;
        } else {
            if (count == 0) return true;
            // In the suffix, c=P+u and 0<=u<C. Rebasing only the proof keeps
            // c-P nonnegative even when P varies or is zero; TIR is unchanged.
            const size_t local = maxima_.size();
            minima_.push_back(0); maxima_.push_back(count - 1);
            upper_.push_back(Constant(count - 1));
            branch_values_.emplace(coordinate.get(), Add(*boundary, count == 1 ? Polynomial{} : Atom(local)));
            Expression(value, initialized);
            branch_values_.erase(coordinate.get());
        }
        index_atoms_ = std::move(outer_index_atoms);
        return true;
    }
    void Branch(const PrimExpr& value, const PrimExpr& condition, bool truth,
                const std::unordered_set<const Object*>& initialized) {
        if (const auto* literal = condition.As<IntImmNode>()) {
            if ((literal->value != 0) == truth) Expression(value, initialized);
            return;
        }
        const auto* comparison = condition.As<LTNode>();
        if (comparison && PartitionedBranch(value, comparison, truth, initialized)) return;
        int64_t constant = 0;
        bool reversed = false;
        PrimExpr coordinate;
        if (comparison) {
            coordinate = comparison->a;
            if (!internal::EvaluateStaticInt64(comparison->b, &constant)) {
                if (internal::EvaluateStaticInt64(comparison->a, &constant)) {
                    coordinate = comparison->b; reversed = true;
                } else coordinate = PrimExpr();
            }
        }
        coordinate = WidenedCoordinate(coordinate);
        const auto variable = variables_.find(coordinate.get());
        if (variable == variables_.end() || upper_.at(variable->second).size() != 1 ||
            !upper_.at(variable->second).begin()->first.empty()) {
            Expression(value, initialized); return;
        }
        const size_t atom = variable->second;
        const int64_t old_min = minima_[atom], old_max = maxima_[atom];
        const auto old_upper = upper_[atom];
        if (reversed ? truth : !truth) {
            if (reversed && constant == INT64_MAX) return;
            minima_[atom] = std::max(old_min, reversed ? constant + 1 : constant);
        } else {
            if (!reversed && constant == INT64_MIN) return;
            maxima_[atom] = std::min(old_max, reversed ? constant : constant - 1);
        }
        if (minima_[atom] <= maxima_[atom]) {
            upper_[atom] = Constant(maxima_[atom]);
            // Quotient atoms created under this narrower coordinate range
            // cannot supply bounds to its sibling branch or later loads.
            auto outer_index_atoms = index_atoms_;
            Expression(value, initialized);
            index_atoms_ = std::move(outer_index_atoms);
        }
        minima_[atom] = old_min; maxima_[atom] = old_max; upper_[atom] = old_upper;
    }
    void CheckExpression(const PrimExpr& expr, const std::unordered_set<const Object*>& initialized) {
        needs_read_range_ = false;
        Expression(expr, initialized);
        if (!needs_read_range_) return;
        auto writes = writers_;
        writes.insert(outputs_.begin(), outputs_.end());
        CheckBoundedCudaReadValue(function_, expr, writes, {
            [&](const LoadNode* load) { return Read(load, initialized); },
            [&](const PrimExpr& a, const PrimExpr& b) { return Parse(a) == Parse(b); },
            [&](const PrimExpr& value) -> std::optional<std::pair<int64_t, int64_t>> {
                const auto parsed = TryParse(value);
                if (!parsed) return std::nullopt;
                return std::make_pair(Minimum(*parsed), Maximum(*parsed));
            }});
    }
    void Statement(const Stmt& stmt, std::unordered_set<const Object*>* initialized) {
        if (const auto* store = stmt.As<StoreNode>()) {
            if (!outputs_.count(store->buffer_var.get()) || store->value.dtype() != store->buffer_var->dtype ||
                Parse(store->index) != owned_ ||
                (store->predicate.defined() && !pass_utils::IsConstOne(store->predicate))) Reject();
            CheckExpression(store->value, *initialized);
            if (writers_.count(store->buffer_var.get())) ProjectAddress(store, store->buffer_var, store->index);
            initialized->insert(store->buffer_var.get());
        } else if (const auto* loop = stmt.As<ForNode>()) {
            const auto extent = Loop(loop);
            if (Maximum(extent) == 0) { empty_loops_.insert(loop); return; }
            BindAxis(loop->loop_var, extent);
            auto inside = *initialized;
            Statement(loop->body, &inside);
            // Only a known positive constant trip count guarantees a store.
            if (extent.size() == 1 && extent.begin()->first.empty() && extent.begin()->second > 0) *initialized = std::move(inside);
            variables_.erase(loop->loop_var.get());
            bound_.erase(loop->loop_var.get());
        } else if (const auto* let = stmt.As<LetStmtNode>()) {
            CheckExpression(let->value, *initialized);
            if (!bound_.insert(let->var.get()).second) Reject();
            Statement(let->body, initialized);
            bound_.erase(let->var.get());
        } else if (const auto* sequence = stmt.As<SeqStmtNode>()) {
            for (const auto& child : sequence->seq) Statement(child, initialized);
        } else Reject();
    }
    const PrimFunc& function_;
    size_t scalar_count_{0};
    int64_t output_start_{0};
    std::unordered_map<const Object*, size_t> scalars_, variables_;
    std::unordered_map<const Object*, Polynomial> elements_;
    std::unordered_map<const Object*, Polynomial> branch_values_;
    std::unordered_map<const Object*, Array<PrimExpr>> shapes_;
    std::unordered_map<const Object*, PrimExpr> addresses_;
    std::unordered_set<const Object*> bound_, outputs_;
    std::unordered_set<const Object*> scratch_, writers_, completed_, empty_loops_;
    std::vector<Var> row_axes_;
    std::vector<int64_t> maxima_, minima_;
    std::vector<Polynomial> upper_;
    std::map<std::tuple<Polynomial, Polynomial, bool>, size_t> index_atoms_;
    Polynomial owned_;
    bool needs_read_range_{false};
};

// A separate read proof complements output ownership without changing indices
// or introducing a runtime fallback. Only branches proved unreachable are skipped.
class CudaIndirectReadProof final {
    struct Range { int64_t low, high; };
    struct Facts { std::unordered_map<size_t, Range> ranges; bool live{true}; };
    struct Atom {
        const Object* buffer;
        std::vector<int64_t> coefficients;
        int64_t constant;
        Range limits;
        PrimExpr index;
    };
public:
    CudaIndirectReadProof(const PrimFunc& function, std::vector<DataAxis> domain,
                         const std::unordered_set<const Object*>& writes,
                         const CudaReadProofHooks* hooks = nullptr)
        : function_(function), domain_(std::move(domain)), writes_(writes), hooks_(hooks) {}

    void Check(const Stmt& stmt) { CheckStatement(stmt); }
    void CheckValue(const PrimExpr& value) { CheckExpr(value, Facts{}); }

private:
    [[noreturn]] static void Reject() {
        throw std::invalid_argument("BindCudaThreads cannot prove bounded read-only indirect Load index expressions");
    }
    static Range Limits(DataType dtype) {
        if (dtype.lanes != 1 || (dtype.code != 0 && dtype.code != 1) ||
            (dtype.bits != 32 && dtype.bits != 64)) Reject();
        if (dtype.code == 1) return {0, dtype.bits == 32 ? int64_t(UINT32_MAX) : INT64_MAX};
        return dtype.bits == 32 ? Range{INT32_MIN, INT32_MAX} : Range{INT64_MIN, INT64_MAX};
    }
    static Range Fit(Range value, DataType dtype) {
        const Range limit = Limits(dtype);
        if (value.low < limit.low || value.high > limit.high) Reject();
        return value;
    }
    static int64_t AddSigned(int64_t a, int64_t b) {
        if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) Reject();
        return a + b;
    }
    static int64_t SubSigned(int64_t a, int64_t b) {
        if ((b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b)) Reject();
        return a - b;
    }
    static int64_t MulSigned(int64_t a, int64_t b) {
        if (a == 0 || b == 0) return 0;
        if (a > 0 ? (b > 0 ? a > INT64_MAX / b : b < INT64_MIN / a)
                  : (b > 0 ? a < INT64_MIN / b : a < INT64_MAX / b)) Reject();
        return a * b;
    }
    int64_t Elements(const Var& var) const {
        if (!function_->buffer_map.count(var)) Reject();
        const auto& buffer = function_->buffer_map.at(var);
        if (buffer->dtype != var->dtype || !buffer->strides.empty() ||
            !pass_utils::IsConstZero(buffer->elem_offset)) Reject();
        int64_t count = 1;
        for (const auto& dimension : buffer->shape) {
            int64_t extent = 0;
            if (!internal::EvaluateStaticInt64(dimension, &extent) || extent < 0) Reject();
            count = CheckedMultiply(count, extent);
        }
        return count;
    }
    size_t ReadAtom(const LoadNode* load) {
        if (load->dtype.code != 0 || writes_.count(load->buffer_var.get()) ||
            (load->predicate.defined() && !pass_utils::IsConstOne(load->predicate))) Reject();
        const Range limits = Limits(load->dtype);
        if (hooks_) {
            if (!hooks_->direct_read(load)) Reject();
            for (size_t i = 0; i < atoms_.size(); ++i) {
                if (atoms_[i].buffer == load->buffer_var.get() &&
                    hooks_->same_index(atoms_[i].index, load->index)) return i;
            }
            atoms_.push_back({load->buffer_var.get(), {}, 0, limits, load->index});
            return atoms_.size() - 1;
        }
        std::vector<int64_t> coefficients;
        int64_t constant = 0;
        if (!ReadOutputIndex(load->index, domain_, &coefficients, &constant)) Reject();
        int64_t maximum = constant;
        for (size_t i = 0; i < domain_.size(); ++i) {
            maximum = CheckedAdd(maximum, CheckedMultiply(coefficients[i],
                CheckedAdd(domain_[i].minimum, domain_[i].extent - 1)));
        }
        if (maximum >= Elements(load->buffer_var)) Reject();
        for (size_t i = 0; i < atoms_.size(); ++i) {
            if (atoms_[i].buffer == load->buffer_var.get() && atoms_[i].constant == constant &&
                atoms_[i].coefficients == coefficients) return i;
        }
        atoms_.push_back({load->buffer_var.get(), std::move(coefficients), constant, limits, load->index});
        return atoms_.size() - 1;
    }
    Range Lookup(const Facts& facts, size_t atom) const {
        const auto found = facts.ranges.find(atom);
        return found == facts.ranges.end() ? atoms_[atom].limits : found->second;
    }
    Facts Join(const Facts& a, const Facts& b) const {
        if (!a.live) return b;
        if (!b.live) return a;
        Facts result;
        for (size_t atom = 0; atom < atoms_.size(); ++atom) {
            const Range x = Lookup(a, atom), y = Lookup(b, atom);
            result.ranges[atom] = {std::min(x.low, y.low), std::max(x.high, y.high)};
        }
        return result;
    }
    Facts Assume(const PrimExpr& condition, bool truth, Facts facts) {
        if (!facts.live || !condition.defined()) return facts;
        if (const auto* value = condition.As<IntImmNode>()) {
            facts.live = (value->value != 0) == truth;
            return facts;
        }
        if (const auto* logical = condition.As<NotNode>()) return Assume(logical->value, !truth, facts);
        if (const auto* logical = condition.As<AndNode>()) {
            return truth ? Assume(logical->b, true, Assume(logical->a, true, facts))
                         : Join(Assume(logical->a, false, facts), Assume(logical->b, false, facts));
        }
        if (const auto* logical = condition.As<OrNode>()) {
            return truth ? Join(Assume(logical->a, true, facts), Assume(logical->b, true, facts))
                         : Assume(logical->b, false, Assume(logical->a, false, facts));
        }
        const auto* comparison = condition.As<LTNode>();
        if (!comparison || comparison->a.dtype().code != 0 || comparison->b.dtype().code != 0) return facts;
        PrimExpr expression = comparison->a;
        int64_t constant = 0;
        bool reversed = false;
        if (!internal::EvaluateStaticInt64(comparison->b, &constant)) {
            if (!internal::EvaluateStaticInt64(comparison->a, &constant)) return facts;
            expression = comparison->b;
            reversed = true;
        }
        while (const auto* cast = expression.As<CallNode>()) {
            if (cast->name != "cast" || cast->args.size() != 1 || cast->dtype.code != 0 ||
                cast->args[0].dtype().code != 0 || cast->dtype.bits < cast->args[0].dtype().bits) return facts;
            expression = cast->args[0];
        }
        const auto* load = expression.As<LoadNode>();
        if (!load) return facts;
        const size_t atom = ReadAtom(load);
        Range range = Lookup(facts, atom);
        if (reversed ? truth : !truth) {
            if (reversed && constant == INT64_MAX) { facts.live = false; return facts; }
            range.low = std::max(range.low, reversed ? constant + 1 : constant);
        } else {
            if (!reversed && constant == INT64_MIN) { facts.live = false; return facts; }
            range.high = std::min(range.high, reversed ? constant : constant - 1);
        }
        facts.ranges[atom] = range;
        facts.live = range.low <= range.high;
        return facts;
    }
    Range Evaluate(const PrimExpr& expr, const Facts& facts) {
        if (!facts.live) Reject();
        if (hooks_) {
            const auto range = hooks_->integer_bounds(expr);
            if (range) return Fit({range->first, range->second}, expr.dtype());
        }
        if (const auto* integer = expr.As<IntImmNode>()) return Fit({integer->value, integer->value}, expr.dtype());
        for (const auto& axis : domain_) {
            if (expr.get() == axis.var.get()) return Fit({axis.minimum,
                CheckedAdd(axis.minimum, axis.extent - 1)}, expr.dtype());
        }
        if (const auto* load = expr.As<LoadNode>()) return Lookup(facts, ReadAtom(load));
        if (const auto* cast = expr.As<CallNode>()) {
            if (cast->name != "cast" || cast->args.size() != 1) Reject();
            return Fit(Evaluate(cast->args[0], facts), cast->dtype);
        }
        if (const auto* select = expr.As<SelectNode>()) {
            const Facts yes = Assume(select->condition, true, facts), no = Assume(select->condition, false, facts);
            if (!yes.live) return Fit(Evaluate(select->false_value, no), expr.dtype());
            if (!no.live) return Fit(Evaluate(select->true_value, yes), expr.dtype());
            const Range a = Evaluate(select->true_value, yes), b = Evaluate(select->false_value, no);
            return Fit({std::min(a.low, b.low), std::max(a.high, b.high)}, expr.dtype());
        }
        const auto* binary = expr.As<BinaryOpNode>();
        if (!binary || binary->a.dtype().code != binary->b.dtype().code) Reject();
        const Range a = Evaluate(binary->a, facts), b = Evaluate(binary->b, facts);
        // Check both the TIR type and the native C++ arithmetic width.
        DataType native = binary->a.dtype();
        native.bits = std::max(binary->a.dtype().bits, binary->b.dtype().bits);
        const auto result = [&](Range range) { return Fit(Fit(range, native), expr.dtype()); };
        if (expr.As<AddNode>()) return result({AddSigned(a.low, b.low), AddSigned(a.high, b.high)});
        if (expr.As<SubNode>()) return result({SubSigned(a.low, b.high), SubSigned(a.high, b.low)});
        if (expr.As<MulNode>()) {
            const int64_t values[] = {MulSigned(a.low, b.low), MulSigned(a.low, b.high),
                                     MulSigned(a.high, b.low), MulSigned(a.high, b.high)};
            return result({*std::min_element(values, values + 4), *std::max_element(values, values + 4)});
        }
        if (b.low != b.high || b.low <= 0) Reject();
        if (expr.As<DivNode>()) return result({a.low / b.low, a.high / b.low});
        if (expr.As<ModNode>() && a.low >= 0) {
            return result(a.low / b.low == a.high / b.low
                ? Range{a.low % b.low, a.high % b.low} : Range{0, b.low - 1});
        }
        Reject();
    }
    void CheckExpr(const PrimExpr& expr, const Facts& facts) {
        if (!expr.defined() || !facts.live) return;
        if (const auto* load = expr.As<LoadNode>()) {
            CheckExpr(load->predicate, facts);
            const Facts active = Assume(load->predicate, true, facts);
            if (!active.live) return;
            CheckExpr(load->index, active);
            if (hooks_ && hooks_->direct_read(load)) return;
            const Range index = Evaluate(load->index, active);
            if (index.low < 0 || index.high >= Elements(load->buffer_var)) Reject();
            return;
        }
        if (const auto* select = expr.As<SelectNode>()) {
            CheckExpr(select->condition, facts);
            CheckExpr(select->true_value, Assume(select->condition, true, facts));
            CheckExpr(select->false_value, Assume(select->condition, false, facts));
            return;
        }
        if (const auto* logical = expr.As<NotNode>()) { CheckExpr(logical->value, facts); return; }
        if (const auto* binary = expr.As<BinaryOpNode>()) {
            CheckExpr(binary->a, facts);
            Facts right = facts;
            if (expr.As<AndNode>()) right = Assume(binary->a, true, facts);
            if (expr.As<OrNode>()) right = Assume(binary->a, false, facts);
            CheckExpr(binary->b, right);
            if (expr.dtype().code != 2 && expr.dtype().bits != 1) (void)Evaluate(expr, facts);
            return;
        }
        if (const auto* call = expr.As<CallNode>()) {
            for (const auto& argument : call->args) CheckExpr(argument, facts);
            if (call->dtype.code != 2 && call->dtype.bits != 1) (void)Evaluate(expr, facts);
            return;
        }
        if (expr.As<IntImmNode>() && expr.dtype().bits != 1) (void)Evaluate(expr, facts);
    }
    void CheckStatement(const Stmt& stmt) {
        if (const auto* store = stmt.As<StoreNode>()) {
            CheckExpr(store->predicate, Facts{});
            CheckExpr(store->value, Assume(store->predicate, true, Facts{}));
        } else if (const auto* loop = stmt.As<ForNode>()) {
            const auto range = StaticLoopDomain(loop, false);
            if (range.second == 0) return;
            domain_.push_back({loop->loop_var, range.first, range.second, 1});
            CheckStatement(loop->body);
            domain_.pop_back();
        } else if (const auto* let = stmt.As<LetStmtNode>()) {
            CheckExpr(let->value, Facts{});
            CheckStatement(let->body);
        } else if (const auto* sequence = stmt.As<SeqStmtNode>()) {
            for (const auto& child : sequence->seq) CheckStatement(child);
        } else Reject();
    }
    const PrimFunc& function_;
    std::vector<DataAxis> domain_;
    const std::unordered_set<const Object*>& writes_;
    std::vector<Atom> atoms_;
    const CudaReadProofHooks* hooks_{nullptr};
};

void CheckBoundedCudaReadValue(const PrimFunc& function, const PrimExpr& value,
    const std::unordered_set<const Object*>& writes, const CudaReadProofHooks& hooks) {
    CudaIndirectReadProof(function, {}, writes, &hooks).CheckValue(value);
}

ProvenIteration ProveSingleStage(const PrimFunc& function) {
    Stmt iteration = function->body;
    std::vector<DataAxis> axes;
    std::unordered_set<const Object*> bound_vars;
    for (const auto& parameter : function->params) bound_vars.insert(parameter.get());
    while (const auto* loop = iteration.As<ForNode>()) {
        int64_t extent = 0;
        if (internal::EvaluateStaticInt64(loop->extent, &extent) && extent == 0) {
            (void)StaticLoopDomain(loop, false);
            // A perfect outer nest containing an empty loop executes no body.
            // Keep an explicit guarded CUDA launch, including for null 0-byte buffers.
            return {Evaluate(IntImm(0)), {}, 0};
        }
        const auto domain = StaticLoopDomain(loop, true);
        if (!bound_vars.insert(loop->loop_var.get()).second) {
            throw std::invalid_argument("BindCudaThreads rejects rebound data variables");
        }
        axes.push_back({loop->loop_var, domain.first, domain.second, 1});
        iteration = loop->body;
    }
    int64_t work_size = 1, first_output = 0;
    for (auto axis = axes.rbegin(); axis != axes.rend(); ++axis) {
        axis->stride = work_size;
        first_output = CheckedAdd(first_output, CheckedMultiply(axis->minimum, axis->stride));
        work_size = CheckedMultiply(work_size, axis->extent);
    }
    const int64_t output_end = CheckedAdd(first_output, work_size);

    std::unordered_set<const Object*> writes;
    size_t store_count = 0;
    CollectIndependentWrites(iteration, axes, &writes, &bound_vars, &store_count);
    if (store_count == 0) {
        throw std::invalid_argument("BindCudaThreads requires at least one Store");
    }
    for (const Object* written : writes) {
        bool found = false;
        for (const auto& item : function->buffer_map) {
            if (item.first.get() != written) continue;
            found = true;
            int64_t elements = 1;
            for (const PrimExpr& dimension : item.second->shape) {
                int64_t extent = 0;
                if (!internal::EvaluateStaticInt64(dimension, &extent) || extent < 0) {
                    throw std::invalid_argument("BindCudaThreads requires static output buffers");
                }
                elements = CheckedMultiply(elements, extent);
            }
            if (!item.second->strides.empty() ||
                !pass_utils::IsConstZero(item.second->elem_offset) || elements < output_end) {
                throw std::invalid_argument("BindCudaThreads output address exceeds its dense buffer");
            }
        }
        if (!found) throw std::invalid_argument("BindCudaThreads written buffer has no parameter contract");
    }
    std::unordered_set<const Object*> initialized;
    ValidateOwnedReads(iteration, writes, axes, &initialized);
    // Existing direct-read schedules retain their proof; guarded indirect reads
    // additionally require every read and integer address expression to be safe.
    if (HasIndirectLoad(iteration)) {
        CudaIndirectReadProof(function, axes, writes).Check(iteration);
    }

    return {iteration, std::move(axes), work_size};
}

struct BufferDomain {
    Var var;
    DataType dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t elements{1};
    bool internal{false};
};

using BufferDomains = std::unordered_map<const Object*, BufferDomain>;

BufferDomain ReadBufferDomain(const Var& var, DataType dtype,
                              const Array<PrimExpr>& shape, bool internal_buffer) {
    BufferDomain result{var, dtype, {}, {}, 1, internal_buffer};
    if (dtype.lanes != 1 || dtype.bits == 0) {
        throw std::invalid_argument("BindCudaThreads requires scalar buffer elements");
    }
    for (const PrimExpr& dimension : shape) {
        std::vector<int64_t> unused;
        int64_t extent = 0;
        if (!ReadOutputIndex(dimension, {}, &unused, &extent) || extent <= 0) {
            throw std::invalid_argument("BindCudaThreads requires positive static stage buffers");
        }
        result.shape.push_back(extent);
    }
    result.strides.resize(result.shape.size());
    for (size_t i = result.shape.size(); i != 0; --i) {
        result.strides[i - 1] = result.elements;
        result.elements = CheckedMultiply(result.elements, result.shape[i - 1]);
    }
    return result;
}

struct AffineAddress {
    std::vector<int64_t> coefficients;
    int64_t constant{0};
};

AffineAddress ReadBoundedAddress(const PrimExpr& expr, const std::vector<DataAxis>& domain,
                                 int64_t elements) {
    AffineAddress result;
    if (!ReadOutputIndex(expr, domain, &result.coefficients, &result.constant)) {
        throw std::invalid_argument("BindCudaThreads cannot prove a stage buffer address");
    }
    int64_t maximum = result.constant;
    for (size_t i = 0; i < domain.size(); ++i) {
        maximum = CheckedAdd(maximum, CheckedMultiply(result.coefficients[i],
            CheckedAdd(domain[i].minimum, domain[i].extent - 1)));
    }
    if (maximum >= elements) {
        throw std::invalid_argument("BindCudaThreads stage address exceeds its buffer");
    }
    return result;
}

struct CudaStage {
    Stmt body;
    Stmt iteration;
    std::vector<DataAxis> axes;
    Var output;
};

// Each stage must completely define one tensor, with only owned-cell recurrence.
CudaStage ReadStage(const Stmt& body, const BufferDomains& buffers) {
    CudaStage stage{body, body, {}, {}};
    std::unordered_set<const Object*> bound;
    for (const auto& item : buffers) bound.insert(item.first);
    while (const auto* loop = stage.iteration.As<ForNode>()) {
        const auto domain = StaticLoopDomain(loop, true);
        if (domain.first != 0 || !bound.insert(loop->loop_var.get()).second) {
            throw std::invalid_argument("BindCudaThreads requires distinct zero-based stage axes");
        }
        stage.axes.push_back({loop->loop_var, 0, domain.second, 1});
        stage.iteration = loop->body;
    }
    int64_t elements = 1;
    for (auto axis = stage.axes.rbegin(); axis != stage.axes.rend(); ++axis) {
        axis->stride = elements;
        elements = CheckedMultiply(elements, axis->extent);
    }
    std::unordered_set<const Object*> writes;
    size_t stores = 0;
    CollectIndependentWrites(stage.iteration, stage.axes, &writes, &bound, &stores);
    if (writes.size() != 1 || buffers.count(*writes.begin()) == 0) {
        throw std::invalid_argument("BindCudaThreads requires one declared tensor writer per stage");
    }
    const auto& buffer = buffers.at(*writes.begin());
    if (buffer.shape.size() != stage.axes.size() || elements != buffer.elements) {
        throw std::invalid_argument("BindCudaThreads stage loops must cover the declared buffer");
    }
    for (size_t i = 0; i < stage.axes.size(); ++i) {
        if (stage.axes[i].extent != buffer.shape[i]) {
            throw std::invalid_argument("BindCudaThreads stage layout does not match its buffer");
        }
    }
    std::unordered_set<const Object*> initialized;
    ValidateOwnedReads(stage.iteration, writes, stage.axes, &initialized);
    if (initialized.count(buffer.var.get()) == 0) {
        throw std::invalid_argument("BindCudaThreads stage does not initialize every output cell");
    }
    RejectIndirectLoads(stage.iteration);
    stage.output = buffer.var;
    return stage;
}

// Eliminate only input copies/casts. A reduction or computed intermediate is
// never duplicated by this inliner; all producer inputs must remain read-only.
class InlineCudaInputs final : public TIRPass {
public:
    InlineCudaInputs(const std::unordered_map<const Object*, CudaStage>& copies,
                     const BufferDomains& buffers, const std::unordered_set<const Object*>& writers,
                     const std::unordered_set<const Object*>& completed, const Var& output)
        : copies_(copies), buffers_(buffers), writers_(writers), completed_(completed), output_(output) {}

protected:
    Stmt VisitFor(const ForNode* op, const Stmt& ref) override {
        const auto range = StaticLoopDomain(op, false);
        if (range.second <= 0) {
            throw std::invalid_argument("BindCudaThreads requires positive multi-stage loops");
        }
        domain_.push_back({op->loop_var, range.first, range.second, 1});
        Stmt result = TIRPass::VisitFor(op, ref);
        domain_.pop_back();
        return result;
    }

    PrimExpr VisitLoad(const LoadNode* op, const PrimExpr& ref) override {
        if (buffers_.count(op->buffer_var.get()) == 0 ||
            (writers_.count(op->buffer_var.get()) && op->buffer_var.get() != output_.get() &&
             completed_.count(op->buffer_var.get()) == 0)) {
            throw std::invalid_argument("BindCudaThreads reads an undeclared or incomplete stage producer");
        }
        const auto found = copies_.find(op->buffer_var.get());
        if (found == copies_.end()) return TIRPass::VisitLoad(op, ref);
        if (op->predicate.defined() && !pass_utils::IsConstOne(op->predicate)) {
            throw std::invalid_argument("BindCudaThreads cannot inline a predicated scratch read");
        }
        const auto& stage = found->second;
        const auto& buffer = buffers_.at(op->buffer_var.get());
        (void)ReadBoundedAddress(op->index, domain_, buffer.elements);
        const DataType i64 = DataType::Int(64);
        const PrimExpr index = op->index.dtype() == i64
            ? op->index : PrimExpr(Call(i64, "cast", {op->index}));
        std::vector<std::pair<Var, PrimExpr>> replacements;
        for (size_t i = 0; i < stage.axes.size(); ++i) {
            PrimExpr coordinate = (index / IntImm(buffer.strides[i], i64)) %
                                  IntImm(buffer.shape[i], i64);
            if (coordinate.dtype() != stage.axes[i].var->dtype) {
                coordinate = Call(stage.axes[i].var->dtype, "cast", {coordinate});
            }
            replacements.emplace_back(stage.axes[i].var, coordinate);
        }
        CudaBodyRewriter substitute(std::move(replacements));
        return substitute.Mutate(stage.iteration.As<StoreNode>()->value);
    }

private:
    const std::unordered_map<const Object*, CudaStage>& copies_;
    const BufferDomains& buffers_;
    const std::unordered_set<const Object*>& writers_;
    const std::unordered_set<const Object*>& completed_;
    Var output_;
    std::vector<DataAxis> domain_;
};

// Project one proven stage onto a thread's independent coordinates. Mixed-radix
// digits are range-checked separately, so a local offset cannot carry into the
// neighboring thread's coordinate even when the flat address stays in bounds.
class CudaStageProjector final : public TIRPass {
public:
    CudaStageProjector(const CudaStage& stage, const BufferDomains& buffers,
                       const std::unordered_set<const Object*>& writers,
                       const std::unordered_set<const Object*>& completed,
                       const std::vector<Var>& rows)
        : stage_(stage), buffers_(buffers), writers_(writers),
          completed_(completed), rows_(rows), domain_(stage.axes) {}

protected:
    PrimExpr VisitVar(const VarNode* op, const PrimExpr& ref) override {
        for (size_t i = 0; i < stage_.axes.size(); ++i) {
            if (rows_[i].defined() && stage_.axes[i].var.get() == op) return rows_[i];
        }
        return ref;
    }

    Stmt VisitFor(const ForNode* op, const Stmt& ref) override {
        (void)ref;
        const auto range = StaticLoopDomain(op, false);
        if (range.second <= 0) {
            throw std::invalid_argument("BindCudaThreads requires positive multi-stage loops");
        }
        for (size_t i = 0; i < stage_.axes.size(); ++i) {
            if (stage_.axes[i].var.get() != op->loop_var.get()) continue;
            if (rows_[i].defined()) return Mutate(op->body);
            return For(op->loop_var, IntImm(range.first, op->loop_var->dtype),
                       IntImm(range.second, op->loop_var->dtype), op->for_type, Mutate(op->body));
        }
        domain_.push_back({op->loop_var, range.first, range.second, 1});
        Stmt body = Mutate(op->body);
        domain_.pop_back();
        return For(op->loop_var, IntImm(range.first, op->loop_var->dtype),
                   IntImm(range.second, op->loop_var->dtype), op->for_type, body);
    }

    PrimExpr VisitLoad(const LoadNode* op, const PrimExpr& ref) override {
        if (buffers_.count(op->buffer_var.get()) == 0) {
            throw std::invalid_argument("BindCudaThreads found an undeclared buffer read");
        }
        if (writers_.count(op->buffer_var.get()) == 0) return TIRPass::VisitLoad(op, ref);
        if (op->buffer_var.get() != stage_.output.get() &&
            completed_.count(op->buffer_var.get()) == 0) {
            throw std::invalid_argument("BindCudaThreads stage reads a producer before completion");
        }
        return Load(op->buffer_var, ProjectAddress(op->index, buffers_.at(op->buffer_var.get())),
                    Mutate(op->predicate));
    }

    Stmt VisitStore(const StoreNode* op, const Stmt& ref) override {
        (void)ref;
        return Store(op->buffer_var, Mutate(op->value),
                     ProjectAddress(op->index, buffers_.at(op->buffer_var.get())),
                     Mutate(op->predicate));
    }

private:
    PrimExpr ProjectAddress(const PrimExpr& expr, const BufferDomain& buffer) {
        AffineAddress address = ReadBoundedAddress(expr, domain_, buffer.elements);
        for (size_t i = 0; i < rows_.size(); ++i) {
            if (!rows_[i].defined()) continue;
            if (address.coefficients[i] != buffer.strides[i]) {
                throw std::invalid_argument("BindCudaThreads stage reads another thread's coordinate");
            }
            address.coefficients[i] = 0;
        }
        int64_t private_stride = 1, private_constant = 0;
        std::vector<int64_t> private_coefficients(domain_.size(), 0);
        for (size_t reverse = buffer.shape.size(); reverse != 0; --reverse) {
            const size_t axis = reverse - 1;
            const int64_t digit = (address.constant / buffer.strides[axis]) % buffer.shape[axis];
            int64_t coordinate_maximum = digit;
            if (rows_[axis].defined() && digit != 0) {
                throw std::invalid_argument("BindCudaThreads scratch offset crosses a thread boundary");
            }
            if (!rows_[axis].defined()) {
                private_constant = CheckedAdd(private_constant, CheckedMultiply(digit, private_stride));
            }
            for (size_t i = 0; i < domain_.size(); ++i) {
                const int64_t coefficient =
                    (address.coefficients[i] / buffer.strides[axis]) % buffer.shape[axis];
                if (rows_[axis].defined() && coefficient != 0) {
                    throw std::invalid_argument("BindCudaThreads scratch index crosses a thread boundary");
                }
                coordinate_maximum = CheckedAdd(coordinate_maximum, CheckedMultiply(coefficient,
                    CheckedAdd(domain_[i].minimum, domain_[i].extent - 1)));
                if (!rows_[axis].defined()) {
                    private_coefficients[i] = CheckedAdd(private_coefficients[i],
                        CheckedMultiply(coefficient, private_stride));
                }
            }
            if (coordinate_maximum >= buffer.shape[axis]) {
                throw std::invalid_argument("BindCudaThreads scratch coordinate may carry into another axis");
            }
            if (!rows_[axis].defined()) private_stride = CheckedMultiply(private_stride, buffer.shape[axis]);
        }
        if (!buffer.internal) return Mutate(expr);
        const DataType i64 = DataType::Int(64);
        PrimExpr result = IntImm(private_constant, i64);
        for (size_t i = 0; i < domain_.size(); ++i) {
            if (private_coefficients[i] == 0) continue;
            PrimExpr coordinate = domain_[i].var;
            if (coordinate.dtype() != i64) coordinate = Call(i64, "cast", {coordinate});
            result = result + coordinate * IntImm(private_coefficients[i], i64);
        }
        return result;
    }

    const CudaStage& stage_;
    const BufferDomains& buffers_;
    const std::unordered_set<const Object*>& writers_;
    const std::unordered_set<const Object*>& completed_;
    const std::vector<Var>& rows_;
    std::vector<DataAxis> domain_;
};

// SeqStmt flattens the init/update of scalar reductions into adjacent children.
// Recover only a contiguous scalar producer; ReadStage still proves initialization.
Var ScalarStageWriter(const Stmt& stmt, const BufferDomains& buffers) {
    if (const auto* loop = stmt.As<ForNode>()) return ScalarStageWriter(loop->body, buffers);
    if (const auto* let = stmt.As<LetStmtNode>()) return ScalarStageWriter(let->body, buffers);
    if (const auto* store = stmt.As<StoreNode>()) {
        const auto found = buffers.find(store->buffer_var.get());
        return found != buffers.end() && found->second.shape.empty() ? store->buffer_var : Var();
    }
    if (const auto* sequence = stmt.As<SeqStmtNode>()) {
        Var writer;
        for (const auto& child : sequence->seq) {
            Var current = ScalarStageWriter(child, buffers);
            if (!current.defined() || (writer.defined() && writer.get() != current.get())) return Var();
            writer = current;
        }
        return writer;
    }
    return Var();
}

ProvenIteration ProveMultiStage(const PrimFunc& function) {
    BufferDomains buffers;
    for (const auto& parameter : function->params) {
        if (!function->buffer_map.count(parameter)) {
            throw std::invalid_argument("BindCudaThreads requires buffer parameters");
        }
        const auto& buffer = function->buffer_map.at(parameter);
        if (!buffer->strides.empty() || !pass_utils::IsConstZero(buffer->elem_offset)) {
            throw std::invalid_argument("BindCudaThreads requires dense stage buffer contracts");
        }
        if (!buffers.emplace(parameter.get(), ReadBufferDomain(parameter, buffer->dtype,
                                                               buffer->shape, false)).second) {
            throw std::invalid_argument("BindCudaThreads rejects duplicate parameters");
        }
    }
    std::vector<Var> allocations;
    Stmt body = function->body;
    while (const auto* allocation = body.As<AllocateNode>()) {
        if (!pass_utils::IsConstOne(allocation->condition) ||
            !buffers.emplace(allocation->buffer_var.get(),
                ReadBufferDomain(allocation->buffer_var, allocation->dtype,
                                 allocation->extents, true)).second) {
            throw std::invalid_argument("BindCudaThreads requires distinct unconditional scratch buffers");
        }
        allocations.push_back(allocation->buffer_var);
        body = allocation->body;
    }
    const auto* sequence = body.As<SeqStmtNode>();
    if (allocations.empty() || !sequence || sequence->seq.size() < 2) {
        throw std::invalid_argument("BindCudaThreads requires an explicit sequence of complete stages");
    }
    std::vector<CudaStage> stages;
    std::unordered_set<const Object*> writers;
    bool public_output = false;
    for (size_t index = 0; index < sequence->seq.size(); ++index) {
        Stmt statement = sequence->seq[index];
        const Var scalar = ScalarStageWriter(statement, buffers);
        if (scalar.defined()) {
            Array<Stmt> group{statement};
            while (index + 1 < sequence->seq.size() &&
                   ScalarStageWriter(sequence->seq[index + 1], buffers).get() == scalar.get()) {
                group.push_back(sequence->seq[++index]);
            }
            if (group.size() > 1) statement = SeqStmt(group);
        }
        CudaStage stage;
        try {
            stage = ReadStage(statement, buffers);
        } catch (const std::invalid_argument& error) {
            throw std::invalid_argument("BindCudaThreads stage " + std::to_string(stages.size()) +
                                        ": " + error.what());
        }
        if (!writers.insert(stage.output.get()).second) {
            throw std::invalid_argument("BindCudaThreads requires one producer stage per buffer");
        }
        public_output |= !buffers.at(stage.output.get()).internal;
        stages.push_back(std::move(stage));
    }
    if (!public_output) throw std::invalid_argument("BindCudaThreads stages have no public output");
    for (const auto& allocation : allocations) {
        if (writers.count(allocation.get()) == 0) {
            throw std::invalid_argument("BindCudaThreads scratch buffer has no complete producer");
        }
    }
    std::unordered_map<const Object*, CudaStage> copies;
    for (const auto& stage : stages) {
        if (!buffers.at(stage.output.get()).internal) continue;
        const auto* store = stage.iteration.As<StoreNode>();
        if (!store || (store->predicate.defined() && !pass_utils::IsConstOne(store->predicate))) continue;
        PrimExpr value = store->value;
        if (const auto* cast = value.As<CallNode>()) {
            if (cast->name != "cast" || cast->args.size() != 1) continue;
            value = cast->args[0];
        }
        const auto* input = value.As<LoadNode>();
        if (input && buffers.count(input->buffer_var.get()) &&
            !buffers.at(input->buffer_var.get()).internal &&
            writers.count(input->buffer_var.get()) == 0 &&
            (!input->predicate.defined() || pass_utils::IsConstOne(input->predicate))) {
            copies.emplace(stage.output.get(), stage);
        }
    }
    std::vector<CudaStage> materialized;
    std::unordered_set<const Object*> original_completed;
    for (const auto& stage : stages) {
        InlineCudaInputs inliner(copies, buffers, writers, original_completed, stage.output);
        Stmt inlined = inliner.Mutate(stage.body);
        if (!copies.count(stage.output.get())) materialized.push_back(ReadStage(inlined, buffers));
        original_completed.insert(stage.output.get());
    }
    const auto& common_shape = buffers.at(materialized.back().output.get()).shape;
    std::vector<Var> rows(common_shape.size());
    for (const auto& stage : materialized) {
        if (buffers.at(stage.output.get()).shape.size() != common_shape.size()) {
            throw std::invalid_argument("BindCudaThreads cannot align the stage coordinate domains");
        }
    }
    std::vector<DataAxis> row_axes;
    for (size_t i = 0; i < common_shape.size(); ++i) {
        bool shared_coordinate = common_shape[i] > 1;
        for (const auto& stage : materialized) {
            shared_coordinate &= buffers.at(stage.output.get()).shape[i] == common_shape[i];
        }
        if (shared_coordinate) {
            rows[i] = Var("cuda_row_" + std::to_string(i), DataType::Int(64));
            row_axes.push_back({rows[i], 0, common_shape[i], 1});
        }
    }
    int64_t work_size = 1;
    for (auto axis = row_axes.rbegin(); axis != row_axes.rend(); ++axis) {
        axis->stride = work_size;
        work_size = CheckedMultiply(work_size, axis->extent);
    }
    Array<Stmt> projected;
    std::unordered_set<const Object*> completed;
    for (const auto& stage : materialized) {
        CudaStageProjector projector(stage, buffers, writers, completed, rows);
        projected.push_back(projector.Mutate(stage.body));
        completed.insert(stage.output.get());
    }
    body = SeqStmt(projected);
    // ponytail: bounded thread-private storage; cooperative/shared scheduling
    // is a separate optimization, not an unbounded local-memory fallback.
    int64_t scratch_bytes = 0;
    for (auto allocation = allocations.rbegin(); allocation != allocations.rend(); ++allocation) {
        if (copies.count(allocation->get())) continue;
        const auto& buffer = buffers.at(allocation->get());
        Array<PrimExpr> private_shape;
        int64_t elements = 1;
        for (size_t i = 0; i < buffer.shape.size(); ++i) {
            const int64_t extent = rows[i].defined() ? 1 : buffer.shape[i];
            private_shape.push_back(IntImm(extent, DataType::Int(64)));
            elements = CheckedMultiply(elements, extent);
        }
        scratch_bytes = CheckedAdd(scratch_bytes,
            CheckedMultiply(elements, (static_cast<int64_t>(buffer.dtype.bits) + 7) / 8));
        if (scratch_bytes > kMaxPrivateScratchBytes) {
            throw std::invalid_argument("BindCudaThreads private scratch exceeds the 64 KiB per-thread budget");
        }
        body = Allocate(buffer.var, buffer.dtype, private_shape, IntImm(1, DataType::Bool()), body);
    }
    return {body, std::move(row_axes), work_size};
}

// 深拷贝 attrs，避免 Map::Set 改写输入 PrimFunc 的共享节点。
Map<String, ObjectRef> CopyAttrs(const Map<String, ObjectRef>& attrs) {
    Map<String, ObjectRef> result;
    for (const auto& item : attrs) result.Set(item.first, item.second);
    return result;
}

// Target 必须是可用 CUDA 能力快照；CPU 或占位 capability 不得静默调度。
void ValidateCudaTarget(const Target& target) {
    if (!target.defined() || !target.As<TargetNode>() ||
        target->device_type != kCUDA || target->kind != "cuda" ||
        target->device_id < 0) {
        throw std::invalid_argument("BindCudaThreads requires a complete CUDA Target");
    }
    if (target->attrs.exists == 0 || target->attrs.max_threads_per_block <= 0 ||
        target->attrs.max_shared_memory_per_block < 0) {
        throw std::invalid_argument("CUDA Target is missing launch capabilities");
    }
}

}  // namespace

// 节点构造器只由强类型结果句柄调用并一次性接管两个同源对象。
CudaScheduleResultNode::CudaScheduleResultNode(
    PrimFunc prim_func, CudaLaunchConfig launch_config)
    : prim_func_(std::move(prim_func)),
      launch_config_(launch_config) {}

// 构造后立即验证 metadata attr 与独立访问器共享同一 Object 节点。
CudaScheduleResult::CudaScheduleResult(
    PrimFunc prim_func, CudaLaunchConfig launch_config) {
    SetData(new CudaScheduleResultNode(std::move(prim_func),
                                       launch_config));
    Validate();
}

// ObjectRef 恢复路径重新验证内容，防止同类型坏节点越过调度边界。
CudaScheduleResult::CudaScheduleResult(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<CudaScheduleResultNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain CudaScheduleResultNode");
    }
    if (defined()) Validate();
}

// 返回调度后函数的共享只读句柄。
PrimFunc CudaScheduleResult::prim_func() const { return operator->()->prim_func_; }

// 返回与函数 attr 同源的启动元数据句柄。
CudaLaunchConfig CudaScheduleResult::launch_config() const {
    return operator->()->launch_config_;
}

// 结果校验以 Object 身份保证 metadata 不存在两个可漂移事实来源。
void CudaScheduleResult::Validate() const {
    const auto* node = operator->();
    if (!node->prim_func_.defined() || !node->prim_func_.As<PrimFuncNode>()) {
        throw std::invalid_argument("CudaScheduleResult requires a PrimFunc");
    }
    const CudaLaunchConfig& config = node->launch_config_;
    if (config.grid_x == 0 || config.grid_y == 0 || config.grid_z == 0 ||
        config.block_x == 0 || config.block_y == 0 || config.block_z == 0) {
        throw std::invalid_argument(
            "CudaScheduleResult requires positive launch dimensions");
    }
    const String key(kCudaLaunchMetadataAttr);
    if (!node->prim_func_->attrs.count(key)) {
        throw std::invalid_argument(
            "CudaScheduleResult launch config attr is missing");
    }
    const auto* values =
        node->prim_func_->attrs.at(key).As<ArrayNode<int64_t>>();
    if (!values || values->data.size() != 7 ||
        values->data[0] != config.grid_x ||
        values->data[1] != config.grid_y ||
        values->data[2] != config.grid_z ||
        values->data[3] != config.block_x ||
        values->data[4] != config.block_y ||
        values->data[5] != config.block_z ||
        values->data[6] !=
            static_cast<int64_t>(config.dynamic_shared_memory_bytes)) {
        throw std::invalid_argument(
            "CudaScheduleResult launch config attr does not match result");
    }
}

// undefined 或其他 Object 类型不能冒充 CUDA 调度结果。
const CudaScheduleResultNode* CudaScheduleResult::operator->() const {
    const auto* node = As<CudaScheduleResultNode>();
    if (!node) throw std::runtime_error("undefined or invalid CudaScheduleResult");
    return node;
}

// Flatten independent data coordinates; reductions remain serial within the owner.
CudaScheduleResult BindCudaThreads(const PrimFunc& function, const Target& target) {
    if (!function.defined() || !function.As<PrimFuncNode>()) {
        throw std::invalid_argument("BindCudaThreads requires a PrimFunc");
    }
    ValidateCudaTarget(target);
    if (function->attrs.count(String(kCudaLaunchMetadataAttr))) {
        throw std::invalid_argument("PrimFunc is already CUDA thread-bound");
    }
    bool bounded = false;
    if (function->attrs.count(String("kxc.runtime_extent_count"))) {
        const auto* count = function->attrs.at(String("kxc.runtime_extent_count")).As<IntImmNode>();
        if (!count || count->value < 0) {
            throw std::invalid_argument("BindCudaThreads requires a valid runtime extent count");
        }
        bounded = count->value > 0;
    }
    const ProvenIteration proof = bounded ? BoundedCudaIterationProof(function).Check()
        : function->body.As<AllocateNode>() ? ProveMultiStage(function) : ProveSingleStage(function);
    const Stmt& iteration = proof.body;
    const auto& axes = proof.axes;
    const int64_t work_size = proof.work_size;

    const int64_t block_size =
        std::min<int64_t>(256, target->attrs.max_threads_per_block);
    const int64_t grid_size = work_size == 0 ? 1 : (work_size - 1) / block_size + 1;
    if (block_size > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        grid_size > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("CUDA launch dimensions exceed Driver API limits");
    }

    // Grid arithmetic is int64 even when individual TE axes use int32.
    const DataType index_dtype = DataType::Int(64);
    Var block_var("block_idx_x", index_dtype);
    Var thread_var("thread_idx_x", index_dtype);
    const PrimExpr block_extent = IntImm(block_size, index_dtype);
    const PrimExpr grid_extent = IntImm(grid_size, index_dtype);
    const PrimExpr linear = block_var * block_extent + thread_var;
    std::vector<std::pair<Var, PrimExpr>> replacements;
    for (size_t i = 0; i < axes.size(); ++i) {
        const auto& axis = axes[i];
        const PrimExpr extent = bounded ? proof.actual_extents[i] : IntImm(axis.extent, index_dtype);
        const PrimExpr stride = bounded ? proof.actual_strides[i] : IntImm(axis.stride, index_dtype);
        PrimExpr coordinate = IntImm(axis.minimum, index_dtype) +
            (linear / stride) % extent;
        if (axis.var->dtype != index_dtype) {
            coordinate = Call(axis.var->dtype, "cast", {coordinate});
        }
        replacements.emplace_back(axis.var, coordinate);
    }
    CudaBodyRewriter rewriter(std::move(replacements), bounded);
    Stmt rewritten_body = rewriter.Mutate(iteration);
    // The actual-work guard dominates tensor accesses and coordinate division;
    // an empty runtime axis therefore never divides by zero or touches data.
    rewritten_body = IfThenElse(linear < (bounded ? proof.actual_work : IntImm(work_size, index_dtype)),
                               rewritten_body);
    Stmt bound_body = ThreadBinding(
        block_var, ThreadIndexKind::kBlockIdxX, grid_extent,
        ThreadBinding(thread_var, ThreadIndexKind::kThreadIdxX,
                      block_extent, rewritten_body));

    CudaLaunchConfig launch_config;
    launch_config.grid_x = static_cast<uint32_t>(grid_size);
    launch_config.block_x = static_cast<uint32_t>(block_size);
    Map<String, ObjectRef> attrs = CopyAttrs(function->attrs);
    const String symbol_key("global_symbol");
    if (attrs.count(symbol_key)) {
        const auto* symbol = attrs.at(symbol_key).As<StringObj>();
        // Relay lowering 默认使用 main；CUDA 禁止 main 成为 __global__ 函数。
        // 调度阶段重命名可保证后续 Signature、NVRTC 和 Driver lookup 同源。
        if (symbol && symbol->data == "main") {
            attrs.Set(symbol_key, String("kxc_cuda_main"));
        }
    }
    attrs.Set(String(kCudaLaunchMetadataAttr),
              Array<int64_t>{
                  launch_config.grid_x, launch_config.grid_y,
                  launch_config.grid_z, launch_config.block_x,
                  launch_config.block_y, launch_config.block_z,
                  static_cast<int64_t>(
                      launch_config.dynamic_shared_memory_bytes)});
    attrs.Set(String(kCudaWorkSizeAttr),
              IntImm(work_size, DataType::Int(64)));
    PrimFunc scheduled(function->params, bound_body, function->buffer_map, attrs);
    return CudaScheduleResult(scheduled, launch_config);
}

// metadata 必须使用专用节点类型，错误 attr 不能通过静态转换造成 UB。
CudaLaunchConfig GetCudaLaunchConfig(const PrimFunc& function) {
    if (!function.defined() || !function.As<PrimFuncNode>()) {
        throw std::invalid_argument("GetCudaLaunchConfig requires a PrimFunc");
    }
    const String key(kCudaLaunchMetadataAttr);
    if (!function->attrs.count(key)) {
        throw std::invalid_argument("PrimFunc has no CUDA launch config");
    }
    const auto* values = function->attrs.at(key).As<ArrayNode<int64_t>>();
    if (!values || values->data.size() != 7) {
        throw std::invalid_argument("PrimFunc CUDA launch config is malformed");
    }
    CudaLaunchConfig config;
    config.grid_x = static_cast<uint32_t>(values->data[0]);
    config.grid_y = static_cast<uint32_t>(values->data[1]);
    config.grid_z = static_cast<uint32_t>(values->data[2]);
    config.block_x = static_cast<uint32_t>(values->data[3]);
    config.block_y = static_cast<uint32_t>(values->data[4]);
    config.block_z = static_cast<uint32_t>(values->data[5]);
    config.dynamic_shared_memory_bytes =
        static_cast<uint64_t>(values->data[6]);
    return config;
}

}  // namespace kxc::tir
