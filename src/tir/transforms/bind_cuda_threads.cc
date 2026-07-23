/*! \file src/tir/transforms/bind_cuda_threads.cc
 * \brief 实现保守的一维 CUDA thread-binding 调度与启动元数据计算。
 */

#include "kxc/tir/transforms/bind_cuda_threads.h"
#include "kxc/support/object_registration.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "kxc/tir/visitor.h"
#include "kxc/tir/pass_utils.h"

namespace kxc::tir {

KXC_OBJECT_DEFINE_WITH_KEY(CudaScheduleResultNode, "kxc.tir.CudaScheduleResultNode")

namespace {

// 判断表达式是否正是指定循环变量；第一阶段不猜测复杂下标的单射性。
bool IsLoopVar(const PrimExpr& expr, const Var& loop_var) {
    return expr.defined() && expr.get() == loop_var.get();
}

// 收集写 buffer 并证明每次写入都由当前循环变量唯一索引。
void CollectIndependentWrites(const Stmt& stmt, const Var& loop_var,
                              std::unordered_set<const Object*>* writes,
                              size_t* store_count) {
    if (const auto* store = stmt.As<StoreNode>()) {
        if (!IsLoopVar(store->index, loop_var)) {
            throw std::invalid_argument(
                "BindCudaThreads cannot prove Store index is iteration-independent");
        }
        writes->insert(store->buffer_var.get());
        ++(*store_count);
        return;
    }
    if (const auto* let_stmt = stmt.As<LetStmtNode>()) {
        CollectIndependentWrites(let_stmt->body, loop_var, writes, store_count);
        return;
    }
    if (const auto* sequence = stmt.As<SeqStmtNode>()) {
        for (const auto& child : sequence->seq) {
            CollectIndependentWrites(child, loop_var, writes, store_count);
        }
        return;
    }
    // For/Allocate/Attr/If/Block/Evaluate 都需要更强的依赖或存储域分析。
    throw std::invalid_argument(
        "BindCudaThreads only supports straight-line elementwise Store bodies");
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

// Generic indirect-load detector: CUDA's first schedule only admits direct indexing.
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

void RejectIndirectLoads(const Stmt& stmt) {
    if (const auto* store = stmt.As<StoreNode>()) {
        if (HasIndirectLoad(store->value) || HasIndirectLoad(store->predicate)) {
            throw std::invalid_argument(
                "BindCudaThreads rejects indirect Load index expressions");
        }
        return;
    }
    if (const auto* let_stmt = stmt.As<LetStmtNode>()) {
        if (HasIndirectLoad(let_stmt->value)) {
            throw std::invalid_argument(
                "BindCudaThreads rejects indirect Load index expressions");
        }
        RejectIndirectLoads(let_stmt->body);
        return;
    }
    if (const auto* sequence = stmt.As<SeqStmtNode>()) {
        for (const auto& child : sequence->seq) RejectIndirectLoads(child);
    }
}

// 检查表达式是否读取某个被并行写入的 buffer，保守拒绝潜在跨线程 RAW 竞争。
class WrittenBufferReadDetector : public TIRExprFunctor<bool> {
public:
    explicit WrittenBufferReadDetector(const std::unordered_set<const Object*>& writes)
        : writes_(writes) {}

protected:
    bool VisitLoad(const LoadNode* op, const PrimExpr& ref) override {
        (void)ref;
        return writes_.count(op->buffer_var.get()) != 0 || VisitExpr(op->index) ||
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
};

// 遍历已接受的直线语句，拒绝读取任何并行写 buffer。
void RejectReadAfterWrite(const Stmt& stmt,
                          const std::unordered_set<const Object*>& writes) {
    WrittenBufferReadDetector detector(writes);
    if (const auto* store = stmt.As<StoreNode>()) {
        if (detector.VisitExpr(store->value) || detector.VisitExpr(store->predicate)) {
            throw std::invalid_argument(
                "BindCudaThreads detected a read from a concurrently written buffer");
        }
        return;
    }
    if (const auto* let_stmt = stmt.As<LetStmtNode>()) {
        if (detector.VisitExpr(let_stmt->value)) {
            throw std::invalid_argument(
                "BindCudaThreads detected a loop-carried value dependency");
        }
        RejectReadAfterWrite(let_stmt->body, writes);
        return;
    }
    const auto* sequence = stmt.As<SeqStmtNode>();
    if (!sequence) return;
    for (const auto& child : sequence->seq) RejectReadAfterWrite(child, writes);
}

// 用 CUDA 线性线程号替换原循环变量，保持 TIR body 不依赖隐式命名规则。
class LoopVarSubstituter final : public TIRPass {
public:
    LoopVarSubstituter(Var loop_var, PrimExpr replacement)
        : loop_var_(std::move(loop_var)), replacement_(std::move(replacement)) {}

protected:
    PrimExpr VisitVar(const VarNode* op, const PrimExpr& ref) override {
        return ref.get() == loop_var_.get() ? replacement_ : ref;
    }

private:
    Var loop_var_;
    PrimExpr replacement_;
};

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

// 将单个可证明独立的外层循环映射为一维 CUDA grid/block。
CudaScheduleResult BindCudaThreads(const PrimFunc& function, const Target& target) {
    if (!function.defined() || !function.As<PrimFuncNode>()) {
        throw std::invalid_argument("BindCudaThreads requires a PrimFunc");
    }
    ValidateCudaTarget(target);
    if (function->attrs.count(String(kCudaLaunchMetadataAttr))) {
        throw std::invalid_argument("PrimFunc is already CUDA thread-bound");
    }
    const auto* loop = function->body.As<ForNode>();
    if (!loop || loop->for_type != ForType::Serial) {
        throw std::invalid_argument(
            "BindCudaThreads requires one outer serial data-parallel loop");
    }

    int64_t min_value = 0;
    int64_t work_size = 0;
    if (!pass_utils::TryGetConstInt64(loop->min, &min_value) ||
        !pass_utils::TryGetConstInt64(loop->extent, &work_size) || work_size <= 0) {
        throw std::invalid_argument(
            "BindCudaThreads requires positive static loop extent and constant min");
    }
    if (loop->loop_var->dtype.lanes != 1 ||
        (loop->loop_var->dtype.code != 0 && loop->loop_var->dtype.code != 1)) {
        throw std::invalid_argument("CUDA loop variable must be a scalar integer");
    }

    std::unordered_set<const Object*> writes;
    size_t store_count = 0;
    CollectIndependentWrites(loop->body, loop->loop_var, &writes, &store_count);
    if (store_count == 0) {
        throw std::invalid_argument("BindCudaThreads requires at least one Store");
    }
    RejectReadAfterWrite(loop->body, writes);
    RejectIndirectLoads(loop->body);

    const int64_t block_size =
        std::min<int64_t>(256, target->attrs.max_threads_per_block);
    const int64_t grid_size = (work_size - 1) / block_size + 1;
    if (block_size > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        grid_size > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("CUDA launch dimensions exceed Driver API limits");
    }

    const DataType index_dtype = loop->loop_var->dtype;
    Var block_var("block_idx_x", index_dtype);
    Var thread_var("thread_idx_x", index_dtype);
    const PrimExpr block_extent = IntImm(block_size, index_dtype);
    const PrimExpr grid_extent = IntImm(grid_size, index_dtype);
    const PrimExpr linear = block_var * block_extent + thread_var;
    const PrimExpr logical_index = IntImm(min_value, index_dtype) + linear;
    LoopVarSubstituter substituter(loop->loop_var, logical_index);
    Stmt rewritten_body = substituter.Mutate(loop->body);
    // 尾块中超出 work_size 的线程必须在任何 Load/Store 前被屏蔽。
    const PrimExpr bound = IntImm(min_value + work_size, index_dtype);
    rewritten_body = IfThenElse(logical_index < bound, rewritten_body);
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
