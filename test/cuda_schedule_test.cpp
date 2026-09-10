/*! \file test/cuda_schedule_test.cpp
 * \brief 验证 CUDA thread-binding 调度的结构、元数据和保守拒绝边界。
 */

#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/pass/context.h"
#include "kxc/relay/op.h"
#include "support/primitive_lowering.h"
#include "kxc/tir/visitor.h"
#include "kxc/tir/printer/print_ir.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"
#include "kxc/tir/transforms/pipeline.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

// 负例只关心调度边界明确失败，不绑定具体异常文本。
bool Throws(const std::function<void()>& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// CPU-only 构建不查询真实 GPU，测试使用显式且完整的 CUDA Target 快照。
kxc::Target MakeCudaTarget(int64_t max_threads = 128, int exists = 1,
                           int64_t max_shared_memory = 48 * 1024) {
    auto* node = new kxc::TargetNode();
    node->kind = "cuda";
    node->device_type = kxc::kCUDA;
    node->device_id = 0;
    node->attrs.exists = exists;
    node->attrs.max_threads_per_block = max_threads;
    node->attrs.max_shared_memory_per_block = max_shared_memory;
    return kxc::Target(kxc::ObjectRef(node));
}

// 构造第一阶段支持的一维 add/relu 形状，Store 下标与循环变量严格一致。
kxc::tir::PrimFunc MakeElementwiseFunction(int64_t extent, bool relu = false) {
    using namespace kxc;
    using namespace kxc::tir;
    const DataType f32 = DataType::Float(32);
    const DataType i64 = DataType::Int(64);
    tir::Var a("a", f32);
    tir::Var b("b", f32);
    tir::Var out("out", f32);
    tir::Var i("i", i64);
    PrimExpr value = relu ? PrimExpr(Max(Load(a, i), FloatImm(0.0, f32)))
                          : PrimExpr(Add(Load(a, i), Load(b, i)));
    Stmt body = For(i, IntImm(0, i64), IntImm(extent, i64), ForType::Serial,
                    Store(out, value, i));
    Array<tir::Var> params =
        relu ? Array<tir::Var>{a, out} : Array<tir::Var>{a, b, out};
    Map<tir::Var, Buffer> buffers;
    for (const auto& parameter : params) {
        buffers.Set(parameter, Buffer(parameter, f32, {IntImm(extent, i64)}, {},
                                      IntImm(0, i64), parameter->name_hint, 4, 0));
    }
    return PrimFunc(params, body, buffers, {});
}

// 通用 TIRPass 不覆写 ThreadBinding 时也必须递归保留新节点。
class IdentityTIRPass final : public kxc::TIRPass {};

// 验证 block/thread 嵌套、尾块 guard、启动尺寸以及 printer/mutator 支持。
bool TestElementwiseSchedule() {
    using namespace kxc;
    using namespace kxc::tir;
    CudaScheduleResult result = BindCudaThreads(MakeElementwiseFunction(1000),
                                                MakeCudaTarget(128));
    PrimFunc scheduled = result.prim_func();
    CudaLaunchConfig launch = result.launch_config();
    TEST_CHECK(launch.grid_x == 8 && launch.block_x == 128,
               "grid/block dimensions were not derived from work size");
    CudaLaunchConfig stored = GetCudaLaunchConfig(scheduled);
    TEST_CHECK(stored.grid_x == launch.grid_x &&
                   stored.block_x == launch.block_x,
               "PrimFunc attr and schedule result must share launch values");

    const auto* block = scheduled->body.As<ThreadBindingNode>();
    TEST_CHECK(block && block->thread_index == ThreadIndexKind::kBlockIdxX,
               "outer blockIdx.x binding is missing");
    const auto* thread = block->body.As<ThreadBindingNode>();
    TEST_CHECK(thread && thread->thread_index == ThreadIndexKind::kThreadIdxX,
               "inner threadIdx.x binding is missing");
    TEST_CHECK(thread->body.As<IfThenElseNode>() != nullptr,
               "tail threads must be protected by a bounds guard");

    std::ostringstream text;
    printer::DumpPrimFunc(scheduled, text);
    TEST_CHECK(text.str().find("blockIdx.x") != std::string::npos &&
                   text.str().find("threadIdx.x") != std::string::npos,
               "IR printer did not expose structured thread bindings");
    IdentityTIRPass identity;
    PrimFunc copied = identity.Mutate(scheduled);
    TEST_CHECK(copied->body.As<ThreadBindingNode>() != nullptr,
               "generic TIR mutator dropped ThreadBinding");
    return true;
}

// relu 使用相同的独立 Store 证明，并验证 pipeline 名称注册和 Target 上下文读取。
bool TestReluPipelineRegistration() {
    using namespace kxc;
    using namespace kxc::tir;
    const Target target = MakeCudaTarget(64);
    PassContext::Scope scope(PassContext::FromTarget(target));
    PrimFunc scheduled = RunTIRPassPipeline(
        MakeElementwiseFunction(65, true), {String("bind_cuda_threads")});
    CudaLaunchConfig launch = GetCudaLaunchConfig(scheduled);
    TEST_CHECK(launch.grid_x == 2 && launch.block_x == 64,
               "pipeline adapter produced incorrect relu launch dimensions");
    return true;
}

// int32 循环变量可安全扩宽成 int64 地址；这种单射变换不应阻断 CUDA 调度。
bool TestWidenedStoreIndexSchedule() {
    using namespace kxc;
    using namespace kxc::tir;
    const DataType i32 = DataType::Int(32);
    const DataType i64 = DataType::Int(64);
    const DataType f32 = DataType::Float(32);
    tir::Var out("out", f32);
    tir::Var i("i", i32);
    PrimExpr widened_i = tir::Call(i64, "cast", {i});
    Stmt body = For(i, IntImm(0, i32), IntImm(8, i32), ForType::Serial,
                    Store(out, FloatImm(1.0, f32), widened_i));
    Map<tir::Var, Buffer> buffers;
    buffers.Set(out, Buffer(out, f32, {IntImm(8, i64)}, {},
                            IntImm(0, i64), "out", 4, 0));

    CudaScheduleResult result =
        BindCudaThreads(PrimFunc({out}, body, buffers, {}), MakeCudaTarget());
    TEST_CHECK(result.launch_config().grid_x == 1 &&
                   result.launch_config().block_x == 128,
               "widening cast around the loop index should preserve injectivity");
    return true;
}

// Canonical TE reduction: initialize each output, then accumulate serially.
kxc::tir::PrimFunc MakeOwnedReduction(const std::vector<int64_t>& shape) {
    using namespace kxc;
    using namespace kxc::tir;
    const DataType i64 = DataType::Int(64), f32 = DataType::Float(32);
    tir::Var data("data", f32), out("out", f32), r("r", i64);
    std::vector<tir::Var> axes;
    Array<PrimExpr> output_shape;
    PrimExpr index = IntImm(0, i64);
    int64_t elements = 1;
    for (int64_t extent : shape) {
        axes.emplace_back("axis", i64);
        output_shape.push_back(IntImm(extent, i64));
        index = index * IntImm(extent, i64) + axes.back();
        elements *= extent;
    }
    Stmt body = SeqStmt({
        Store(out, FloatImm(0.0f), index),
        For(r, IntImm(0, i64), IntImm(4, i64), ForType::Serial,
            Store(out, Load(out, index) + Load(data, index * IntImm(4, i64) + r), index))});
    for (size_t i = axes.size(); i > 0; --i) {
        body = For(axes[i - 1], IntImm(0, i64), output_shape[i - 1], ForType::Serial, body);
    }
    Map<tir::Var, Buffer> buffers;
    buffers.Set(data, Buffer(data, f32, {IntImm(elements * 4, i64)}, {}, IntImm(0), "data", 4, 0));
    buffers.Set(out, Buffer(out, f32, output_shape, {}, IntImm(0), "out", 4, 0));
    return PrimFunc({data, out}, body, buffers, {});
}

bool TestOwnedReductionAndScalar() {
    using namespace kxc::tir;
    const auto result = BindCudaThreads(MakeOwnedReduction({17, 17}), MakeCudaTarget());
    TEST_CHECK(result.launch_config().grid_x == 3 && result.launch_config().block_x == 128,
               "all 289 output coordinates must be assigned across tail-safe blocks");
    const auto* block = result.prim_func()->body.As<ThreadBindingNode>();
    const auto* thread = block->body.As<ThreadBindingNode>();
    const auto* guard = thread->body.As<IfThenElseNode>();
    const auto* sequence = guard->then_case.As<SeqStmtNode>();
    TEST_CHECK(sequence && sequence->seq.size() == 2 &&
                   sequence->seq[0].As<StoreNode>() && sequence->seq[1].As<ForNode>(),
               "initialization and the serial reduction must remain inside the thread guard");
    const auto scalar = BindCudaThreads(MakeOwnedReduction({}), MakeCudaTarget());
    TEST_CHECK(scalar.launch_config().grid_x == 1,
               "a scalar reduction still needs an explicitly guarded CUDA owner");
    return true;
}

bool TestRejectUnsafeOwnedReads() {
    using namespace kxc;
    using namespace kxc::tir;
    const PrimFunc base = MakeOwnedReduction({8});
    const auto* outer = base->body.As<ForNode>();
    const auto* sequence = outer->body.As<SeqStmtNode>();
    const auto* init = sequence->seq[0].As<StoreNode>();
    const auto* reduce = sequence->seq[1].As<ForNode>();
    const auto* update = reduce->body.As<StoreNode>();
    const auto reject = [&](const Stmt& body) {
        return Throws([&] {
            BindCudaThreads(PrimFunc(base->params,
                For(outer->loop_var, outer->min, outer->extent, outer->for_type, body),
                base->buffer_map, {}), MakeCudaTarget());
        });
    };
    TEST_CHECK(reject(SeqStmt({Store(init->buffer_var, Load(init->buffer_var, init->index),
                                       init->index), sequence->seq[1]})),
               "reading an output before initialization must fail");
    TEST_CHECK(reject(SeqStmt({Store(init->buffer_var, init->value, init->index,
                                       Load(base->params[0], IntImm(0)) < FloatImm(0.0f)),
                              sequence->seq[1]})),
               "a conditional initializer does not dominate a later read");
    TEST_CHECK(reject(SeqStmt({sequence->seq[0],
        For(reduce->loop_var, reduce->min, reduce->extent, reduce->for_type,
            Store(update->buffer_var, Load(update->buffer_var, update->index + IntImm(1)),
                  update->index))})),
               "even an initialized output must not read a neighboring thread's cell");
    TEST_CHECK(reject(SeqStmt({
        For(reduce->loop_var, IntImm(0), IntImm(0), ForType::Serial, sequence->seq[0]),
        sequence->seq[1]})), "a zero-trip loop does not initialize its output");
    TEST_CHECK(reject(SeqStmt({sequence->seq[0],
        For(outer->loop_var, reduce->min, reduce->extent, ForType::Serial, reduce->body)})),
        "reduction must not rebind a parallel coordinate");
    TEST_CHECK(reject(LetStmt(base->params[0], FloatImm(0.0f), outer->body)),
               "a local binder must not shadow a buffer parameter identity");
    TEST_CHECK(reject(SeqStmt({sequence->seq[0],
        For(reduce->loop_var, reduce->min, reduce->extent, ForType::Serial,
            Store(update->buffer_var, FloatImm(1.0f), reduce->loop_var))})),
        "reduction coordinates must not index the output");
    const tir::Var dynamic_extent("n", DataType::Int(64));
    TEST_CHECK(reject(SeqStmt({sequence->seq[0],
        For(reduce->loop_var, reduce->min, dynamic_extent, ForType::Serial, reduce->body)})),
        "a dynamic reduction bound has no static CUDA contract");
    const PrimExpr narrow_overflow = IntImm(2147483647, DataType::Int(32)) +
                                     IntImm(1, DataType::Int(32));
    TEST_CHECK(reject(SeqStmt({sequence->seq[0],
        For(reduce->loop_var, narrow_overflow, IntImm(1), ForType::Serial, reduce->body)})),
        "a wider induction variable cannot justify overflow inside a loop bound");
    Map<tir::Var, Buffer> small_buffers;
    for (const auto& item : base->buffer_map) small_buffers.Set(item.first, item.second);
    small_buffers.Set(init->buffer_var, Buffer(init->buffer_var, DataType::Float(32),
        {IntImm(7)}, {}, IntImm(0), "too_small", 4, 0));
    TEST_CHECK(Throws([&] { BindCudaThreads(PrimFunc(base->params, base->body, small_buffers, {}),
                                          MakeCudaTarget()); }),
               "a valid ownership index still must fit the output allocation");
    return true;
}

bool TestRejectNarrowIndexOverflow() {
    using namespace kxc;
    using namespace kxc::tir;
    const DataType i32 = DataType::Int(32), i64 = DataType::Int(64), f32 = DataType::Float(32);
    tir::Var i("i", i32), j("j", i32), out("out", f32);
    Map<tir::Var, Buffer> buffers;
    buffers.Set(out, Buffer(out, f32, {IntImm(65536, i64), IntImm(65536, i64)}, {},
                            IntImm(0), "out", 4, 0));
    const auto function = [&](PrimExpr row) {
        return PrimFunc({out},
            For(i, IntImm(0), IntImm(65536), ForType::Serial,
                For(j, IntImm(0), IntImm(65536), ForType::Serial,
                    Store(out, FloatImm(1.0f), row * IntImm(65536, row.dtype()) + j))),
            buffers, {});
    };
    TEST_CHECK(Throws([&] { BindCudaThreads(function(i), MakeCudaTarget()); }),
               "row-major coefficients cannot justify overflowing int32 arithmetic");
    const auto wide = BindCudaThreads(function(tir::Call(i64, "cast", {i})), MakeCudaTarget());
    const auto* block = wide.prim_func()->body.As<ThreadBindingNode>();
    TEST_CHECK(block->thread_var->dtype == i64 && wide.launch_config().grid_x == 33554432,
               "a proven wide address must use int64 grid arithmetic");
    return true;
}

// Missing initialization and non-injective writes remain unsupported.
bool TestRejectReductionAndWriteConflict() {
    using namespace kxc;
    using namespace kxc::tir;
    const DataType i64 = DataType::Int(64);
    const DataType f32 = DataType::Float(32);
    tir::Var out("out", f32);
    tir::Var i("i", i64);
    tir::Var r("r", i64);
    Stmt reduction = For(
        i, IntImm(0, i64), IntImm(8, i64), ForType::Serial,
        For(r, IntImm(0, i64), IntImm(4, i64), ForType::Serial,
            Store(out, Add(Load(out, i), FloatImm(1.0, f32)), i)));
    TEST_CHECK(Throws([&] {
                   BindCudaThreads(PrimFunc({out}, reduction), MakeCudaTarget());
               }),
               "a nested update without initialization should be rejected");

    Stmt conflict = For(
        i, IntImm(0, i64), IntImm(8, i64), ForType::Serial,
        Store(out, FloatImm(1.0, f32), IntImm(0, i64)));
    TEST_CHECK(Throws([&] {
                   BindCudaThreads(PrimFunc({out}, conflict), MakeCudaTarget());
               }),
               "non-injective Store index should be rejected");
    return true;
}

// Indirect Load indices are gather-like accesses and must be rejected independently of op names.
bool TestRejectIndirectGatherLoad() {
    using namespace kxc;
    using namespace kxc::tir;
    const DataType i64 = DataType::Int(64);
    const DataType f32 = DataType::Float(32);
    tir::Var data("data", f32);
    tir::Var indices("indices", i64);
    tir::Var out("out", f32);
    tir::Var i("i", i64);
    Stmt body = For(i, IntImm(0, i64), IntImm(8, i64), ForType::Serial,
                    Store(out, Load(data, Load(indices, i)), i));
    Map<tir::Var, Buffer> buffers;
    for (const auto& parameter : Array<tir::Var>{data, indices, out}) {
        buffers.Set(parameter, Buffer(parameter, parameter->dtype, {IntImm(8, i64)}, {},
                                      IntImm(0, i64), parameter->name_hint, 4, 0));
    }
    TEST_CHECK(Throws([&] {
                   BindCudaThreads(PrimFunc({data, indices, out}, body, buffers, {}),
                                   MakeCudaTarget());
               }),
               "rank-1 indirect gather load should be rejected");
    return true;
}

bool TestProductionGuardedGather() {
    using namespace kxc;
    struct Case { Array<int64_t> data, indices; int axis; int64_t work; };
    const std::vector<Case> cases{
        {{2, 7, 5}, {3, 4}, 1, 120}, {{7, 5}, {257}, 0, 1285},
        {{7}, {}, -1, 1}, {{0, 5}, {3}, 0, 15}, {{7, 5}, {0}, 0, 0},
        {{262144, 8960}, {5}, 0, 44800}};
    for (const auto& item : cases) for (const std::string& dtype : {"int32", "int64"}) {
        Var data("data", TensorType(item.data, "float32")), indices("indices", TensorType(item.indices, dtype));
        const auto target = MakeCudaTarget();
        const auto fixture = test_support::LowerPrimitivesForTest(
            Function({data, indices}, Call(relay::Op::Get("gather"), {data, indices},
                relay::GatherAttrs::Create(item.axis))), target);
        TEST_CHECK(fixture.lowered.size() == 1, "Gather must remain an ordinary primitive");
        const auto scheduled = tir::RunTIRPassPipeline(fixture.lowered[0]->prim_func,
            {"fold_constant", "simplify_expr", "force_narrow_index_to_i32", "remove_no_op", "bind_cuda_threads"},
            PassContext::FromTarget(target));
        TEST_CHECK(scheduled->attrs.at(String(tir::kCudaWorkSizeAttr)).As<tir::IntImmNode>()->value == item.work,
                   "Gather output ownership count is incorrect");
        if (item.work == 0) TEST_CHECK(tir::GetCudaLaunchConfig(scheduled).grid_x == 1,
                                      "empty Gather still needs a valid guarded launch");
    }
    return true;
}

kxc::tir::PrimFunc MakeGuardedIndirectCase(int mode) {
    using namespace kxc;
    using namespace kxc::tir;
    const auto f32 = DataType::Float(32), i64 = DataType::Int(64);
    tir::Var data("data", f32), indices("indices", i64), out("out", f32), i("i", i64);
    const auto imm = [&](int64_t value) { return IntImm(value, i64); };
    PrimExpr index = Load(indices, i);
    PrimExpr compared = index;
    if (mode == 4) compared = tir::Call(DataType::Int(32), "cast", {index});
    if (mode == 5) compared = index + imm(8);
    PrimExpr lower = !(compared < imm(0)), upper = compared < imm(8);
    PrimExpr guard = lower && upper;
    if (mode == 1) guard = upper;
    if (mode == 2) guard = lower;
    if (mode == 3) index = Load(indices, i + imm(1));
    if (mode == 7) index = Load(indices, i * imm(1) + imm(0));
    if (mode == 8) index = tir::Call(DataType::Int(32), "cast", {index});
    Stmt body = Store(out, Select(guard, Load(data, index), FloatImm(0, f32)), i);
    Map<tir::Var, Buffer> buffers;
    buffers.Set(data, Buffer(data, f32, {imm(mode == 6 ? 7 : 8)}, {}, imm(0), "data", 0, 0));
    buffers.Set(indices, Buffer(indices, i64, {imm(9)}, {}, imm(0), "indices", 0, 0));
    buffers.Set(out, Buffer(out, f32, {imm(8)}, {}, imm(0), "out", 0, 0));
    return PrimFunc({data, indices, out}, For(i, imm(0), imm(8), ForType::Serial, body), buffers);
}

bool TestGuardedIndirectBoundaries() {
    using namespace kxc::tir;
    for (int mode : {0, 7, 8}) {
        TEST_CHECK(BindCudaThreads(MakeGuardedIndirectCase(mode), MakeCudaTarget()).prim_func().defined(),
                   "complete guards and equivalent affine index reads must be accepted");
    }
    for (int mode = 1; mode <= 6; ++mode) {
        TEST_CHECK(Throws([&] { BindCudaThreads(MakeGuardedIndirectCase(mode), MakeCudaTarget()); }),
                   "partial/mismatched/narrowed/overflowing guards or incorrect table bounds must fail closed");
    }
    const auto empty = BindCudaThreads(MakeElementwiseFunction(0), MakeCudaTarget(1));
    TEST_CHECK(empty.launch_config().grid_x == 1 && empty.launch_config().block_x == 1,
               "empty output launch must also work on a one-thread target");
    return true;
}

// 动态工作量、缺失 capability、CPU target 和非法 thread index 都必须明确失败。
bool TestRejectInvalidContracts() {
    using namespace kxc;
    using namespace kxc::tir;
    PrimFunc valid = MakeElementwiseFunction(8);
    const auto* loop = valid->body.As<ForNode>();
    tir::Var n("n", DataType::Int(64));
    PrimFunc dynamic(valid->params,
                     For(loop->loop_var, loop->min, n, loop->for_type, loop->body),
                     valid->buffer_map, valid->attrs);
    TEST_CHECK(Throws([&] { BindCudaThreads(dynamic, MakeCudaTarget()); }),
               "dynamic extent should be rejected");
    TEST_CHECK(Throws([&] { BindCudaThreads(valid, MakeCudaTarget(0)); }),
               "missing max_threads capability should be rejected");
    TEST_CHECK(Throws([&] { BindCudaThreads(valid, MakeCudaTarget(128, 1, -1)); }),
               "missing shared-memory capability should be rejected");
    TEST_CHECK(Throws([&] { BindCudaThreads(valid, MakeCudaTarget(128, 0)); }),
               "unavailable CUDA target should be rejected");
    TEST_CHECK(Throws([&] { BindCudaThreads(valid, BuildTarget(Device::CPU())); }),
               "CPU target should be rejected");
    TEST_CHECK(Throws([&] {
                   ThreadBinding(tir::Var("tx", DataType::Int(32)),
                                 static_cast<ThreadIndexKind>(99), IntImm(1),
                                 Evaluate(IntImm(0)));
               }),
               "unknown thread index should be rejected");
    return true;
}

// 第一阶段不支持循环内分配；同时必须在生成 uint32 grid 前拒绝超大工作量。
bool TestRejectAllocationAndLaunchOverflow() {
    using namespace kxc;
    using namespace kxc::tir;
    const DataType i64 = DataType::Int(64);
    const DataType f32 = DataType::Float(32);
    PrimFunc base = MakeElementwiseFunction(8);
    const auto* loop = base->body.As<ForNode>();
    TEST_CHECK(loop != nullptr, "elementwise fixture lost its outer loop");

    tir::Var scratch("scratch", f32);
    Stmt allocation = Allocate(
        scratch, f32, {IntImm(8, i64)}, IntImm(1, DataType::Bool()), loop->body);
    PrimFunc allocated(
        base->params,
        For(loop->loop_var, loop->min, loop->extent, loop->for_type, allocation),
        base->buffer_map, base->attrs);
    TEST_CHECK(Throws([&] { BindCudaThreads(allocated, MakeCudaTarget()); }),
               "loop-local allocation should be rejected before CUDA codegen");

    const int64_t too_large_work =
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) * 128 + 1;
    TEST_CHECK(
        Throws([&] {
            BindCudaThreads(MakeElementwiseFunction(too_large_work),
                            MakeCudaTarget(128));
        }),
        "grid dimension beyond uint32 should be rejected");
    return true;
}

class ScratchAudit final : public kxc::TIRPass {
public:
    int64_t bytes{0};
    size_t allocations{0};
    std::vector<std::vector<int64_t>> shapes;
protected:
    kxc::tir::Stmt VisitAllocate(const kxc::tir::AllocateNode* op,
                                 const kxc::tir::Stmt& ref) override {
        int64_t elements = 1;
        std::vector<int64_t> shape;
        for (const auto& extent : op->extents) {
            const auto* value = extent.As<kxc::tir::IntImmNode>();
            if (!value || value->value <= 0) throw std::runtime_error("invalid private extent");
            shape.push_back(value->value);
            elements *= value->value;
        }
        shapes.push_back(std::move(shape));
        bytes += elements * ((op->dtype.bits + 7) / 8);
        ++allocations;
        return TIRPass::VisitAllocate(op, ref);
    }
};

// Use the real CUDA target-dependent lowering path even in a CPU-only test build.
bool TestProductionMultiStageSchedule() {
    using namespace kxc;
    const Target target = MakeCudaTarget(128);
    Var x("x", TensorType({2, 3, 5}, "float32"));
    Var mask("mask", TensorType({1, 3, 1}, "bool"));
    Var scale("scale", TensorType({3, 5}, "float32"));
    Var bias("bias", TensorType({3, 5}, "float32"));
    const std::vector<std::pair<Function, int64_t>> cases{
        {Function({x}, Call(relay::Op::Get("softmax"), {x}, relay::SoftmaxAttrs::Create(1))), 10},
        {Function({x}, Call(relay::Op::Get("softmax"), {x}, relay::SoftmaxAttrs::Create(0))), 15},
        {Function({x}, Call(relay::Op::Get("softmax"), {x}, relay::SoftmaxAttrs::Create(-1))), 6},
        {Function({x, mask}, Call(relay::Op::Get("masked_softmax"), {x, mask},
                                 relay::SoftmaxAttrs::Create(1))), 10},
        {Function({x, scale, bias}, Call(relay::Op::Get("nn_layer_norm"), {x, scale, bias},
                relay::LayerNormAttrs::Create(1, 1e-5, "float64"))), 2},
        {Function({x}, Call(relay::Op::Get("reduce_mean"), {x},
                            relay::ReduceMeanAttrs::Create({1}, false))), 10},
        {Function({x}, Call(relay::Op::Get("reduce_mean"), {x},
                            relay::ReduceMeanAttrs::Create({0, 1, 2}, false))), 1}};
    for (size_t index = 0; index < cases.size(); ++index) {
        std::cout << "[STRUCTURAL] multistage_case=" << index << '\n';
        const auto fixture = test_support::LowerPrimitivesForTest(cases[index].first, target);
        TEST_CHECK(fixture.lowered.size() == 1, "expected one production primitive");
        const auto original = fixture.lowered[0]->prim_func;
        std::ostringstream before, after;
        tir::printer::DumpPrimFunc(original, before);
        const auto scheduled = tir::RunTIRPassPipeline(original,
            {"fold_constant", "simplify_expr", "force_narrow_index_to_i32", "remove_no_op", "bind_cuda_threads"},
            PassContext::FromTarget(target));
        const auto* work = scheduled->attrs.at(String(tir::kCudaWorkSizeAttr)).As<tir::IntImmNode>();
        TEST_CHECK(work && work->value == cases[index].second, "incorrect independent coordinate count");
        ScratchAudit audit;
        audit.Mutate(scheduled);
        TEST_CHECK(audit.allocations > 0 && audit.bytes < 1024, "scratch did not become thread-private");
        if (index == 0) {
            TEST_CHECK(audit.bytes == 32 && audit.allocations == 4,
                       "middle-axis softmax requires two scalar and two length-3 private buffers");
            for (const auto& shape : audit.shapes) {
                TEST_CHECK(shape.size() == 3 && shape[0] == 1 && shape[2] == 1,
                           "both non-reduced coordinates must be private");
            }
        }
        tir::printer::DumpPrimFunc(original, after);
        TEST_CHECK(before.str() == after.str(), "scheduling mutated the input PrimFunc");
    }
    Var large("large", TensorType({2, 10000}, "float32"));
    const auto large_fixture = test_support::LowerPrimitivesForTest(
        Function({large}, Call(relay::Op::Get("softmax"), {large}, relay::SoftmaxAttrs::Create(-1))), target);
    TEST_CHECK(Throws([&] { tir::BindCudaThreads(large_fixture.lowered[0]->prim_func, target); }),
               "oversized private storage must be rejected before CUDA codegen");
    return true;
}

// A two-stage sum permits partitioning on axes 0 and 2; r*4 stays flat-in-bounds
// but crosses the neighboring axis-2 owner. This catches a merely flat bounds proof.
kxc::tir::PrimFunc MakeMultiStageProofCase(int mode) {
    using namespace kxc;
    using namespace kxc::tir;
    const auto f32 = DataType::Float(32), i64 = DataType::Int(64);
    tir::Var input("input", f32), scratch("scratch", f32), out("out", f32);
    tir::Var i("i", i64), j("j", i64), k("k", i64), r("r", i64);
    const auto imm = [&](int64_t value) { return IntImm(value, i64); };
    const auto nest = [&](int64_t middle, Stmt stmt) -> Stmt {
        return For(i, imm(0), imm(2), ForType::Serial,
            For(j, imm(0), imm(middle), ForType::Serial,
                For(k, imm(0), imm(5), ForType::Serial, stmt)));
    };
    PrimExpr full = i * imm(15) + j * imm(5) + k;
    Stmt producer = nest(3, Store(scratch, mode == 8 ? PrimExpr(Load(input, full))
        : PrimExpr(Load(input, full) + FloatImm(1, f32)), full));
    PrimExpr output_index = i * imm(5) + j * imm(5) + k;
    PrimExpr read = i * imm(15) + r * imm(mode == 1 ? 4 : 5) + k;
    Stmt init = Store(out, FloatImm(0, f32), output_index,
                     mode == 4 ? PrimExpr(IntImm(0, DataType::Bool())) : PrimExpr());
    tir::Var read_buffer = mode == 9 ? tir::Var("undeclared", f32) : scratch;
    Stmt update = For(r, imm(0), imm(mode == 6 ? 0 : 3), ForType::Serial,
        Store(out, Load(out, output_index) + Load(read_buffer, read), output_index));
    Stmt consumer = nest(1, mode == 3 ? update : Stmt(SeqStmt({init, update})));
    Array<Stmt> stages = mode == 2 || mode == 8 ? Array<Stmt>{consumer, producer}
                                  : Array<Stmt>{producer, consumer};
    if (mode == 5) stages.push_back(producer);
    Map<tir::Var, Buffer> buffers;
    buffers.Set(input, Buffer(input, f32, {imm(2), imm(3), imm(5)}, {}, imm(0), "input", 0, 0));
    buffers.Set(out, Buffer(out, f32, {imm(2), imm(1), imm(5)}, {}, imm(0), "out", 0, 0));
    return PrimFunc({input, out}, Allocate(scratch, f32, {imm(2), imm(3), imm(5)},
        IntImm(mode == 7 ? 0 : 1, DataType::Bool()), SeqStmt(stages)), buffers);
}

bool TestMultiStageOwnershipRejections() {
    using namespace kxc::tir;
    const auto valid = BindCudaThreads(MakeMultiStageProofCase(0), MakeCudaTarget());
    TEST_CHECK(valid.prim_func()->attrs.at(kxc::String(kCudaWorkSizeAttr)).As<IntImmNode>()->value == 10,
               "valid two-stage reduction must have ten independent owners");
    for (int mode = 1; mode <= 9; ++mode) {
        TEST_CHECK(Throws([&] { BindCudaThreads(MakeMultiStageProofCase(mode), MakeCudaTarget()); }),
                   "unsafe stage ownership, ordering, initialization or allocation was accepted");
    }
    return true;
}

// Rectangular N x K @ K x M uses distinct symbolic axes. Every output has
// one owner; K may be zero, so initialization must precede the runtime loop.
kxc::tir::PrimFunc MakeBoundedMatmulProofCase(int mode = 0) {
    using namespace kxc;
    using namespace kxc::tir;
    const DataType f32 = DataType::Float(32), i32 = DataType::Int(32),
                   i64 = DataType::Int(64), u64 = DataType::UInt(64);
    tir::Var a("a", f32), b("b", f32), out("out", f32);
    tir::Var n("n", u64), m("m", u64), k("k", u64);
    tir::Var i("i", i32), j("j", i32), r("r", i32);
    const auto wide = [i64](const PrimExpr& value) { return PrimExpr(tir::Call(i64, "cast", {value})); };
    const PrimExpr N = wide(Load(n, IntImm(0))), M = wide(Load(m, IntImm(0))),
                   K = wide(Load(k, IntImm(0)));
    const PrimExpr index = wide(i) * M + wide(j);
    PrimExpr a_index = wide(i) * K + wide(r);
    if (mode == 5) a_index = a_index + IntImm(1, i64);
    if (mode == 10) a_index = tir::Call(i64, "cast", {Load(b, IntImm(0))});
    PrimExpr read = mode == 15 ? PrimExpr(tir::Var("unbound", f32))
                              : PrimExpr(Load(a, a_index));
    const PrimExpr value = Load(out, mode == 7 ? index + IntImm(1, i64) : index) +
                          read * Load(b, wide(r) * M + wide(j));
    const PrimExpr store_index = mode == 6 ? index + IntImm(1, i64)
        : mode == 12 ? PrimExpr(tir::Call(i32, "cast", {index})) : index;
    Stmt initialize = Store(out, FloatImm(0.0f), store_index,
                           mode == 17 ? PrimExpr(IntImm(0, DataType::Bool())) : PrimExpr());
    Stmt accumulate = For(r, IntImm(0), mode == 18 ? K + IntImm(1, i64) : K,
        ForType::Serial, Store(mode == 9 ? a : out, value, store_index));
    Stmt body = mode == 8 ? Stmt(SeqStmt({accumulate, initialize}))
                         : Stmt(SeqStmt({initialize, accumulate}));
    body = For(i, IntImm(0), N, ForType::Serial,
        For(j, IntImm(0), M, ForType::Serial, body));
    Map<tir::Var, Buffer> buffers;
    buffers.Set(a, Buffer(a, f32, mode == 11 ? Array<PrimExpr>{N} : Array<PrimExpr>{N, K}, {}, IntImm(0), "a", 4, 0));
    buffers.Set(b, Buffer(b, f32, {K, M}, {}, IntImm(0), "b", 4, 0));
    buffers.Set(out, Buffer(out, f32, {N, M}, {}, IntImm(0), "out", 4, 0));
    for (const auto& scalar : Array<tir::Var>{n, m, k}) {
        buffers.Set(scalar, Buffer(scalar, u64, {IntImm(1)}, {}, IntImm(0), scalar->name_hint, 8, 0));
    }
    Map<String, ObjectRef> attrs;
    attrs.Set("kxc.input_count", IntImm(2));
    attrs.Set("kxc.runtime_extent_param_start", IntImm(2));
    attrs.Set("kxc.runtime_extent_count", IntImm(3));
    attrs.Set("kxc.constant_count", IntImm(0));
    attrs.Set("kxc.output_param_start", IntImm(5));
    attrs.Set("kxc.output_count", IntImm(1));
    if (mode != 1) {
        Array<int64_t> bounds{19, 17, 13};
        if (mode == 2) bounds = {19, 17};
        if (mode == 3) bounds = {-1, 17, 13};
        if (mode == 4) bounds = {int64_t(INT32_MAX) + 1, 17, 13};
        if (mode == 12 || mode == 13) bounds = {INT32_MAX, INT32_MAX, 13};
        if (mode == 14) bounds = {0, 17, 13};
        attrs.Set(kCudaRuntimeExtentBoundsAttr, mode == 16 ? ObjectRef(String("untyped")) : ObjectRef(bounds));
    }
    return PrimFunc({a, b, n, m, k, out}, body, buffers, attrs);
}

bool TestBoundedCompactOwnership() {
    using namespace kxc::tir;
    const auto scheduled = BindCudaThreads(MakeBoundedMatmulProofCase(), MakeCudaTarget());
    TEST_CHECK(scheduled.launch_config().grid_x == 3,
               "323 maximum output owners must span three blocks");
    const auto* block = scheduled.prim_func()->body.As<ThreadBindingNode>();
    const auto* thread = block->body.As<ThreadBindingNode>();
    const auto* guard = thread->body.As<IfThenElseNode>();
    TEST_CHECK(guard && !guard->condition.As<LTNode>()->b.As<IntImmNode>(),
               "the launch upper bound must not replace actual-work guarding");
    const auto* body = guard->then_case.As<SeqStmtNode>();
    TEST_CHECK(body && body->seq[0].As<StoreNode>() && body->seq[1].As<ForNode>() &&
               !body->seq[1].As<ForNode>()->extent.As<IntImmNode>(),
               "initialization and runtime K reduction must remain within each owner");
    for (int mode = 1; mode <= 18; ++mode) {
        if (mode == 14) continue;
        TEST_CHECK(Throws([&] { BindCudaThreads(MakeBoundedMatmulProofCase(mode), MakeCudaTarget()); }),
                   "unsafe bounded address, initialization, extent ABI or launch was admitted (mode " + std::to_string(mode) + ")");
    }
    const auto empty = BindCudaThreads(MakeBoundedMatmulProofCase(14), MakeCudaTarget());
    TEST_CHECK(empty.prim_func()->attrs.at(kxc::String(kCudaWorkSizeAttr)).As<IntImmNode>()->value == 0,
               "a zero upper bound must produce a guarded empty kernel");
    return true;
}

kxc::tir::PrimFunc MakeBoundedMultiStageProofCase(int mode) {
    using namespace kxc;
    using namespace kxc::tir;
    const auto f32 = DataType::Float(32), i32 = DataType::Int(32),
               i64 = DataType::Int(64), u64 = DataType::UInt(64);
    tir::Var input("input", f32), output("output", f32), scratch("scratch", f32), sum("sum", f32);
    tir::Var rows("rows", u64), columns("columns", u64), i("i", i32), j("j", i32), r("r", i32);
    const auto wide = [i64](const PrimExpr& x) { return PrimExpr(tir::Call(i64, "cast", {x})); };
    const PrimExpr N = wide(Load(rows, IntImm(0))), S = wide(Load(columns, IntImm(0)));
    const PrimExpr index = wide(i) * S + wide(j);
    const auto full = [&](Stmt body) { return For(i, IntImm(0), N, ForType::Serial,
        For(j, IntImm(0), S, ForType::Serial, body)); };
    Stmt copy = full(Store(scratch, mode == 5 ? PrimExpr(Load(scratch, index)) : PrimExpr(Load(input, index)),
        index, mode == 6 ? PrimExpr(IntImm(0, DataType::Bool())) : PrimExpr()));
    PrimExpr reduction_index = wide(i) * S + wide(r);
    if (mode == 2) reduction_index = wide(r);  // flat-in-bounds, wrong owner's row
    if (mode == 3) reduction_index = reduction_index + IntImm(1, i64);
    if (mode == 13) reduction_index = tir::Call(i64, "cast", {Load(input, IntImm(0))});
    const Stmt reduction = For(i, IntImm(0), N, ForType::Serial,
        For(j, IntImm(0), IntImm(1), ForType::Serial, SeqStmt({
            Store(sum, FloatImm(0.0f), wide(i)),
            For(r, IntImm(0), S, ForType::Serial,
                Store(sum, Load(sum, wide(i)) + Load(scratch, reduction_index), wide(i)))})));
    const Stmt consume = full(Store(output, Load(scratch, index) / (Load(sum, wide(i)) + FloatImm(1.0f)),
        mode == 4 ? index + IntImm(1, i64) : index));
    Array<Stmt> stages{copy, reduction, consume};
    if (mode == 1) stages = {reduction, copy, consume};
    if (mode == 10) stages = {copy, reduction, copy, consume};
    if (mode == 14) stages = {reduction, consume};
    if (mode == 15) stages = {copy, reduction};
    Stmt body = Allocate(sum, f32, {N, IntImm(1)}, IntImm(1, DataType::Bool()), SeqStmt(stages));
    body = Allocate(scratch, f32, {N, mode == 7 ? S + IntImm(1, i64) : S},
        IntImm(mode == 9 ? 0 : 1, DataType::Bool()), body);
    Map<tir::Var, Buffer> buffers;
    buffers.Set(input, Buffer(input, f32, {N, S}, {}, IntImm(0), "input", 4, 0));
    buffers.Set(output, Buffer(output, f32, {N, S}, {}, IntImm(0), "output", 4, 0));
    for (const auto& scalar : Array<tir::Var>{rows, columns})
        buffers.Set(scalar, Buffer(scalar, u64, {IntImm(1)}, {}, IntImm(0), scalar->name_hint, 8, 0));
    Map<String, ObjectRef> attrs;
    attrs.Set("kxc.input_count", IntImm(1));
    attrs.Set("kxc.runtime_extent_param_start", IntImm(1));
    attrs.Set("kxc.runtime_extent_count", IntImm(2));
    attrs.Set("kxc.constant_count", IntImm(0));
    attrs.Set("kxc.output_param_start", IntImm(3));
    attrs.Set("kxc.output_count", IntImm(1));
    attrs.Set(kCudaRuntimeExtentBoundsAttr, Array<int64_t>{mode == 11 ? 0 : 19,
        mode == 12 ? 0 : mode == 8 ? 20000 : 17});
    return PrimFunc({input, rows, columns, output}, body, buffers, attrs);
}

bool TestBoundedMultiStageOwnership() {
    using namespace kxc::tir;
    const auto original = MakeBoundedMultiStageProofCase(0);
    std::ostringstream before, after;
    kxc::tir::printer::DumpPrimFunc(original, before);
    const auto scheduled = BindCudaThreads(original, MakeCudaTarget(128));
    ScratchAudit scratch;
    scratch.Mutate(scheduled.prim_func());
    TEST_CHECK(scratch.allocations == 2 && scratch.bytes == 72 && scheduled.launch_config().grid_x == 1,
               "bounded stages must allocate one length-17 row plus a scalar per owner");
    kxc::tir::printer::DumpPrimFunc(original, after);
    TEST_CHECK(before.str() == after.str(), "bounded projection mutated its input function");
    for (int mode = 1; mode <= 15; ++mode) {
        if (mode == 11 || mode == 12) {
            (void)BindCudaThreads(MakeBoundedMultiStageProofCase(mode), MakeCudaTarget());
        } else {
            TEST_CHECK(Throws([&] { BindCudaThreads(MakeBoundedMultiStageProofCase(mode), MakeCudaTarget()); }),
                "unsafe bounded stage was admitted (mode " + std::to_string(mode) + ")");
        }
    }
    return true;
}

kxc::tir::PrimFunc MakeBoundedGatherProofCase(int mode) {
    using namespace kxc;
    using namespace kxc::tir;
    const auto f32 = DataType::Float(32), i64 = DataType::Int(64), u64 = DataType::UInt(64);
    const auto index_type = mode == 1 ? DataType::Int(32) : mode == 9 ? f32 : i64;
    tir::Var ids("ids", index_type), table("table", f32), output("output", f32);
    tir::Var n("n", u64), s("s", u64), i("i", i64), j("j", i64), c("c", i64);
    const auto imm = [&](int64_t value) { return IntImm(value, i64); };
    const auto wide = [&](const PrimExpr& value) { return PrimExpr(tir::Call(i64, "cast", {value})); };
    const PrimExpr N = wide(Load(n, imm(0))), S = wide(Load(s, imm(0)));
    const PrimExpr position = i * S + j;
    const PrimExpr index = wide(Load(ids, position));
    PrimExpr compared = mode == 5 ? PrimExpr(tir::Call(DataType::Int(32), "cast", {index})) : index;
    if (mode == 6) compared = compared + imm(7);
    const PrimExpr lower = !(compared < imm(mode == 16 ? -8 : -7)), upper = compared < imm(7);
    PrimExpr guard = lower && upper;
    if (mode == 2) guard = upper;
    if (mode == 3) guard = lower;
    if (mode == 17) guard = IntImm(1, DataType::Bool());
    // Both addresses are within N*S; only the guarded value has range facts.
    PrimExpr used = mode == 4 ? wide(Load(ids, j * N + i)) : index;
    if (mode == 15) used = wide(Load(ids, j + S * i));
    if (mode == 19) used = wide(Load(ids, position + imm(1)));
    PrimExpr normalized = Select(used < imm(0), used + imm(mode == 7 ? 8 : 7), used);
    if (mode == 14) normalized = tir::Call(DataType::Int(32), "cast", {normalized});
    const PrimExpr address = position * imm(4) + c;
    PrimExpr value = Select(guard, Load(table, normalized * imm(4) + c), FloatImm(0, f32));
    if (mode == 11) value = Select(guard, Load(output, wide(Load(ids, position))), FloatImm(0, f32));
    Stmt body = For(i, imm(0), N, ForType::Serial,
        For(j, imm(0), S, ForType::Serial,
            For(c, imm(0), imm(4), ForType::Serial, Store(mode == 10 ? table : output, value, address))));
    Map<tir::Var, Buffer> buffers;
    buffers.Set(ids, Buffer(ids, index_type, {N, S}, {}, imm(0), "ids", 8, 0));
    buffers.Set(table, Buffer(table, f32, {mode == 18 ? N : PrimExpr(imm(7)), imm(mode == 8 ? 3 : 4)}, {}, imm(0), "table", 4, 0));
    buffers.Set(output, Buffer(output, f32, {N, S, imm(4)}, {}, imm(0), "output", 4, 0));
    for (const auto& scalar : Array<tir::Var>{n, s})
        buffers.Set(scalar, Buffer(scalar, u64, {imm(1)}, {}, imm(0), scalar->name_hint, 8, 0));
    Map<String, ObjectRef> attrs;
    attrs.Set("kxc.input_count", IntImm(2));
    attrs.Set("kxc.runtime_extent_param_start", IntImm(2));
    attrs.Set("kxc.runtime_extent_count", IntImm(2));
    attrs.Set("kxc.constant_count", IntImm(0));
    attrs.Set("kxc.output_param_start", IntImm(4));
    attrs.Set("kxc.output_count", IntImm(1));
    attrs.Set(kCudaRuntimeExtentBoundsAttr, Array<int64_t>{mode == 12 ? 0 : 19, mode == 13 ? 0 : 17});
    return PrimFunc({ids, table, n, s, output}, body, buffers, attrs);
}

bool TestBoundedGuardedGather() {
    using namespace kxc::tir;
    for (int mode = 0; mode <= 19; ++mode) {
        const bool valid = mode == 0 || mode == 1 || mode == 12 || mode == 13 || mode == 14 || mode == 15;
        if (!valid) {
            TEST_CHECK(Throws([&] { BindCudaThreads(MakeBoundedGatherProofCase(mode), MakeCudaTarget()); }),
                "unsafe bounded Gather admitted (mode " + std::to_string(mode) + ")");
            continue;
        }
        const auto original = MakeBoundedGatherProofCase(mode);
        std::ostringstream before, after;
        kxc::tir::printer::DumpPrimFunc(original, before);
        const auto scheduled = BindCudaThreads(original, MakeCudaTarget());
        kxc::tir::printer::DumpPrimFunc(original, after);
        TEST_CHECK(before.str() == after.str() && scheduled.prim_func().defined(),
                   "bounded Gather proof mutated its input");
    }
    return true;
}

kxc::tir::PrimFunc MakeBoundedShapeProofCase(int mode) {
    using namespace kxc;
    using namespace kxc::tir;
    const auto f32 = DataType::Float(32), i64 = DataType::Int(64), u64 = DataType::UInt(64);
    tir::Var input("input", f32), output("output", f32), n("n", u64), s("s", u64);
    tir::Var i("i", i64), j("j", i64), c("c", i64);
    const auto imm = [&](int64_t value) { return IntImm(value, i64); };
    const auto wide = [&](const PrimExpr& value) { return PrimExpr(tir::Call(i64, "cast", {value})); };
    const PrimExpr N = wide(Load(n, imm(0))), S = wide(Load(s, imm(0)));
    const PrimExpr columns = mode == 12 ? PrimExpr(imm(1)) : mode == 18 ? S + imm(1) : S;
    const PrimExpr row = i * columns + j, flat = row * imm(96) + c;
    Array<PrimExpr> input_shape{N, columns, imm(96)};
    PrimExpr index = flat;
    if (mode == 0) {  // Reshape's actual symbolic delinearization/relinearization.
        index = (((flat / imm(96) / S) % N) * S + (flat / imm(96)) % S) * imm(96) + flat % imm(96);
    } else if (mode == 1 || mode == 14) {
        index = row * imm(96) + (c / imm(12)) * imm(mode == 14 ? 13 : 12) + c % imm(12);
    } else if (mode == 2) {  // Unsqueeze retains the singleton axis in its index.
        index = ((row * imm(1) * imm(96) + c) / imm(96)) * imm(96) + flat % imm(96);
    } else if (mode == 3 || mode == 4) {
        index = (Select((mode == 4 ? S : N) == imm(1), imm(0), i) * S +
                 Select(S == imm(1), imm(0), j)) * imm(96) + c;
    } else if (mode == 5) {
        input_shape = {N, S, imm(1)};
        index = row + Select(IntImm(1, DataType::Bool()), imm(0), c);
    } else if (mode == 6 || mode == 7) {
        input_shape = {imm(mode == 6 ? 128 : 7), imm(96)};
        index = j * imm(96) + c;  // Slice a prefix of a static position table.
    } else if (mode == 12 || mode == 18) {
        index = flat / S;  // S can be zero even though the output is nonempty.
    } else if (mode == 13) {
        index = flat % (N * S * imm(96)) + imm(1);
    } else if (mode == 20) {
        index = (i * imm(INT64_MAX)) / imm(96) + c;
    } else if (mode == 21) {
        index = flat - imm(1);
    } else if (mode >= 22 && mode <= 25) {
        input_shape = {imm(mode < 24 ? (mode == 22 ? 12 : 11) : (mode == 24 ? 8 : 7))};
        index = mode < 24 ? c % imm(12) : c / imm(12);
    }
    PrimExpr value = Load(input, index);
    if ((mode >= 8 && mode <= 11) || mode == 26 || mode == 27) {
        input_shape = {N, S, imm(48)};
        const PrimExpr coordinate = mode == 11 ? PrimExpr(tir::Call(DataType::Int(32), "cast", {c})) : PrimExpr(c);
        const PrimExpr left = Load(input, row * imm(48) + c);
        const PrimExpr right = Load(input, row * imm(48) + c - imm(mode == 10 ? 47 : 48));
        value = mode == 27 ? PrimExpr(Select(imm(47) < coordinate, right, left))
            : PrimExpr(Select(coordinate < imm(mode == 9 ? 49 : 48), mode == 26 ? right : left, right));
    } else if (mode >= 15 && mode <= 17) {
        input_shape = {imm(2)};
        const PrimExpr read = Load(input, c / imm(4));
        value = Select(c < imm(8), read, mode == 16 ? read : PrimExpr(FloatImm(0, f32)));
        // A quotient proved in c<8 must not retain that bound outside it or
        // inside its sibling branch, where c may reach 95.
        if (mode == 15) value = value + read;
    }
    const Stmt body = For(i, imm(0), N, ForType::Serial,
        For(j, imm(0), columns, ForType::Serial,
            For(c, imm(0), imm(96), ForType::Serial, Store(output, value, flat))));
    Map<tir::Var, Buffer> buffers;
    buffers.Set(input, Buffer(input, f32, input_shape, {}, imm(0), "input", 4, 0));
    buffers.Set(output, Buffer(output, f32, {N, columns, imm(96)}, {}, imm(0), "output", 4, 0));
    for (const auto& scalar : Array<tir::Var>{n, s})
        buffers.Set(scalar, Buffer(scalar, u64, {imm(1)}, {}, imm(0), scalar->name_hint, 8, 0));
    Map<String, ObjectRef> attrs;
    attrs.Set("kxc.input_count", IntImm(1));
    attrs.Set("kxc.runtime_extent_param_start", IntImm(1));
    attrs.Set("kxc.runtime_extent_count", IntImm(2));
    attrs.Set("kxc.constant_count", IntImm(0));
    attrs.Set("kxc.output_param_start", IntImm(3));
    attrs.Set("kxc.output_count", IntImm(1));
    attrs.Set(kCudaRuntimeExtentBoundsAttr, Array<int64_t>{3, mode == 19 ? 0 : 8});
    return PrimFunc({input, n, s, output}, body, buffers, attrs);
}

bool TestBoundedShapeAddresses() {
    using namespace kxc::tir;
    for (int mode = 0; mode <= 27; ++mode) {
        const bool valid = mode <= 3 || mode == 5 || mode == 6 || mode == 8 ||
            mode == 17 || mode == 19 || mode == 22 || mode == 24 || mode == 27;
        if (!valid) {
            TEST_CHECK(Throws([&] { BindCudaThreads(MakeBoundedShapeProofCase(mode), MakeCudaTarget()); }),
                "unsafe/unsupported bounded shape address admitted (mode " + std::to_string(mode) + ")");
            continue;
        }
        const auto original = MakeBoundedShapeProofCase(mode);
        std::ostringstream before, after;
        kxc::tir::printer::DumpPrimFunc(original, before);
        const auto scheduled = BindCudaThreads(original, MakeCudaTarget());
        kxc::tir::printer::DumpPrimFunc(original, after);
        TEST_CHECK(before.str() == after.str() && scheduled.prim_func().defined(),
                   "bounded shape proof mutated its input");
    }
    return true;
}

kxc::tir::PrimFunc MakeBoundedAppendProofCase(int mode) {
    using namespace kxc;
    using namespace kxc::tir;
    const auto f32 = DataType::Float(32), i64 = DataType::Int(64), u64 = DataType::UInt(64);
    tir::Var past("past", f32), next("next", f32), output("output", f32), n("n", u64), p("p", u64);
    tir::Var i("i", i64), j("j", i64), c("c", i64);
    const auto imm = [&](int64_t value) { return IntImm(value, i64); };
    const auto wide = [&](const PrimExpr& value) { return PrimExpr(tir::Call(i64, "cast", {value})); };
    const PrimExpr N = wide(Load(n, imm(0))), P = wide(Load(p, imm(0)));
    const int64_t count = mode == 2 ? 0 : mode == 1 || mode == 5 || mode == 11 ? 2 : mode == 20 ? 9 : 1;
    const PrimExpr C = imm(count), extent = P + C;
    const PrimExpr index = (i * extent + j) * imm(4) + c;
    PrimExpr prefix = (i * P + (mode == 4 ? j % P : PrimExpr(j))) * imm(4) + c;
    const PrimExpr offset = mode == 9 ? P + imm(1) : mode == 10 ? P - imm(1) : P;
    PrimExpr local = j - offset;
    if (mode == 5) local = local % C;
    const PrimExpr suffix = (i * C + local) * imm(4) + c;
    PrimExpr coordinate = mode == 15 ? PrimExpr(tir::Call(DataType::Int(32), "cast", {j})) : PrimExpr(j);
    if (mode == 16) coordinate = i;
    const PrimExpr boundary = mode == 7 ? P + imm(1) : mode == 8 ? PrimExpr(imm(1)) : P;
    PrimExpr left = Load(past, prefix), right = Load(next, suffix);
    if (mode == 13) left = Load(output, index);
    if (mode == 19) right = Load(past, (i * P + j % P) * imm(4) + c);
    if (mode == 20) left = Load(past, j / imm(4));
    PrimExpr value = Select(coordinate < boundary, left, right);
    if (mode == 6) value = left;
    if (mode == 17) value = value + right;
    if (mode == 18) value = value + Load(next, j / P);
    if (mode == 20) value = value + left;
    Stmt body = For(i, imm(0), N, ForType::Serial,
        For(j, imm(0), extent, ForType::Serial,
            For(c, imm(0), imm(4), ForType::Serial,
                Store(mode == 14 ? past : output, value, mode == 12 ? index + imm(1) : index))));
    Map<tir::Var, Buffer> buffers;
    buffers.Set(past, Buffer(past, f32, mode == 20 ? Array<PrimExpr>{imm(2)} : Array<PrimExpr>{N,P,imm(4)}, {}, imm(0), "past", 4, 0));
    buffers.Set(next, Buffer(next, f32, {N,imm(mode == 11 ? count - 1 : count),imm(4)}, {}, imm(0), "next", 4, 0));
    buffers.Set(output, Buffer(output, f32, {N,extent,imm(4)}, {}, imm(0), "output", 4, 0));
    for (const auto& scalar : Array<tir::Var>{n,p})
        buffers.Set(scalar, Buffer(scalar, u64, {imm(1)}, {}, imm(0), scalar->name_hint, 8, 0));
    Map<String, ObjectRef> attrs;
    attrs.Set("kxc.input_count", IntImm(2));
    attrs.Set("kxc.runtime_extent_param_start", IntImm(2));
    attrs.Set("kxc.runtime_extent_count", IntImm(2));
    attrs.Set("kxc.constant_count", IntImm(0));
    attrs.Set("kxc.output_param_start", IntImm(4));
    attrs.Set("kxc.output_count", IntImm(1));
    attrs.Set(kCudaRuntimeExtentBoundsAttr, Array<int64_t>{3,mode == 3 ? 0 : 8});
    return PrimFunc({past,next,n,p,output},body,buffers,attrs);
}

bool TestBoundedAppendAddresses() {
    using namespace kxc::tir;
    for (int mode = 0; mode <= 20; ++mode) {
        if (mode >= 6) {
            TEST_CHECK(Throws([&] { BindCudaThreads(MakeBoundedAppendProofCase(mode), MakeCudaTarget()); }),
                       "unsafe bounded append admitted (mode " + std::to_string(mode) + ")");
            continue;
        }
        const auto original = MakeBoundedAppendProofCase(mode);
        std::ostringstream before, after;
        kxc::tir::printer::DumpPrimFunc(original, before);
        const auto scheduled = BindCudaThreads(original, MakeCudaTarget());
        kxc::tir::printer::DumpPrimFunc(original, after);
        TEST_CHECK(before.str() == after.str() && scheduled.prim_func().defined(),
                   "bounded append proof mutated its input");
    }
    return true;
}

}  // namespace

// 顺序运行全部契约测试，使 CI 输出保留具体失败类别。
int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"elementwise_schedule", TestElementwiseSchedule},
        {"relu_pipeline_registration", TestReluPipelineRegistration},
        {"widened_store_index_schedule", TestWidenedStoreIndexSchedule},
        {"owned_reduction_and_scalar", TestOwnedReductionAndScalar},
        {"production_multistage_schedule", TestProductionMultiStageSchedule},
        {"multistage_ownership_rejections", TestMultiStageOwnershipRejections},
        {"bounded_compact_ownership", TestBoundedCompactOwnership},
        {"bounded_multistage_ownership", TestBoundedMultiStageOwnership},
        {"bounded_guarded_gather", TestBoundedGuardedGather},
        {"bounded_shape_addresses", TestBoundedShapeAddresses},
        {"bounded_append_addresses", TestBoundedAppendAddresses},
        {"reject_unsafe_owned_reads", TestRejectUnsafeOwnedReads},
        {"reject_narrow_index_overflow", TestRejectNarrowIndexOverflow},
        {"reject_reduction_and_conflict", TestRejectReductionAndWriteConflict},
        {"reject_indirect_gather_load", TestRejectIndirectGatherLoad},
        {"production_guarded_gather", TestProductionGuardedGather},
        {"guarded_indirect_boundaries", TestGuardedIndirectBoundaries},
        {"reject_invalid_contracts", TestRejectInvalidContracts},
        {"reject_allocation_and_launch_overflow",
         TestRejectAllocationAndLaunchOverflow},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
            } else {
                std::cout << "[PASS] " << test.first << "\n";
            }
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
