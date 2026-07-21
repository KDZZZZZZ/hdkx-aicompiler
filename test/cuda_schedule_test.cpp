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

#include "base/pass.h"
#include "tir/pass/print_ir.h"
#include "tir/transforms/bind_cuda_threads.h"
#include "tir/transforms/pipeline.h"

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
    codegen::KernelLaunchMetadata metadata = result.launch_metadata();
    TEST_CHECK(metadata->grid.x == 8 && metadata->block.x == 128,
               "grid/block dimensions were not derived from work size");
    TEST_CHECK(GetCudaLaunchMetadata(scheduled).get() == metadata.get(),
               "PrimFunc attr and schedule result must share metadata identity");

    const auto* block = scheduled->body.As<ThreadBindingNode>();
    TEST_CHECK(block && block->thread_index == ThreadIndexKind::kBlockIdxX,
               "outer blockIdx.x binding is missing");
    const auto* thread = block->body.As<ThreadBindingNode>();
    TEST_CHECK(thread && thread->thread_index == ThreadIndexKind::kThreadIdxX,
               "inner threadIdx.x binding is missing");
    TEST_CHECK(thread->body.As<IfThenElseNode>() != nullptr,
               "tail threads must be protected by a bounds guard");

    std::ostringstream text;
    pass::DumpPrimFunc(scheduled, text);
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
    codegen::KernelLaunchMetadata metadata = GetCudaLaunchMetadata(scheduled);
    TEST_CHECK(metadata->grid.x == 2 && metadata->block.x == 64,
               "pipeline adapter produced incorrect relu launch dimensions");
    return true;
}

// reduction 具有内层循环和输出读写依赖，第一阶段必须在 codegen 前拒绝。
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
               "nested reduction loop should be rejected");

    Stmt conflict = For(
        i, IntImm(0, i64), IntImm(8, i64), ForType::Serial,
        Store(out, FloatImm(1.0, f32), IntImm(0, i64)));
    TEST_CHECK(Throws([&] {
                   BindCudaThreads(PrimFunc({out}, conflict), MakeCudaTarget());
               }),
               "non-injective Store index should be rejected");
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

}  // namespace

// 顺序运行全部契约测试，使 CI 输出保留具体失败类别。
int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"elementwise_schedule", TestElementwiseSchedule},
        {"relu_pipeline_registration", TestReluPipelineRegistration},
        {"reject_reduction_and_conflict", TestRejectReductionAndWriteConflict},
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
