/*! \file test/te_schedule_test.cpp
 * \brief Verifies that TE schedules are validated, consumed, identified, and target-aware.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/kernel_abi_builder.h"
#include "../src/compiler/internal/primitive_cache.h"
#include "../src/compiler/internal/te_to_tir.h"
#include "kxc/compiler/compiler.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/ndarray.h"
#include "kxc/te/te.h"
#include "kxc/tir/printer/print_ir.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"

#if KXC_USE_LLVM
#include <llvm/IR/LLVMContext.h>

#include "../src/codegen/llvm/internal/codegen_llvm.h"
#include "../src/codegen/llvm/internal/llvm_jit.h"
#endif

namespace {

#define CHECK(condition, message)                                                \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << '\n'; \
            return false;                                                         \
        }                                                                         \
    } while (false)

bool Throws(const std::function<void()>& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

kxc::Target SyntheticCudaTarget() {
    auto* node = new kxc::TargetNode();
    node->kind = "cuda";
    node->device_type = kxc::kCUDA;
    node->device_id = 0;
    node->attrs.exists = 1;
    node->attrs.max_threads_per_block = 128;
    node->attrs.max_shared_memory_per_block = 0;
    node->attrs.warp_size = 32;
    node->attrs.compute_version = "7.5";
    node->attrs.compute_version_major = 7;
    node->attrs.compute_version_minor = 5;
    return kxc::Target(kxc::ObjectRef(node));
}

kxc::relay::LoweredFunction Lower(
    const kxc::te::Tensor& input, const kxc::te::Tensor& output,
    const kxc::te::Schedule& schedule, const kxc::Target& target,
    const char* symbol) {
    return kxc::relay::internal::LowerTensorGraphToTIR(
        {input}, {}, {output}, schedule, target,
        kxc::relay::internal::PrimFuncIdentity{kxc::String(symbol)});
}

void CollectForTypes(const kxc::tir::Stmt& statement,
                     std::vector<kxc::tir::ForType>* types,
                     std::vector<std::string>* names) {
    if (!statement.defined()) return;
    if (const auto* loop = statement.As<kxc::tir::ForNode>()) {
        types->push_back(loop->for_type);
        names->push_back(loop->loop_var->name_hint);
        CollectForTypes(loop->body, types, names);
        return;
    }
    if (const auto* allocation = statement.As<kxc::tir::AllocateNode>()) {
        CollectForTypes(allocation->body, types, names);
        return;
    }
    if (const auto* branch = statement.As<kxc::tir::IfThenElseNode>()) {
        CollectForTypes(branch->then_case, types, names);
        CollectForTypes(branch->else_case, types, names);
        return;
    }
    if (const auto* sequence = statement.As<kxc::tir::SeqStmtNode>()) {
        for (const kxc::tir::Stmt& child : sequence->seq) {
            CollectForTypes(child, types, names);
        }
        return;
    }
    if (const auto* let = statement.As<kxc::tir::LetStmtNode>()) {
        CollectForTypes(let->body, types, names);
    }
}

bool ContainsIf(const kxc::tir::Stmt& statement) {
    if (!statement.defined()) return false;
    if (statement.As<kxc::tir::IfThenElseNode>()) return true;
    if (const auto* loop = statement.As<kxc::tir::ForNode>()) {
        return ContainsIf(loop->body);
    }
    if (const auto* allocation = statement.As<kxc::tir::AllocateNode>()) {
        return ContainsIf(allocation->body);
    }
    if (const auto* sequence = statement.As<kxc::tir::SeqStmtNode>()) {
        for (const kxc::tir::Stmt& child : sequence->seq) {
            if (ContainsIf(child)) return true;
        }
    }
    return false;
}

bool ContainsThreadBinding(const kxc::tir::Stmt& statement) {
    if (!statement.defined()) return false;
    if (statement.As<kxc::tir::ThreadBindingNode>()) return true;
    if (const auto* loop = statement.As<kxc::tir::ForNode>()) {
        return ContainsThreadBinding(loop->body);
    }
    if (const auto* allocation = statement.As<kxc::tir::AllocateNode>()) {
        return ContainsThreadBinding(allocation->body);
    }
    if (const auto* branch = statement.As<kxc::tir::IfThenElseNode>()) {
        return ContainsThreadBinding(branch->then_case) ||
               ContainsThreadBinding(branch->else_case);
    }
    if (const auto* sequence = statement.As<kxc::tir::SeqStmtNode>()) {
        for (const kxc::tir::Stmt& child : sequence->seq) {
            if (ContainsThreadBinding(child)) return true;
        }
    }
    return false;
}

bool TestScheduleCollectsProducerDag() {
    using namespace kxc;
    const te::Tensor input = te::placeholder({8}, tir::DataType::Float(32), "input");
    const te::Tensor intermediate = te::compute(
        {8}, [input](const Array<tir::Var>& axis) {
            return input(axis) + tir::FloatImm(1.0f);
        }, "intermediate");
    const te::Tensor output = te::compute(
        {8}, [intermediate](const Array<tir::Var>& axis) {
            return intermediate(axis) * tir::FloatImm(2.0f);
        }, "output");
    const te::Schedule schedule = te::create_schedule({output->op});
    CHECK(schedule->stages.size() == 3 &&
              schedule->stages[0]->op.get() == input->op.get() &&
              schedule->stages[1]->op.get() == intermediate->op.get() &&
              schedule->stages[2]->op.get() == output->op.get(),
          "create_schedule did not retain producer-first DAG stages");
    CHECK(schedule->op_map.count(input->op) &&
              schedule->op_map.count(intermediate->op) &&
              schedule->op_map.count(output->op),
          "producer DAG operations are absent from op_map");
    return true;
}

bool TestManualScheduleChangesTIRAndIdentity() {
    using namespace kxc;
    const Target cpu = BuildTarget(Device::CPU());
    const te::Tensor input = te::placeholder(
        {2, 3, 8}, tir::DataType::Float(32), "manual_input");
    const te::Tensor output = te::compute(
        {2, 3, 8}, [input](const Array<tir::Var>& axis) {
            return input(axis) + tir::FloatImm(1.0f);
        }, "manual_output");

    const te::Schedule identity = te::create_schedule({output->op});
    const te::Schedule schedule = te::create_schedule({output->op});
    te::Stage stage = schedule[output->op];
    const te::IterVar axis0 = stage->root_iter_vars[0];
    const te::IterVar axis1 = stage->root_iter_vars[1];
    const te::IterVar axis2 = stage->root_iter_vars[2];
    te::IterVar outer;
    te::IterVar inner;
    stage.split(axis2, tir::IntImm(4), &outer, &inner);
    stage.reorder({axis0, outer, axis1, inner});
    stage.parallel(axis0);
    stage.unroll(axis1);
    stage.vectorize(inner);

    const relay::LoweredFunction lowered =
        Lower(input, output, schedule, cpu, "manual_schedule");
    std::vector<tir::ForType> types;
    std::vector<std::string> names;
    CollectForTypes(lowered->prim_func->body, &types, &names);
    const std::vector<tir::ForType> expected_types{
        tir::ForType::Parallel, tir::ForType::Serial,
        tir::ForType::Unrolled, tir::ForType::Vectorized};
    CHECK(types == expected_types,
          "split/reorder/parallel/unroll/vectorize were not materialized");
    CHECK(names.size() == 4 && names[0] == "ax0" &&
              names[1] == "ax2.outer" && names[2] == "ax1" &&
              names[3] == "ax2.inner",
          "scheduled leaf order is not observable in TIR");

    std::ostringstream text;
    tir::printer::DumpPrimFunc(lowered->prim_func, text);
    CHECK(text.str().find("for[parallel]") != std::string::npos &&
              text.str().find("for[unrolled]") != std::string::npos &&
              text.str().find("for[vectorized]") != std::string::npos,
          "TIR text hides scheduled loop kinds");

    const std::string actual =
        relay::internal::GetTEScheduleContract(lowered->prim_func);
    const std::string expected =
        relay::internal::CanonicalTEScheduleContract(schedule, cpu);
    const std::string baseline =
        relay::internal::CanonicalTEScheduleContract(identity, cpu);
    CHECK(actual == expected && actual != baseline,
          "PrimFunc does not retain its exact schedule contract");
    const api::PrimitiveArtifactKey scheduled_key =
        api::internal::BuildPrimitiveArtifactKey(
            api::UnitSemanticKey("te-schedule-unit"), cpu, "pipeline", actual.c_str(),
            "llvm-test");
    const api::PrimitiveArtifactKey baseline_key =
        api::internal::BuildPrimitiveArtifactKey(
            api::UnitSemanticKey("te-schedule-unit"), cpu, "pipeline", baseline.c_str(),
            "llvm-test");
    CHECK(scheduled_key != baseline_key,
          "artifact identity aliases distinct real schedules");
    return true;
}

bool TestScheduleValidation() {
    using namespace kxc;
    const te::Tensor input = te::placeholder({2, 8}, tir::DataType::Float(32), "input");
    const te::Tensor output = te::compute(
        {2, 8}, [input](const Array<tir::Var>& axis) { return input(axis); },
        "output");
    te::Schedule schedule = te::create_schedule({output->op});
    te::Stage stage = schedule[output->op];
    const te::IterVar axis0 = stage->root_iter_vars[0];
    const te::IterVar axis1 = stage->root_iter_vars[1];
    CHECK(Throws([&] { stage.split(axis1, tir::IntImm(0)); }),
          "zero split factor was accepted");
    CHECK(Throws([&] { stage.reorder({axis0, axis0}); }),
          "duplicate reorder was accepted");
    te::IterVar outer;
    te::IterVar inner;
    stage.split(axis1, tir::IntImm(4), &outer, &inner);
    CHECK(Throws([&] { stage.split(axis1, tir::IntImm(2)); }),
          "a non-leaf parent was split twice");
    stage.vectorize(inner);
    CHECK(Throws([&] { stage.unroll(inner); }),
          "conflicting leaf annotations were accepted");

    const te::IterVar reduction = te::reduce_axis(0, 8, "reduce");
    const te::Tensor reduced = te::compute(
        {2}, [input, reduction](const Array<tir::Var>& axis) {
            return te::sum(input(axis[0], reduction), {reduction});
        }, "reduced");
    te::Schedule reduction_schedule = te::create_schedule({reduced->op});
    te::Stage reduction_stage = reduction_schedule[reduced->op];
    const te::IterVar data = reduction_stage->root_iter_vars[0];
    const te::IterVar reduce = reduction_stage->root_iter_vars[1];
    CHECK(Throws([&] { reduction_stage.parallel(reduce); }) &&
              Throws([&] { reduction_stage.vectorize(reduce); }) &&
              Throws([&] { reduction_stage.reorder({reduce, data}); }),
          "unsafe reduction scheduling was accepted");
    reduction_stage.unroll(reduce);
    return true;
}

bool TestCudaKeepsSingleBindingAuthority() {
    using namespace kxc;
    const Target cuda = SyntheticCudaTarget();
    const te::Tensor input = te::placeholder({8}, tir::DataType::Float(32), "cuda_input");
    const te::Tensor output = te::compute(
        {8}, [input](const Array<tir::Var>& axis) {
            return input(axis) + tir::FloatImm(1.0f);
        }, "cuda_output");
    const te::Schedule schedule =
        relay::internal::BuildDefaultTESchedule({output}, cuda);
    const relay::LoweredFunction lowered =
        Lower(input, output, schedule, cuda, "cuda_schedule");
    const auto* loop = lowered->prim_func->body.As<tir::ForNode>();
    CHECK(loop && loop->for_type == tir::ForType::Serial &&
              !ContainsThreadBinding(lowered->prim_func->body) &&
              !lowered->prim_func->attrs.count(
                  String(tir::kCudaLaunchMetadataAttr)),
          "TE scheduling duplicated CUDA thread/launch binding");
    const tir::CudaScheduleResult bound =
        tir::BindCudaThreads(lowered->prim_func, cuda);
    CHECK(ContainsThreadBinding(bound.prim_func()->body) &&
              bound.launch_config().block_x > 0,
          "existing BindCudaThreads did not remain the launch authority");

    te::Schedule conflicting = te::create_schedule({output->op});
    conflicting[output->op].parallel(
        conflicting[output->op]->root_iter_vars[0]);
    CHECK(Throws([&] {
              (void)Lower(input, output, conflicting, cuda,
                          "bad_cuda_schedule");
          }),
          "CUDA accepted a competing TE parallel schedule");

    auto* bad_node = new TargetNode();
    bad_node->kind = "cuda";
    bad_node->device_type = kCPU;
    bad_node->device_id = 0;
    const Target inconsistent{ObjectRef(bad_node)};
    CHECK(Throws([&] {
              (void)relay::internal::BuildDefaultTESchedule({output},
                                                            inconsistent);
          }),
          "inconsistent Target did not fail closed");
    return true;
}

#if KXC_USE_LLVM
bool ExecuteScheduledLLVM(const kxc::te::Tensor& input,
                          const kxc::te::Tensor& output,
                          const kxc::te::Schedule& schedule,
                          const kxc::Target& target,
                          const std::string& symbol,
                          const std::vector<float>& values) {
    using namespace kxc;
    const relay::LoweredFunction lowered =
        Lower(input, output, schedule, target, symbol.c_str());
    const codegen::KernelSignature signature =
        codegen::BuildKernelSignature(lowered->prim_func, {}, target, symbol);
    auto context = std::make_unique<llvm::LLVMContext>();
    codegen::CodeGenLLVM codegen(*context);
    codegen.AddFunction(lowered->prim_func, symbol);
    const codegen::CompiledKernel kernel = codegen::LLVMJITEngine().Compile(
        codegen.TakeModule(), std::move(context), signature,
        codegen::KernelLaunchMetadata(
            Device::CPU(), codegen::CodeGenBackend::kLLVM),
        2);
    runtime::NDArray source = runtime::NDArray::Empty(
        {static_cast<int64_t>(values.size())},
        runtime::DataTypeFromString("float32"), Device::CPU());
    runtime::NDArray destination = runtime::NDArray::Empty(
        {static_cast<int64_t>(values.size())},
        runtime::DataTypeFromString("float32"), Device::CPU());
    source.CopyFromBytes(values.data(), values.size() * sizeof(float));
    kernel.Launch({source, destination},
                  DeviceStream::Default(Device::CPU())).Wait();
    std::vector<float> actual(values.size());
    destination.CopyToBytes(actual.data(), actual.size() * sizeof(float));
    for (size_t index = 0; index < values.size(); ++index) {
        if (actual[index] != values[index] + 1.0f) return false;
    }
    return true;
}
#endif

bool TestCpuDefaultScheduleAndLLVMNumerics() {
    using namespace kxc;
    const Target cpu = BuildTarget(Device::CPU());
    const te::Tensor input = te::placeholder({16}, tir::DataType::Float(32), "input");
    const te::Tensor output = te::compute(
        {16}, [input](const Array<tir::Var>& axis) {
            return input(axis) + tir::FloatImm(1.0f);
        }, "output");
    const te::Schedule schedule =
        relay::internal::BuildDefaultTESchedule({output}, cpu);
    const te::Stage stage = schedule[output->op];
    CHECK(schedule->policy ==
              relay::internal::kDefaultTESchedulePolicy &&
              stage->split_relations.size() == 1 &&
              stage->leaf_iter_vars.size() == 2 &&
              stage->leaf_iter_vars[0]->iter_type ==
                  te::IterVarType::kParallel &&
              stage->leaf_iter_vars[1]->iter_type ==
                  te::IterVarType::kVectorized,
          "CPU default policy did not produce its target-aware skeleton");

    const te::Tensor tail_input = te::placeholder(
        {10}, tir::DataType::Float(32), "tail_input");
    const te::Tensor tail_output = te::compute(
        {10}, [tail_input](const Array<tir::Var>& axis) {
            return tail_input(axis) + tir::FloatImm(1.0f);
        }, "tail_output");
    te::Schedule tail_schedule = te::create_schedule({tail_output->op});
    te::Stage tail_stage = tail_schedule[tail_output->op];
    te::IterVar tail_outer;
    te::IterVar tail_inner;
    tail_stage.split(tail_stage->root_iter_vars[0], tir::IntImm(4),
                     &tail_outer, &tail_inner);
    tail_stage.parallel(tail_outer);
    tail_stage.vectorize(tail_inner);
    const relay::LoweredFunction tail_lowered = Lower(
        tail_input, tail_output, tail_schedule, cpu, "tail_schedule");
    CHECK(ContainsIf(tail_lowered->prim_func->body),
          "non-divisible split did not materialize a tail predicate");
#if KXC_USE_LLVM
    std::vector<float> values(16);
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<float>(index * 3);
    }
    CHECK(ExecuteScheduledLLVM(input, output, schedule, cpu,
                               "default_schedule_numeric", values),
          "LLVM numerical result changed under the CPU default schedule");
    const Var relay_input("relay_input", TensorType({16}, "float32"));
    const api::CompiledGraph compiled = api::Compiler::Compile(
        Function({relay_input},
                 Call(relay::Op::Get("nn_relu"), {relay_input})),
        api::CompileConfig::Create(cpu, 0));
    CHECK(compiled.artifact_pins().size() == 1,
          "production schedule test expected one primitive artifact");
    const std::string artifact_identity =
        compiled.artifact_pins()[0].record().artifact_key.canonical_bytes();
    CHECK(artifact_identity.find("kxc.te.schedule.v1") !=
                  std::string::npos &&
              artifact_identity.find(
                  relay::internal::kDefaultTESchedulePolicy) !=
                  std::string::npos,
          "production artifact key omits the real TE schedule contract");
    std::vector<float> tail_values(10);
    for (size_t index = 0; index < tail_values.size(); ++index) {
        tail_values[index] = static_cast<float>(index * 5);
    }
    CHECK(ExecuteScheduledLLVM(tail_input, tail_output, tail_schedule, cpu,
                               "tail_schedule_numeric", tail_values),
          "predicated split tail changed the LLVM numerical result");
#else
    std::cout << "[SKIP] LLVM numerical schedule test: KXC_USE_LLVM=0\n";
#endif
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"producer_dag", TestScheduleCollectsProducerDag},
        {"manual_schedule_tir_identity", TestManualScheduleChangesTIRAndIdentity},
        {"schedule_validation", TestScheduleValidation},
        {"cuda_single_binding_authority", TestCudaKeepsSingleBindingAuthority},
        {"cpu_default_schedule_llvm_numeric", TestCpuDefaultScheduleAndLLVMNumerics},
    };
    for (const auto& [name, test] : tests) {
        try {
            if (!test()) return 1;
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
