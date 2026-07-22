/*! \file src/codegen/llvm/llvm_jit.cc
 * \brief 实现 LLVM 私有 ABI launcher 与每模块独占的 ORC JIT 生命周期。
 */

#if KXC_USE_LLVM

#include "internal/llvm_jit.h"

#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kxc::codegen {
namespace {

#ifndef KXC_MINGW_LIBGCC_PATH
#define KXC_MINGW_LIBGCC_PATH ""
#endif

/*! \brief LLVM 后端唯一允许的私有 call-frame ABI。 */
using LLVMKernelFunction = int32_t (*)(void** data, uint64_t count);

// LLVM 原生目标初始化是进程级操作，call_once 避免并发编译重复初始化全局状态。
void InitializeNativeLLVM() {
    static std::once_flag once;
    std::call_once(once, [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
    });
}

// 让 JIT 内核可以解析当前进程导出的数学库和运行时符号。
void AddHostRuntimeSymbols(llvm::orc::LLJIT* jit) {
    auto generator = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
        jit->getDataLayout().getGlobalPrefix());
    if (!generator) {
        throw std::runtime_error(
            "Failed to add current-process symbols to LLVM JIT: " +
            llvm::toString(generator.takeError()));
    }
    jit->getMainJITDylib().addGenerator(std::move(*generator));
}

// MinGW 的除法等指令可能依赖 libgcc；仅在配置给出归档路径时静态链接。
void AddMinGWRuntimeArchive(llvm::orc::LLJIT* jit) {
#ifdef __MINGW32__
    const char* libgcc_path = KXC_MINGW_LIBGCC_PATH;
    if (libgcc_path[0] == '\0') return;
    if (auto error = jit->linkStaticLibraryInto(jit->getMainJITDylib(), libgcc_path)) {
        throw std::runtime_error("Failed to link MinGW libgcc into LLVM JIT: " +
                                 llvm::toString(std::move(error)));
    }
#else
    (void)jit;
#endif
}

// 在优化前后统一验证 module，错误文本保留 LLVM verifier 的完整上下文。
void VerifyModule(const llvm::Module& module, const char* stage) {
    std::string error;
    llvm::raw_string_ostream stream(error);
    if (llvm::verifyModule(module, &stream)) {
        throw std::runtime_error(std::string("LLVM module verification failed ") + stage +
                                 ": " + error);
    }
}

/*!
 * \brief 持有单个 LLJIT 与解析后的强类型函数地址。
 *
 * launcher 析构前 LLJIT 始终存活。调用帧在每次 Launch 的栈上独立构造，
 * 因此同一编译模块可以被多个 CPU 线程并发执行。
 */
class LLVMKernelLauncher final : public KernelLauncher {
public:
    /*! \brief 接管 JIT、函数地址、符号和固定参数数量。 */
    LLVMKernelLauncher(std::unique_ptr<llvm::orc::LLJIT> jit,
                       LLVMKernelFunction function,
                       String symbol,
                       uint64_t expected_count)
        : jit_(std::move(jit)),
          function_(function),
          symbol_(std::move(symbol)),
          expected_count_(expected_count) {}

    // JIT、函数地址和非空 symbol 同时存在时才允许进入机器码。
    bool IsReady() const noexcept override {
        return jit_ != nullptr && function_ != nullptr && !std::string(symbol_).empty();
    }

    // 将已验证 NDArray 转为 CPU 私有 call frame，并同步调用机器码。
    AsyncOperation Launch(const Array<runtime::NDArray>& arguments,
                          const DeviceStream& stream,
                          const ObjectRef& executable_owner) const override {
        (void)executable_owner;
        if (!IsReady()) throw std::runtime_error("LLVM launcher is not ready");
        if (!stream.defined() || stream.device().device_type() != kCPU) {
            throw std::invalid_argument("LLVM launcher requires a CPU DeviceStream");
        }
        if (arguments.size() != expected_count_) {
            throw std::invalid_argument(
                "LLVM launcher argument count does not match compiled ABI");
        }

        std::vector<void*> call_frame;
        call_frame.reserve(arguments.size());
        Array<Storage> retained;
        for (const auto& argument : arguments) {
            const Storage storage = argument.storage();
            retained.push_back(storage);
            // CompiledModule 已验证 range、溢出和 alignment；此处只形成后端地址值。
            const uintptr_t address =
                reinterpret_cast<uintptr_t>(storage.data()) + argument->byte_offset;
            call_frame.push_back(reinterpret_cast<void*>(address));
        }

        const int32_t result = function_(call_frame.data(), expected_count_);
        if (result != 0) {
            throw std::runtime_error("LLVM kernel '" + std::string(symbol_) +
                                     "' returned ABI error " +
                                     std::to_string(result));
        }
        return AsyncOperation::Completed(stream, std::move(retained));
    }

private:
    /*! \brief 机器码、JITDylib 和 module 的唯一所有者。 */
    std::unique_ptr<llvm::orc::LLJIT> jit_;
    /*! \brief 在 jit_ 生命周期内有效的强类型入口地址。 */
    LLVMKernelFunction function_{nullptr};
    /*! \brief 用于错误诊断的稳定入口符号。 */
    String symbol_;
    /*! \brief 与 LLVM IR 入口 count 门禁一致的参数数。 */
    uint64_t expected_count_{0};
};

}  // namespace

// opt_level=0 不运行优化；其余等级使用 LLVM 官方默认 module pipeline。
void LLVMJITEngine::Optimize(llvm::Module* module, int opt_level) const {
    if (!module) throw std::invalid_argument("LLVM Optimize requires a module");
    if (opt_level < 0 || opt_level > 3) {
        throw std::invalid_argument("LLVM opt_level must be between 0 and 3");
    }
    if (opt_level == 0) return;

    llvm::PassBuilder builder;
    llvm::LoopAnalysisManager loop_analyses;
    llvm::FunctionAnalysisManager function_analyses;
    llvm::CGSCCAnalysisManager cgscc_analyses;
    llvm::ModuleAnalysisManager module_analyses;
    builder.registerModuleAnalyses(module_analyses);
    builder.registerCGSCCAnalyses(cgscc_analyses);
    builder.registerFunctionAnalyses(function_analyses);
    builder.registerLoopAnalyses(loop_analyses);
    builder.crossRegisterProxies(loop_analyses, function_analyses,
                                 cgscc_analyses, module_analyses);

    const llvm::OptimizationLevel level =
        opt_level == 1 ? llvm::OptimizationLevel::O1
                       : (opt_level == 2 ? llvm::OptimizationLevel::O2
                                         : llvm::OptimizationLevel::O3);
    llvm::ModulePassManager pipeline = builder.buildPerModuleDefaultPipeline(level);
    pipeline.run(*module, module_analyses);
}

// 创建独立 LLJIT，转移 module/context，并返回拥有全部 ORC 资源的 launcher。
CompiledKernel LLVMJITEngine::Compile(
    std::unique_ptr<llvm::Module> module,
    std::unique_ptr<llvm::LLVMContext> context,
    const KernelSignature& signature,
    const KernelLaunchMetadata& launch_metadata,
    int opt_level) const {
    if (!module || !context) {
        throw std::invalid_argument("LLVM Compile requires module and context");
    }
    signature.Validate();
    launch_metadata.Validate();
    if (launch_metadata->backend != CodeGenBackend::kLLVM ||
        launch_metadata->device.device_type() != kCPU) {
        throw std::invalid_argument("LLVM Compile requires LLVM/CPU launch metadata");
    }
    for (const auto& argument : signature.arguments()) {
        if (argument->device != launch_metadata->device) {
            throw std::invalid_argument(
                "LLVM signature argument device does not match launch metadata");
        }
        // 当前标量 TIR lowering 不生成 LLVM 向量 load/store，不能静默接受 lanes。
        if (argument->dtype.lanes != 1) {
            throw std::invalid_argument(
                "LLVM backend does not support vector-lane NDArray arguments");
        }
    }

    VerifyModule(*module, "before optimization");
    Optimize(module.get(), opt_level);
    VerifyModule(*module, "after optimization");

    InitializeNativeLLVM();
    auto jit_result = llvm::orc::LLJITBuilder().create();
    if (!jit_result) {
        throw std::runtime_error("Failed to create LLJIT: " +
                                 llvm::toString(jit_result.takeError()));
    }
    std::unique_ptr<llvm::orc::LLJIT> jit = std::move(*jit_result);
    AddHostRuntimeSymbols(jit.get());
    AddMinGWRuntimeArchive(jit.get());

    if (auto error = jit->addIRModule(
            llvm::orc::ThreadSafeModule(std::move(module), std::move(context)))) {
        throw std::runtime_error("Failed to add module to LLVM JIT: " +
                                 llvm::toString(std::move(error)));
    }

    const std::string symbol = signature->symbol;
    auto address = jit->lookup(symbol);
    if (!address) {
        throw std::runtime_error("Failed to lookup LLVM function '" + symbol +
                                 "': " + llvm::toString(address.takeError()));
    }
    LLVMKernelFunction function = address->toPtr<LLVMKernelFunction>();
    auto launcher = std::make_shared<LLVMKernelLauncher>(
        std::move(jit), function, signature->symbol,
        static_cast<uint64_t>(signature.arguments().size()));
    return CompiledKernel(signature, launch_metadata, std::move(launcher));
}

}  // namespace kxc::codegen

#endif  // KXC_USE_LLVM
