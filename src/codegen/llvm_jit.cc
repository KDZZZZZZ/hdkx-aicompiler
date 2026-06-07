/*! \file src/codegen/llvm_jit.cc
 * \brief 实现 C/LLVM codegen、LLVM JIT 和 compiled kernel 调用封装。
 */

#if KXC_USE_LLVM

#include "codegen/llvm_jit.h"

#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>

#include <stdexcept>

namespace kxc {
namespace codegen {

#ifndef KXC_MINGW_LIBGCC_PATH
#define KXC_MINGW_LIBGCC_PATH ""
#endif

namespace {

void AddHostRuntimeSymbols(llvm::orc::LLJIT* jit) {
    auto generator = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
        jit->getDataLayout().getGlobalPrefix());
    if (!generator) {
        throw std::runtime_error("Failed to add current-process symbols to LLVM JIT: " +
                                 llvm::toString(generator.takeError()));
    }
    jit->getMainJITDylib().addGenerator(std::move(*generator));
}

void AddMinGWRuntimeArchive(llvm::orc::LLJIT* jit) {
#ifdef __MINGW32__
    const char* libgcc_path = KXC_MINGW_LIBGCC_PATH;
    if (libgcc_path[0] == '\0') {
        return;
    }
    if (auto err = jit->linkStaticLibraryInto(jit->getMainJITDylib(), libgcc_path)) {
        throw std::runtime_error("Failed to link MinGW libgcc into LLVM JIT: " +
                                 llvm::toString(std::move(err)));
    }
#else
    (void)jit;
#endif
}

}  // namespace

class LLVMJITEngine::Impl {
public:
    std::unique_ptr<llvm::orc::LLJIT> jit;
    llvm::LLVMContext ctx;

    Impl() {
        // 初始化LLVM目标
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        // 创建JIT实例
        auto jit_or_err = llvm::orc::LLJITBuilder().create();
        if (!jit_or_err) {
            throw std::runtime_error("Failed to create LLJIT: " +
                                     llvm::toString(jit_or_err.takeError()));
        }
        jit = std::move(*jit_or_err);
        AddHostRuntimeSymbols(jit.get());
        AddMinGWRuntimeArchive(jit.get());
    }
};

LLVMJITEngine::LLVMJITEngine() : impl_(std::make_unique<Impl>()) {}

LLVMJITEngine::~LLVMJITEngine() = default;

void LLVMJITEngine::Optimize(llvm::Module* module, int opt_level) {
    if (opt_level == 0) return;  // 不优化

    llvm::PassBuilder PB;
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    llvm::ModulePassManager MPM;
    if (opt_level == 1) {
        MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O1);
    } else if (opt_level == 2) {
        MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
    } else {
        MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    }

    MPM.run(*module, MAM);
}

CompiledKernel LLVMJITEngine::Compile(std::unique_ptr<llvm::Module> module,
                                       std::unique_ptr<llvm::LLVMContext> context,
                                       const std::string& func_name,
                                       int opt_level) {
    // 优化
    Optimize(module.get(), opt_level);

    // 添加到JIT (transfer ownership of both module and context)
    auto& jit = impl_->jit;
    auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(
        std::move(module), std::move(context)));
    if (err) {
        throw std::runtime_error("Failed to add module to JIT: " +
                                 llvm::toString(std::move(err)));
    }

    // 查找函数符号
    auto sym_or_err = jit->lookup(func_name);
    if (!sym_or_err) {
        throw std::runtime_error("Failed to lookup function '" + func_name +
                                 "': " + llvm::toString(sym_or_err.takeError()));
    }

    void* func_ptr = reinterpret_cast<void*>(sym_or_err->getValue());

    // 创建CompiledKernel
    // 重要: JIT引擎必须比kernel存活更久，否则函数指针无效
    // 通过jit_resource保存shared_ptr到JIT实例
    auto jit_shared = std::make_shared<std::unique_ptr<llvm::orc::LLJIT>>(std::move(impl_->jit));

    auto* kernel_node = new CompiledKernelNode();
    kernel_node->kernel_name = func_name;
    kernel_node->backend = CodeGenBackend::kLLVM;
    kernel_node->func_ptr = func_ptr;
    // Store shared_ptr as void* to keep JIT alive
    kernel_node->jit_resource = new std::shared_ptr<std::unique_ptr<llvm::orc::LLJIT>>(jit_shared);

    return CompiledKernel(kernel_node);
}

}  // namespace codegen
}  // namespace kxc

#endif  // KXC_USE_LLVM
