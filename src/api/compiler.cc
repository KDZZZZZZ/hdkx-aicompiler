#include "api/compiler.h"

#include <fstream>
#include <stdexcept>

#include "codegen/codegen_c.h"
#include "relay/transforms/lower.h"
#include "runtime/runtime_session.h"
#include "tir/transforms/pipeline.h"

#ifdef KXC_USE_LLVM
#include <llvm/IR/LLVMContext.h>

#include "codegen/codegen_llvm.h"
#include "codegen/llvm_jit.h"
#endif

namespace kxc {
namespace api {

// ==================== Compiler ====================

CompiledModule Compiler::Compile(Function func, CompileConfig config) {
    // Adaptive模式：创建RuntimeSession，延迟编译
    if (config->mode == CompileMode::kAdaptive) {
        auto session = std::make_shared<runtime::RuntimeSession>(func, config);

        // 如果有建议shape，预热
        for (const auto& shape : config->suggested_input_shapes) {
            std::vector<std::vector<int64_t>> shapes_vec;
            std::vector<int64_t> single_shape;
            for (const auto& dim : shape) single_shape.push_back(dim);
            shapes_vec.push_back(single_shape);
            session->WarmUp(shapes_vec);
        }

        return CompiledModule(config, func, session);
    }

    // AOT/JIT模式：同步编译
    // Step 1: Relay → TIR
    tir::PrimFunc prim_func = relay::LowerToTIR(func);

    // Step 2: TIR优化
    Array<String> passes;
    if (config->mode == CompileMode::kAOT) {
        passes = {String("optimize_default")};
    } else {
        passes = {String("fold_constant"), String("simplify_expr")};
    }
    prim_func = RunTIRPassPipeline(prim_func, passes);

    // Step 3: Codegen
    codegen::CompiledKernel kernel;

#ifdef KXC_USE_LLVM
    if (config->backend == codegen::CodeGenBackend::kLLVM) {
        auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
        codegen::CodeGenLLVM codegen(*llvm_ctx);
        codegen.AddFunction(prim_func, "main");
        auto module = codegen.TakeModule();

        codegen::LLVMJITEngine jit;
        kernel = jit.Compile(std::move(module), std::move(llvm_ctx),
                            "main", config->opt_level);
    } else
#endif
    {
        throw std::runtime_error(
            "C backend compilation not fully implemented. Use LLVM backend.");
    }

    return CompiledModule(config, prim_func, kernel);
}

// ==================== CompiledModule ====================

CompiledModule::CompiledModule(CompileConfig config,
                               tir::PrimFunc prim_func,
                               codegen::CompiledKernel kernel)
    : config_(config), prim_func_(prim_func), kernel_(kernel) {
    codegen::CodeGenC codegen_c;
    c_source_ = codegen_c.Generate(prim_func, "main");
}

CompiledModule::CompiledModule(CompileConfig config,
                               Function relay_func,
                               std::shared_ptr<runtime::RuntimeSession> session)
    : config_(config), relay_func_(relay_func), session_(session) {}

void CompiledModule::Run(const std::vector<void*>& packed_args) {
    if (session_) {
        throw std::runtime_error(
            "Adaptive mode requires Run(args, input_shapes). "
            "Use the overload with input_shapes parameter.");
    }
    if (!kernel_.IsReady()) {
        throw std::runtime_error("CompiledModule: kernel not ready");
    }
    kernel_(packed_args);
}

void CompiledModule::Run(const std::vector<void*>& packed_args,
                         const std::vector<std::vector<int64_t>>& input_shapes) {
    if (session_) {
        session_->Run(packed_args, input_shapes);
        return;
    }
    // 非Adaptive模式：忽略input_shapes，直接执行
    Run(packed_args);
}

void CompiledModule::WarmUp(const std::vector<std::vector<int64_t>>& input_shapes) {
    if (session_) {
        session_->WarmUp(input_shapes);
    }
}

void CompiledModule::WaitAll() {
    if (session_) {
        session_->WaitAll();
    }
}

bool CompiledModule::IsReady() const {
    if (session_) return true;  // Adaptive模式总是"ready"（按需编译）
    return kernel_.IsReady();
}

std::string CompiledModule::GetStatus() const {
    if (session_) return session_->GetStatus();
    if (!kernel_.IsReady()) return "not_ready";
    if (config_->mode == CompileMode::kAOT) return "aot_compiled";
    if (config_->mode == CompileMode::kJIT) return "jit_compiled";
    return "compiled";
}

void CompiledModule::SaveCSource(const std::string& path) const {
    if (c_source_.empty() && prim_func_.defined()) {
        codegen::CodeGenC codegen_c;
        const_cast<std::string&>(c_source_) = codegen_c.Generate(prim_func_, "main");
    }
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error("Failed to open file: " + path);
    ofs << c_source_;
}

}  // namespace api
}  // namespace kxc
