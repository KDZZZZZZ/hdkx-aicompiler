#pragma once

#include <memory>
#include <utility>

#include "kxc/profiling/profiling.h"
#include "kxc/target/target.h"
#include "kxc/runtime/compiled_module.h"
#include "../../codegen/internal/compiled_kernel.h"
#include "kxc/tir/stmt.h"

namespace kxc::api {

class CompiledModuleNode final : public Object {
public:
    CompiledModuleNode(Target target,
                       tir::PrimFunc prim_func,
                       codegen::KernelSignature signature,
                       codegen::KernelLaunchMetadata launch_metadata,
                       Map<String, runtime::NDArray> constants,
                       codegen::CompiledKernel executable,
                       std::shared_ptr<profiling::ProfileContext> profile_context)
        : target_(std::move(target)),
          prim_func_(std::move(prim_func)),
          signature_(std::move(signature)),
          launch_metadata_(std::move(launch_metadata)),
          constants_(std::move(constants)),
          executable_(std::move(executable)),
          profile_context_(std::move(profile_context)) {}

    KXC_OBJECT_DECLARE

    Target target_;
    tir::PrimFunc prim_func_;
    codegen::KernelSignature signature_;
    codegen::KernelLaunchMetadata launch_metadata_;
    Map<String, runtime::NDArray> constants_;
    codegen::CompiledKernel executable_;
    std::shared_ptr<profiling::ProfileContext> profile_context_;
};

namespace internal {

CompiledModule BuildCompiledModule(
    Target target,
    tir::PrimFunc prim_func,
    codegen::KernelSignature signature,
    codegen::KernelLaunchMetadata launch_metadata,
    Map<String, runtime::NDArray> constants,
    codegen::CompiledKernel executable,
    std::shared_ptr<profiling::ProfileContext> profile_context = nullptr);

}  // namespace internal
}  // namespace kxc::api
