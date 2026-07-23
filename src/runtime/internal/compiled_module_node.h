#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kxc/profiling/profiling.h"
#include "kxc/target/target.h"
#include "kxc/runtime/compiled_module.h"
#include "../../codegen/internal/compiled_kernel.h"
#include "kxc/tir/stmt.h"

namespace kxc::api {

namespace internal {

struct CompiledModuleEntry final {
    tir::PrimFunc prim_func;
    codegen::KernelSignature signature;
    codegen::KernelLaunchMetadata launch_metadata;
    codegen::CompiledKernel executable;
};

}  // namespace internal

class CompiledModuleNode final : public Object {
public:
    CompiledModuleNode(Target target,
                       std::vector<internal::CompiledModuleEntry> entries,
                       Map<String, runtime::NDArray> constants,
                       std::shared_ptr<profiling::ProfileContext> profile_context)
        : target_(std::move(target)),
          constants_(std::move(constants)),
          profile_context_(std::move(profile_context)) {
        for (auto& entry : entries) {
            entries_.emplace(std::string(entry.signature->symbol),
                             std::move(entry));
        }
    }

    KXC_OBJECT_DECLARE

    Target target_;
    std::unordered_map<std::string, internal::CompiledModuleEntry> entries_;
    Map<String, runtime::NDArray> constants_;
    std::shared_ptr<profiling::ProfileContext> profile_context_;
};

namespace internal {

CompiledModule BuildCompiledModule(
    Target target,
    std::vector<CompiledModuleEntry> entries,
    Map<String, runtime::NDArray> constants,
    std::shared_ptr<profiling::ProfileContext> profile_context = nullptr);

/*! \brief Internal immutable borrow; public constants() returns deep copies. */
const Map<String, runtime::NDArray>& BorrowCompiledModuleConstants(
    const CompiledModule& module);

}  // namespace internal
}  // namespace kxc::api
