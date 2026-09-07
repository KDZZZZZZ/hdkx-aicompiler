#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kxc/profiling/profiling.h"
#include "kxc/target/target.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/execution_observer.h"
#include "module_invocation_contract.h"
#include "codegen/internal/compiled_kernel.h"

namespace kxc::api {

namespace internal {

struct CompiledModuleEntry final {
    codegen::KernelSignature signature;
    codegen::KernelLaunchMetadata launch_metadata;
    codegen::CompiledKernel executable;
    /*! Empty means BuildCompiledModule creates the static specialization. */
    std::shared_ptr<const ModuleInvocationContract> invocation_contract;
};

}  // namespace internal

class CompiledModuleNode final : public Object {
public:
    CompiledModuleNode(Target target,
                       std::vector<internal::CompiledModuleEntry> entries,
                       Map<String, runtime::NDArray> constants,
                       std::shared_ptr<profiling::ProfileContext> profile_context,
                       std::shared_ptr<runtime::ExecutionObserver> execution_observer)
        : target_(std::move(target)),
          constants_(std::move(constants)),
          profile_context_(std::move(profile_context)),
          execution_observer_(std::move(execution_observer)) {
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
    /*! \brief BuildCompiledModule 装配的执行观测器；观测器与 ProfileContext
     *  都不参与模块 identity。 */
    std::shared_ptr<runtime::ExecutionObserver> execution_observer_;
};

namespace internal {

CompiledModule BuildCompiledModule(
    Target target,
    std::vector<CompiledModuleEntry> entries,
    Map<String, runtime::NDArray> constants,
    std::shared_ptr<profiling::ProfileContext> profile_context = nullptr,
    std::shared_ptr<runtime::ExecutionObserver> execution_observer = nullptr);

/*! \brief Internal immutable borrow; public constants() returns deep copies. */
const Map<String, runtime::NDArray>& BorrowCompiledModuleConstants(
    const CompiledModule& module);

/*! \brief Source-private contract inspection for runtime planning and identity. */
const ModuleInvocationContract& BorrowCompiledModuleInvocationContract(
    const CompiledModule& module, const String& symbol);

/*! \brief Execution observer attached at module build; empty when unset. */
std::shared_ptr<runtime::ExecutionObserver> BorrowCompiledModuleExecutionObserver(
    const CompiledModule& module);

/*! Internal preallocated-output hook used by RuntimeSession/control paths. */
AsyncOperation InvokeCompiledModuleWithOutputs(
    const CompiledModule& module, const String& symbol,
    const Array<runtime::NDArray>& data_inputs,
    const Array<runtime::NDArray>& outputs, const DeviceStream& stream,
    std::size_t run_byte_budget = 0);

}  // namespace internal
}  // namespace kxc::api
