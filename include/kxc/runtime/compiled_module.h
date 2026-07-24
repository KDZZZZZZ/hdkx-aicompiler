/*! \file include/kxc/runtime/compiled_module.h
 * \brief 定义不暴露编译器和后端实现细节的可执行模块句柄。
 */

#pragma once

#include "kxc/runtime/device_stream.h"
#include "kxc/runtime/ndarray.h"
#include "kxc/runtime/kernel_abi.h"
#include "kxc/runtime/module_invocation.h"

namespace kxc::api {

class CompiledModuleNode;

class CompiledModule : public ObjectRef {
public:
    explicit CompiledModule(const ObjectRef& ref);

    /*! \brief Resolves a module-owned ABI, allocates outputs, injects constants,
     * and launches the real CompiledKernel. */
    ModuleInvocationResult Invoke(const String& symbol,
                                  const Array<runtime::NDArray>& data_inputs,
                                  const DeviceStream& stream,
                                  std::size_t run_byte_budget = 0) const;
    codegen::KernelSignature signature(const String& symbol) const;
    codegen::KernelLaunchMetadata launch_metadata(const String& symbol) const;
    /*! \brief Returns independent payload snapshots; callers cannot mutate
     * the module. */
    Map<String, runtime::NDArray> constants() const;
    bool HasFunction(const String& symbol) const;
    size_t entry_count() const;
    Array<String> symbols() const;
    bool IsReady() const noexcept;
    String GetStatus() const;
    String GetProfileBundlePath() const;
};

}  // namespace kxc::api
