/*! \file include/kxc/runtime/compiled_module.h
 * \brief 定义不暴露编译器和后端实现细节的可执行模块句柄。
 */

#pragma once

#include "kxc/runtime/device_stream.h"
#include "kxc/runtime/ndarray.h"
#include "kxc/runtime/kernel_abi.h"

namespace kxc::api {

class CompiledModuleNode;

class CompiledModule : public ObjectRef {
public:
    explicit CompiledModule(const ObjectRef& ref);

    AsyncOperation Launch(const String& symbol,
                          const Array<runtime::NDArray>& ordered_arguments,
                          const DeviceStream& stream) const;
    codegen::KernelSignature signature(const String& symbol) const;
    codegen::KernelLaunchMetadata launch_metadata(const String& symbol) const;
    Map<String, runtime::NDArray> constants() const;
    bool HasFunction(const String& symbol) const;
    size_t entry_count() const;
    Array<String> symbols() const;
    bool IsReady() const noexcept;
    String GetStatus() const;
    String GetProfileBundlePath() const;
};

}  // namespace kxc::api
