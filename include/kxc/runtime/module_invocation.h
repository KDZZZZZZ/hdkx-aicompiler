/*! \file include/kxc/runtime/module_invocation.h
 * \brief Public results and failures for CompiledModule::Invoke.
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "kxc/runtime/device_stream.h"
#include "kxc/runtime/ndarray.h"

namespace kxc::api {

using ModuleExtent = std::uint64_t;

struct ModuleInvocationOutput {
    runtime::NDArray storage;
    std::vector<ModuleExtent> logical, physical, valid;
};

struct ModuleInvocationResult {
    std::vector<ModuleInvocationOutput> outputs;
    AsyncOperation operation;
};

enum class ModuleInvocationFailureKind {
    kInvalidContract, kDisabled, kGuard, kResource, kPreallocatedMismatch, kLaunch
};

class ModuleInvocationError : public std::runtime_error {
public:
    ModuleInvocationError(ModuleInvocationFailureKind kind, const std::string& message)
        : std::runtime_error(message), kind_(kind) {}
    ModuleInvocationFailureKind kind() const noexcept { return kind_; }
private:
    ModuleInvocationFailureKind kind_;
};

}  // namespace kxc::api
