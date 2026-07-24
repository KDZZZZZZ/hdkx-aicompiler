#pragma once

#include "kxc/runtime/control_execution_plan.h"

namespace kxc::runtime::internal {

struct BoundControlKernelAccess final {
    static AsyncOperation Launch(
        const BoundControlKernel& kernel,
        const Array<NDArray>& ordered_arguments,
        const DeviceStream& stream) {
        return kernel.Launch(ordered_arguments, stream);
    }
};

}  // namespace kxc::runtime::internal
