/*! \file include/kxc/compiler/restricted_shape_runtime_bridge.h
 * \brief Default-off bridge from restricted symbolic Shape decisions to local runtime plans.
 *
 * This binds only trusted synchronous local launch evidence. It is neither a
 * CompiledModule dynamic-output ABI nor LLVM/JIT compilation proof.
 */
#pragma once

#include <memory>
#include <string>

#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/runtime/runtime_shape_plan.h"

namespace kxc::api::experimental::restricted_shape_runtime_bridge::v1 {

inline constexpr uint32_t kRestrictedShapeRuntimeBridgeVersion = 1;

/*! \brief Preselected local launcher; it cannot represent async or JIT evidence. */
struct TrustedSynchronousLauncherDescriptor final {
    std::string module_label;
    std::string entry_symbol;
    runtime::RuntimeShapeBoundLauncher launcher;
    std::shared_ptr<void> module_lease;
};

class RestrictedShapeRuntimeBridge final {
public:
    static bool IsEnabled() noexcept;

    /*! \brief Freeze the selected decision's final tree output into a runtime plan.
     *
     * The supplied launcher is trusted local synchronous evidence only. Shape
     * guards, output contracts, byte limits, and canonical ABI bytes are all
     * minted here before RuntimeShapeSession can allocate or launch.
     */
    static runtime::RuntimeShapePlan Bind(
        const restricted_symbolic_shape::v1::PreparedRestrictedSymbolicTemplate& prepared,
        const restricted_symbolic_shape::v1::RestrictedDispatchDecision& decision,
        TrustedSynchronousLauncherDescriptor launcher);
};

}  // namespace kxc::api::experimental::restricted_shape_runtime_bridge::v1
