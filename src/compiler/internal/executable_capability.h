/*! \file src/compiler/internal/executable_capability.h
 * \brief Static Relay subset accepted by executable graph construction.
 */

#pragma once

#include <cstdint>

#include "kxc/relay/relay.h"

namespace kxc::api::internal {

/*! \brief Versioned, deliberately static capability contract for Relay executables. */
struct ExecutableCapabilityOptions {
    static constexpr uint32_t kVersion = 1;

    uint32_t version{kVersion};
    bool allow_if{false};
};

/*! \brief The static-dataflow contract used before ValueGraph construction. */
ExecutableCapabilityOptions StaticDataflowExecutableCapabilities();

/*! \brief Verifies the typed, static-exact Relay subset required by an executable. */
void VerifyExecutableCapability(const Function& function,
                                const ExecutableCapabilityOptions& options);

}  // namespace kxc::api::internal
