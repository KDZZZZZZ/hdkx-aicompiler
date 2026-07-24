/*! \file src/compiler/internal/executable_capability.h
 * \brief Static Relay subset accepted by executable graph construction.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kxc/relay/relay.h"
#include "kxc/runtime/device.h"

namespace kxc::api::internal {

/*! \brief Versioned, deliberately static capability contract for Relay executables. */
struct ExecutableCapabilityOptions {
    static constexpr uint32_t kVersion = 1;

    uint32_t version{kVersion};
    bool allow_if{false};
    bool allow_while{false};
    bool allow_tuple_parameters{false};
    bool allow_nested_tuple_call_outputs{false};
    bool allow_device_regions{false};
    // Compiler entry can receive Relay before InferType; typed boundaries cannot.
    bool require_checked_types{true};
    Device execution_device;
};

/*! \brief A structured rejection from the shared executable Relay policy. */
struct ExecutableCapabilityIssue final {
    std::string path;
    std::string node_kind;
    std::string capability;
    std::string detail;
};

/*! \brief The static-dataflow contract used before ValueGraph construction. */
ExecutableCapabilityOptions StaticDataflowExecutableCapabilities(
    Device execution_device = Device());

/*! \brief Returns shared-policy rejections without formatting exception text. */
std::vector<ExecutableCapabilityIssue> CollectExecutableCapabilityIssues(
    const Function& function, const ExecutableCapabilityOptions& options);

/*! \brief Verifies the static-exact Relay subset required by an executable. */
void VerifyExecutableCapability(const Function& function,
                                const ExecutableCapabilityOptions& options);

}  // namespace kxc::api::internal
