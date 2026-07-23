/*! \file include/kxc/compiler/capability.h
 * \brief Fail-closed executable capability contract for compiler control paths.
 */

#pragma once

#include <string>
#include <vector>

#include "kxc/relay/relay.h"
#include "kxc/target/target.h"

namespace kxc::api {

/*! \brief Compiler boundary at which an executable capability check runs. */
enum class CapabilityBoundary {
    kCompilerEntry,
    kPostGraphPass,
    kPrePartition,
};

/*! \brief Requested execution model. Only static-exact is implemented today. */
enum class CapabilityMode {
    kStaticExact,
    kShapeSpecialization,
    kControlFlow,
    kRegion,
};

/*! \brief One stable, node-local reason that a request is not executable. */
struct CapabilityIssue final {
    std::string diagnostic_locator;
    std::string relay_node_kind;
    std::string missing_capability;
    std::string detail;
};

/*! \brief Input contract shared by production checks and deterministic fakes. */
struct CapabilityRequest final {
    Function function;
    Target target;
    std::string graph_locator{"graph"};
    std::string pipeline_fingerprint;
    CapabilityBoundary boundary{CapabilityBoundary::kCompilerEntry};
    CapabilityMode requested_mode{CapabilityMode::kStaticExact};
    bool require_checked_types{false};
};

/*! \brief Complete fail-closed result; callers must not infer fallback support. */
struct CapabilityResult final {
    bool supported{false};
    std::vector<std::string> missing_capabilities;
    std::string diagnostic_locator;
    std::string target_identity;
    std::string pipeline_fingerprint;
    CapabilityMode requested_mode{CapabilityMode::kStaticExact};
    std::vector<std::string> normalized_requirements;
    std::vector<CapabilityIssue> issues;

    std::string Diagnostic() const;
};

/*! \brief Verifies the currently executable Relay static-exact dialect. */
class CapabilityVerifier final {
public:
    static CapabilityResult Verify(const CapabilityRequest& request);
    static void Require(const CapabilityRequest& request);
};

const char* ToString(CapabilityBoundary boundary);
const char* ToString(CapabilityMode mode);

}  // namespace kxc::api
