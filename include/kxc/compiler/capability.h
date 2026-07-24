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

/*! \brief Truth state for capability verification. */
enum class CapabilityStatus {
    kUnsupported,
    kEligibleButNotExecutable,
    kExecutable,
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
    bool require_checked_types{false};
    int opt_level{2};
};

/*! \brief Complete fail-closed result; callers must not infer fallback support. */
struct CapabilityResult final {
    CapabilityStatus status{CapabilityStatus::kUnsupported};
    std::string target_identity;
    std::string pipeline_fingerprint;
    std::vector<CapabilityIssue> issues;

    std::string Diagnostic() const;
};

/*! \brief Verifies the currently executable Relay static-exact dialect. */
class CapabilityVerifier final {
public:
    /*! \brief Proves executability by running the real production compiler.
     *
     * This is an executable probe, not a metadata-only query: it participates
     * in the production artifact cache/singleflight and may publish a ready
     * artifact or a bounded failure record.
     */
    static CapabilityResult Verify(const CapabilityRequest& request);
    // Requires only structural eligibility; suitable before per-unit lowering.
    static void RequireEligible(const CapabilityRequest& request);
};

const char* ToString(CapabilityBoundary boundary);
const char* ToString(CapabilityStatus status);

}  // namespace kxc::api
