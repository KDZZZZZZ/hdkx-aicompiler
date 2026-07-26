/*! \file src/compiler/internal/relay_program.h
 * \brief Compiler-owned Relay preparation and topology selection contracts.
 */

#pragma once

#include <bitset>
#include <cstddef>
#include <string>
#include <vector>

#include "execution_contract.h"
#include "kxc/compiler/compile_config.h"
#include "kxc/relay/relay.h"

namespace kxc::api::internal {

enum class RelayControlCapability : std::size_t {
    kConditionalBranch,
    kBoundedPreTestLoop,
    kCount,
};

using RelayControlCapabilitySet =
    std::bitset<static_cast<std::size_t>(RelayControlCapability::kCount)>;

const char* RelayControlCapabilityName(
    RelayControlCapability capability) noexcept;

class ControlFlowPolicy final {
public:
    static ControlFlowPolicy StaticOnly();
    static ControlFlowPolicy NativeExact();

    bool Allows(RelayControlCapability capability) const noexcept;
    const RelayControlCapabilitySet& allowed_capabilities() const noexcept;

private:
    explicit ControlFlowPolicy(RelayControlCapabilitySet allowed);

    RelayControlCapabilitySet allowed_;
};

class PreparedRelayProgram;

class RelayProgramProfile final {
public:
    bool Requires(RelayControlCapability capability) const noexcept;
    bool requires_control_topology() const noexcept;
    const RelayControlCapabilitySet& required_control_capabilities() const noexcept;
    std::vector<std::string> RequiredCapabilityNames() const;

private:
    friend class PreparedRelayProgram;
    friend PreparedRelayProgram PrepareRelayProgram(
        Function function, const CompileConfig& config,
        const ControlFlowPolicy& policy);

    explicit RelayProgramProfile(RelayControlCapabilitySet required);

    RelayControlCapabilitySet required_;
};

class PreparedRelayProgram final {
public:
    const Function& typed_anf() const noexcept;
    const RelayProgramProfile& residual_profile() const noexcept;
    const Target& target() const noexcept;
    const CompilerExecutionContract& execution_contract() const noexcept;
    size_t capability_boundary_checks() const noexcept;

private:
    friend PreparedRelayProgram PrepareRelayProgram(
        Function function, const CompileConfig& config,
        const ControlFlowPolicy& policy);

    PreparedRelayProgram(Function typed_anf, RelayProgramProfile residual_profile,
                         Target target,
                         CompilerExecutionContract execution_contract,
                         size_t capability_boundary_checks);

    Function typed_anf_;
    RelayProgramProfile residual_profile_;
    Target target_;
    CompilerExecutionContract execution_contract_;
    size_t capability_boundary_checks_{0};
};

PreparedRelayProgram PrepareRelayProgram(
    Function function, const CompileConfig& config,
    const ControlFlowPolicy& policy);

}  // namespace kxc::api::internal
