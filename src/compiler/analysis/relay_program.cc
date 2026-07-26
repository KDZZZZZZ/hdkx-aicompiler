/*! \file src/compiler/analysis/relay_program.cc
 * \brief Implements the single Relay preparation and topology selection path.
 */

#include "../internal/relay_program.h"

#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "kxc/pass/context.h"
#include "kxc/relay/visitor.h"

namespace kxc::api::internal {
namespace {

constexpr std::size_t CapabilityIndex(
    RelayControlCapability capability) noexcept {
    return static_cast<std::size_t>(capability);
}

void ScanRelayControlCapabilities(
    const Expr& expression, RelayControlCapabilitySet* required,
    std::unordered_set<const Object*>* visited) {
    if (!expression.defined() || !required || !visited ||
        !visited->insert(expression.get()).second) {
        return;
    }
    if (const auto* function = expression.As<FunctionNode>()) {
        ScanRelayControlCapabilities(function->body, required, visited);
        return;
    }
    if (const auto* call = expression.As<CallNode>()) {
        for (const Expr& argument : call->args) {
            ScanRelayControlCapabilities(argument, required, visited);
        }
        return;
    }
    if (const auto* conditional = expression.As<IfNode>()) {
        required->set(CapabilityIndex(
            RelayControlCapability::kConditionalBranch));
        ScanRelayControlCapabilities(conditional->cond, required, visited);
        ScanRelayControlCapabilities(
            conditional->true_branch, required, visited);
        ScanRelayControlCapabilities(
            conditional->false_branch, required, visited);
        return;
    }
    if (const auto* loop = expression.As<WhileNode>()) {
        required->set(CapabilityIndex(
            RelayControlCapability::kBoundedPreTestLoop));
        ScanRelayControlCapabilities(loop->initial_state, required, visited);
        ScanRelayControlCapabilities(loop->condition, required, visited);
        ScanRelayControlCapabilities(loop->body, required, visited);
        return;
    }
    if (const auto* let = expression.As<LetNode>()) {
        ScanRelayControlCapabilities(let->value, required, visited);
        ScanRelayControlCapabilities(let->body, required, visited);
        return;
    }
    if (const auto* tuple = expression.As<TupleNode>()) {
        for (const Expr& field : tuple->fields) {
            ScanRelayControlCapabilities(field, required, visited);
        }
        return;
    }
    if (const auto* item = expression.As<TupleGetItemNode>()) {
        ScanRelayControlCapabilities(item->tuple, required, visited);
    }
}

RelayControlCapabilitySet AnalyzeRelayControlCapabilities(
    const Function& function) {
    RelayControlCapabilitySet required;
    std::unordered_set<const Object*> visited;
    ScanRelayControlCapabilities(
        Expr(ObjectRef(function)), &required, &visited);
    return required;
}

Array<String> CapabilityNames(
    const RelayControlCapabilitySet& capabilities) {
    Array<String> names;
    for (std::size_t index = 0;
         index < CapabilityIndex(RelayControlCapability::kCount); ++index) {
        if (!capabilities.test(index)) continue;
        names.push_back(String(RelayControlCapabilityName(
            static_cast<RelayControlCapability>(index))));
    }
    return names;
}

void RequireAllowedCapabilities(
    const RelayProgramProfile& profile, const ControlFlowPolicy& policy) {
    RelayControlCapabilitySet missing =
        profile.required_control_capabilities() &
        ~policy.allowed_capabilities();
    if (!missing.any()) return;

    std::ostringstream message;
    message << "PrepareRelayProgram control-flow policy is missing capabilities: ";
    bool first = true;
    for (std::size_t index = 0;
         index < CapabilityIndex(RelayControlCapability::kCount); ++index) {
        if (!missing.test(index)) continue;
        if (!first) message << ", ";
        first = false;
        message << RelayControlCapabilityName(
            static_cast<RelayControlCapability>(index));
    }
    throw std::invalid_argument(message.str());
}

}  // namespace

const char* RelayControlCapabilityName(
    RelayControlCapability capability) noexcept {
    switch (capability) {
        case RelayControlCapability::kConditionalBranch:
            return "conditional_branch";
        case RelayControlCapability::kBoundedPreTestLoop:
            return "bounded_pre_test_loop";
        case RelayControlCapability::kCount:
            break;
    }
    return "unknown";
}

ControlFlowPolicy::ControlFlowPolicy(RelayControlCapabilitySet allowed)
    : allowed_(std::move(allowed)) {}

ControlFlowPolicy ControlFlowPolicy::StaticOnly() {
    return ControlFlowPolicy(RelayControlCapabilitySet{});
}

ControlFlowPolicy ControlFlowPolicy::NativeExact() {
    RelayControlCapabilitySet allowed;
    allowed.set(CapabilityIndex(
        RelayControlCapability::kConditionalBranch));
    allowed.set(CapabilityIndex(
        RelayControlCapability::kBoundedPreTestLoop));
    return ControlFlowPolicy(std::move(allowed));
}

bool ControlFlowPolicy::Allows(
    RelayControlCapability capability) const noexcept {
    const std::size_t index = CapabilityIndex(capability);
    return index < allowed_.size() && allowed_.test(index);
}

const RelayControlCapabilitySet&
ControlFlowPolicy::allowed_capabilities() const noexcept {
    return allowed_;
}

RelayProgramProfile::RelayProgramProfile(
    RelayControlCapabilitySet required)
    : required_(std::move(required)) {}

bool RelayProgramProfile::Requires(
    RelayControlCapability capability) const noexcept {
    const std::size_t index = CapabilityIndex(capability);
    return index < required_.size() && required_.test(index);
}

bool RelayProgramProfile::requires_control_topology() const noexcept {
    return required_.any();
}

const RelayControlCapabilitySet&
RelayProgramProfile::required_control_capabilities() const noexcept {
    return required_;
}

std::vector<std::string>
RelayProgramProfile::RequiredCapabilityNames() const {
    std::vector<std::string> names;
    for (const String& name : CapabilityNames(required_)) {
        names.push_back(static_cast<std::string>(name));
    }
    return names;
}

PreparedRelayProgram::PreparedRelayProgram(
    Function typed_anf, RelayProgramProfile residual_profile,
    Target target, CompilerExecutionContract execution_contract,
    size_t capability_boundary_checks)
    : typed_anf_(std::move(typed_anf)),
      residual_profile_(std::move(residual_profile)),
      target_(std::move(target)),
      execution_contract_(std::move(execution_contract)),
      capability_boundary_checks_(capability_boundary_checks) {}

const Function& PreparedRelayProgram::typed_anf() const noexcept {
    return typed_anf_;
}

const RelayProgramProfile&
PreparedRelayProgram::residual_profile() const noexcept {
    return residual_profile_;
}

const Target& PreparedRelayProgram::target() const noexcept {
    return target_;
}

const CompilerExecutionContract&
PreparedRelayProgram::execution_contract() const noexcept {
    return execution_contract_;
}

size_t PreparedRelayProgram::capability_boundary_checks() const noexcept {
    return capability_boundary_checks_;
}

PreparedRelayProgram PrepareRelayProgram(
    Function function, const CompileConfig& config,
    const ControlFlowPolicy& policy) {
    if (!function.defined()) {
        throw std::invalid_argument(
            "PrepareRelayProgram requires a defined Function");
    }

    size_t capability_boundary_checks = 0;
    const RelayControlCapabilitySet input_capabilities =
        AnalyzeRelayControlCapabilities(function);
    ++capability_boundary_checks;
    CompilerExecutionContract contract =
        ResolveCompilerExecutionContract(
            config, CapabilityNames(input_capabilities));

    const PassContext pass_context = PassContext::MergeTarget(
        relay::PassContextFromRelay(function), config->target);
    PassContext::Scope pass_scope(pass_context);
    Function typed_anf = PipelineExecutor::ExecuteRelay(
        contract.relay_pipeline, function, config->target);

    RelayProgramProfile residual_profile(
        AnalyzeRelayControlCapabilities(typed_anf));
    ++capability_boundary_checks;
    RequireAllowedCapabilities(residual_profile, policy);

    return PreparedRelayProgram(
        std::move(typed_anf), std::move(residual_profile),
        config->target, std::move(contract),
        capability_boundary_checks);
}

}  // namespace kxc::api::internal
