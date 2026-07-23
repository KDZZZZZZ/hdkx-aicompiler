/*! \file src/compiler/pipeline_resolver.cc
 * \brief Resolves the generated pass contract into one production pipeline.
 */

#include "kxc/compiler/pipeline.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kxc/profiling/profiling.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/tir/transforms/pipeline.h"
#include "../pass/generated/pass_contract.inc"

namespace kxc::api {
namespace {

std::string AsString(const String& value) {
    return value.defined() ? static_cast<std::string>(value) : std::string();
}

void AppendField(std::string* canonical, const std::string& name,
                 const std::string& value) {
    *canonical += std::to_string(name.size()) + ":" + name + "=" +
                  std::to_string(value.size()) + ":" + value + ";";
}

Array<String> CompilerPolicy(const PipelineRequest& request) {
    const std::string dialect = ToString(request.dialect);
    if (request.opt_level < 0 || request.opt_level > 3) {
        throw std::invalid_argument(
            "PipelineResolver requires opt_level in [0, 3]");
    }
    std::string name = dialect + ".compiler.";
    if (request.dialect == IRDialect::kTIR && request.opt_level == 3) {
        if (!request.target.defined() || !request.target.As<TargetNode>()) {
            throw std::invalid_argument(
                "TIR O3 pipeline requires a defined Target");
        }
        if (request.target->kind == "cuda" &&
            request.target->device_type == kCUDA) {
            name += "cuda.o3";
        } else if (request.target->kind == "llvm" &&
                   request.target->device_type == kCPU) {
            name += "cpu.o3";
        } else {
            throw std::invalid_argument(
                "TIR O3 pipeline has no policy for the requested Target");
        }
    } else {
        name += "o" + std::to_string(request.opt_level);
    }
    return pass_contract_generated::Pipeline(name);
}

Array<String> BasePipeline(const PipelineRequest& request) {
    const std::string named = AsString(request.named_pipeline);
    if (named.empty() || named == "compiler") return CompilerPolicy(request);
    if (named == "optimize_default") {
        return pass_contract_generated::Pipeline(
            ToString(request.dialect) + ".optimize_default");
    }
    return pass_contract_generated::Pipeline(named);
}

std::unordered_set<std::string> UniqueNames(const Array<String>& values,
                                             const char* field) {
    std::unordered_set<std::string> result;
    for (const String& value : values) {
        const std::string name = AsString(value);
        if (name.empty() || !result.insert(name).second) {
            throw std::invalid_argument(std::string("PipelineResolver ") + field +
                                        " contains an empty or duplicate pass");
        }
    }
    return result;
}

int PhaseRank(const PassSpec& spec) {
    const std::string phase = AsString(spec.phase);
    if (spec.dialect == IRDialect::kRelay && phase == "relay_optimize") return 0;
    if (spec.dialect == IRDialect::kTIR && phase == "tir_optimize") return 0;
    if (spec.dialect == IRDialect::kTIR && phase == "tir_schedule") return 1;
    throw std::invalid_argument("PipelineResolver has no phase ordering for " +
                                PassSpecKey(spec.dialect, spec.name));
}

Array<String> SetToArray(const std::set<std::string>& values) {
    Array<String> result;
    for (const std::string& value : values) result.push_back(String(value));
    return result;
}

void RequireTarget(const PipelineRequest& request) {
    if (!request.target.defined() || !request.target.As<TargetNode>()) {
        throw std::invalid_argument(
            "PipelineResolver requires a defined Target capability snapshot");
    }
    const bool cpu = request.target->kind == "llvm" &&
                     request.target->device_type == kCPU;
    const bool cuda = request.target->kind == "cuda" &&
                      request.target->device_type == kCUDA;
    if (!cpu && !cuda) {
        throw std::invalid_argument(
            "PipelineResolver target kind and device type are inconsistent");
    }
}

}  // namespace

bool NormalizedPipeline::defined() const noexcept {
    return dialect != IRDialect::kUnknown && scope != PassScope::kUnknown &&
           canonical_bytes.defined() && !std::string(canonical_bytes).empty() &&
           fingerprint.defined() && !std::string(fingerprint).empty();
}

NormalizedPipeline PipelineResolver::Resolve(
    const PipelineRequest& request) {
    if (request.dialect != IRDialect::kRelay &&
        request.dialect != IRDialect::kTIR) {
        throw std::invalid_argument(
            "PipelineResolver requires Relay or TIR dialect");
    }
    const PassScope expected_scope = request.dialect == IRDialect::kRelay
                                         ? PassScope::kGraph
                                         : PassScope::kPrimFunc;
    if (request.requested_scope != expected_scope) {
        throw std::invalid_argument(
            "PipelineResolver requested scope does not match dialect");
    }
    RequireTarget(request);
    if (request.dialect == IRDialect::kRelay) {
        (void)relay::RelayRegisteredPassSpecs();
    } else {
        (void)tir::TIRRegisteredPassSpecs();
    }

    const auto enabled = UniqueNames(request.enabled, "enabled");
    const auto disabled = UniqueNames(request.disabled, "disabled");
    for (const std::string& name : enabled) {
        if (disabled.count(name)) {
            throw std::invalid_argument(
                "PipelineResolver cannot both enable and disable pass: " + name);
        }
    }

    Array<String> ordered;
    std::unordered_set<std::string> seen;
    for (const String& pass : BasePipeline(request)) {
        const std::string name = AsString(pass);
        if (name.empty() || !seen.insert(name).second) {
            throw std::invalid_argument(
                "generated pipeline contains an empty or duplicate pass");
        }
        if (!disabled.count(name)) ordered.push_back(pass);
    }
    for (const String& pass : request.enabled) {
        const std::string name = AsString(pass);
        if (seen.insert(name).second) ordered.push_back(pass);
    }

    std::set<std::string> invariants;
    for (const String& invariant : request.initial_invariants) {
        const std::string name = AsString(invariant);
        if (name.empty() || !invariants.insert(name).second) {
            throw std::invalid_argument(
                "PipelineResolver initial invariants must be unique and non-empty");
        }
    }
    std::set<std::string> analyses;
    for (const String& analysis : request.initial_analyses) {
        const std::string name = AsString(analysis);
        if (name.empty() || !analyses.insert(name).second) {
            throw std::invalid_argument(
                "PipelineResolver initial analyses must be unique and non-empty");
        }
    }

    NormalizedPipeline result;
    result.dialect = request.dialect;
    result.scope = expected_scope;
    result.ordered_passes = ordered;
    result.contract_versions.push_back(String(
        "pass-contract-v" +
        std::to_string(pass_contract_generated::kContractVersion)));
    result.target_requirements.push_back(String(
        request.target->kind + ":" +
        std::to_string(static_cast<int>(request.target->device_type))));

    std::string canonical;
    AppendField(&canonical, "kind", "normalized-pipeline-v1");
    AppendField(&canonical, "contract_version",
                std::to_string(pass_contract_generated::kContractVersion));
    AppendField(&canonical, "dialect", ToString(request.dialect));
    AppendField(&canonical, "scope", ToString(expected_scope));
    AppendField(&canonical, "target_kind", request.target->kind);
    AppendField(&canonical, "target_device",
                std::to_string(static_cast<int>(request.target->device_type)));
    AppendField(&canonical, "opt_level", std::to_string(request.opt_level));
    AppendField(&canonical, "named_pipeline",
                AsString(request.named_pipeline));
    for (const std::string& invariant : invariants) {
        AppendField(&canonical, "initial_invariant", invariant);
    }
    for (const std::string& analysis : analyses) {
        AppendField(&canonical, "initial_analysis", analysis);
    }

    int previous_phase = -1;
    for (const String& name : ordered) {
        const PassSpec& spec =
            PassRegistry::Global().Get(request.dialect, name);
        ValidatePassSpec(spec);
        if (spec.scope != expected_scope) {
            throw std::invalid_argument(
                "PipelineResolver pass scope does not match request: " +
                PassSpecKey(spec.dialect, spec.name));
        }
        const int phase = PhaseRank(spec);
        if (phase < previous_phase) {
            throw std::invalid_argument(
                "PipelineResolver pass phases are out of order");
        }
        previous_phase = phase;
        if (spec.target_dependent) {
            if (!(request.target->kind == "cuda" &&
                  request.target->device_type == kCUDA)) {
                throw std::invalid_argument(
                    "target-dependent pass requires a supported CUDA Target: " +
                    AsString(spec.name));
            }
            result.target_requirements.push_back(
                String("cuda_thread_binding"));
        }

        PipelineInvariantTransition transition;
        transition.pass_name = spec.name;
        transition.phase = spec.phase;
        transition.required = spec.required_invariants;
        transition.produced = spec.produced_invariants;
        transition.preserved_analyses = spec.preserved_analyses;
        transition.invalidated_analyses = spec.invalidated_analyses;
        transition.invariants_before = SetToArray(invariants);
        transition.analyses_before = SetToArray(analyses);
        for (const String& required : spec.required_invariants) {
            if (!invariants.count(AsString(required))) {
                throw std::invalid_argument(
                    "PipelineResolver missing required invariant '" +
                    AsString(required) + "' before pass " +
                    AsString(spec.name));
            }
        }
        for (const String& invalidated : spec.invalidated_analyses) {
            analyses.erase(AsString(invalidated));
        }
        for (const String& produced : spec.produced_invariants) {
            invariants.insert(AsString(produced));
        }
        transition.invariants_after = SetToArray(invariants);
        transition.analyses_after = SetToArray(analyses);
        result.invariant_transitions.push_back(std::move(transition));
        result.contract_versions.push_back(String(
            PassSpecKey(spec.dialect, spec.name) + "@v" +
            std::to_string(spec.schema_version)));
        AppendField(&canonical, "pass", AsString(spec.name));
        AppendField(&canonical, "pass_schema",
                    std::to_string(spec.schema_version));
        AppendField(&canonical, "phase", AsString(spec.phase));
        for (const std::string& invariant : invariants) {
            AppendField(&canonical, "invariant", invariant);
        }
        for (const std::string& analysis : analyses) {
            AppendField(&canonical, "analysis", analysis);
        }
    }
    result.canonical_bytes = String(canonical);
    result.fingerprint = String(profiling::HashText(canonical));
    return result;
}

}  // namespace kxc::api
