/*! \file src/compiler/pipeline_resolver.cc
 * \brief Resolves and executes the generated pass contract without hidden passes.
 */

#include "kxc/compiler/pipeline.h"

#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kxc/pass/context.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/normalize_to_anf.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"
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

void AppendArray(std::string* canonical, const std::string& name,
                 const Array<String>& values) {
    for (const String& value : values) AppendField(canonical, name, AsString(value));
}

std::string StepKindName(PipelineExecutionStepKind kind) {
    switch (kind) {
        case PipelineExecutionStepKind::kPass:
            return "pass";
    }
    throw std::invalid_argument("PipelineExecutor has an unknown step kind");
}

bool EqualArray(const Array<String>& lhs, const Array<String>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!(lhs[i] == rhs[i])) return false;
    }
    return true;
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

std::set<std::string> UniqueValues(const Array<String>& values,
                                   const char* field) {
    std::set<std::string> result;
    for (const String& value : values) {
        const std::string name = AsString(value);
        if (name.empty() || !result.insert(name).second) {
            throw std::invalid_argument(std::string("PipelineResolver ") + field +
                                        " must be unique and non-empty");
        }
    }
    return result;
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

void RequireTarget(const Target& target, const char* caller) {
    if (!target.defined() || !target.As<TargetNode>()) {
        throw std::invalid_argument(std::string(caller) +
                                    " requires a defined Target capability snapshot");
    }
    const bool cpu = target->kind == "llvm" && target->device_type == kCPU;
    const bool cuda = target->kind == "cuda" && target->device_type == kCUDA;
    if (!cpu && !cuda) {
        throw std::invalid_argument(std::string(caller) +
                                    " target kind and device type are inconsistent");
    }
}

Array<String> CompilerPolicy(const PipelineRequest& request) {
    const std::string dialect = ToString(request.dialect);
    if (request.opt_level < 0 || request.opt_level > 3) {
        throw std::invalid_argument("PipelineResolver requires opt_level in [0, 3]");
    }
    std::string name = dialect + ".compiler.";
    if (request.dialect == IRDialect::kTIR && request.opt_level == 3) {
        if (request.target->kind == "cuda" && request.target->device_type == kCUDA) {
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
        return pass_contract_generated::Pipeline(ToString(request.dialect) +
                                                  ".optimize_default");
    }
    return pass_contract_generated::Pipeline(named);
}

bool IsCompilerPipeline(const PipelineRequest& request) {
    const std::string named = AsString(request.named_pipeline);
    return named.empty() || named == "compiler";
}

std::string CanonicalBytes(const NormalizedPipeline& pipeline) {
    std::string canonical;
    AppendField(&canonical, "kind", "normalized-pipeline-v3");
    AppendField(&canonical, "contract_version",
                std::to_string(pass_contract_generated::kContractVersion));
    AppendField(&canonical, "dialect", ToString(pipeline.dialect));
    AppendField(&canonical, "scope", ToString(pipeline.scope));
    AppendField(&canonical, "opt_level", std::to_string(pipeline.opt_level));
    AppendField(&canonical, "named_pipeline", AsString(pipeline.named_pipeline));
    AppendArray(&canonical, "initial_invariant", pipeline.initial_invariants);
    AppendArray(&canonical, "initial_analysis", pipeline.initial_analyses);
    AppendArray(&canonical, "target_requirement", pipeline.target_requirements);
    for (const PipelineExecutionStep& step : pipeline.execution_steps) {
        AppendField(&canonical, "step_kind", StepKindName(step.kind));
        AppendField(&canonical, "step_dialect", ToString(step.dialect));
        AppendField(&canonical, "step_scope", ToString(step.scope));
        AppendField(&canonical, "step_occurrence", std::to_string(step.occurrence));
        AppendField(&canonical, "step_pass", AsString(step.pass_name));
        AppendField(&canonical, "step_phase", AsString(step.phase));
        AppendField(&canonical, "step_schema", std::to_string(step.schema_version));
        AppendField(&canonical, "step_implementation", AsString(step.implementation_key));
    }
    for (const PipelineInvariantTransition& transition :
         pipeline.invariant_transitions) {
        AppendField(&canonical, "transition_pass", AsString(transition.pass_name));
        AppendField(&canonical, "transition_phase", AsString(transition.phase));
        AppendArray(&canonical, "transition_required", transition.required);
        AppendArray(&canonical, "transition_produced", transition.produced);
        AppendArray(&canonical, "transition_declarative_only",
                    transition.declarative_only);
        AppendArray(&canonical, "transition_preserved", transition.preserved_analyses);
        AppendArray(&canonical, "transition_invalidated", transition.invalidated_analyses);
        AppendArray(&canonical, "transition_invariants_before",
                    transition.invariants_before);
        AppendArray(&canonical, "transition_invariants_after",
                    transition.invariants_after);
        AppendArray(&canonical, "transition_analyses_before", transition.analyses_before);
        AppendArray(&canonical, "transition_analyses_after", transition.analyses_after);
    }
    return canonical;
}

void EnsureRegistered(IRDialect dialect) {
    if (dialect == IRDialect::kRelay) {
        (void)relay::RelayRegisteredPassSpecs();
    } else if (dialect == IRDialect::kTIR) {
        (void)tir::TIRRegisteredPassSpecs();
    }
}

void ValidateTargetRequirements(const NormalizedPipeline& pipeline,
                                const Target& target, bool has_cuda_schedule) {
    RequireTarget(target, "PipelineExecutor");
    Array<String> expected{
        String(target->kind + ":" + std::to_string(static_cast<int>(target->device_type)))};
    if (has_cuda_schedule) expected.push_back(String("cuda_thread_binding"));
    if (!EqualArray(pipeline.target_requirements, expected)) {
        throw std::invalid_argument(
            "PipelineExecutor target requirements do not match execution steps");
    }
}

using CheckedTypeSnapshot = std::pair<Expr, Type>;

bool CollectCompleteCheckedTypes(
    const Expr& expr, std::unordered_set<const Object*>* visited,
    std::vector<CheckedTypeSnapshot>* snapshots) {
    if (!expr.defined()) return false;
    if (expr.As<relay::OpNode>()) return true;
    if (!expr.checked_type().defined()) return false;
    if (!visited->insert(expr.get()).second) return true;
    snapshots->emplace_back(expr, expr.checked_type());
    if (expr.As<ConstantNode>() || expr.As<VarNode>()) return true;
    if (const auto* call = expr.As<CallNode>()) {
        for (const Expr& argument : call->args) {
            if (!CollectCompleteCheckedTypes(argument, visited, snapshots)) {
                return false;
            }
        }
        return true;
    }
    if (const auto* function = expr.As<FunctionNode>()) {
        for (const Var& parameter : function->params) {
            if (!CollectCompleteCheckedTypes(parameter, visited, snapshots)) {
                return false;
            }
        }
        return CollectCompleteCheckedTypes(function->body, visited,
                                           snapshots);
    }
    if (const auto* branch = expr.As<IfNode>()) {
        return CollectCompleteCheckedTypes(branch->cond, visited, snapshots) &&
               CollectCompleteCheckedTypes(branch->true_branch, visited,
                                           snapshots) &&
               CollectCompleteCheckedTypes(branch->false_branch, visited,
                                           snapshots);
    }
    if (const auto* let = expr.As<LetNode>()) {
        return CollectCompleteCheckedTypes(let->var, visited, snapshots) &&
               CollectCompleteCheckedTypes(let->value, visited, snapshots) &&
               CollectCompleteCheckedTypes(let->body, visited, snapshots);
    }
    if (const auto* tuple = expr.As<TupleNode>()) {
        for (const Expr& field : tuple->fields) {
            if (!CollectCompleteCheckedTypes(field, visited, snapshots)) {
                return false;
            }
        }
        return true;
    }
    if (const auto* item = expr.As<TupleGetItemNode>()) {
        return CollectCompleteCheckedTypes(item->tuple, visited, snapshots);
    }
    return false;
}

class CheckedTypeRestore final {
public:
    explicit CheckedTypeRestore(
        const std::vector<CheckedTypeSnapshot>* snapshots)
        : snapshots_(snapshots) {}

    ~CheckedTypeRestore() {
        for (const auto& [expr, type] : *snapshots_) {
            SetCheckedType(expr, type);
        }
    }

private:
    const std::vector<CheckedTypeSnapshot>* snapshots_;
};

bool ContainsName(const Array<String>& values, const std::string& expected) {
    for (const String& value : values) {
        if (AsString(value) == expected) return true;
    }
    return false;
}

void ValidateInvariantContract(const PassSpec& spec) {
    for (const String& required : spec.required_invariants) {
        const std::string name = AsString(required);
        if (ContainsName(spec.declarative_only_invariants, name) ||
            !PipelineInvariantValidator::IsExecutable(spec.dialect, required)) {
            throw std::invalid_argument(
                "PipelineResolver cannot use declarative-only or unsupported invariant '" +
                name + "' as a production precondition for " +
                PassSpecKey(spec.dialect, spec.name));
        }
    }
    for (const String& produced : spec.produced_invariants) {
        const std::string name = AsString(produced);
        const bool executable =
            PipelineInvariantValidator::IsExecutable(spec.dialect, produced);
        const bool declarative =
            ContainsName(spec.declarative_only_invariants, name);
        if (executable == declarative) {
            throw std::invalid_argument(
                "PassSpec " + PassSpecKey(spec.dialect, spec.name) +
                " must classify produced invariant '" + name +
                " as exactly one of executable or declarative-only");
        }
    }
}

void ValidateExecutableInvariantNames(IRDialect dialect,
                                      const Array<String>& invariants,
                                      const char* context) {
    for (const String& invariant : invariants) {
        if (!PipelineInvariantValidator::IsExecutable(dialect, invariant)) {
            throw std::invalid_argument(std::string(context) +
                                        " contains unsupported production invariant '" +
                                        AsString(invariant) + "'");
        }
    }
}

}  // namespace

bool NormalizedPipeline::defined() const noexcept {
    return dialect != IRDialect::kUnknown && scope != PassScope::kUnknown &&
           canonical_bytes.defined() && !std::string(canonical_bytes).empty() &&
           fingerprint.defined() && !std::string(fingerprint).empty();
}

NormalizedPipeline PipelineResolver::Resolve(const PipelineRequest& request) {
    if (request.dialect != IRDialect::kRelay && request.dialect != IRDialect::kTIR) {
        throw std::invalid_argument("PipelineResolver requires Relay or TIR dialect");
    }
    const PassScope expected_scope = request.dialect == IRDialect::kRelay
                                         ? PassScope::kGraph
                                         : PassScope::kPrimFunc;
    if (request.requested_scope != expected_scope) {
        throw std::invalid_argument(
            "PipelineResolver requested scope does not match dialect");
    }
    RequireTarget(request.target, "PipelineResolver");
    EnsureRegistered(request.dialect);

    const auto enabled = UniqueNames(request.enabled, "enabled");
    const auto disabled = UniqueNames(request.disabled, "disabled");
    for (const std::string& name : enabled) {
        if (disabled.count(name)) {
            throw std::invalid_argument(
                "PipelineResolver cannot both enable and disable pass: " + name);
        }
    }
    if (IsCompilerPipeline(request) &&
        (disabled.count("infer_type") || disabled.count("normalize_to_anf"))) {
        throw std::invalid_argument(
            "PipelineResolver cannot disable mandatory compiler infer_type or ANF steps");
    }

    Array<String> ordered;
    std::unordered_set<std::string> seen;
    const auto append_unique = [&ordered, &seen](const String& pass) {
        const std::string name = AsString(pass);
        if (name.empty() || !seen.insert(name).second) {
            throw std::invalid_argument(
                "generated pipeline contains an empty or duplicate pass");
        }
        ordered.push_back(pass);
    };
    const bool compiler_pipeline = IsCompilerPipeline(request);
    if (compiler_pipeline && request.dialect == IRDialect::kRelay) {
        // These are real contract passes, deliberately duplicated around Relay optimization.
        ordered.push_back(String("infer_type"));
        seen.insert("infer_type");
    }
    for (const String& pass : BasePipeline(request)) {
        const std::string name = AsString(pass);
        if (!disabled.count(name)) append_unique(pass);
    }
    for (const String& pass : request.enabled) {
        const std::string name = AsString(pass);
        if (seen.insert(name).second) ordered.push_back(pass);
    }
    if (compiler_pipeline && request.dialect == IRDialect::kRelay) {
        ordered.push_back(String("infer_type"));
        append_unique(String("normalize_to_anf"));
    }

    std::set<std::string> invariants =
        UniqueValues(request.initial_invariants, "initial invariants");
    ValidateExecutableInvariantNames(request.dialect,
                                     request.initial_invariants,
                                     "PipelineResolver initial invariants");
    std::set<std::string> analyses =
        UniqueValues(request.initial_analyses, "initial analyses");

    NormalizedPipeline result;
    result.dialect = request.dialect;
    result.scope = expected_scope;
    result.opt_level = request.opt_level;
    result.named_pipeline = request.named_pipeline;
    result.initial_invariants = request.initial_invariants;
    result.initial_analyses = request.initial_analyses;
    result.ordered_passes = ordered;
    result.target_requirements = {
        String(request.target->kind + ":" +
               std::to_string(static_cast<int>(request.target->device_type)))};
    result.contract_versions.push_back(String(
        "pass-contract-v" + std::to_string(pass_contract_generated::kContractVersion)));

    int previous_phase = -1;
    std::unordered_map<std::string, size_t> occurrences;
    for (const String& name : ordered) {
        const PassSpec& spec = PassRegistry::Global().Get(request.dialect, name);
        PipelineInvariantValidator::ValidateProductionContract(spec);
        if (spec.scope != expected_scope) {
            throw std::invalid_argument(
                "PipelineResolver pass scope does not match request: " +
                PassSpecKey(spec.dialect, spec.name));
        }
        const int phase = PhaseRank(spec);
        if (phase < previous_phase) {
            throw std::invalid_argument("PipelineResolver pass phases are out of order");
        }
        previous_phase = phase;
        if (spec.target_dependent) {
            if (!(request.target->kind == "cuda" && request.target->device_type == kCUDA)) {
                throw std::invalid_argument(
                    "target-dependent pass requires a supported CUDA Target: " +
                    AsString(spec.name));
            }
            result.target_requirements.push_back(String("cuda_thread_binding"));
        }

        PipelineExecutionStep step;
        step.dialect = spec.dialect;
        step.scope = spec.scope;
        step.occurrence = occurrences[AsString(spec.name)]++;
        step.pass_name = spec.name;
        step.phase = spec.phase;
        step.schema_version = spec.schema_version;
        step.implementation_key = spec.implementation_key;
        result.execution_steps.push_back(std::move(step));

        PipelineInvariantTransition transition;
        transition.pass_name = spec.name;
        transition.phase = spec.phase;
        transition.required = spec.required_invariants;
        transition.produced = spec.produced_invariants;
        transition.declarative_only = spec.declarative_only_invariants;
        transition.preserved_analyses = spec.preserved_analyses;
        transition.invalidated_analyses = spec.invalidated_analyses;
        transition.invariants_before = SetToArray(invariants);
        transition.analyses_before = SetToArray(analyses);
        for (const String& required : spec.required_invariants) {
            if (!invariants.count(AsString(required))) {
                throw std::invalid_argument("PipelineResolver missing required invariant '" +
                                            AsString(required) + "' before pass " +
                                            AsString(spec.name));
            }
        }
        for (const String& invalidated : spec.invalidated_analyses) {
            analyses.erase(AsString(invalidated));
        }
        for (const String& produced : spec.produced_invariants) {
            if (PipelineInvariantValidator::IsExecutable(spec.dialect,
                                                         produced)) {
                invariants.insert(AsString(produced));
            }
        }
        transition.invariants_after = SetToArray(invariants);
        transition.analyses_after = SetToArray(analyses);
        result.invariant_transitions.push_back(std::move(transition));
        result.contract_versions.push_back(String(
            PassSpecKey(spec.dialect, spec.name) + "@v" +
            std::to_string(spec.schema_version)));
    }
    result.canonical_bytes = String(CanonicalBytes(result));
    result.fingerprint =
        String(profiling::HashText(static_cast<std::string>(result.canonical_bytes)));
    return result;
}

void PipelineExecutor::Validate(const NormalizedPipeline& pipeline,
                                const Target& target) {
    if (!pipeline.defined()) {
        throw std::invalid_argument("PipelineExecutor requires a defined normalized pipeline");
    }
    if (pipeline.dialect != IRDialect::kRelay && pipeline.dialect != IRDialect::kTIR) {
        throw std::invalid_argument("PipelineExecutor pipeline dialect is invalid");
    }
    const PassScope expected_scope = pipeline.dialect == IRDialect::kRelay
                                         ? PassScope::kGraph
                                         : PassScope::kPrimFunc;
    if (pipeline.scope != expected_scope ||
        pipeline.ordered_passes.size() != pipeline.execution_steps.size() ||
        pipeline.execution_steps.size() != pipeline.invariant_transitions.size()) {
        throw std::invalid_argument("PipelineExecutor pipeline step vectors disagree");
    }
    EnsureRegistered(pipeline.dialect);
    std::set<std::string> invariants =
        UniqueValues(pipeline.initial_invariants, "initial invariants");
    ValidateExecutableInvariantNames(pipeline.dialect,
                                     pipeline.initial_invariants,
                                     "PipelineExecutor initial invariants");
    std::set<std::string> analyses =
        UniqueValues(pipeline.initial_analyses, "initial analyses");
    std::unordered_map<std::string, size_t> occurrences;
    int previous_phase = -1;
    bool has_cuda_schedule = false;
    for (size_t i = 0; i < pipeline.execution_steps.size(); ++i) {
        const PipelineExecutionStep& step = pipeline.execution_steps[i];
        const PipelineInvariantTransition& transition =
            pipeline.invariant_transitions[i];
        if (step.kind != PipelineExecutionStepKind::kPass ||
            step.dialect != pipeline.dialect || step.scope != pipeline.scope ||
            !(pipeline.ordered_passes[i] == step.pass_name) ||
            step.occurrence != occurrences[AsString(step.pass_name)]++) {
            throw std::invalid_argument("PipelineExecutor step identity was tampered");
        }
        const PassSpec& spec = PassRegistry::Global().Get(step.dialect, step.pass_name);
        PipelineInvariantValidator::ValidateProductionContract(spec);
        if (spec.scope != expected_scope || step.phase != spec.phase ||
            step.schema_version != spec.schema_version ||
            step.implementation_key != spec.implementation_key ||
            transition.pass_name != spec.name || transition.phase != spec.phase ||
            !EqualArray(transition.required, spec.required_invariants) ||
            !EqualArray(transition.produced, spec.produced_invariants) ||
            !EqualArray(transition.declarative_only,
                        spec.declarative_only_invariants) ||
            !EqualArray(transition.preserved_analyses, spec.preserved_analyses) ||
            !EqualArray(transition.invalidated_analyses, spec.invalidated_analyses) ||
            !EqualArray(transition.invariants_before, SetToArray(invariants)) ||
            !EqualArray(transition.analyses_before, SetToArray(analyses))) {
            throw std::invalid_argument("PipelineExecutor pass contract transition was tampered");
        }
        const int phase = PhaseRank(spec);
        if (phase < previous_phase) {
            throw std::invalid_argument("PipelineExecutor pass phases are out of order");
        }
        previous_phase = phase;
        for (const String& required : spec.required_invariants) {
            if (!invariants.count(AsString(required))) {
                throw std::invalid_argument("PipelineExecutor missing required invariant '" +
                                            AsString(required) + "'");
            }
        }
        for (const String& invalidated : spec.invalidated_analyses) {
            analyses.erase(AsString(invalidated));
        }
        for (const String& produced : spec.produced_invariants) {
            if (PipelineInvariantValidator::IsExecutable(spec.dialect,
                                                         produced)) {
                invariants.insert(AsString(produced));
            }
        }
        if (!EqualArray(transition.invariants_after, SetToArray(invariants)) ||
            !EqualArray(transition.analyses_after, SetToArray(analyses))) {
            throw std::invalid_argument("PipelineExecutor post-pass transition was tampered");
        }
        has_cuda_schedule = has_cuda_schedule || spec.target_dependent;
    }
    ValidateTargetRequirements(pipeline, target, has_cuda_schedule);
    const std::string canonical = CanonicalBytes(pipeline);
    if (canonical != AsString(pipeline.canonical_bytes) ||
        profiling::HashText(canonical) != AsString(pipeline.fingerprint)) {
        throw std::invalid_argument("PipelineExecutor canonical pipeline identity was tampered");
    }
}

bool PipelineInvariantValidator::IsExecutable(
    IRDialect dialect, const String& invariant) {
    if (dialect != IRDialect::kRelay) return false;
    const std::string name = AsString(invariant);
    return name == "checked_type" || name == "anf";
}

void PipelineInvariantValidator::ValidateProductionContract(
    const PassSpec& spec) {
    ValidatePassSpec(spec);
    ValidateInvariantContract(spec);
}

void PipelineInvariantValidator::ValidateRelay(
    const Array<String>& invariants, const Function& function) {
    if (!function.defined()) {
        throw std::invalid_argument(
            "Relay invariant validation requires a defined Function");
    }
    for (const String& invariant : invariants) {
        const std::string name = AsString(invariant);
        if (!IsExecutable(IRDialect::kRelay, invariant)) {
            throw std::invalid_argument(
                "no executable Relay invariant validator for '" + name + "'");
        }
        if (name == "checked_type") {
            std::unordered_set<const Object*> visited;
            std::vector<CheckedTypeSnapshot> snapshots;
            if (!CollectCompleteCheckedTypes(Expr(ObjectRef(function)),
                                             &visited, &snapshots)) {
                throw std::runtime_error(
                    "Relay invariant 'checked_type' is incomplete");
            }
            const CheckedTypeRestore restore(&snapshots);
            (void)relay::InferTypePass(function);
            for (const auto& [expr, type] : snapshots) {
                if (!TypeEqual(type, expr.checked_type())) {
                    throw std::runtime_error(
                        "Relay invariant 'checked_type' is stale or inconsistent");
                }
            }
        } else if (name == "anf") {
            relay::VerifyANF(function);
        }
    }
}

void PipelineInvariantValidator::ValidateTIR(
    const Array<String>& invariants, const tir::PrimFunc& function) {
    if (!function.defined()) {
        throw std::invalid_argument(
            "TIR invariant validation requires a defined PrimFunc");
    }
    for (const String& invariant : invariants) {
        throw std::invalid_argument(
            "no executable TIR invariant validator for '" +
            AsString(invariant) + "'");
    }
}

Function PipelineExecutor::ExecuteRelay(const NormalizedPipeline& pipeline,
                                        const Function& function,
                                        const Target& target) {
    Validate(pipeline, target);
    if (pipeline.dialect != IRDialect::kRelay || !function.defined()) {
        throw std::invalid_argument("PipelineExecutor ExecuteRelay requires Relay pipeline and Function");
    }
    PassContext::Scope scope(PassContext::MergeTarget(PassContext::Current(), target));
    Function current = function;
    PipelineInvariantValidator::ValidateRelay(pipeline.initial_invariants,
                                              current);
    for (size_t i = 0; i < pipeline.execution_steps.size(); ++i) {
        current = relay::RunRelayPassPipeline(
            current, {pipeline.execution_steps[i].pass_name});
        PipelineInvariantValidator::ValidateRelay(
            pipeline.invariant_transitions[i].invariants_after, current);
    }
    return current;
}

tir::PrimFunc PipelineExecutor::ExecuteTIR(const NormalizedPipeline& pipeline,
                                           const tir::PrimFunc& function,
                                           const Target& target) {
    Validate(pipeline, target);
    if (pipeline.dialect != IRDialect::kTIR || !function.defined()) {
        throw std::invalid_argument("PipelineExecutor ExecuteTIR requires TIR pipeline and PrimFunc");
    }
    PassContext::Scope scope(PassContext::MergeTarget(PassContext::Current(), target));
    tir::PrimFunc current = function;
    PipelineInvariantValidator::ValidateTIR(pipeline.initial_invariants,
                                            current);
    for (size_t i = 0; i < pipeline.execution_steps.size(); ++i) {
        const PipelineExecutionStep& step = pipeline.execution_steps[i];
        current = tir::RunTIRPassPipeline(current, {step.pass_name});
        PipelineInvariantValidator::ValidateTIR(
            pipeline.invariant_transitions[i].invariants_after, current);
        const PassSpec& spec = PassRegistry::Global().Get(step.dialect, step.pass_name);
        if (spec.target_dependent) {
            const tir::CudaLaunchConfig launch = tir::GetCudaLaunchConfig(current);
            if (launch.grid_x == 0 || launch.block_x == 0) {
                throw std::runtime_error("PipelineExecutor CUDA schedule produced invalid launch metadata");
            }
        }
    }
    return current;
}

}  // namespace kxc::api
