#include "kxc/compiler/restricted_shape_runtime_bridge.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE
#define KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE 0
#endif
#ifndef KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
#define KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE 0
#endif
#ifndef KXC_ENABLE_RUNTIME_SHAPE_TASKS
#define KXC_ENABLE_RUNTIME_SHAPE_TASKS 0
#endif

namespace kxc::api::experimental::restricted_shape_runtime_bridge::v1 {
namespace restricted = kxc::api::experimental::restricted_symbolic_shape::v1;
namespace shape = kxc::shape::experimental::v1;
namespace runtime = kxc::runtime;
namespace {

[[noreturn]] void Reject(const std::string& message) {
    throw std::invalid_argument("RestrictedShapeRuntimeBridge: " + message);
}

void RequireEnabled() {
#if !KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE || !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE || !KXC_ENABLE_RUNTIME_SHAPE_TASKS
    throw std::runtime_error("RestrictedShapeRuntimeBridge is disabled; configure with "
                             "-DKXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE=ON, "
                             "-DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON, and "
                             "-DKXC_ENABLE_RUNTIME_SHAPE_TASKS=ON");
#endif
}

std::string DType(const shape::TensorAbiDescriptor& abi) {
    if (abi.dtype() == shape::DataType::kFloat16) return "float16";
    if (abi.dtype() == shape::DataType::kFloat32) return "float32";
    Reject("unsupported tensor dtype");
}

void RequireCpuAbi(const shape::TensorAbiDescriptor& abi) {
    if (abi.device().kind() != shape::DeviceKind::kCpu || abi.device().id() != 0 ||
        abi.target_backend_abi().abi_version() != runtime::RuntimeShapePlan::kAbiVersion) {
        Reject("only CPU:0 shape ABI version 1 is bridgeable");
    }
}

const shape::NamedTensorContract& NamedOutput(const shape::GraphTemplate& graph,
                                              const std::string& name) {
    const auto& values = graph.shape_program().outputs();
    const auto it = std::find_if(values.begin(), values.end(), [&name](const auto& value) {
        return value.name == name;
    });
    if (it == values.end()) Reject("last unit output is not a graph output");
    return *it;
}

const shape::BucketValueBoundary& BucketBoundary(const shape::BucketPolicy& policy,
                                                  const std::string& name) {
    const auto it = std::find_if(policy.boundaries().begin(), policy.boundaries().end(),
                                 [&name](const auto& value) { return value.name == name; });
    if (it == policy.boundaries().end()) Reject("bucket policy omitted final output boundary");
    return *it;
}

const shape::SymbolicBoundaryContract& SymbolicBoundary(const shape::PolymorphicPolicy& policy,
                                                         const std::string& name) {
    const auto it = std::find_if(policy.boundaries().begin(), policy.boundaries().end(),
                                 [&name](const auto& value) { return value.name == name; });
    if (it == policy.boundaries().end()) Reject("polymorphic policy omitted final output boundary");
    return *it;
}

struct Domain {
    runtime::RuntimeShapeExtent lower;
    runtime::RuntimeShapeExtent upper;
    runtime::RuntimeShapeExtent divisible_by;
};

runtime::RuntimeShapeExtent ToExtent(int64_t value, const char* label) {
    if (value < 0) Reject(std::string(label) + " must be non-negative");
    return static_cast<runtime::RuntimeShapeExtent>(value);
}

runtime::RuntimeShapeExtent CheckedLcm(runtime::RuntimeShapeExtent left,
                                       runtime::RuntimeShapeExtent right) {
    const auto gcd = [](runtime::RuntimeShapeExtent a, runtime::RuntimeShapeExtent b) {
        while (b != 0) {
            const auto next = a % b;
            a = b;
            b = next;
        }
        return a;
    };
    const auto factor = left / gcd(left, right);
    if (factor > std::numeric_limits<runtime::RuntimeShapeExtent>::max() / right) {
        Reject("input divisibility guard overflows");
    }
    return factor * right;
}

std::map<std::string, Domain> Domains(
    const restricted::PreparedRestrictedSymbolicTemplate& prepared,
    const shape::ApplicabilityGuard* policy_guard) {
    std::map<std::string, Domain> result;
    for (const auto& symbol : prepared.input_axis_symbols()) {
        const Domain next{ToExtent(symbol.lower, "input guard lower"),
                          ToExtent(symbol.upper, "input guard upper"),
                          ToExtent(symbol.divisible_by, "input guard divisor")};
        const auto [it, inserted] = result.emplace(symbol.symbol, next);
        if (!inserted && (it->second.lower != next.lower || it->second.upper != next.upper ||
                          it->second.divisible_by != next.divisible_by)) {
            Reject("input-axis symbol has inconsistent runtime guards");
        }
    }
    if (!policy_guard) return result;
    for (const auto& constraint : policy_guard->constraints()) {
        const auto symbols = constraint.Symbols();
        if (symbols.size() != 1 || result.count(symbols[0]) == 0 ||
            constraint.left().kind() != shape::DimExpr::Kind::kSymbol ||
            constraint.left().CanonicalString() != shape::DimExpr::Symbol(symbols[0]).CanonicalString()) {
            Reject("policy guard is not a direct explicit input-axis guard");
        }
        Domain& domain = result.at(symbols[0]);
        if (constraint.kind() == shape::Constraint::Kind::kRange) {
            domain.lower = std::max(domain.lower, ToExtent(constraint.lower(), "policy lower"));
            domain.upper = std::min(domain.upper, ToExtent(constraint.upper(), "policy upper"));
        } else if (constraint.kind() == shape::Constraint::Kind::kDivisibleBy) {
            domain.divisible_by = CheckedLcm(domain.divisible_by,
                                              ToExtent(constraint.divisor(), "policy divisor"));
        } else {
            Reject("policy guard kind is unsupported by runtime input guards");
        }
    }
    for (const auto& [symbol, domain] : result) {
        if (domain.lower > domain.upper) Reject("policy guard has an empty input-axis domain");
    }
    return result;
}

struct AxisReference {
    std::size_t input_index;
    std::size_t axis;
};

std::optional<std::string> DirectSymbol(const shape::DimExpr& expression) {
    if (expression.kind() != shape::DimExpr::Kind::kSymbol) return std::nullopt;
    const auto symbols = expression.Symbols();
    return symbols.size() == 1 ? std::optional<std::string>(symbols[0]) : std::nullopt;
}

runtime::RuntimeShapeExpr Expression(const shape::DimExpr& expression,
                                     runtime::RuntimeShapeExtent exact,
                                     const std::map<std::string, AxisReference>& references) {
    if (const auto symbol = DirectSymbol(expression)) {
        const auto it = references.find(*symbol);
        if (it == references.end()) Reject("output uses a symbol without an explicit input axis");
        return runtime::RuntimeShapeExpr::InputAxis(it->second.input_index, it->second.axis);
    }
    return runtime::RuntimeShapeExpr::Const(exact);
}

std::vector<runtime::RuntimeShapeExpr> Expressions(
    const std::vector<shape::DimExpr>& dimensions,
    const std::vector<int64_t>& exact,
    const std::map<std::string, AxisReference>& references) {
    if (dimensions.size() != exact.size()) Reject("symbolic and exact output ranks differ");
    std::vector<runtime::RuntimeShapeExpr> result;
    result.reserve(dimensions.size());
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
        result.push_back(Expression(dimensions[axis], ToExtent(exact[axis], "output extent"), references));
    }
    return result;
}

std::vector<runtime::RuntimeShapeExpr> Constants(const std::vector<int64_t>& extents) {
    std::vector<runtime::RuntimeShapeExpr> result;
    result.reserve(extents.size());
    for (const auto extent : extents) result.push_back(runtime::RuntimeShapeExpr::Const(ToExtent(extent, "output extent")));
    return result;
}

std::size_t ByteCount(const runtime::RuntimeShapeTensorContract& output,
                      const std::vector<std::vector<runtime::RuntimeShapeExtent>>& maximum_inputs) {
    runtime::RuntimeShapeExtent elements = 1;
    for (const auto& expression : output.physical) {
        const auto extent = expression.Evaluate(maximum_inputs);
        if (extent != 0 && elements > std::numeric_limits<runtime::RuntimeShapeExtent>::max() / extent) {
            Reject("physical output capacity overflows");
        }
        elements *= extent;
    }
    const std::size_t bytes = output.dtype == "float16" ? 2 : 4;
    if (elements > std::numeric_limits<std::size_t>::max() / bytes) {
        Reject("physical output byte capacity overflows");
    }
    return static_cast<std::size_t>(elements) * bytes;
}

}  // namespace

bool RestrictedShapeRuntimeBridge::IsEnabled() noexcept {
#if KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE && KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE && KXC_ENABLE_RUNTIME_SHAPE_TASKS
    return true;
#else
    return false;
#endif
}

runtime::RuntimeShapePlan RestrictedShapeRuntimeBridge::Bind(
    const restricted::PreparedRestrictedSymbolicTemplate& prepared,
    const restricted::RestrictedDispatchDecision& decision,
    TrustedSynchronousLauncherDescriptor launcher) {
    RequireEnabled();
    if (launcher.module_label.empty() || launcher.entry_symbol.empty() ||
        launcher.artifact_identity.empty() || !launcher.launcher) {
        Reject("trusted synchronous launcher descriptor is incomplete");
    }
    const auto& graph = decision.graph_template();
    if (graph.CanonicalBytes() != prepared.graph_template().CanonicalBytes()) {
        Reject("decision was not minted from the supplied prepared template");
    }
    const auto& units = graph.ordered_units();
    if (units.empty() || units.back().output_value_names.size() != 1) {
        Reject("restricted tree has no single final output");
    }
    const std::string& final_name = units.back().output_value_names.front();
    const auto& final_template = NamedOutput(graph, final_name);
    const std::string artifact_identity = decision.kind() == restricted::DispatchKind::kExact
        ? decision.exact_requests().back().artifact_key.CanonicalBytes()
        : decision.guarded_requests().back().artifact_key.CanonicalBytes();
    const auto* bucket_policy = decision.bucket_policy();
    if (decision.kind() == restricted::DispatchKind::kBucket && !bucket_policy) {
        Reject("bucket decision lacks its validated policy");
    }
    const std::string tail_policy_identity = bucket_policy ? bucket_policy->CanonicalString() : std::string{};
    if (launcher.artifact_identity != artifact_identity ||
        launcher.tail_policy_identity != tail_policy_identity) {
        Reject("trusted launcher artifact or tail-policy binding does not match decision");
    }
    RequireCpuAbi(final_template.contract.abi());
    const auto& exact = decision.exact_oracle().profile();
    const auto& exact_final = exact.Value(final_name).contract;
    const shape::ApplicabilityGuard* policy_guard = nullptr;
    if (const auto* profile = decision.guarded_profile()) policy_guard = &profile->guard();
    const auto domains = Domains(prepared, policy_guard);

    runtime::RuntimeShapePlanSpec spec;
    std::map<std::pair<std::size_t, std::size_t>, const restricted::InputAxisSymbol*> symbols;
    for (const auto& symbol : prepared.input_axis_symbols()) {
        symbols.emplace(std::make_pair(symbol.parameter_index, symbol.axis), &symbol);
    }
    std::map<std::string, AxisReference> references;
    std::vector<std::vector<runtime::RuntimeShapeExtent>> maximum_inputs;
    const auto& template_inputs = graph.shape_program().inputs();
    maximum_inputs.resize(template_inputs.size());
    for (std::size_t input_index = 0; input_index < template_inputs.size(); ++input_index) {
        const auto& source = template_inputs[input_index];
        RequireCpuAbi(source.contract.abi());
        const auto& concrete = exact.Value(source.name).contract;
        if (concrete.logical.size() != source.contract.logical().dimensions().size()) {
            Reject("exact input rank differs from template");
        }
        runtime::RuntimeShapeInputContract input;
        input.dtype = DType(source.contract.abi());
        input.rank = concrete.logical.size();
        input.device = "CPU:0";
        input.abi_version = runtime::RuntimeShapePlan::kAbiVersion;
        input.requires_data = true;
        maximum_inputs[input_index].resize(input.rank);
        for (std::size_t axis = 0; axis < input.rank; ++axis) {
            runtime::RuntimeShapeInputAxisGuard guard;
            guard.axis = axis;
            const auto found = symbols.find({input_index, axis});
            if (found == symbols.end()) {
                const auto value = ToExtent(concrete.logical[axis], "exact input extent");
                guard.lower = value;
                guard.upper = value;
                guard.exact = value;
                maximum_inputs[input_index][axis] = value;
            } else {
                const auto& declared = *found->second;
                const auto domain = domains.at(declared.symbol);
                guard.lower = domain.lower;
                guard.upper = domain.upper;
                guard.divisible_by = domain.divisible_by;
                if (decision.kind() == restricted::DispatchKind::kExact) {
                    const auto bound = exact.key().bindings().Find(declared.symbol);
                    if (!bound) Reject("exact decision omitted explicit input-axis binding");
                    guard.exact = ToExtent(*bound, "exact input-axis binding");
                    guard.lower = *guard.exact;
                    guard.upper = *guard.exact;
                }
                const auto [reference, inserted] = references.emplace(
                    declared.symbol, AxisReference{input_index, axis});
                if (!inserted) guard.equal_to = runtime::RuntimeShapeInputAxisReference{
                    reference->second.input_index, reference->second.axis};
                maximum_inputs[input_index][axis] = guard.upper;
            }
            input.axis_guards.push_back(std::move(guard));
        }
        spec.inputs.push_back(std::move(input));
    }

    if (decision.kind() == restricted::DispatchKind::kPolymorphic) {
        const auto* policy = decision.polymorphic_policy();
        if (!policy) Reject("polymorphic decision lacks its validated policy");
        if (policy->runtime_extent_abi().size() != domains.size()) {
            Reject("polymorphic runtime extent ABI does not cover input-axis guards");
        }
        for (std::size_t index = 0; index < policy->runtime_extent_abi().size(); ++index) {
            const auto& scalar = policy->runtime_extent_abi()[index];
            const auto domain = domains.find(scalar.symbol);
            const auto reference = references.find(scalar.symbol);
            if (scalar.ordinal != index || domain == domains.end() || reference == references.end() ||
                scalar.lower != static_cast<int64_t>(domain->second.lower) ||
                scalar.upper != static_cast<int64_t>(domain->second.upper) ||
                scalar.divisible_by != static_cast<int64_t>(domain->second.divisible_by)) {
                Reject("polymorphic runtime extent ABI disagrees with input-axis guards");
            }
            spec.runtime_extent_abi.push_back(runtime::RuntimeShapeExtentScalar{
                scalar.ordinal, scalar.name, scalar.symbol, reference->second.input_index,
                reference->second.axis, domain->second.lower, domain->second.upper,
                domain->second.divisible_by});
        }
    }

    runtime::RuntimeShapeTensorContract output;
    output.dtype = DType(final_template.contract.abi());
    output.alignment = static_cast<std::size_t>(final_template.contract.physical().alignment());
    output.layout = final_template.contract.physical().layout();
    output.scope = final_template.contract.physical().memory_scope();
    output.device = "CPU:0";
    output.abi_version = runtime::RuntimeShapePlan::kAbiVersion;
    if (decision.kind() == restricted::DispatchKind::kExact) {
        output.logical = Constants(exact_final.logical);
        output.physical = Constants(exact_final.physical);
        output.valid = Constants(exact_final.valid);
    } else if (decision.kind() == restricted::DispatchKind::kBucket) {
        const auto* policy = decision.bucket_policy();
        if (!policy) Reject("bucket decision lacks its validated policy");
        const auto& boundary = BucketBoundary(*policy, final_name);
        const auto& tail = policy->tail_contracts().back();
        bool padded = false;
        for (std::size_t axis = 0; axis < boundary.physical.size(); ++axis) {
            padded = padded || boundary.physical[axis] > exact_final.logical[axis];
        }
        if (padded && (!tail.staging_pad || !tail.mask || !tail.predicate || !tail.output_crop)) {
            Reject("bucket final output lacks validated tail behavior");
        }
        output.logical = Expressions(final_template.contract.logical().dimensions(), exact_final.logical, references);
        output.physical = Constants(boundary.physical);
        output.valid = Expressions(final_template.contract.valid().dimensions(), exact_final.valid, references);
    } else {
        const auto* policy = decision.polymorphic_policy();
        if (!policy) Reject("polymorphic decision lacks its validated policy");
        (void)SymbolicBoundary(*policy, final_name);
        output.logical = Expressions(final_template.contract.logical().dimensions(), exact_final.logical, references);
        output.physical = Expressions(final_template.contract.physical().capacity(), exact_final.physical, references);
        output.valid = Expressions(final_template.contract.valid().dimensions(), exact_final.valid, references);
    }
    output.max_bytes = ByteCount(output, maximum_inputs);
    spec.outputs.push_back(std::move(output));
    spec.run_byte_budget = spec.outputs.front().max_bytes;
    spec.entry.module_label = std::move(launcher.module_label);
    spec.entry.entry_symbol = std::move(launcher.entry_symbol);
    spec.entry.abi_version = runtime::RuntimeShapePlan::kAbiVersion;
    spec.entry.ready = true;
    spec.entry.artifact_identity = std::move(launcher.artifact_identity);
    spec.entry.tail_policy_identity = std::move(launcher.tail_policy_identity);
    spec.entry.launcher = std::move(launcher.launcher);
    spec.entry.module_lease = std::move(launcher.module_lease);
    spec.entry.exact_abi_fingerprint = runtime::RuntimeShapePlan::ExactAbiFingerprint(
        spec.inputs, spec.outputs, spec.runtime_extent_abi, spec.entry.artifact_identity,
        spec.entry.tail_policy_identity);
    return runtime::RuntimeShapePlan(std::move(spec));
}

}  // namespace kxc::api::experimental::restricted_shape_runtime_bridge::v1
