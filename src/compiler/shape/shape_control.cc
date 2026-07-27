#include "kxc/compiler/shape_control.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

#include "runtime/internal/memory_plan.h"

namespace kxc::api::experimental::shape_control::v1 {
namespace {

[[noreturn]] void Reject(const std::string& message) {
    throw std::invalid_argument("shape control: " + message);
}

void VerifyOracle(const specialization::GraphTemplate& graph_template,
                  const specialization::ExactOracle& oracle) {
    // This existing consumer verifies template content, graph identity,
    // profile policy, every exact contract, and request routing.
    (void)specialization::MakeExactSpecializationRequests(graph_template,
                                                           oracle);
}

std::vector<ConcreteInputShape> ExactInputs(
    const specialization::GraphTemplate& graph_template,
    const specialization::ExactOracle& oracle) {
    VerifyOracle(graph_template, oracle);
    std::vector<ConcreteInputShape> result;
    result.reserve(graph_template.shape_program().inputs().size());
    for (const auto& input : graph_template.shape_program().inputs()) {
        result.push_back(
            {input.name, oracle.profile().Value(input.name).contract.logical});
    }
    return result;
}

bool SameInputs(const std::vector<ConcreteInputShape>& left,
                const std::vector<ConcreteInputShape>& right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin(),
                      [](const ConcreteInputShape& a,
                         const ConcreteInputShape& b) {
                          return a.name == b.name &&
                                 a.logical == b.logical;
                      });
}

std::vector<OrderedArtifactIdentity> OrderedArtifacts(
    const CompiledGraph& compiled) {
    const auto calls = compiled.plan().calls();
    const auto& pins = compiled.artifact_pins();
    if (calls.empty() || calls.size() != pins.size()) {
        Reject("compiled variant has no complete ordered artifacts");
    }
    std::vector<OrderedArtifactIdentity> result;
    result.reserve(calls.size());
    for (std::size_t index = 0; index < calls.size(); ++index) {
        if (!pins[index].defined()) {
            Reject("compiled variant has an undefined artifact pin");
        }
        result.push_back({index, std::string(calls[index]->symbol),
                          pins[index].record().artifact_key});
    }
    return result;
}

}  // namespace

specialization::BindingSet BindExactInputShapes(
    const specialization::GraphTemplate& graph_template,
    const std::vector<ConcreteInputShape>& inputs) {
    graph_template.Verify();
    const auto& expected = graph_template.shape_program().inputs();
    if (inputs.size() != expected.size()) {
        Reject("input count differs from the template");
    }

    std::map<std::string, const ConcreteInputShape*> by_name;
    for (const auto& input : inputs) {
        if (input.name.empty() || !by_name.emplace(input.name, &input).second) {
            Reject("input names must be nonempty and unique");
        }
        if (std::any_of(input.logical.begin(), input.logical.end(),
                        [](int64_t extent) { return extent < 0; })) {
            Reject("input extents must be non-negative");
        }
    }

    std::map<std::string, int64_t> bound;
    for (const auto& named : expected) {
        const auto found = by_name.find(named.name);
        if (found == by_name.end()) {
            Reject("an input name is missing or unknown");
        }
        const auto& dimensions = named.contract.logical().dimensions();
        const auto& concrete = found->second->logical;
        if (dimensions.size() != concrete.size()) {
            Reject("input rank differs from the template");
        }
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
            const auto& expression = dimensions[axis];
            if (expression.kind() == specialization::DimExpr::Kind::kConst) {
                if (expression.Evaluate(specialization::BindingSet()) !=
                    concrete[axis]) {
                    Reject("static input axis differs from the template");
                }
                continue;
            }
            if (expression.kind() !=
                specialization::DimExpr::Kind::kSymbol) {
                Reject("input axes must be direct symbols or constants");
            }
            const auto symbols = expression.Symbols();
            if (symbols.size() != 1) {
                Reject("direct symbolic axis is malformed");
            }
            const auto inserted = bound.emplace(symbols[0], concrete[axis]);
            if (!inserted.second &&
                inserted.first->second != concrete[axis]) {
                Reject("shared symbolic input axes disagree");
            }
        }
    }

    std::vector<specialization::Binding> bindings;
    bindings.reserve(bound.size());
    for (const auto& [symbol, value] : bound) {
        bindings.push_back({symbol, value});
    }
    const auto evaluated = graph_template.shape_program().Evaluate(
        specialization::BindingSet(std::move(bindings)));
    const auto require_exact = [](const auto& values) {
        for (const auto& value : values) {
            if (ContractDefect(value.contract) != nullptr ||
                !IsExactContract(value.contract)) {
                Reject("binding does not produce exact template contracts");
            }
        }
    };
    require_exact(evaluated.inputs);
    require_exact(evaluated.outputs);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (evaluated.inputs[index].contract.logical !=
            by_name.at(expected[index].name)->logical) {
            Reject("binding does not reproduce the concrete input shape");
        }
    }
    return evaluated.bindings;
}

PublishedExactVariant::PublishedExactVariant(
    CompiledGraph compiled, DispatchKey dispatch, ShapeProfileKey profile,
    PlanVariantKey variant, PlanAbiFingerprint abi,
    std::vector<ConcreteInputShape> inputs)
    : compiled_(std::move(compiled)), dispatch_(std::move(dispatch)),
      profile_(std::move(profile)), variant_(std::move(variant)),
      abi_(std::move(abi)), inputs_(std::move(inputs)) {}

const CompiledGraph& PublishedExactVariant::compiled_graph() const noexcept {
    return compiled_;
}

const DispatchKey& PublishedExactVariant::dispatch_key() const noexcept {
    return dispatch_;
}

const ShapeProfileKey&
PublishedExactVariant::shape_profile_key() const noexcept {
    return profile_;
}

const PlanVariantKey&
PublishedExactVariant::plan_variant_key() const noexcept {
    return variant_;
}

const PlanAbiFingerprint& PublishedExactVariant::plan_abi() const noexcept {
    return abi_;
}

ExactProfileRouteTable::ExactProfileRouteTable(
    specialization::GraphTemplate graph_template,
    std::string target_capability_fingerprint)
    : graph_template_(std::move(graph_template)),
      target_capability_fingerprint_(
          std::move(target_capability_fingerprint)) {
    graph_template_.Verify();
    if (target_capability_fingerprint_.empty()) {
        Reject("route table requires a target capability fingerprint");
    }
}

void ExactProfileRouteTable::Publish(
    const specialization::ExactOracle& oracle, CompiledGraph compiled,
    PlanAbiFingerprint expected_plan_abi) {
    std::vector<ConcreteInputShape> inputs =
        ExactInputs(graph_template_, oracle);
    const ShapeProfileKey& profile = oracle.profile().key();
    const DispatchKey dispatch =
        BuildStaticExactDispatchKey(graph_template_.key(), profile);
    if (!compiled.defined()) {
        Reject("compiled variant is undefined");
    }
    if (!expected_plan_abi.defined()) {
        Reject("published variant requires an expected plan ABI");
    }
    const std::vector<OrderedArtifactIdentity> artifacts =
        OrderedArtifacts(compiled);
    std::vector<OrderedArtifactSelectionIdentity> selections;
    selections.reserve(artifacts.size());
    for (const auto& artifact : artifacts) {
        if (artifact.artifact_key.target_capability_fingerprint() !=
            target_capability_fingerprint_) {
            Reject("compiled variant target is incompatible with this route table");
        }
        selections.push_back({artifact.call_index, artifact.link_symbol,
                              artifact.artifact_key, 0});
    }

    PlanAbiFingerprint actual_plan_abi = BuildPlanAbiFingerprint(compiled);
    if (actual_plan_abi != expected_plan_abi) {
        Reject("compiled variant plan ABI differs from publication metadata");
    }
    PlanVariantKey variant = BuildPlanVariantKey(
        graph_template_.key(), profile, selections,
        runtime::internal::kStaticMemoryPlanVersion);

    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& existing : variants_) {
        if (existing.dispatch_key() == dispatch) {
            Reject("duplicate exact profile route");
        }
        if (SameInputs(existing.inputs_, inputs)) {
            Reject("exact profiles overlap on concrete inputs");
        }
    }
    variants_.push_back(PublishedExactVariant(
        std::move(compiled), dispatch, profile, std::move(variant),
        std::move(actual_plan_abi), std::move(inputs)));
}

std::optional<PublishedExactVariant> ExactProfileRouteTable::TryLookup(
    const specialization::ExactOracle& oracle) const {
    const std::vector<ConcreteInputShape> inputs =
        ExactInputs(graph_template_, oracle);
    const DispatchKey dispatch = BuildStaticExactDispatchKey(
        graph_template_.key(), oracle.profile().key());
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = std::find_if(
        variants_.begin(), variants_.end(), [&dispatch](const auto& variant) {
            return variant.dispatch_key() == dispatch;
        });
    if (found == variants_.end()) {
        return std::nullopt;
    }
    if (found->shape_profile_key() != oracle.profile().key() ||
        found->plan_variant_key().graph_semantic_key() !=
            graph_template_.key() ||
        found->plan_variant_key().shape_profile_key() !=
            oracle.profile().key() ||
        !found->plan_abi().defined() || !found->compiled_graph().defined() ||
        !SameInputs(found->inputs_, inputs)) {
        Reject("published variant identity is inconsistent with its route");
    }
    return *found;
}

PublishedExactVariant ExactProfileRouteTable::Lookup(
    const specialization::ExactOracle& oracle) const {
    std::optional<PublishedExactVariant> found = TryLookup(oracle);
    if (!found) {
        Reject("no published variant for this exact profile");
    }
    return std::move(*found);
}

std::size_t ExactProfileRouteTable::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return variants_.size();
}

}  // namespace kxc::api::experimental::shape_control::v1
