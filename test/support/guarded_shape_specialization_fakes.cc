#include "shape_compiler_foundation_fakes.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace kxc::api::experimental::shape_specialization::v1::fakes::
    compiler_foundation_v1 {
namespace {

[[noreturn]] void Invalid(const std::string& message) {
  throw std::invalid_argument("guarded shape specialization fake v1: " + message);
}

PrimitiveArtifactKey PrimitiveKey(
    const GuardedUnitSpecializationRequest& request) {
  return PrimitiveArtifactKey(
      request.unit_semantic_key, "test-only-target",
      request.specialization_canonical,
      1, "test-only-guarded-schedule-v1",
      "test-only-guarded-backend-v1");
}

std::string EntrySymbol(const PrimitiveArtifactKey& key) {
  return "guarded_fake_v1_entry_" + key.digest();
}

void VerifyRequest(const GuardedUnitSpecializationRequest& request,
                   const GuardedUnitSpecializationRequest& expected) {
  if (request.ordered_call_index != expected.ordered_call_index) {
    Invalid("request ordered call index does not match the guarded profile");
  }
  if (!(request.call_locator == expected.call_locator)) {
    Invalid("request call locator does not match the graph template");
  }
  if (!(request.shape_profile_key == expected.shape_profile_key) ||
      !(request.exact_oracle_key == expected.exact_oracle_key) ||
      request.kind != expected.kind) {
    Invalid("request profile identity does not match the guarded profile");
  }
  if (request.guard_canonical != expected.guard_canonical) {
    Invalid("request guard does not match the canonical profile guard");
  }
  if (request.specialization_canonical !=
          expected.specialization_canonical ||
      request.unit_semantic_key != expected.unit_semantic_key) {
    Invalid("request specialization does not match the template and policy");
  }
  if (request.ordered_inputs != expected.ordered_inputs) {
    Invalid("request ordered input concrete contracts do not match the guarded profile");
  }
  if (request.ordered_outputs != expected.ordered_outputs) {
    Invalid("request ordered output concrete contracts do not match the guarded profile");
  }
}

}  // namespace

GuardedFakeSelectedArtifact::GuardedFakeSelectedArtifact(const GuardedUnitSpecializationRequest& request,
                                                         PrimitiveArtifactKey artifact_key,
                                                         std::string entry_symbol)
    : request_(request), artifact_key_(std::move(artifact_key)),
      entry_symbol_(std::move(entry_symbol)) {}
size_t GuardedFakeSelectedArtifact::ordered_call_index() const noexcept { return request_.ordered_call_index; }
const ShapeProfileKey& GuardedFakeSelectedArtifact::shape_profile_key() const noexcept { return request_.shape_profile_key; }
const ShapeProfileKey& GuardedFakeSelectedArtifact::exact_oracle_key() const noexcept { return request_.exact_oracle_key; }
GuardedProfileKind GuardedFakeSelectedArtifact::kind() const noexcept { return request_.kind; }
const std::string& GuardedFakeSelectedArtifact::guard_canonical() const noexcept { return request_.guard_canonical; }
const PrimitiveArtifactKey&
GuardedFakeSelectedArtifact::artifact_key() const noexcept {
  return artifact_key_;
}
const std::string& GuardedFakeSelectedArtifact::entry_symbol() const noexcept { return entry_symbol_; }
uint64_t GuardedFakeSelectedArtifact::generation() const noexcept { return 0; }

std::vector<GuardedFakeSelectedArtifact> GuardedDeterministicMockCoordinator::Resolve(
    const GraphTemplate& graph_template, const GuardedShapeProfile& profile,
    const std::vector<GuardedUnitSpecializationRequest>& requests) {
  const auto expected = MakeGuardedSpecializationRequests(graph_template, profile);
  if (requests.size() != expected.size()) {
    Invalid("request count does not match the graph template");
  }
  for (size_t index = 0; index < requests.size(); ++index) {
    VerifyRequest(requests[index], expected[index]);
  }

  std::vector<GuardedFakeSelectedArtifact> result;
  result.reserve(expected.size());
  for (const auto& request : expected) {
    const PrimitiveArtifactKey artifact_key = PrimitiveKey(request);
    if (std::find(unique_artifacts_.begin(), unique_artifacts_.end(),
                  artifact_key) == unique_artifacts_.end()) {
      unique_artifacts_.push_back(artifact_key);
    }
    result.push_back(
        GuardedFakeSelectedArtifact(request, artifact_key,
                                    EntrySymbol(artifact_key)));
  }
  return result;
}
size_t GuardedDeterministicMockCoordinator::unique_resolve_count() const noexcept { return unique_artifacts_.size(); }

GuardedFakeFrozenPlan::GuardedFakeFrozenPlan(PlanVariantKey key, GuardedShapeProfile profile,
                                             std::vector<GuardedFakeFrozenPlanCall> ordered_calls,
                                             std::vector<GuardedFakeSelectedArtifact> retained_artifacts)
    : key_(std::move(key)), profile_(std::move(profile)), ordered_calls_(std::move(ordered_calls)),
      retained_artifacts_(std::move(retained_artifacts)) {}
const PlanVariantKey& GuardedFakeFrozenPlan::key() const noexcept {
  return key_;
}
const GuardedShapeProfile& GuardedFakeFrozenPlan::profile() const noexcept { return profile_; }
const std::vector<GuardedFakeFrozenPlanCall>& GuardedFakeFrozenPlan::ordered_calls() const noexcept { return ordered_calls_; }
const std::vector<GuardedFakeSelectedArtifact>& GuardedFakeFrozenPlan::retained_artifacts() const noexcept { return retained_artifacts_; }

GuardedFakeFrozenPlan GuardedDeterministicMockPlanAssembler::Assemble(
    const GraphTemplate& graph_template, const GuardedShapeProfile& profile,
    const std::vector<GuardedFakeSelectedArtifact>& selected_artifacts) const {
  const auto expected = MakeGuardedSpecializationRequests(graph_template, profile);
  if (expected.size() != selected_artifacts.size()) Invalid("missing or extra selected guarded artifact");
  std::vector<GuardedFakeFrozenPlanCall> calls;
  std::vector<OrderedArtifactSelectionIdentity> selections;
  for (size_t index = 0; index < expected.size(); ++index) {
    const auto& request = expected[index];
    const auto& selected = selected_artifacts[index];
    if (selected.ordered_call_index() != index || !(selected.shape_profile_key() == request.shape_profile_key) ||
        !(selected.exact_oracle_key() == request.exact_oracle_key) || selected.kind() != request.kind ||
        selected.guard_canonical() != request.guard_canonical ||
        !(selected.artifact_key() == PrimitiveKey(request)) ||
        selected.generation() != 0) {
      Invalid("selected artifact does not match ordered guarded request");
    }
    const UnitSkeleton& unit = graph_template.ordered_units()[index];
    GuardedFakeFrozenPlanCall call{unit.call_locator, selected.entry_symbol(), 0, {}, {}, request.guard_canonical,
                                   std::nullopt, {}};
    for (const std::string& name : unit.input_value_names) call.inputs.push_back(profile.Value(name));
    for (const std::string& name : unit.output_value_names) call.outputs.push_back(profile.Value(name));
    if (profile.kind() == GuardedProfileKind::kBucket) call.tail_contract = profile.bucket_policy()->tail_contracts()[index];
    else call.runtime_extent_abi = profile.polymorphic_policy()->runtime_extent_abi();
    selections.push_back(OrderedArtifactSelectionIdentity{
        index, selected.entry_symbol(), selected.artifact_key(), 0});
    calls.push_back(std::move(call));
  }
  return GuardedFakeFrozenPlan(
      BuildPlanVariantKey(graph_template.key(), profile.key(), selections,
                          "test-only-guarded-memory-plan-v1"),
      profile, std::move(calls), selected_artifacts);
}

}  // namespace kxc::api::experimental::shape_specialization::v1::fakes::compiler_foundation_v1
