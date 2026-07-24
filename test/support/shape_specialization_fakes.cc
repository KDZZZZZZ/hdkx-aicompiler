#include "shape_compiler_foundation_fakes.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace kxc::api::experimental::shape_specialization::v1::fakes::
    compiler_foundation_v1 {
namespace {

[[noreturn]] void Invalid(const std::string& message) {
  throw std::invalid_argument("shape specialization fake v1: " + message);
}

PrimitiveArtifactKey PrimitiveKey(
    const UnitSpecializationRequest& request) {
  return PrimitiveArtifactKey(
      request.unit_semantic_key, "test-only-target",
      request.signature_digest.value(),
      1, "test-only-schedule-v1", "test-only-backend-v1");
}

std::string EntrySymbol(const PrimitiveArtifactKey& key) {
  return "fake_v1_entry_" + key.digest();
}

void VerifyRequest(const UnitSpecializationRequest& request) {
  if (!request.shape_profile_key.defined() ||
      !request.unit_semantic_key.defined() ||
      !MatchesExactSignatureDigest(request.signature_digest,
                                   request.ordered_inputs,
                                   request.ordered_outputs)) {
    Invalid("request has incompatible exact specialization data");
  }
}

}  // namespace

FakeSelectedArtifact::FakeSelectedArtifact(
    size_t ordered_call_index, ShapeProfileKey shape_profile_key,
    PrimitiveArtifactKey artifact_key,
    UnitSignatureDigest signature_digest,
    std::string entry_symbol)
    : ordered_call_index_(ordered_call_index),
      shape_profile_key_(std::move(shape_profile_key)),
      artifact_key_(std::move(artifact_key)),
      signature_digest_(std::move(signature_digest)),
      entry_symbol_(std::move(entry_symbol)) {}
size_t FakeSelectedArtifact::ordered_call_index() const noexcept { return ordered_call_index_; }
const ShapeProfileKey& FakeSelectedArtifact::shape_profile_key() const noexcept {
  return shape_profile_key_;
}
const PrimitiveArtifactKey&
FakeSelectedArtifact::artifact_key() const noexcept {
  return artifact_key_;
}
const UnitSignatureDigest& FakeSelectedArtifact::signature_digest() const noexcept { return signature_digest_; }
const std::string& FakeSelectedArtifact::entry_symbol() const noexcept { return entry_symbol_; }
uint64_t FakeSelectedArtifact::generation() const noexcept { return 0; }

std::vector<FakeSelectedArtifact> DeterministicMockCoordinator::Resolve(
    const std::vector<UnitSpecializationRequest>& requests) {
  std::vector<FakeSelectedArtifact> resolved;
  resolved.reserve(requests.size());
  for (const UnitSpecializationRequest& request : requests) {
    VerifyRequest(request);
    const PrimitiveArtifactKey request_key = PrimitiveKey(request);
    const auto existing = std::find_if(
        unique_artifacts_.begin(), unique_artifacts_.end(),
        [&request_key](const UniqueArtifact& artifact) {
          return artifact.artifact_key == request_key;
        });
    const UniqueArtifact* canonical = nullptr;
    if (existing == unique_artifacts_.end()) {
      unique_artifacts_.push_back(UniqueArtifact{
          request_key, request.signature_digest,
          EntrySymbol(request_key)});
      canonical = &unique_artifacts_.back();
    } else {
      canonical = &*existing;
    }
    FakeSelectedArtifact selected(
        request.ordered_call_index, request.shape_profile_key,
        canonical->artifact_key, canonical->signature_digest,
        canonical->entry_symbol);
    resolve_records_.push_back(ResolveRecord{request, selected});
    resolved.push_back(std::move(selected));
  }
  return resolved;
}
size_t DeterministicMockCoordinator::unique_resolve_count() const noexcept { return unique_artifacts_.size(); }
const std::vector<ResolveRecord>& DeterministicMockCoordinator::resolve_records() const noexcept {
  return resolve_records_;
}

FakeFrozenPlan::FakeFrozenPlan(PlanVariantKey key, ShapeProfileKey shape_profile_key,
                               std::vector<FakeFrozenPlanCall> ordered_calls,
                               std::vector<FakeSelectedArtifact> retained_artifacts)
    : key_(std::move(key)), shape_profile_key_(std::move(shape_profile_key)),
      ordered_calls_(std::move(ordered_calls)), retained_artifacts_(std::move(retained_artifacts)) {}
const PlanVariantKey& FakeFrozenPlan::key() const noexcept { return key_; }
const ShapeProfileKey& FakeFrozenPlan::shape_profile_key() const noexcept { return shape_profile_key_; }
const std::vector<FakeFrozenPlanCall>& FakeFrozenPlan::ordered_calls() const noexcept {
  return ordered_calls_;
}
const std::vector<FakeSelectedArtifact>& FakeFrozenPlan::retained_artifacts() const noexcept {
  return retained_artifacts_;
}

FakeFrozenPlan DeterministicMockPlanAssembler::Assemble(
    const GraphTemplate& graph_template, const ExactOracle& oracle,
    const std::vector<FakeSelectedArtifact>& selected_artifacts) const {
  const std::vector<UnitSpecializationRequest> expected = MakeExactSpecializationRequests(graph_template, oracle);
  if (selected_artifacts.size() != expected.size()) Invalid("missing or extra selected artifact");

  std::vector<FakeFrozenPlanCall> calls;
  std::vector<OrderedArtifactSelectionIdentity> selections;
  calls.reserve(expected.size());
  selections.reserve(expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    const UnitSpecializationRequest& request = expected[index];
    const FakeSelectedArtifact& selected = selected_artifacts[index];
    if (selected.ordered_call_index() != index ||
        !(selected.shape_profile_key() == request.shape_profile_key) ||
        !(selected.artifact_key() == PrimitiveKey(request)) ||
        !(selected.signature_digest() == request.signature_digest) ||
        selected.generation() != 0) {
      Invalid("selected artifact does not match the ordered exact request");
    }
    const UnitSkeleton& unit = graph_template.ordered_units()[index];
    FakeFrozenPlanCall call{unit.call_locator, selected.entry_symbol(), selected.generation(), {}, {}};
    for (const std::string& name : unit.input_value_names) call.inputs.push_back(oracle.profile().Value(name));
    for (const std::string& name : unit.output_value_names) call.outputs.push_back(oracle.profile().Value(name));
    calls.push_back(std::move(call));
    selections.push_back(OrderedArtifactSelectionIdentity{
        index, selected.entry_symbol(), selected.artifact_key(), 0});
  }
  return FakeFrozenPlan(
      BuildPlanVariantKey(graph_template.key(), oracle.profile().key(),
                          selections, "test-only-memory-plan-v1"),
      oracle.profile().key(), std::move(calls), selected_artifacts);
}

}  // namespace kxc::api::experimental::shape_specialization::v1::fakes::compiler_foundation_v1
