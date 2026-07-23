#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "kxc/shape/specialization.h"

namespace kxc::shape::fakes::compiler_foundation_v1 {

inline constexpr uint32_t kContractVersion = 1;

class FakeSelectedArtifact {
 public:
  [[nodiscard]] size_t ordered_call_index() const noexcept;
  [[nodiscard]] const ShapeProfileKey& shape_profile_key() const noexcept;
  [[nodiscard]] const KernelArtifactKey& artifact_key() const noexcept;
  [[nodiscard]] const UnitSignatureDigest& signature_digest() const noexcept;
  [[nodiscard]] const std::string& entry_symbol() const noexcept;
  [[nodiscard]] uint64_t generation() const noexcept;

 private:
  FakeSelectedArtifact(size_t ordered_call_index,
                       ShapeProfileKey shape_profile_key,
                       KernelArtifactKey artifact_key,
                       UnitSignatureDigest signature_digest,
                       std::string entry_symbol);

  size_t ordered_call_index_;
  ShapeProfileKey shape_profile_key_;
  KernelArtifactKey artifact_key_;
  UnitSignatureDigest signature_digest_;
  std::string entry_symbol_;

  friend class DeterministicMockCoordinator;
};

struct ResolveRecord {
  UnitSpecializationRequest request;
  FakeSelectedArtifact selected_artifact;
};

// Synchronous deterministic contract fake; it is not a queue, cache, or runtime.
class DeterministicMockCoordinator {
 public:
  [[nodiscard]] std::vector<FakeSelectedArtifact> Resolve(
      const std::vector<UnitSpecializationRequest>& requests);
  [[nodiscard]] size_t unique_resolve_count() const noexcept;
  [[nodiscard]] const std::vector<ResolveRecord>& resolve_records() const noexcept;

 private:
  struct UniqueArtifact {
    KernelArtifactKey artifact_key;
    UnitSignatureDigest signature_digest;
    std::string entry_symbol;
  };

  std::vector<UniqueArtifact> unique_artifacts_;
  std::vector<ResolveRecord> resolve_records_;
};

struct FakeFrozenPlanCall {
  GraphLocalCallLocator call_locator;
  std::string entry_symbol;
  uint64_t generation;
  std::vector<NamedConcreteTensorContract> inputs;
  std::vector<NamedConcreteTensorContract> outputs;
};

class FakeFrozenPlan {
 public:
  [[nodiscard]] const PlanVariantKey& key() const noexcept;
  [[nodiscard]] const ShapeProfileKey& shape_profile_key() const noexcept;
  [[nodiscard]] const std::vector<FakeFrozenPlanCall>& ordered_calls() const noexcept;
  [[nodiscard]] const std::vector<FakeSelectedArtifact>& retained_artifacts() const noexcept;

 private:
  FakeFrozenPlan(PlanVariantKey key, ShapeProfileKey shape_profile_key,
                 std::vector<FakeFrozenPlanCall> ordered_calls,
                 std::vector<FakeSelectedArtifact> retained_artifacts);

  PlanVariantKey key_;
  ShapeProfileKey shape_profile_key_;
  std::vector<FakeFrozenPlanCall> ordered_calls_;
  std::vector<FakeSelectedArtifact> retained_artifacts_;

  friend class DeterministicMockPlanAssembler;
};

class DeterministicMockPlanAssembler {
 public:
  [[nodiscard]] FakeFrozenPlan Assemble(const GraphTemplate& graph_template,
                                        const ExactOracle& oracle,
                                        const std::vector<FakeSelectedArtifact>& selected_artifacts) const;
};

}  // namespace kxc::shape::fakes::compiler_foundation_v1
