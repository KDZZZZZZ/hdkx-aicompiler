#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "kxc/shape/guarded_specialization.h"

namespace kxc::shape::experimental::v1::fakes::compiler_foundation_v1 {

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

class GuardedFakeSelectedArtifact {
 public:
  [[nodiscard]] size_t ordered_call_index() const noexcept;
  [[nodiscard]] const ShapeProfileKey& shape_profile_key() const noexcept;
  [[nodiscard]] const ShapeProfileKey& exact_oracle_key() const noexcept;
  [[nodiscard]] GuardedProfileKind kind() const noexcept;
  [[nodiscard]] const std::string& guard_canonical() const noexcept;
  [[nodiscard]] const GuardedArtifactKey& artifact_key() const noexcept;
  [[nodiscard]] const std::string& entry_symbol() const noexcept;
  [[nodiscard]] uint64_t generation() const noexcept;

 private:
  GuardedFakeSelectedArtifact(const GuardedUnitSpecializationRequest& request,
                              std::string entry_symbol);
  GuardedUnitSpecializationRequest request_;
  std::string entry_symbol_;
  friend class GuardedDeterministicMockCoordinator;
};

class GuardedDeterministicMockCoordinator {
 public:
  [[nodiscard]] std::vector<GuardedFakeSelectedArtifact> Resolve(
      const std::vector<GuardedUnitSpecializationRequest>& requests);
  [[nodiscard]] size_t unique_resolve_count() const noexcept;

 private:
  std::vector<GuardedArtifactKey> unique_artifacts_;
};

struct GuardedFakeFrozenPlanCall {
  GraphLocalCallLocator call_locator;
  std::string entry_symbol;
  uint64_t generation;
  std::vector<GuardedValueContract> inputs;
  std::vector<GuardedValueContract> outputs;
  std::string guard_canonical;
  std::optional<TailContract> tail_contract;
  std::vector<RuntimeExtentScalar> runtime_extent_abi;
};

class GuardedFakeFrozenPlan {
 public:
  [[nodiscard]] const GuardedPlanVariantKey& key() const noexcept;
  [[nodiscard]] const GuardedShapeProfile& profile() const noexcept;
  [[nodiscard]] const std::vector<GuardedFakeFrozenPlanCall>& ordered_calls() const noexcept;
  [[nodiscard]] const std::vector<GuardedFakeSelectedArtifact>& retained_artifacts() const noexcept;

 private:
  GuardedFakeFrozenPlan(GuardedPlanVariantKey key, GuardedShapeProfile profile,
                        std::vector<GuardedFakeFrozenPlanCall> ordered_calls,
                        std::vector<GuardedFakeSelectedArtifact> retained_artifacts);
  GuardedPlanVariantKey key_;
  GuardedShapeProfile profile_;
  std::vector<GuardedFakeFrozenPlanCall> ordered_calls_;
  std::vector<GuardedFakeSelectedArtifact> retained_artifacts_;
  friend class GuardedDeterministicMockPlanAssembler;
};

// Frozen contract fake only; it executes nothing and is not a cache or runtime.
class GuardedDeterministicMockPlanAssembler {
 public:
  [[nodiscard]] GuardedFakeFrozenPlan Assemble(
      const GraphTemplate& graph_template, const GuardedShapeProfile& profile,
      const std::vector<GuardedFakeSelectedArtifact>& selected_artifacts) const;
};

}  // namespace kxc::shape::experimental::v1::fakes::compiler_foundation_v1
