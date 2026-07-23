#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "kxc/shape/specialization.h"

namespace kxc::shape {

// Versioned contract DTOs only.  They do not allocate, compile, launch, or
// integrate with RuntimeSession.
inline constexpr uint32_t kGuardedSpecializationContractVersion = 1;

enum class GuardedProfileKind { kBucket, kPolymorphic };

class ApplicabilityGuard {
 public:
  ApplicabilityGuard(std::vector<std::string> declared_symbols,
                     std::vector<Constraint> constraints);

  [[nodiscard]] const std::vector<std::string>& declared_symbols() const noexcept;
  [[nodiscard]] const std::vector<Constraint>& constraints() const noexcept;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool Matches(const GraphTemplate& graph_template,
                             const BindingSet& bindings) const noexcept;
  [[nodiscard]] bool operator==(const ApplicabilityGuard& other) const;

 private:
  std::vector<std::string> declared_symbols_;
  std::vector<Constraint> constraints_;
};

struct BucketValueBoundary {
  std::string name;  // Routing only; excluded from artifact identity.
  std::vector<int64_t> physical;
  std::vector<int64_t> strides;
  std::string layout;
  int64_t alignment;
  std::string memory_scope;
};

struct TailContract {
  size_t ordered_unit_index;
  UnitSemanticKey unit_semantic_key;
  bool staging_pad;
  bool mask;
  bool predicate;
  bool output_crop;
};

class BucketPolicy {
 public:
  BucketPolicy(std::string bucket_id, uint32_t policy_version,
               ApplicabilityGuard guard,
               std::vector<BucketValueBoundary> boundaries,
               std::vector<TailContract> tail_contracts,
               uint64_t workspace_bytes);

  [[nodiscard]] const std::string& bucket_id() const noexcept;
  [[nodiscard]] uint32_t policy_version() const noexcept;
  [[nodiscard]] const ApplicabilityGuard& guard() const noexcept;
  [[nodiscard]] const std::vector<BucketValueBoundary>& boundaries() const noexcept;
  [[nodiscard]] const std::vector<TailContract>& tail_contracts() const noexcept;
  [[nodiscard]] uint64_t workspace_bytes() const noexcept;
  [[nodiscard]] std::string CanonicalString() const;

 private:
  std::string bucket_id_;
  uint32_t policy_version_;
  ApplicabilityGuard guard_;
  std::vector<BucketValueBoundary> boundaries_;
  std::vector<TailContract> tail_contracts_;
  uint64_t workspace_bytes_;
};

struct RuntimeExtentScalar {
  uint32_t ordinal;
  std::string name;
  std::string symbol;
  int64_t lower;
  int64_t upper;
  int64_t divisible_by;
};

struct SymbolicBoundaryContract {
  std::string name;  // Routing only; excluded from artifact identity.
  std::vector<DimExpr> dimensions;
  std::string layout;
  int64_t alignment;
  std::string memory_scope;
};

struct PolymorphicUnitProof {
  size_t ordered_unit_index;
  UnitSemanticKey unit_semantic_key;
  std::string proof;
};

class PolymorphicPolicy {
 public:
  PolymorphicPolicy(uint32_t policy_version, ApplicabilityGuard guard,
                    std::vector<PolymorphicUnitProof> allowlist_proofs,
                    std::vector<RuntimeExtentScalar> runtime_extent_abi,
                    std::vector<SymbolicBoundaryContract> boundaries,
                    std::string target, uint64_t workspace_upper_bound);

  [[nodiscard]] uint32_t policy_version() const noexcept;
  [[nodiscard]] const ApplicabilityGuard& guard() const noexcept;
  [[nodiscard]] const std::vector<PolymorphicUnitProof>& allowlist_proofs() const noexcept;
  [[nodiscard]] const std::vector<RuntimeExtentScalar>& runtime_extent_abi() const noexcept;
  [[nodiscard]] const std::vector<SymbolicBoundaryContract>& boundaries() const noexcept;
  [[nodiscard]] const std::string& target() const noexcept;
  [[nodiscard]] uint64_t workspace_upper_bound() const noexcept;
  [[nodiscard]] std::string CanonicalString() const;

 private:
  uint32_t policy_version_;
  ApplicabilityGuard guard_;
  std::vector<PolymorphicUnitProof> allowlist_proofs_;
  std::vector<RuntimeExtentScalar> runtime_extent_abi_;
  std::vector<SymbolicBoundaryContract> boundaries_;
  std::string target_;
  uint64_t workspace_upper_bound_;
};

struct GuardedValueContract {
  std::string name;
  ConcreteTensorShapeContract contract;
};

class GuardedShapeProfile {
 public:
  [[nodiscard]] GuardedProfileKind kind() const noexcept;
  [[nodiscard]] const ShapeProfileKey& key() const noexcept;
  [[nodiscard]] const ShapeProfileKey& exact_oracle_key() const noexcept;
  [[nodiscard]] const ApplicabilityGuard& guard() const noexcept;
  [[nodiscard]] const std::vector<GuardedValueContract>& values() const noexcept;
  [[nodiscard]] const GuardedValueContract& Value(const std::string& name) const;
  [[nodiscard]] const BucketPolicy* bucket_policy() const noexcept;
  [[nodiscard]] const PolymorphicPolicy* polymorphic_policy() const noexcept;

 private:
  GuardedShapeProfile(GuardedProfileKind kind, ShapeProfileKey key,
                      ShapeProfileKey exact_oracle_key, ApplicabilityGuard guard,
                      std::vector<GuardedValueContract> values,
                      std::optional<BucketPolicy> bucket_policy,
                      std::optional<PolymorphicPolicy> polymorphic_policy);

  GuardedProfileKind kind_;
  ShapeProfileKey key_;
  ShapeProfileKey exact_oracle_key_;
  ApplicabilityGuard guard_;
  std::vector<GuardedValueContract> values_;
  std::optional<BucketPolicy> bucket_policy_;
  std::optional<PolymorphicPolicy> polymorphic_policy_;

  friend GuardedShapeProfile BuildBucketProfile(const GraphTemplate&, const ExactOracle&,
                                                 const BucketPolicy&);
  friend GuardedShapeProfile BuildPolymorphicProfile(const GraphTemplate&, const ExactOracle&,
                                                      const PolymorphicPolicy&);
};

// ExactOracle is deliberately required at the profile boundary.  The builder
// verifies its template and request binding before inspecting policy data.
[[nodiscard]] GuardedShapeProfile BuildBucketProfile(const GraphTemplate& graph_template,
                                                      const ExactOracle& oracle,
                                                      const BucketPolicy& policy);
[[nodiscard]] GuardedShapeProfile BuildPolymorphicProfile(const GraphTemplate& graph_template,
                                                           const ExactOracle& oracle,
                                                           const PolymorphicPolicy& policy);

class GuardedArtifactKey {
 public:
  [[nodiscard]] GuardedProfileKind kind() const noexcept;
  [[nodiscard]] const UnitSemanticKey& unit_semantic_key() const noexcept;
  [[nodiscard]] std::string CanonicalBytes() const;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool operator==(const GuardedArtifactKey& other) const noexcept;

 private:
  GuardedArtifactKey(GuardedProfileKind kind, UnitSemanticKey unit_semantic_key,
                     std::string canonical_payload);

  GuardedProfileKind kind_;
  UnitSemanticKey unit_semantic_key_;
  std::string canonical_payload_;

  friend std::vector<struct GuardedUnitSpecializationRequest>
  MakeGuardedSpecializationRequests(const GraphTemplate&, const GuardedShapeProfile&);
};

struct GuardedUnitSpecializationRequest {
  size_t ordered_call_index;
  GraphLocalCallLocator call_locator;
  ShapeProfileKey shape_profile_key;
  ShapeProfileKey exact_oracle_key;
  GuardedProfileKind kind;
  std::string guard_canonical;
  GuardedArtifactKey artifact_key;
  std::vector<ConcreteTensorShapeContract> ordered_inputs;
  std::vector<ConcreteTensorShapeContract> ordered_outputs;
};

[[nodiscard]] std::vector<GuardedUnitSpecializationRequest>
MakeGuardedSpecializationRequests(const GraphTemplate& graph_template,
                                  const GuardedShapeProfile& profile);

class GuardedPlanVariantKey {
 public:
  GuardedPlanVariantKey(GraphTemplateKey graph_template_key, ShapeProfileKey shape_profile_key,
                        std::vector<std::string> ordered_call_identities);
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool operator==(const GuardedPlanVariantKey& other) const noexcept;

 private:
  GraphTemplateKey graph_template_key_;
  ShapeProfileKey shape_profile_key_;
  std::vector<std::string> ordered_call_identities_;
};

}  // namespace kxc::shape

namespace kxc::shape::fakes::compiler_foundation_v1 {

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

}  // namespace kxc::shape::fakes::compiler_foundation_v1
