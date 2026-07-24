#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "kxc/compiler/shape_specialization.h"

// Experimental compiler policy over the authoritative Shape and Identity
// contracts. It does not define tensor/kernel ABI or identity types.
namespace kxc::api::experimental::shape_specialization::v1 {

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
  // Empty means unnamed axes; otherwise this must match dimensions rank.
  std::vector<std::optional<std::string>> axis_names{};
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
                    uint64_t workspace_upper_bound);

  [[nodiscard]] uint32_t policy_version() const noexcept;
  [[nodiscard]] const ApplicabilityGuard& guard() const noexcept;
  [[nodiscard]] const std::vector<PolymorphicUnitProof>& allowlist_proofs() const noexcept;
  [[nodiscard]] const std::vector<RuntimeExtentScalar>& runtime_extent_abi() const noexcept;
  [[nodiscard]] const std::vector<SymbolicBoundaryContract>& boundaries() const noexcept;
  [[nodiscard]] uint64_t workspace_upper_bound() const noexcept;
  [[nodiscard]] std::string CanonicalString() const;

 private:
  uint32_t policy_version_;
  ApplicabilityGuard guard_;
  std::vector<PolymorphicUnitProof> allowlist_proofs_;
  std::vector<RuntimeExtentScalar> runtime_extent_abi_;
  std::vector<SymbolicBoundaryContract> boundaries_;
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
  [[nodiscard]] const BindingSet& bindings() const noexcept;
  [[nodiscard]] const ApplicabilityGuard& guard() const noexcept;
  [[nodiscard]] const std::vector<GuardedValueContract>& values() const noexcept;
  [[nodiscard]] const GuardedValueContract& Value(const std::string& name) const;
  [[nodiscard]] const BucketPolicy* bucket_policy() const noexcept;
  [[nodiscard]] const PolymorphicPolicy* polymorphic_policy() const noexcept;

 private:
  GuardedShapeProfile(GuardedProfileKind kind, ShapeProfileKey key,
                      ShapeProfileKey exact_oracle_key, BindingSet bindings,
                      ApplicabilityGuard guard,
                      std::vector<GuardedValueContract> values,
                      std::optional<BucketPolicy> bucket_policy,
                      std::optional<PolymorphicPolicy> polymorphic_policy);

  GuardedProfileKind kind_;
  ShapeProfileKey key_;
  ShapeProfileKey exact_oracle_key_;
  BindingSet bindings_;
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

struct GuardedUnitSpecializationRequest {
  size_t ordered_call_index;
  GraphLocalCallLocator call_locator;
  ShapeProfileKey shape_profile_key;
  ShapeProfileKey exact_oracle_key;
  GuardedProfileKind kind;
  std::string guard_canonical;
  std::string specialization_canonical;
  UnitSemanticKey unit_semantic_key;
  std::vector<ConcreteTensorShapeContract> ordered_inputs;
  std::vector<ConcreteTensorShapeContract> ordered_outputs;
};

[[nodiscard]] std::vector<GuardedUnitSpecializationRequest>
MakeGuardedSpecializationRequests(const GraphTemplate& graph_template,
                                  const GuardedShapeProfile& profile);

}  // namespace kxc::api::experimental::shape_specialization::v1
