#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kxc/shape/shape.h"

namespace kxc::shape::experimental::v1 {

class ExactOracle;
struct UnitSpecializationRequest;

inline constexpr uint32_t kExactSpecializationContractVersion = 1;

// Graph-local only: this identifies a call in a frozen plan or diagnostic.
class GraphLocalCallLocator {
 public:
  explicit GraphLocalCallLocator(std::string value);

  [[nodiscard]] const std::string& value() const noexcept;
  [[nodiscard]] bool operator==(const GraphLocalCallLocator& other) const noexcept;

 private:
  std::string value_;
};

// Reusable unit meaning. It intentionally has no graph value, storage, or link identity.
class UnitSemanticKey {
 public:
  UnitSemanticKey(uint32_t version, std::string normalized_unit_fingerprint);

  [[nodiscard]] uint32_t version() const noexcept;
  [[nodiscard]] const std::string& normalized_unit_fingerprint() const noexcept;
  [[nodiscard]] std::string CanonicalBytes() const;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool operator==(const UnitSemanticKey& other) const noexcept;

 private:
  uint32_t version_;
  std::string normalized_unit_fingerprint_;
};

struct UnitSkeleton {
  GraphLocalCallLocator call_locator;
  UnitSemanticKey semantic_key;
  std::vector<std::string> input_value_names;
  std::vector<std::string> output_value_names;
};

class GraphTemplate {
 public:
  GraphTemplate(GraphTemplateKey key, ShapeProgram shape_program,
                std::vector<UnitSkeleton> ordered_units);

  [[nodiscard]] const GraphTemplateKey& key() const noexcept;
  [[nodiscard]] const ShapeProgram& shape_program() const noexcept;
  [[nodiscard]] const std::vector<UnitSkeleton>& ordered_units() const noexcept;
  [[nodiscard]] GraphTemplateContentKey content_key() const;
  [[nodiscard]] std::string CanonicalBytes() const;
  void Verify() const;

 private:
  GraphTemplateKey key_;
  ShapeProgram shape_program_;
  std::vector<UnitSkeleton> ordered_units_;
};

class ExactShapeProfile {
 public:
  [[nodiscard]] const ShapeProfileKey& key() const noexcept;
  [[nodiscard]] const std::vector<NamedConcreteTensorContract>& values() const noexcept;
  [[nodiscard]] const NamedConcreteTensorContract& Value(const std::string& name) const;

 private:
  ExactShapeProfile(ShapeProfileKey key, std::vector<NamedConcreteTensorContract> values);

  ShapeProfileKey key_;
  std::vector<NamedConcreteTensorContract> values_;

  friend class ExactOracle;
  friend ExactOracle InstantiateExactProfile(const GraphTemplate&, const BindingSet&);
};

// The only shape applicability proof accepted by this exact-specialization API.
class ExactOracle {
 public:
  [[nodiscard]] const ExactShapeProfile& profile() const noexcept;

 private:
  explicit ExactOracle(ExactShapeProfile profile);

  ExactShapeProfile profile_;

  friend ExactOracle InstantiateExactProfile(const GraphTemplate&, const BindingSet&);
};

[[nodiscard]] ExactOracle InstantiateExactProfile(const GraphTemplate& graph_template,
                                                   const BindingSet& bindings);

class UnitSignatureDigest {
 public:
  [[nodiscard]] static UnitSignatureDigest ForExactContracts(
      const std::vector<ConcreteTensorShapeContract>& ordered_inputs,
      const std::vector<ConcreteTensorShapeContract>& ordered_outputs);
  [[nodiscard]] const std::string& value() const noexcept;
  [[nodiscard]] bool operator==(const UnitSignatureDigest& other) const noexcept;

 private:
  explicit UnitSignatureDigest(std::string value);

  std::string value_;

  friend class KernelArtifactKey;
  friend std::vector<UnitSpecializationRequest> MakeExactSpecializationRequests(
      const GraphTemplate&, const ExactOracle&);
};

// Canonical artifact identity. It deliberately contains no call locator or value name.
class KernelArtifactKey {
 public:
  KernelArtifactKey(UnitSemanticKey unit_semantic_key,
                    uint32_t shape_abi_version,
                    std::string pipeline_fingerprint,
                    std::string capability_fingerprint,
                    TargetBackendAbiDescriptor target_backend_abi,
                    std::vector<ConcreteTensorShapeContract> ordered_inputs,
                    std::vector<ConcreteTensorShapeContract> ordered_outputs);

  [[nodiscard]] const UnitSemanticKey& unit_semantic_key() const noexcept;
  [[nodiscard]] uint32_t shape_abi_version() const noexcept;
  [[nodiscard]] const std::string& pipeline_fingerprint() const noexcept;
  [[nodiscard]] const std::string& capability_fingerprint() const noexcept;
  [[nodiscard]] const TargetBackendAbiDescriptor& target_backend_abi() const noexcept;
  [[nodiscard]] const std::vector<ConcreteTensorShapeContract>& ordered_inputs() const noexcept;
  [[nodiscard]] const std::vector<ConcreteTensorShapeContract>& ordered_outputs() const noexcept;
  [[nodiscard]] std::string CanonicalBytes() const;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool operator==(const KernelArtifactKey& other) const noexcept;

 private:
  UnitSemanticKey unit_semantic_key_;
  uint32_t shape_abi_version_;
  std::string pipeline_fingerprint_;
  std::string capability_fingerprint_;
  TargetBackendAbiDescriptor target_backend_abi_;
  std::vector<ConcreteTensorShapeContract> ordered_inputs_;
  std::vector<ConcreteTensorShapeContract> ordered_outputs_;
};

struct UnitSpecializationRequest {
  size_t ordered_call_index;
  GraphLocalCallLocator call_locator;
  ShapeProfileKey shape_profile_key;
  KernelArtifactKey artifact_key;
  UnitSignatureDigest signature_digest;
  std::vector<ConcreteTensorShapeContract> ordered_inputs;
  std::vector<ConcreteTensorShapeContract> ordered_outputs;
};

[[nodiscard]] std::vector<UnitSpecializationRequest> MakeExactSpecializationRequests(
    const GraphTemplate& graph_template, const ExactOracle& oracle);
[[nodiscard]] bool MatchesExactSignatureDigest(
    const UnitSignatureDigest& digest,
    const std::vector<ConcreteTensorShapeContract>& ordered_inputs,
    const std::vector<ConcreteTensorShapeContract>& ordered_outputs);

// A plan identity may use graph-local call and selected-entry identity, unlike artifact identity.
class PlanVariantKey {
 public:
  PlanVariantKey(GraphTemplateKey graph_template_key, ShapeProfileKey shape_profile_key,
                 std::vector<std::string> ordered_call_identities);

  [[nodiscard]] const GraphTemplateKey& graph_template_key() const noexcept;
  [[nodiscard]] const ShapeProfileKey& shape_profile_key() const noexcept;
  [[nodiscard]] const std::vector<std::string>& ordered_call_identities() const noexcept;
  [[nodiscard]] std::string CanonicalBytes() const;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool operator==(const PlanVariantKey& other) const noexcept;

 private:
  GraphTemplateKey graph_template_key_;
  ShapeProfileKey shape_profile_key_;
  std::vector<std::string> ordered_call_identities_;
};

}  // namespace kxc::shape::experimental::v1
