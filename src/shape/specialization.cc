#include "kxc/shape/fakes/compiler_foundation_v1.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace kxc::shape::experimental::v1 {
namespace {

[[noreturn]] void Invalid(const std::string& message) {
  throw std::invalid_argument("shape specialization: " + message);
}

void CheckName(const std::string& value, const char* label) {
  if (value.empty()) Invalid(std::string(label) + " must not be empty");
}

void AppendU64(std::string* bytes, uint64_t value) {
  for (int shift = 0; shift != 64; shift += 8) {
    bytes->push_back(static_cast<char>((value >> shift) & 0xffU));
  }
}

void AppendField(std::string* bytes, std::string_view value) {
  AppendU64(bytes, value.size());
  bytes->append(value.data(), value.size());
}

std::string Hex(const std::string& bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (const unsigned char byte : bytes) {
    result.push_back(kDigits[byte >> 4U]);
    result.push_back(kDigits[byte & 0xfU]);
  }
  return result;
}

bool IsExactContract(const ConcreteTensorShapeContract& contract) {
  return contract.logical == contract.physical && contract.logical == contract.valid;
}

std::vector<int64_t> RowMajorStrides(const std::vector<int64_t>& physical) {
  std::vector<int64_t> strides(physical.size(), 1);
  int64_t stride = 1;
  for (size_t index = physical.size(); index > 0; --index) {
    strides[index - 1] = stride;
    if (physical[index - 1] != 0 &&
        stride > std::numeric_limits<int64_t>::max() / physical[index - 1]) {
      Invalid("concrete row-major stride overflow");
    }
    stride *= physical[index - 1];
  }
  return strides;
}

void VerifyConcreteContract(const ConcreteTensorShapeContract& contract, bool require_exact) {
  if (contract.logical.size() != contract.physical.size() ||
      contract.logical.size() != contract.valid.size() ||
      contract.logical.size() != contract.strides.size() ||
      (!contract.axis_names.empty() && contract.axis_names.size() != contract.logical.size())) {
    Invalid("concrete tensor contract rank mismatch");
  }
  if (contract.layout != "contiguous.row_major" || contract.memory_scope.empty() ||
      contract.alignment <= 0 || (contract.alignment & (contract.alignment - 1)) != 0) {
    Invalid("experimental exact v1 requires contiguous.row_major layout, scope, and alignment");
  }
  for (size_t i = 0; i < contract.logical.size(); ++i) {
    if (contract.logical[i] < 0 || contract.physical[i] < 0 || contract.valid[i] < 0 ||
        contract.strides[i] < 0) {
      Invalid("concrete tensor contract contains a negative dimension or stride");
    }
    if (contract.valid[i] > contract.logical[i] || contract.logical[i] > contract.physical[i]) {
      Invalid("concrete tensor contract violates valid <= logical <= physical");
    }
  }
  if (contract.strides != RowMajorStrides(contract.physical)) {
    Invalid("contiguous.row_major requires canonical non-overlapping writable strides");
  }
  int64_t element_count = 1;
  for (const int64_t extent : contract.physical) {
    if (extent != 0 && element_count > std::numeric_limits<int64_t>::max() / extent) {
      Invalid("concrete physical element count overflow");
    }
    element_count *= extent;
  }
  if (element_count != 0 &&
      element_count > std::numeric_limits<int64_t>::max() /
                          static_cast<int64_t>(contract.abi.element_bytes())) {
    Invalid("concrete physical byte extent overflow");
  }
  if (require_exact && !IsExactContract(contract)) {
    Invalid("exact specialization requires logical == physical == valid");
  }
}

std::string ContractBytes(const ConcreteTensorShapeContract& contract) {
  VerifyConcreteContract(contract, false);
  std::string bytes("kxc.shape.contract.v1");
  const auto append_dimensions = [&bytes](const std::vector<int64_t>& values) {
    AppendU64(&bytes, values.size());
    for (const int64_t value : values) AppendU64(&bytes, static_cast<uint64_t>(value));
  };
  append_dimensions(contract.logical);
  append_dimensions(contract.physical);
  append_dimensions(contract.valid);
  append_dimensions(contract.strides);
  AppendU64(&bytes, contract.axis_names.size());
  for (const auto& axis_name : contract.axis_names) {
    AppendU64(&bytes, axis_name.has_value() ? 1 : 0);
    if (axis_name) AppendField(&bytes, *axis_name);
  }
  AppendField(&bytes, contract.layout);
  AppendU64(&bytes, static_cast<uint64_t>(contract.alignment));
  AppendField(&bytes, contract.memory_scope);
  AppendField(&bytes, contract.abi.CanonicalBytes());
  return bytes;
}

std::string SignatureBytes(const std::vector<ConcreteTensorShapeContract>& inputs,
                           const std::vector<ConcreteTensorShapeContract>& outputs) {
  std::string bytes("kxc.shape.exact-signature.v1");
  const auto append_contracts = [&bytes](const std::vector<ConcreteTensorShapeContract>& contracts) {
    AppendU64(&bytes, contracts.size());
    for (const ConcreteTensorShapeContract& contract : contracts) {
      const std::string encoded = ContractBytes(contract);
      AppendField(&bytes, encoded);
    }
  };
  append_contracts(inputs);
  append_contracts(outputs);
  return bytes;
}

std::string SignatureValue(const std::vector<ConcreteTensorShapeContract>& inputs,
                           const std::vector<ConcreteTensorShapeContract>& outputs) {
  for (const ConcreteTensorShapeContract& contract : inputs) VerifyConcreteContract(contract, true);
  for (const ConcreteTensorShapeContract& contract : outputs) VerifyConcreteContract(contract, true);
  return "exact.v1:" + Hex(SignatureBytes(inputs, outputs));
}

const NamedConcreteTensorContract& FindValue(const ExactShapeProfile& profile,
                                             const std::string& name) {
  return profile.Value(name);
}

void VerifyExactProfileForTemplate(const GraphTemplate& graph_template, const ExactOracle& oracle) {
  graph_template.Verify();
  const ExactShapeProfile& profile = oracle.profile();
  if (!(profile.key().graph_template() == graph_template.key()) ||
      !(profile.key().graph_template_content() == graph_template.content_key()) ||
      profile.key().policy_id() != "exact" ||
      profile.key().shape_abi_version() != kShapeAbiVersion) {
    Invalid("exact oracle does not belong to this graph template");
  }
  for (const NamedConcreteTensorContract& value : profile.values()) {
    VerifyConcreteContract(value.contract, true);
  }
}

}  // namespace

GraphLocalCallLocator::GraphLocalCallLocator(std::string value) : value_(std::move(value)) {
  CheckName(value_, "graph-local call locator");
}
const std::string& GraphLocalCallLocator::value() const noexcept { return value_; }
bool GraphLocalCallLocator::operator==(const GraphLocalCallLocator& other) const noexcept {
  return value_ == other.value_;
}

UnitSemanticKey::UnitSemanticKey(uint32_t version, std::string normalized_unit_fingerprint)
    : version_(version), normalized_unit_fingerprint_(std::move(normalized_unit_fingerprint)) {
  if (version_ == 0) Invalid("unit semantic key version must be nonzero");
  CheckName(normalized_unit_fingerprint_, "unit semantic fingerprint");
}
uint32_t UnitSemanticKey::version() const noexcept { return version_; }
const std::string& UnitSemanticKey::normalized_unit_fingerprint() const noexcept {
  return normalized_unit_fingerprint_;
}
std::string UnitSemanticKey::CanonicalBytes() const {
  std::string bytes("kxc.shape.unit-semantic.v1");
  AppendU64(&bytes, version_);
  AppendField(&bytes, normalized_unit_fingerprint_);
  return bytes;
}
std::string UnitSemanticKey::CanonicalString() const {
  return "UnitSemanticKey(" + Hex(CanonicalBytes()) + ")";
}
bool UnitSemanticKey::operator==(const UnitSemanticKey& other) const noexcept {
  return version_ == other.version_ && normalized_unit_fingerprint_ == other.normalized_unit_fingerprint_;
}

GraphTemplate::GraphTemplate(GraphTemplateKey key, ShapeProgram shape_program,
                             std::vector<UnitSkeleton> ordered_units)
    : key_(std::move(key)), shape_program_(std::move(shape_program)), ordered_units_(std::move(ordered_units)) {
  Verify();
}
const GraphTemplateKey& GraphTemplate::key() const noexcept { return key_; }
const ShapeProgram& GraphTemplate::shape_program() const noexcept { return shape_program_; }
const std::vector<UnitSkeleton>& GraphTemplate::ordered_units() const noexcept { return ordered_units_; }
GraphTemplateContentKey GraphTemplate::content_key() const {
  return GraphTemplateContentKey(CanonicalBytes());
}
std::string GraphTemplate::CanonicalBytes() const {
  std::string bytes("kxc.shape.graph-template-content.v1");
  AppendField(&bytes, key_.CanonicalBytes());
  AppendField(&bytes, shape_program_.CanonicalString());
  AppendU64(&bytes, ordered_units_.size());
  for (const UnitSkeleton& unit : ordered_units_) {
    AppendField(&bytes, unit.call_locator.value());
    AppendField(&bytes, unit.semantic_key.CanonicalBytes());
    AppendU64(&bytes, unit.input_value_names.size());
    for (const std::string& name : unit.input_value_names) AppendField(&bytes, name);
    AppendU64(&bytes, unit.output_value_names.size());
    for (const std::string& name : unit.output_value_names) AppendField(&bytes, name);
  }
  return bytes;
}
void GraphTemplate::Verify() const {
  shape_program_.Verify();
  const auto verify_abi = [this](const std::vector<NamedTensorContract>& values) {
    for (const NamedTensorContract& value : values) {
      if (!(value.contract.abi().target_backend_abi() == key_.target_backend_abi())) {
        Invalid("tensor target/backend ABI does not match graph template");
      }
    }
  };
  verify_abi(shape_program_.inputs());
  verify_abi(shape_program_.outputs());
  std::set<std::string> declared_values;
  std::set<std::string> available_values;
  for (const NamedTensorContract& value : shape_program_.inputs()) {
    declared_values.insert(value.name);
    available_values.insert(value.name);
  }
  for (const NamedTensorContract& value : shape_program_.outputs()) declared_values.insert(value.name);

  std::set<std::string> locators;
  std::set<std::string> produced_values;
  for (const UnitSkeleton& unit : ordered_units_) {
    if (!locators.insert(unit.call_locator.value()).second) {
      Invalid("duplicate graph-local call locator '" + unit.call_locator.value() + "'");
    }
    if (unit.input_value_names.empty() && unit.output_value_names.empty()) {
      Invalid("unit must have an explicit boundary");
    }
    if (unit.output_value_names.empty()) Invalid("unit must produce at least one value");
    std::set<std::string> unit_outputs;
    for (const std::string& value_name : unit.input_value_names) {
      CheckName(value_name, "unit input value name");
      if (declared_values.count(value_name) == 0 || available_values.count(value_name) == 0) {
        Invalid("unit input '" + value_name + "' is not available in ordered routing");
      }
    }
    for (const std::string& value_name : unit.output_value_names) {
      CheckName(value_name, "unit output value name");
      if (declared_values.count(value_name) == 0) {
        Invalid("unit output '" + value_name + "' has no shape contract");
      }
      if (!unit_outputs.insert(value_name).second || !produced_values.insert(value_name).second) {
        Invalid("value '" + value_name + "' has multiple producers");
      }
      if (available_values.count(value_name) != 0) {
        Invalid("unit output '" + value_name + "' overwrites an available value");
      }
    }
    available_values.insert(unit.output_value_names.begin(), unit.output_value_names.end());
  }
  for (const NamedTensorContract& value : shape_program_.outputs()) {
    if (available_values.count(value.name) == 0) {
      Invalid("shape-program output '" + value.name + "' has no producer");
    }
  }
}

ExactShapeProfile::ExactShapeProfile(ShapeProfileKey key,
                                     std::vector<NamedConcreteTensorContract> values)
    : key_(std::move(key)), values_(std::move(values)) {}
const ShapeProfileKey& ExactShapeProfile::key() const noexcept { return key_; }
const std::vector<NamedConcreteTensorContract>& ExactShapeProfile::values() const noexcept { return values_; }
const NamedConcreteTensorContract& ExactShapeProfile::Value(const std::string& name) const {
  const auto found = std::find_if(values_.begin(), values_.end(), [&name](const auto& value) {
    return value.name == name;
  });
  if (found == values_.end()) Invalid("exact profile has no value named '" + name + "'");
  return *found;
}

ExactOracle::ExactOracle(ExactShapeProfile profile) : profile_(std::move(profile)) {}
const ExactShapeProfile& ExactOracle::profile() const noexcept { return profile_; }

ExactOracle InstantiateExactProfile(const GraphTemplate& graph_template, const BindingSet& bindings) {
  graph_template.Verify();
  const EvaluatedShapeProgram evaluated = graph_template.shape_program().Evaluate(bindings);
  std::vector<NamedConcreteTensorContract> values;
  values.reserve(evaluated.inputs.size() + evaluated.outputs.size());
  values.insert(values.end(), evaluated.inputs.begin(), evaluated.inputs.end());
  values.insert(values.end(), evaluated.outputs.begin(), evaluated.outputs.end());
  for (const NamedConcreteTensorContract& value : values) VerifyConcreteContract(value.contract, true);
  return ExactOracle(ExactShapeProfile(
      ShapeProfileKey(graph_template.key(), graph_template.content_key(),
                      evaluated.bindings, "exact", kShapeAbiVersion),
      std::move(values)));
}

UnitSignatureDigest::UnitSignatureDigest(std::string value) : value_(std::move(value)) {}
UnitSignatureDigest UnitSignatureDigest::ForExactContracts(
    const std::vector<ConcreteTensorShapeContract>& ordered_inputs,
    const std::vector<ConcreteTensorShapeContract>& ordered_outputs) {
  return UnitSignatureDigest(SignatureValue(ordered_inputs, ordered_outputs));
}
const std::string& UnitSignatureDigest::value() const noexcept { return value_; }
bool UnitSignatureDigest::operator==(const UnitSignatureDigest& other) const noexcept {
  return value_ == other.value_;
}
bool MatchesExactSignatureDigest(
    const UnitSignatureDigest& digest,
    const std::vector<ConcreteTensorShapeContract>& ordered_inputs,
    const std::vector<ConcreteTensorShapeContract>& ordered_outputs) {
  return digest.value() == SignatureValue(ordered_inputs, ordered_outputs);
}

KernelArtifactKey::KernelArtifactKey(
    UnitSemanticKey unit_semantic_key, uint32_t shape_abi_version,
    std::string pipeline_fingerprint, std::string capability_fingerprint,
    TargetBackendAbiDescriptor target_backend_abi,
    std::vector<ConcreteTensorShapeContract> ordered_inputs,
    std::vector<ConcreteTensorShapeContract> ordered_outputs)
    : unit_semantic_key_(std::move(unit_semantic_key)),
      shape_abi_version_(shape_abi_version),
      pipeline_fingerprint_(std::move(pipeline_fingerprint)),
      capability_fingerprint_(std::move(capability_fingerprint)),
      target_backend_abi_(std::move(target_backend_abi)),
      ordered_inputs_(std::move(ordered_inputs)),
      ordered_outputs_(std::move(ordered_outputs)) {
  if (shape_abi_version_ != kShapeAbiVersion) {
    Invalid("artifact key has an unsupported shape ABI version");
  }
  CheckName(pipeline_fingerprint_, "artifact pipeline fingerprint");
  CheckName(capability_fingerprint_, "artifact capability fingerprint");
  const auto verify_contract = [this](const ConcreteTensorShapeContract& contract) {
    VerifyConcreteContract(contract, true);
    if (!(contract.abi.target_backend_abi() == target_backend_abi_)) {
      Invalid("artifact tensor ABI does not match target/backend ABI");
    }
  };
  for (const ConcreteTensorShapeContract& contract : ordered_inputs_) verify_contract(contract);
  for (const ConcreteTensorShapeContract& contract : ordered_outputs_) verify_contract(contract);
}
const UnitSemanticKey& KernelArtifactKey::unit_semantic_key() const noexcept {
  return unit_semantic_key_;
}
uint32_t KernelArtifactKey::shape_abi_version() const noexcept {
  return shape_abi_version_;
}
const std::string& KernelArtifactKey::pipeline_fingerprint() const noexcept {
  return pipeline_fingerprint_;
}
const std::string& KernelArtifactKey::capability_fingerprint() const noexcept {
  return capability_fingerprint_;
}
const TargetBackendAbiDescriptor& KernelArtifactKey::target_backend_abi() const noexcept {
  return target_backend_abi_;
}
const std::vector<ConcreteTensorShapeContract>& KernelArtifactKey::ordered_inputs() const noexcept {
  return ordered_inputs_;
}
const std::vector<ConcreteTensorShapeContract>& KernelArtifactKey::ordered_outputs() const noexcept {
  return ordered_outputs_;
}
std::string KernelArtifactKey::CanonicalBytes() const {
  std::string bytes("kxc.shape.exact-artifact.v1");
  const std::string semantic = unit_semantic_key_.CanonicalBytes();
  AppendField(&bytes, semantic);
  AppendU64(&bytes, shape_abi_version_);
  AppendField(&bytes, pipeline_fingerprint_);
  AppendField(&bytes, capability_fingerprint_);
  AppendField(&bytes, target_backend_abi_.CanonicalBytes());
  const std::string signature = SignatureBytes(ordered_inputs_, ordered_outputs_);
  AppendField(&bytes, signature);
  return bytes;
}
std::string KernelArtifactKey::CanonicalString() const {
  return "KernelArtifactKey(" + Hex(CanonicalBytes()) + ")";
}
bool KernelArtifactKey::operator==(const KernelArtifactKey& other) const noexcept {
  return unit_semantic_key_ == other.unit_semantic_key_ &&
         shape_abi_version_ == other.shape_abi_version_ &&
         pipeline_fingerprint_ == other.pipeline_fingerprint_ &&
         capability_fingerprint_ == other.capability_fingerprint_ &&
         target_backend_abi_ == other.target_backend_abi_ &&
         ordered_inputs_ == other.ordered_inputs_ &&
         ordered_outputs_ == other.ordered_outputs_;
}

std::vector<UnitSpecializationRequest> MakeExactSpecializationRequests(
    const GraphTemplate& graph_template, const ExactOracle& oracle) {
  VerifyExactProfileForTemplate(graph_template, oracle);
  std::vector<UnitSpecializationRequest> requests;
  requests.reserve(graph_template.ordered_units().size());
  for (size_t index = 0; index < graph_template.ordered_units().size(); ++index) {
    const UnitSkeleton& unit = graph_template.ordered_units()[index];
    std::vector<ConcreteTensorShapeContract> inputs;
    std::vector<ConcreteTensorShapeContract> outputs;
    inputs.reserve(unit.input_value_names.size());
    outputs.reserve(unit.output_value_names.size());
    for (const std::string& name : unit.input_value_names) inputs.push_back(FindValue(oracle.profile(), name).contract);
    for (const std::string& name : unit.output_value_names) outputs.push_back(FindValue(oracle.profile(), name).contract);
    requests.push_back(UnitSpecializationRequest{
        index,
        unit.call_locator,
        oracle.profile().key(),
        KernelArtifactKey(unit.semantic_key, kShapeAbiVersion,
                          graph_template.key().pipeline_fingerprint(),
                          graph_template.key().capability_fingerprint(),
                          graph_template.key().target_backend_abi(), inputs, outputs),
        UnitSignatureDigest::ForExactContracts(inputs, outputs),
        std::move(inputs),
        std::move(outputs),
    });
  }
  return requests;
}

PlanVariantKey::PlanVariantKey(GraphTemplateKey graph_template_key,
                               ShapeProfileKey shape_profile_key,
                               std::vector<std::string> ordered_call_identities)
    : graph_template_key_(std::move(graph_template_key)),
      shape_profile_key_(std::move(shape_profile_key)),
      ordered_call_identities_(std::move(ordered_call_identities)) {
  if (!(shape_profile_key_.graph_template() == graph_template_key_) ||
      shape_profile_key_.policy_id() != "exact" ||
      shape_profile_key_.shape_abi_version() != kShapeAbiVersion) {
    Invalid("plan variant key has an incompatible exact profile");
  }
  for (const std::string& identity : ordered_call_identities_) CheckName(identity, "plan call identity");
}
const GraphTemplateKey& PlanVariantKey::graph_template_key() const noexcept { return graph_template_key_; }
const ShapeProfileKey& PlanVariantKey::shape_profile_key() const noexcept { return shape_profile_key_; }
const std::vector<std::string>& PlanVariantKey::ordered_call_identities() const noexcept {
  return ordered_call_identities_;
}
std::string PlanVariantKey::CanonicalBytes() const {
  std::string bytes("kxc.shape.exact-plan.v1");
  const std::string graph = graph_template_key_.CanonicalBytes();
  const std::string profile = shape_profile_key_.CanonicalBytes();
  AppendField(&bytes, graph);
  AppendField(&bytes, profile);
  AppendU64(&bytes, ordered_call_identities_.size());
  for (const std::string& identity : ordered_call_identities_) AppendField(&bytes, identity);
  return bytes;
}
std::string PlanVariantKey::CanonicalString() const {
  return "PlanVariantKey(" + Hex(CanonicalBytes()) + ")";
}
bool PlanVariantKey::operator==(const PlanVariantKey& other) const noexcept {
  return graph_template_key_ == other.graph_template_key_ &&
         shape_profile_key_ == other.shape_profile_key_ &&
         ordered_call_identities_ == other.ordered_call_identities_;
}

}  // namespace kxc::shape::experimental::v1

namespace kxc::shape::experimental::v1::fakes::compiler_foundation_v1 {
namespace {

[[noreturn]] void Invalid(const std::string& message) {
  throw std::invalid_argument("shape specialization fake v1: " + message);
}

std::string EntrySymbol(const KernelArtifactKey& key) {
  return "fake_v1_entry_" + Hex(key.CanonicalBytes());
}

void VerifyRequest(const UnitSpecializationRequest& request) {
  const GraphTemplateKey& graph_key = request.shape_profile_key.graph_template();
  if (request.shape_profile_key.policy_id() != "exact" ||
      request.shape_profile_key.shape_abi_version() != kShapeAbiVersion ||
      request.artifact_key.shape_abi_version() != kShapeAbiVersion ||
      request.artifact_key.pipeline_fingerprint() !=
          graph_key.pipeline_fingerprint() ||
      request.artifact_key.capability_fingerprint() !=
          graph_key.capability_fingerprint() ||
      !(request.artifact_key.target_backend_abi() ==
        graph_key.target_backend_abi()) ||
      request.artifact_key.ordered_inputs() != request.ordered_inputs ||
      request.artifact_key.ordered_outputs() != request.ordered_outputs ||
      !MatchesExactSignatureDigest(request.signature_digest,
                                   request.ordered_inputs,
                                   request.ordered_outputs)) {
    Invalid("request has incompatible exact artifact data");
  }
  // Rebuild and compare the complete key; hashes never decide fake identity.
  const KernelArtifactKey rebuilt(
      request.artifact_key.unit_semantic_key(),
      request.artifact_key.shape_abi_version(),
      request.artifact_key.pipeline_fingerprint(),
      request.artifact_key.capability_fingerprint(),
      request.artifact_key.target_backend_abi(), request.ordered_inputs,
      request.ordered_outputs);
  if (!(rebuilt == request.artifact_key)) {
    Invalid("request artifact key is not canonical");
  }
}

}  // namespace

FakeSelectedArtifact::FakeSelectedArtifact(
    size_t ordered_call_index, ShapeProfileKey shape_profile_key,
    KernelArtifactKey artifact_key, UnitSignatureDigest signature_digest,
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
const KernelArtifactKey& FakeSelectedArtifact::artifact_key() const noexcept { return artifact_key_; }
const UnitSignatureDigest& FakeSelectedArtifact::signature_digest() const noexcept { return signature_digest_; }
const std::string& FakeSelectedArtifact::entry_symbol() const noexcept { return entry_symbol_; }
uint64_t FakeSelectedArtifact::generation() const noexcept { return 0; }

std::vector<FakeSelectedArtifact> DeterministicMockCoordinator::Resolve(
    const std::vector<UnitSpecializationRequest>& requests) {
  std::vector<FakeSelectedArtifact> resolved;
  resolved.reserve(requests.size());
  for (const UnitSpecializationRequest& request : requests) {
    VerifyRequest(request);
    const auto existing = std::find_if(
        unique_artifacts_.begin(), unique_artifacts_.end(),
        [&request](const UniqueArtifact& artifact) {
          return artifact.artifact_key == request.artifact_key;
        });
    const UniqueArtifact* canonical = nullptr;
    if (existing == unique_artifacts_.end()) {
      unique_artifacts_.push_back(UniqueArtifact{
          request.artifact_key, request.signature_digest,
          EntrySymbol(request.artifact_key)});
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
  std::vector<std::string> call_identities;
  calls.reserve(expected.size());
  call_identities.reserve(expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    const UnitSpecializationRequest& request = expected[index];
    const FakeSelectedArtifact& selected = selected_artifacts[index];
    if (selected.ordered_call_index() != index ||
        !(selected.shape_profile_key() == request.shape_profile_key) ||
        !(selected.artifact_key() == request.artifact_key) ||
        !(selected.signature_digest() == request.signature_digest) ||
        selected.generation() != 0) {
      Invalid("selected artifact does not match the ordered exact request");
    }
    const UnitSkeleton& unit = graph_template.ordered_units()[index];
    FakeFrozenPlanCall call{unit.call_locator, selected.entry_symbol(), selected.generation(), {}, {}};
    for (const std::string& name : unit.input_value_names) call.inputs.push_back(oracle.profile().Value(name));
    for (const std::string& name : unit.output_value_names) call.outputs.push_back(oracle.profile().Value(name));
    calls.push_back(std::move(call));
    call_identities.push_back(unit.call_locator.value() + "|" + selected.entry_symbol() + "|0");
  }
  return FakeFrozenPlan(PlanVariantKey(graph_template.key(), oracle.profile().key(), std::move(call_identities)),
                        oracle.profile().key(), std::move(calls), selected_artifacts);
}

}  // namespace kxc::shape::experimental::v1::fakes::compiler_foundation_v1
