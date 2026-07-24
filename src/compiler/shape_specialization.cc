#include "kxc/compiler/shape_specialization.h"
#include "../../test/support/shape_compiler_foundation_fakes.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace kxc::api::experimental::shape_specialization::v1 {
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

void VerifyConcreteContract(const ConcreteTensorShapeContract& contract, bool require_exact) {
  if (contract.logical.size() != contract.physical.size() ||
      contract.logical.size() != contract.valid.size() ||
      (!contract.axis_names.empty() && contract.axis_names.size() != contract.logical.size())) {
    Invalid("concrete tensor contract rank mismatch");
  }
  for (size_t i = 0; i < contract.logical.size(); ++i) {
    if (contract.logical[i] < 0 || contract.physical[i] < 0 ||
        contract.valid[i] < 0) {
      Invalid("concrete tensor contract contains a negative dimension");
    }
    if (contract.valid[i] > contract.logical[i] || contract.logical[i] > contract.physical[i]) {
      Invalid("concrete tensor contract violates valid <= logical <= physical");
    }
  }
  if (require_exact && !IsExactContract(contract)) {
    Invalid("exact specialization requires logical == physical == valid");
  }
}

std::string ContractBytes(const ConcreteTensorShapeContract& contract) {
  VerifyConcreteContract(contract, false);
  std::string bytes("kxc.shape.contract.v2");
  const auto append_dimensions = [&bytes](const std::vector<int64_t>& values) {
    AppendU64(&bytes, values.size());
    for (const int64_t value : values) AppendU64(&bytes, static_cast<uint64_t>(value));
  };
  append_dimensions(contract.logical);
  append_dimensions(contract.physical);
  append_dimensions(contract.valid);
  AppendU64(&bytes, contract.axis_names.size());
  for (const auto& axis_name : contract.axis_names) {
    AppendU64(&bytes, axis_name.has_value() ? 1 : 0);
    if (axis_name) AppendField(&bytes, *axis_name);
  }
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
  const ShapeProfileKey expected_key = BuildShapeProfileKey(
      graph_template.key(), graph_template.CanonicalBytes(),
      profile.bindings().CanonicalString(), profile.policy_id(),
      profile.shape_abi_version());
  if (profile.key() != expected_key ||
      profile.policy_id() != "exact" ||
      profile.shape_abi_version() != kShapeProfileAbiVersion) {
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

GraphTemplate::GraphTemplate(GraphSemanticKey key, ShapeProgram shape_program,
                             std::vector<UnitSkeleton> ordered_units)
    : key_(std::move(key)), shape_program_(std::move(shape_program)), ordered_units_(std::move(ordered_units)) {
  Verify();
}
const GraphSemanticKey& GraphTemplate::key() const noexcept { return key_; }
const ShapeProgram& GraphTemplate::shape_program() const noexcept { return shape_program_; }
const std::vector<UnitSkeleton>& GraphTemplate::ordered_units() const noexcept { return ordered_units_; }
std::string GraphTemplate::CanonicalBytes() const {
  std::string bytes("kxc.compiler.shape-template.v2");
  AppendField(&bytes, key_.canonical_bytes());
  AppendField(&bytes, shape_program_.CanonicalString());
  AppendU64(&bytes, ordered_units_.size());
  for (const UnitSkeleton& unit : ordered_units_) {
    AppendField(&bytes, unit.call_locator.value());
    AppendField(&bytes, unit.semantic_key.canonical_bytes());
    AppendU64(&bytes, unit.input_value_names.size());
    for (const std::string& name : unit.input_value_names) AppendField(&bytes, name);
    AppendU64(&bytes, unit.output_value_names.size());
    for (const std::string& name : unit.output_value_names) AppendField(&bytes, name);
  }
  return bytes;
}
void GraphTemplate::Verify() const {
  if (!key_.defined()) Invalid("graph template requires graph semantics");
  shape_program_.Verify();
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

ExactShapeProfile::ExactShapeProfile(
    ShapeProfileKey key, BindingSet bindings, std::string policy_id,
    uint32_t shape_abi_version,
    std::vector<NamedConcreteTensorContract> values)
    : key_(std::move(key)), bindings_(std::move(bindings)),
      policy_id_(std::move(policy_id)),
      shape_abi_version_(shape_abi_version), values_(std::move(values)) {}
const ShapeProfileKey& ExactShapeProfile::key() const noexcept { return key_; }
const BindingSet& ExactShapeProfile::bindings() const noexcept {
  return bindings_;
}
const std::string& ExactShapeProfile::policy_id() const noexcept {
  return policy_id_;
}
uint32_t ExactShapeProfile::shape_abi_version() const noexcept {
  return shape_abi_version_;
}
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
      BuildShapeProfileKey(
          graph_template.key(), graph_template.CanonicalBytes(),
          evaluated.bindings.CanonicalString(), "exact",
          kShapeProfileAbiVersion),
      evaluated.bindings, "exact", kShapeProfileAbiVersion,
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
        unit.semantic_key,
        UnitSignatureDigest::ForExactContracts(inputs, outputs),
        std::move(inputs),
        std::move(outputs),
    });
  }
  return requests;
}

}  // namespace kxc::api::experimental::shape_specialization::v1

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
