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
  throw std::invalid_argument("guarded shape specialization: " + message);
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
  for (unsigned char byte : bytes) {
    result.push_back(kDigits[byte >> 4U]);
    result.push_back(kDigits[byte & 0xfU]);
  }
  return result;
}

std::vector<std::string> SortedSymbols(std::vector<std::string> symbols) {
  for (const std::string& symbol : symbols) CheckName(symbol, "declared guard symbol");
  std::sort(symbols.begin(), symbols.end());
  if (std::adjacent_find(symbols.begin(), symbols.end()) != symbols.end()) {
    Invalid("duplicate declared guard symbol");
  }
  return symbols;
}

std::string TailString(const TailContract& tail) {
  std::string result("tail.v1");
  AppendU64(&result, tail.ordered_unit_index);
  AppendField(&result, tail.unit_semantic_key.CanonicalBytes());
  AppendU64(&result, tail.staging_pad ? 1 : 0);
  AppendU64(&result, tail.mask ? 1 : 0);
  AppendU64(&result, tail.predicate ? 1 : 0);
  AppendU64(&result, tail.output_crop ? 1 : 0);
  return result;
}

std::string TailArtifactString(const TailContract& tail) {
  std::string result("tail-artifact.v1");
  AppendField(&result, tail.unit_semantic_key.CanonicalBytes());
  AppendU64(&result, tail.staging_pad ? 1 : 0);
  AppendU64(&result, tail.mask ? 1 : 0);
  AppendU64(&result, tail.predicate ? 1 : 0);
  AppendU64(&result, tail.output_crop ? 1 : 0);
  return result;
}

std::vector<int64_t> RowMajorStrides(const std::vector<int64_t>& physical) {
  std::vector<int64_t> strides(physical.size(), 1);
  int64_t stride = 1;
  for (size_t index = physical.size(); index > 0; --index) {
    strides[index - 1] = stride;
    if (physical[index - 1] != 0 &&
        stride > std::numeric_limits<int64_t>::max() / physical[index - 1]) {
      Invalid("bucket row-major stride overflow");
    }
    stride *= physical[index - 1];
  }
  return strides;
}

void VerifyPhysicalByteExtent(const std::vector<int64_t>& physical,
                              const TensorAbiDescriptor& abi) {
  int64_t element_count = 1;
  for (const int64_t extent : physical) {
    if (extent != 0 && element_count > std::numeric_limits<int64_t>::max() / extent) {
      Invalid("bucket physical element count overflow");
    }
    element_count *= extent;
  }
  if (element_count != 0 &&
      element_count > std::numeric_limits<int64_t>::max() /
                          static_cast<int64_t>(abi.element_bytes())) {
    Invalid("bucket physical byte extent overflow");
  }
}

std::string BucketBoundaryString(const BucketValueBoundary& boundary) {
  std::string result("bucket-boundary.v1");
  AppendU64(&result, boundary.physical.size());
  for (int64_t extent : boundary.physical) AppendU64(&result, static_cast<uint64_t>(extent));
  AppendU64(&result, boundary.strides.size());
  for (int64_t stride : boundary.strides) AppendU64(&result, static_cast<uint64_t>(stride));
  AppendField(&result, boundary.layout);
  AppendU64(&result, static_cast<uint64_t>(boundary.alignment));
  AppendField(&result, boundary.memory_scope);
  AppendField(&result, boundary.abi.CanonicalBytes());
  return result;
}

std::string SymbolicBoundaryString(const SymbolicBoundaryContract& boundary) {
  std::string result("symbolic-boundary.v1");
  AppendU64(&result, boundary.dimensions.size());
  for (const DimExpr& dimension : boundary.dimensions) AppendField(&result, dimension.CanonicalString());
  AppendField(&result, boundary.layout);
  AppendU64(&result, static_cast<uint64_t>(boundary.alignment));
  AppendField(&result, boundary.memory_scope);
  AppendField(&result, boundary.abi.CanonicalBytes());
  return result;
}

std::string ScalarString(const RuntimeExtentScalar& scalar) {
  std::string result("runtime-extent.v1");
  AppendU64(&result, scalar.ordinal);
  AppendField(&result, scalar.name);
  AppendField(&result, scalar.symbol);
  AppendU64(&result, static_cast<uint64_t>(scalar.lower));
  AppendU64(&result, static_cast<uint64_t>(scalar.upper));
  AppendU64(&result, static_cast<uint64_t>(scalar.divisible_by));
  return result;
}

const NamedConcreteTensorContract& ExactValue(const ExactOracle& oracle, const std::string& name) {
  return oracle.profile().Value(name);
}

const GuardedValueContract& GuardedValue(const GuardedShapeProfile& profile, const std::string& name) {
  return profile.Value(name);
}

std::vector<std::string> AllValueNames(const GraphTemplate& graph_template) {
  std::vector<std::string> names;
  for (const NamedTensorContract& value : graph_template.shape_program().inputs()) names.push_back(value.name);
  for (const NamedTensorContract& value : graph_template.shape_program().outputs()) names.push_back(value.name);
  return names;
}

const NamedTensorContract& TemplateValue(const GraphTemplate& graph_template, const std::string& name) {
  const auto find = [&name](const std::vector<NamedTensorContract>& values)
      -> const NamedTensorContract* {
    const auto it = std::find_if(values.begin(), values.end(), [&name](const NamedTensorContract& value) {
      return value.name == name;
    });
    return it == values.end() ? nullptr : &*it;
  };
  if (const auto* input = find(graph_template.shape_program().inputs())) return *input;
  if (const auto* output = find(graph_template.shape_program().outputs())) return *output;
  Invalid("unknown shape-program value '" + name + "'");
}

void VerifyOracleFirst(const GraphTemplate& graph_template, const ExactOracle& oracle) {
  graph_template.Verify();
  const ShapeProfileKey& key = oracle.profile().key();
  if (!(key.graph_template() == graph_template.key()) ||
      !(key.graph_template_content() == graph_template.content_key()) ||
      key.policy_id() != "exact" || key.shape_abi_version() != kShapeAbiVersion) {
    Invalid("exact oracle is not minted for this graph template");
  }
  // Re-evaluate the request so a copied oracle cannot be paired with a different binding.
  const ExactOracle differential = InstantiateExactProfile(graph_template, key.bindings());
  if (!(differential.profile().key() == key) ||
      differential.profile().values().size() != oracle.profile().values().size()) {
    Invalid("exact oracle request does not match the graph template");
  }
  for (size_t index = 0; index < differential.profile().values().size(); ++index) {
    const auto& expected = differential.profile().values()[index];
    const auto& actual = oracle.profile().values()[index];
    if (expected.name != actual.name || !(expected.contract == actual.contract)) {
      Invalid("exact oracle request does not match the graph template");
    }
  }
}

void VerifyGuardForTemplate(const ApplicabilityGuard& guard, const GraphTemplate& graph_template,
                            const BindingSet& bindings) {
  if (!guard.Matches(graph_template, bindings)) {
    Invalid("applicability guard does not cover the exact request");
  }
}

void VerifyBoundaryNames(const std::vector<std::string>& expected, const std::vector<std::string>& provided,
                         const char* label) {
  std::set<std::string> expected_set(expected.begin(), expected.end());
  std::set<std::string> provided_set(provided.begin(), provided.end());
  if (expected.size() != provided.size() || expected_set != provided_set ||
      provided_set.size() != provided.size()) {
    Invalid(std::string(label) + " must cover every named shape-program value exactly once");
  }
}

const BucketValueBoundary& BucketBoundary(const BucketPolicy& policy, const std::string& name) {
  const auto it = std::find_if(policy.boundaries().begin(), policy.boundaries().end(), [&name](const auto& value) {
    return value.name == name;
  });
  if (it == policy.boundaries().end()) Invalid("missing bucket capacity for '" + name + "'");
  return *it;
}

const SymbolicBoundaryContract& SymbolicBoundary(const PolymorphicPolicy& policy,
                                                  const std::string& name) {
  const auto it = std::find_if(policy.boundaries().begin(), policy.boundaries().end(), [&name](const auto& value) {
    return value.name == name;
  });
  if (it == policy.boundaries().end()) Invalid("missing symbolic boundary for '" + name + "'");
  return *it;
}

void VerifyTails(const GraphTemplate& graph_template, const BucketPolicy& policy, bool padded) {
  if (policy.tail_contracts().size() != graph_template.ordered_units().size()) {
    Invalid("bucket policy requires one tail contract for every ordered unit");
  }
  for (size_t index = 0; index < graph_template.ordered_units().size(); ++index) {
    const TailContract& tail = policy.tail_contracts()[index];
    if (tail.ordered_unit_index != index ||
        !(tail.unit_semantic_key == graph_template.ordered_units()[index].semantic_key)) {
      Invalid("bucket tail contract does not match ordered unit");
    }
    if (padded && (!tail.staging_pad || !tail.mask || !tail.predicate || !tail.output_crop)) {
      Invalid("padded bucket requires declared staging pad, mask, predicate, and output crop");
    }
  }
}

void VerifyBucketPolicy(const GraphTemplate& graph_template, const ExactOracle& oracle,
                        const BucketPolicy& policy) {
  VerifyGuardForTemplate(policy.guard(), graph_template, oracle.profile().key().bindings());
  const std::vector<std::string> names = AllValueNames(graph_template);
  std::vector<std::string> provided;
  for (const auto& boundary : policy.boundaries()) provided.push_back(boundary.name);
  VerifyBoundaryNames(names, provided, "bucket boundaries");
  bool padded = false;
  for (const std::string& name : names) {
    const ConcreteTensorShapeContract& exact = ExactValue(oracle, name).contract;
    const BucketValueBoundary& boundary = BucketBoundary(policy, name);
    CheckName(boundary.name, "bucket value name");
    CheckName(boundary.layout, "bucket layout");
    CheckName(boundary.memory_scope, "bucket memory scope");
    if (boundary.alignment <= 0 || (boundary.alignment & (boundary.alignment - 1)) != 0 ||
        boundary.physical.size() != exact.logical.size() ||
        boundary.strides.size() != exact.logical.size() ||
        boundary.layout != exact.layout || boundary.alignment != exact.alignment ||
        boundary.memory_scope != exact.memory_scope || !(boundary.abi == exact.abi)) {
      Invalid("bucket boundary has incompatible rank or layout");
    }
    if (boundary.layout == "contiguous.row_major" &&
        boundary.strides != RowMajorStrides(boundary.physical)) {
      Invalid("bucket boundary has invalid contiguous row-major strides");
    }
    VerifyPhysicalByteExtent(boundary.physical, boundary.abi);
    for (size_t axis = 0; axis < boundary.physical.size(); ++axis) {
      if (boundary.physical[axis] < exact.logical[axis]) {
        Invalid("bucket physical capacity is smaller than exact logical extent");
      }
      padded = padded || boundary.physical[axis] > exact.logical[axis];
    }
  }
  VerifyTails(graph_template, policy, padded);
}

bool ScalarMatches(const RuntimeExtentScalar& scalar, const BindingSet& bindings) {
  const auto value = bindings.Find(scalar.symbol);
  return value.has_value() && *value >= scalar.lower && *value <= scalar.upper &&
         *value % scalar.divisible_by == 0;
}

void VerifyPolymorphicPolicy(const GraphTemplate& graph_template, const ExactOracle& oracle,
                             const PolymorphicPolicy& policy) {
  VerifyGuardForTemplate(policy.guard(), graph_template, oracle.profile().key().bindings());
  if (!(policy.target_backend_abi() == graph_template.key().target_backend_abi())) {
    Invalid("polymorphic target/backend ABI does not match graph template");
  }
  if (policy.allowlist_proofs().size() != graph_template.ordered_units().size()) {
    Invalid("polymorphic policy requires an allowlist proof for every ordered unit");
  }
  for (size_t index = 0; index < graph_template.ordered_units().size(); ++index) {
    const auto& proof = policy.allowlist_proofs()[index];
    if (proof.ordered_unit_index != index ||
        !(proof.unit_semantic_key == graph_template.ordered_units()[index].semantic_key) || proof.proof.empty()) {
      Invalid("polymorphic allowlist proof does not match ordered unit");
    }
  }
  const std::vector<std::string>& symbols = graph_template.shape_program().declared_symbols();
  if (policy.runtime_extent_abi().size() != symbols.size()) {
    Invalid("runtime extent ABI must bind every declared symbolic boundary extent");
  }
  std::set<std::string> scalar_symbols;
  std::set<std::string> scalar_names;
  for (size_t index = 0; index < policy.runtime_extent_abi().size(); ++index) {
    const RuntimeExtentScalar& scalar = policy.runtime_extent_abi()[index];
    const bool guard_covers_symbol = std::any_of(policy.guard().constraints().begin(), policy.guard().constraints().end(),
        [&scalar](const Constraint& constraint) {
          const std::vector<std::string> symbols = constraint.Symbols();
          return std::find(symbols.begin(), symbols.end(), scalar.symbol) != symbols.end();
        });
    if (scalar.ordinal != index || scalar.name.empty() || scalar.symbol.empty() || scalar.lower < 0 ||
        scalar.upper < scalar.lower || scalar.divisible_by <= 0 || !scalar_names.insert(scalar.name).second ||
        !scalar_symbols.insert(scalar.symbol).second || !guard_covers_symbol ||
        !ScalarMatches(scalar, oracle.profile().key().bindings())) {
      Invalid("runtime extent ABI has omitted, reordered, unbound, or out-of-domain scalar");
    }
  }
  if (scalar_symbols != std::set<std::string>(symbols.begin(), symbols.end())) {
    Invalid("runtime extent ABI does not cover all symbolic boundary shapes");
  }
  const std::vector<std::string> names = AllValueNames(graph_template);
  std::vector<std::string> provided;
  for (const auto& boundary : policy.boundaries()) provided.push_back(boundary.name);
  VerifyBoundaryNames(names, provided, "polymorphic boundaries");
  for (const std::string& name : names) {
    const NamedTensorContract& expected = TemplateValue(graph_template, name);
    const SymbolicBoundaryContract& actual = SymbolicBoundary(policy, name);
    if (actual.dimensions != expected.contract.logical().dimensions() || actual.layout != expected.contract.physical().layout() ||
        actual.alignment != expected.contract.physical().alignment() ||
        actual.memory_scope != expected.contract.physical().memory_scope() ||
        !(actual.abi == expected.contract.abi())) {
      Invalid("polymorphic symbolic boundary has incompatible rank or layout");
    }
  }
}

std::string BucketPayload(const GraphTemplate& graph_template, const GuardedShapeProfile& profile,
                          const UnitSkeleton& unit, size_t index) {
  const BucketPolicy& policy = *profile.bucket_policy();
  std::string bytes("kxc.shape.guarded-bucket-artifact.v1");
  AppendField(&bytes, unit.semantic_key.CanonicalBytes());
  AppendU64(&bytes, kShapeAbiVersion);
  AppendField(&bytes, graph_template.key().pipeline_fingerprint());
  AppendField(&bytes, graph_template.key().capability_fingerprint());
  AppendField(&bytes, graph_template.key().target_backend_abi().CanonicalBytes());
  AppendField(&bytes, policy.bucket_id());
  AppendU64(&bytes, policy.policy_version());
  AppendField(&bytes, policy.guard().CanonicalString());
  AppendU64(&bytes, policy.workspace_bytes());
  const auto append = [&bytes, &policy](const std::vector<std::string>& names) {
    AppendU64(&bytes, names.size());
    for (const std::string& name : names) AppendField(&bytes, BucketBoundaryString(BucketBoundary(policy, name)));
  };
  append(unit.input_value_names);
  append(unit.output_value_names);
  // The ordered index routes the plan only; reusable artifact identity retains
  // the semantic tail behavior, never a graph-local call identity.
  AppendField(&bytes, TailArtifactString(policy.tail_contracts()[index]));
  return bytes;
}

std::string PolymorphicPayload(const GraphTemplate& graph_template, const GuardedShapeProfile& profile,
                               const UnitSkeleton& unit, size_t index) {
  const PolymorphicPolicy& policy = *profile.polymorphic_policy();
  std::string bytes("kxc.shape.guarded-polymorphic-artifact.v1");
  AppendField(&bytes, unit.semantic_key.CanonicalBytes());
  AppendU64(&bytes, kShapeAbiVersion);
  AppendField(&bytes, graph_template.key().pipeline_fingerprint());
  AppendField(&bytes, graph_template.key().capability_fingerprint());
  AppendField(&bytes, graph_template.key().target_backend_abi().CanonicalBytes());
  AppendU64(&bytes, policy.policy_version());
  AppendField(&bytes, policy.guard().CanonicalString());
  AppendField(&bytes, policy.target_backend_abi().CanonicalBytes());
  AppendU64(&bytes, policy.workspace_upper_bound());
  AppendField(&bytes, policy.allowlist_proofs()[index].proof);
  AppendU64(&bytes, policy.runtime_extent_abi().size());
  for (const auto& scalar : policy.runtime_extent_abi()) AppendField(&bytes, ScalarString(scalar));
  const auto append = [&bytes, &policy](const std::vector<std::string>& names) {
    AppendU64(&bytes, names.size());
    for (const std::string& name : names) AppendField(&bytes, SymbolicBoundaryString(SymbolicBoundary(policy, name)));
  };
  append(unit.input_value_names);
  append(unit.output_value_names);
  return bytes;
}

}  // namespace

ApplicabilityGuard::ApplicabilityGuard(std::vector<std::string> declared_symbols,
                                       std::vector<Constraint> constraints)
    : declared_symbols_(SortedSymbols(std::move(declared_symbols))), constraints_(std::move(constraints)) {
  if (declared_symbols_.empty() || constraints_.empty()) {
    Invalid("applicability guard must declare a nonempty constrained domain");
  }
  std::sort(constraints_.begin(), constraints_.end(), [](const Constraint& left, const Constraint& right) {
    return left.CanonicalString() < right.CanonicalString();
  });
  constraints_.erase(std::unique(constraints_.begin(), constraints_.end(), [](const Constraint& left, const Constraint& right) {
    return left.CanonicalString() == right.CanonicalString();
  }), constraints_.end());
  try {
    for (const Constraint& constraint : constraints_) {
      for (const std::string& symbol : constraint.Symbols()) {
        if (!std::binary_search(declared_symbols_.begin(), declared_symbols_.end(), symbol)) {
          Invalid("applicability guard constraint references undeclared symbol '" + symbol + "'");
        }
      }
    }
  } catch (const std::invalid_argument&) {
    throw;
  }
}
const std::vector<std::string>& ApplicabilityGuard::declared_symbols() const noexcept { return declared_symbols_; }
const std::vector<Constraint>& ApplicabilityGuard::constraints() const noexcept { return constraints_; }
std::string ApplicabilityGuard::CanonicalString() const {
  std::string result("ApplicabilityGuard(v1|");
  for (const std::string& symbol : declared_symbols_) result += std::to_string(symbol.size()) + ":" + symbol + ";";
  result += "|";
  for (const Constraint& constraint : constraints_) result += constraint.CanonicalString() + ";";
  return result + ")";
}
bool ApplicabilityGuard::Matches(const GraphTemplate& graph_template, const BindingSet& bindings) const noexcept {
  try {
    graph_template.Verify();
    if (declared_symbols_ != graph_template.shape_program().declared_symbols()) return false;
    const BindingSet solved = ExactConstraintSolver::Solve(declared_symbols_, bindings, constraints_);
    return solved == bindings;
  } catch (const std::exception&) {
    return false;
  }
}
bool ApplicabilityGuard::operator==(const ApplicabilityGuard& other) const {
  return CanonicalString() == other.CanonicalString();
}

BucketPolicy::BucketPolicy(std::string bucket_id, uint32_t policy_version, ApplicabilityGuard guard,
                           std::vector<BucketValueBoundary> boundaries,
                           std::vector<TailContract> tail_contracts, uint64_t workspace_bytes)
    : bucket_id_(std::move(bucket_id)), policy_version_(policy_version), guard_(std::move(guard)),
      boundaries_(std::move(boundaries)), tail_contracts_(std::move(tail_contracts)), workspace_bytes_(workspace_bytes) {
  CheckName(bucket_id_, "bucket id");
  if (policy_version_ == 0 || boundaries_.empty() || tail_contracts_.empty()) {
    Invalid("bucket policy requires nonzero version, capacities, and tail contracts");
  }
  std::sort(boundaries_.begin(), boundaries_.end(), [](const auto& left, const auto& right) { return left.name < right.name; });
  for (size_t index = 0; index < boundaries_.size(); ++index) {
    const auto& boundary = boundaries_[index];
    CheckName(boundary.name, "bucket value name");
    CheckName(boundary.layout, "bucket layout");
    CheckName(boundary.memory_scope, "bucket memory scope");
    if (boundary.strides.size() != boundary.physical.size() ||
        std::any_of(boundary.physical.begin(), boundary.physical.end(),
                    [](int64_t extent) { return extent < 0; }) ||
        std::any_of(boundary.strides.begin(), boundary.strides.end(),
                    [](int64_t stride) { return stride < 0; }) ||
        boundary.alignment <= 0 ||
        (boundary.alignment & (boundary.alignment - 1)) != 0 ||
        (index != 0 && boundaries_[index - 1].name == boundary.name)) {
      Invalid("invalid bucket boundary");
    }
  }
}
const std::string& BucketPolicy::bucket_id() const noexcept { return bucket_id_; }
uint32_t BucketPolicy::policy_version() const noexcept { return policy_version_; }
const ApplicabilityGuard& BucketPolicy::guard() const noexcept { return guard_; }
const std::vector<BucketValueBoundary>& BucketPolicy::boundaries() const noexcept { return boundaries_; }
const std::vector<TailContract>& BucketPolicy::tail_contracts() const noexcept { return tail_contracts_; }
uint64_t BucketPolicy::workspace_bytes() const noexcept { return workspace_bytes_; }
std::string BucketPolicy::CanonicalString() const {
  std::string bytes("kxc.shape.bucket-policy.v1");
  AppendField(&bytes, bucket_id_);
  AppendU64(&bytes, policy_version_);
  AppendField(&bytes, guard_.CanonicalString());
  AppendU64(&bytes, boundaries_.size());
  for (const auto& boundary : boundaries_) {
    AppendField(&bytes, boundary.name);
    AppendField(&bytes, BucketBoundaryString(boundary));
  }
  AppendU64(&bytes, tail_contracts_.size());
  for (const auto& tail : tail_contracts_) {
    AppendField(&bytes, TailString(tail));
  }
  AppendU64(&bytes, workspace_bytes_);
  return "BucketPolicy(" + Hex(bytes) + ")";
}

PolymorphicPolicy::PolymorphicPolicy(uint32_t policy_version, ApplicabilityGuard guard,
                                     std::vector<PolymorphicUnitProof> allowlist_proofs,
                                     std::vector<RuntimeExtentScalar> runtime_extent_abi,
                                     std::vector<SymbolicBoundaryContract> boundaries,
                                     TargetBackendAbiDescriptor target_backend_abi,
                                     uint64_t workspace_upper_bound)
    : policy_version_(policy_version), guard_(std::move(guard)), allowlist_proofs_(std::move(allowlist_proofs)),
      runtime_extent_abi_(std::move(runtime_extent_abi)), boundaries_(std::move(boundaries)),
      target_backend_abi_(std::move(target_backend_abi)), workspace_upper_bound_(workspace_upper_bound) {
  if (policy_version_ == 0 || allowlist_proofs_.empty() || runtime_extent_abi_.empty() || boundaries_.empty()) {
    Invalid("polymorphic policy requires version, allowlist, runtime ABI, and boundaries");
  }
  std::sort(boundaries_.begin(), boundaries_.end(), [](const auto& left, const auto& right) { return left.name < right.name; });
  for (size_t i = 0; i < boundaries_.size(); ++i) {
    const auto& boundary = boundaries_[i];
    CheckName(boundary.name, "polymorphic boundary name");
    CheckName(boundary.layout, "polymorphic layout");
    CheckName(boundary.memory_scope, "polymorphic memory scope");
    if (boundary.alignment <= 0 ||
        (boundary.alignment & (boundary.alignment - 1)) != 0 ||
        (i != 0 && boundaries_[i - 1].name == boundary.name)) {
      Invalid("invalid polymorphic boundary");
    }
  }
}
uint32_t PolymorphicPolicy::policy_version() const noexcept { return policy_version_; }
const ApplicabilityGuard& PolymorphicPolicy::guard() const noexcept { return guard_; }
const std::vector<PolymorphicUnitProof>& PolymorphicPolicy::allowlist_proofs() const noexcept { return allowlist_proofs_; }
const std::vector<RuntimeExtentScalar>& PolymorphicPolicy::runtime_extent_abi() const noexcept { return runtime_extent_abi_; }
const std::vector<SymbolicBoundaryContract>& PolymorphicPolicy::boundaries() const noexcept { return boundaries_; }
const TargetBackendAbiDescriptor& PolymorphicPolicy::target_backend_abi() const noexcept {
  return target_backend_abi_;
}
uint64_t PolymorphicPolicy::workspace_upper_bound() const noexcept { return workspace_upper_bound_; }
std::string PolymorphicPolicy::CanonicalString() const {
  std::string bytes("kxc.shape.polymorphic-policy.v1");
  AppendU64(&bytes, policy_version_);
  AppendField(&bytes, guard_.CanonicalString());
  AppendU64(&bytes, allowlist_proofs_.size());
  for (const auto& proof : allowlist_proofs_) {
    AppendU64(&bytes, proof.ordered_unit_index);
    AppendField(&bytes, proof.unit_semantic_key.CanonicalBytes());
    AppendField(&bytes, proof.proof);
  }
  AppendU64(&bytes, runtime_extent_abi_.size());
  for (const auto& scalar : runtime_extent_abi_) {
    AppendField(&bytes, ScalarString(scalar));
  }
  AppendU64(&bytes, boundaries_.size());
  for (const auto& boundary : boundaries_) {
    AppendField(&bytes, boundary.name);
    AppendField(&bytes, SymbolicBoundaryString(boundary));
  }
  AppendField(&bytes, target_backend_abi_.CanonicalBytes());
  AppendU64(&bytes, workspace_upper_bound_);
  return "PolymorphicPolicy(" + Hex(bytes) + ")";
}

GuardedShapeProfile::GuardedShapeProfile(GuardedProfileKind kind, ShapeProfileKey key,
                                         ShapeProfileKey exact_oracle_key, ApplicabilityGuard guard,
                                         std::vector<GuardedValueContract> values,
                                         std::optional<BucketPolicy> bucket_policy,
                                         std::optional<PolymorphicPolicy> polymorphic_policy)
    : kind_(kind), key_(std::move(key)), exact_oracle_key_(std::move(exact_oracle_key)), guard_(std::move(guard)),
      values_(std::move(values)), bucket_policy_(std::move(bucket_policy)), polymorphic_policy_(std::move(polymorphic_policy)) {}
GuardedProfileKind GuardedShapeProfile::kind() const noexcept { return kind_; }
const ShapeProfileKey& GuardedShapeProfile::key() const noexcept { return key_; }
const ShapeProfileKey& GuardedShapeProfile::exact_oracle_key() const noexcept { return exact_oracle_key_; }
const ApplicabilityGuard& GuardedShapeProfile::guard() const noexcept { return guard_; }
const std::vector<GuardedValueContract>& GuardedShapeProfile::values() const noexcept { return values_; }
const GuardedValueContract& GuardedShapeProfile::Value(const std::string& name) const {
  const auto it = std::find_if(values_.begin(), values_.end(), [&name](const auto& value) { return value.name == name; });
  if (it == values_.end()) Invalid("guarded profile has no value named '" + name + "'");
  return *it;
}
const BucketPolicy* GuardedShapeProfile::bucket_policy() const noexcept { return bucket_policy_ ? &*bucket_policy_ : nullptr; }
const PolymorphicPolicy* GuardedShapeProfile::polymorphic_policy() const noexcept { return polymorphic_policy_ ? &*polymorphic_policy_ : nullptr; }

GuardedShapeProfile BuildBucketProfile(const GraphTemplate& graph_template, const ExactOracle& oracle,
                                       const BucketPolicy& policy) {
  VerifyOracleFirst(graph_template, oracle);
  VerifyBucketPolicy(graph_template, oracle, policy);
  std::vector<GuardedValueContract> values;
  for (const std::string& name : AllValueNames(graph_template)) {
    ConcreteTensorShapeContract contract = ExactValue(oracle, name).contract;
    const BucketValueBoundary& boundary = BucketBoundary(policy, name);
    contract.physical = boundary.physical;
    contract.strides = boundary.strides;
    values.push_back(GuardedValueContract{name, std::move(contract)});
  }
  return GuardedShapeProfile(GuardedProfileKind::kBucket,
      ShapeProfileKey(graph_template.key(), graph_template.content_key(),
                      oracle.profile().key().bindings(),
                      "bucket:" + policy.bucket_id() + ":" + std::to_string(policy.policy_version()), kShapeAbiVersion),
      oracle.profile().key(), policy.guard(), std::move(values), policy, std::nullopt);
}

GuardedShapeProfile BuildPolymorphicProfile(const GraphTemplate& graph_template, const ExactOracle& oracle,
                                            const PolymorphicPolicy& policy) {
  VerifyOracleFirst(graph_template, oracle);
  VerifyPolymorphicPolicy(graph_template, oracle, policy);
  std::vector<GuardedValueContract> values;
  for (const std::string& name : AllValueNames(graph_template)) {
    values.push_back(GuardedValueContract{name, ExactValue(oracle, name).contract});
  }
  return GuardedShapeProfile(GuardedProfileKind::kPolymorphic,
      ShapeProfileKey(graph_template.key(), graph_template.content_key(),
                      oracle.profile().key().bindings(),
                      "polymorphic:" + std::to_string(policy.policy_version()), kShapeAbiVersion),
      oracle.profile().key(), policy.guard(), std::move(values), std::nullopt, policy);
}

GuardedArtifactKey::GuardedArtifactKey(GuardedProfileKind kind, UnitSemanticKey unit_semantic_key,
                                       std::string canonical_payload)
    : kind_(kind), unit_semantic_key_(std::move(unit_semantic_key)), canonical_payload_(std::move(canonical_payload)) {
  if (canonical_payload_.empty()) Invalid("guarded artifact payload must not be empty");
}
GuardedProfileKind GuardedArtifactKey::kind() const noexcept { return kind_; }
const UnitSemanticKey& GuardedArtifactKey::unit_semantic_key() const noexcept { return unit_semantic_key_; }
std::string GuardedArtifactKey::CanonicalBytes() const { return canonical_payload_; }
std::string GuardedArtifactKey::CanonicalString() const { return "GuardedArtifactKey(" + Hex(canonical_payload_) + ")"; }
bool GuardedArtifactKey::operator==(const GuardedArtifactKey& other) const noexcept {
  return kind_ == other.kind_ && unit_semantic_key_ == other.unit_semantic_key_ && canonical_payload_ == other.canonical_payload_;
}

std::vector<GuardedUnitSpecializationRequest> MakeGuardedSpecializationRequests(
    const GraphTemplate& graph_template, const GuardedShapeProfile& profile) {
  graph_template.Verify();
  if (!(profile.key().graph_template() == graph_template.key()) ||
      !(profile.key().graph_template_content() == graph_template.content_key()) ||
      !(profile.exact_oracle_key().graph_template() == graph_template.key()) ||
      !(profile.exact_oracle_key().graph_template_content() == graph_template.content_key()) ||
      profile.key().shape_abi_version() != kShapeAbiVersion || profile.exact_oracle_key().policy_id() != "exact" ||
      !(profile.key().bindings() == profile.exact_oracle_key().bindings()) ||
      !profile.guard().Matches(graph_template, profile.exact_oracle_key().bindings())) {
    Invalid("guarded profile does not retain a matching exact request and guard");
  }
  if ((profile.kind() == GuardedProfileKind::kBucket && profile.bucket_policy() == nullptr) ||
      (profile.kind() == GuardedProfileKind::kPolymorphic && profile.polymorphic_policy() == nullptr)) {
    Invalid("guarded profile kind and policy disagree");
  }
  std::vector<GuardedUnitSpecializationRequest> requests;
  for (size_t index = 0; index < graph_template.ordered_units().size(); ++index) {
    const UnitSkeleton& unit = graph_template.ordered_units()[index];
    std::vector<ConcreteTensorShapeContract> inputs;
    std::vector<ConcreteTensorShapeContract> outputs;
    for (const std::string& name : unit.input_value_names) inputs.push_back(GuardedValue(profile, name).contract);
    for (const std::string& name : unit.output_value_names) outputs.push_back(GuardedValue(profile, name).contract);
    const std::string payload = profile.kind() == GuardedProfileKind::kBucket
        ? BucketPayload(graph_template, profile, unit, index)
        : PolymorphicPayload(graph_template, profile, unit, index);
    requests.push_back(GuardedUnitSpecializationRequest{index, unit.call_locator, profile.key(),
        profile.exact_oracle_key(), profile.kind(), profile.guard().CanonicalString(),
        GuardedArtifactKey(profile.kind(), unit.semantic_key, payload), std::move(inputs), std::move(outputs)});
  }
  return requests;
}

GuardedPlanVariantKey::GuardedPlanVariantKey(GraphTemplateKey graph_template_key,
                                             ShapeProfileKey shape_profile_key,
                                             std::vector<std::string> ordered_call_identities)
    : graph_template_key_(std::move(graph_template_key)), shape_profile_key_(std::move(shape_profile_key)),
      ordered_call_identities_(std::move(ordered_call_identities)) {
  if (!(shape_profile_key_.graph_template() == graph_template_key_) || ordered_call_identities_.empty()) {
    Invalid("guarded plan key has incompatible profile or empty call sequence");
  }
  for (const std::string& identity : ordered_call_identities_) CheckName(identity, "guarded plan call identity");
}
std::string GuardedPlanVariantKey::CanonicalString() const {
  std::string bytes("kxc.shape.guarded-plan.v1");
  AppendField(&bytes, graph_template_key_.CanonicalBytes());
  AppendField(&bytes, shape_profile_key_.CanonicalBytes());
  AppendU64(&bytes, ordered_call_identities_.size());
  for (const std::string& identity : ordered_call_identities_) AppendField(&bytes, identity);
  return "GuardedPlanVariantKey(" + Hex(bytes) + ")";
}
bool GuardedPlanVariantKey::operator==(const GuardedPlanVariantKey& other) const noexcept {
  return graph_template_key_ == other.graph_template_key_ && shape_profile_key_ == other.shape_profile_key_ &&
         ordered_call_identities_ == other.ordered_call_identities_;
}

}  // namespace kxc::shape::experimental::v1

namespace kxc::shape::experimental::v1::fakes::compiler_foundation_v1 {
namespace {

[[noreturn]] void Invalid(const std::string& message) {
  throw std::invalid_argument("guarded shape specialization fake v1: " + message);
}

std::string EntrySymbol(const GuardedArtifactKey& key) {
  std::string bytes = key.CanonicalBytes();
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(bytes.size() * 2);
  for (unsigned char byte : bytes) { hex.push_back(kDigits[byte >> 4U]); hex.push_back(kDigits[byte & 0xfU]); }
  return "guarded_fake_v1_entry_" + hex;
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
  if (request.ordered_inputs != expected.ordered_inputs) {
    Invalid("request ordered input concrete contracts do not match the guarded profile");
  }
  if (request.ordered_outputs != expected.ordered_outputs) {
    Invalid("request ordered output concrete contracts do not match the guarded profile");
  }
  if (!(request.artifact_key == expected.artifact_key)) {
    Invalid("request guarded artifact key payload does not match the template and policy");
  }
}

}  // namespace

GuardedFakeSelectedArtifact::GuardedFakeSelectedArtifact(const GuardedUnitSpecializationRequest& request,
                                                         std::string entry_symbol)
    : request_(request), entry_symbol_(std::move(entry_symbol)) {}
size_t GuardedFakeSelectedArtifact::ordered_call_index() const noexcept { return request_.ordered_call_index; }
const ShapeProfileKey& GuardedFakeSelectedArtifact::shape_profile_key() const noexcept { return request_.shape_profile_key; }
const ShapeProfileKey& GuardedFakeSelectedArtifact::exact_oracle_key() const noexcept { return request_.exact_oracle_key; }
GuardedProfileKind GuardedFakeSelectedArtifact::kind() const noexcept { return request_.kind; }
const std::string& GuardedFakeSelectedArtifact::guard_canonical() const noexcept { return request_.guard_canonical; }
const GuardedArtifactKey& GuardedFakeSelectedArtifact::artifact_key() const noexcept { return request_.artifact_key; }
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
    if (std::find(unique_artifacts_.begin(), unique_artifacts_.end(), request.artifact_key) == unique_artifacts_.end()) {
      unique_artifacts_.push_back(request.artifact_key);
    }
    result.push_back(GuardedFakeSelectedArtifact(request, EntrySymbol(request.artifact_key)));
  }
  return result;
}
size_t GuardedDeterministicMockCoordinator::unique_resolve_count() const noexcept { return unique_artifacts_.size(); }

GuardedFakeFrozenPlan::GuardedFakeFrozenPlan(GuardedPlanVariantKey key, GuardedShapeProfile profile,
                                             std::vector<GuardedFakeFrozenPlanCall> ordered_calls,
                                             std::vector<GuardedFakeSelectedArtifact> retained_artifacts)
    : key_(std::move(key)), profile_(std::move(profile)), ordered_calls_(std::move(ordered_calls)),
      retained_artifacts_(std::move(retained_artifacts)) {}
const GuardedPlanVariantKey& GuardedFakeFrozenPlan::key() const noexcept { return key_; }
const GuardedShapeProfile& GuardedFakeFrozenPlan::profile() const noexcept { return profile_; }
const std::vector<GuardedFakeFrozenPlanCall>& GuardedFakeFrozenPlan::ordered_calls() const noexcept { return ordered_calls_; }
const std::vector<GuardedFakeSelectedArtifact>& GuardedFakeFrozenPlan::retained_artifacts() const noexcept { return retained_artifacts_; }

GuardedFakeFrozenPlan GuardedDeterministicMockPlanAssembler::Assemble(
    const GraphTemplate& graph_template, const GuardedShapeProfile& profile,
    const std::vector<GuardedFakeSelectedArtifact>& selected_artifacts) const {
  const auto expected = MakeGuardedSpecializationRequests(graph_template, profile);
  if (expected.size() != selected_artifacts.size()) Invalid("missing or extra selected guarded artifact");
  std::vector<GuardedFakeFrozenPlanCall> calls;
  std::vector<std::string> identities;
  for (size_t index = 0; index < expected.size(); ++index) {
    const auto& request = expected[index];
    const auto& selected = selected_artifacts[index];
    if (selected.ordered_call_index() != index || !(selected.shape_profile_key() == request.shape_profile_key) ||
        !(selected.exact_oracle_key() == request.exact_oracle_key) || selected.kind() != request.kind ||
        selected.guard_canonical() != request.guard_canonical || !(selected.artifact_key() == request.artifact_key) ||
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
    identities.push_back(unit.call_locator.value() + "|" + selected.entry_symbol() + "|0");
    calls.push_back(std::move(call));
  }
  return GuardedFakeFrozenPlan(GuardedPlanVariantKey(graph_template.key(), profile.key(), std::move(identities)),
                               profile, std::move(calls), selected_artifacts);
}

}  // namespace kxc::shape::experimental::v1::fakes::compiler_foundation_v1
