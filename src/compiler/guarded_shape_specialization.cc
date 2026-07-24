#include "kxc/compiler/guarded_shape_specialization.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace kxc::api::experimental::shape_specialization::v1 {
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
  AppendField(&result, tail.unit_semantic_key.canonical_bytes());
  AppendU64(&result, tail.staging_pad ? 1 : 0);
  AppendU64(&result, tail.mask ? 1 : 0);
  AppendU64(&result, tail.predicate ? 1 : 0);
  AppendU64(&result, tail.output_crop ? 1 : 0);
  return result;
}

std::string TailArtifactString(const TailContract& tail) {
  std::string result("tail-artifact.v1");
  AppendField(&result, tail.unit_semantic_key.canonical_bytes());
  AppendU64(&result, tail.staging_pad ? 1 : 0);
  AppendU64(&result, tail.mask ? 1 : 0);
  AppendU64(&result, tail.predicate ? 1 : 0);
  AppendU64(&result, tail.output_crop ? 1 : 0);
  return result;
}

std::string BucketBoundaryString(const BucketValueBoundary& boundary) {
  std::string result("bucket-boundary.v2");
  AppendU64(&result, boundary.physical.size());
  for (int64_t extent : boundary.physical) AppendU64(&result, static_cast<uint64_t>(extent));
  return result;
}

std::string SymbolicBoundaryString(const SymbolicBoundaryContract& boundary) {
  std::string result("symbolic-boundary.v2");
  AppendU64(&result, boundary.dimensions.size());
  for (const DimExpr& dimension : boundary.dimensions) AppendField(&result, dimension.CanonicalString());
  AppendU64(&result, boundary.axis_names.size());
  for (const auto& axis_name : boundary.axis_names) {
    AppendU64(&result, axis_name.has_value() ? 1 : 0);
    if (axis_name) AppendField(&result, *axis_name);
  }
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
  const ExactShapeProfile& profile = oracle.profile();
  if (profile.key().graph_semantic_key() != graph_template.key() ||
      profile.policy_id() != "exact" ||
      profile.shape_abi_version() != kShapeProfileAbiVersion) {
    Invalid("exact oracle is not minted for this graph template");
  }
  // Re-evaluate the request so a copied oracle cannot be paired with a different binding.
  const ExactOracle differential =
      InstantiateExactProfile(graph_template, profile.bindings());
  if (!(differential.profile().key() == profile.key()) ||
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
  VerifyGuardForTemplate(policy.guard(), graph_template,
                         oracle.profile().bindings());
  const std::vector<std::string> names = AllValueNames(graph_template);
  std::vector<std::string> provided;
  for (const auto& boundary : policy.boundaries()) provided.push_back(boundary.name);
  VerifyBoundaryNames(names, provided, "bucket boundaries");
  bool padded = false;
  for (const std::string& name : names) {
    const ConcreteTensorShapeContract& exact = ExactValue(oracle, name).contract;
    const BucketValueBoundary& boundary = BucketBoundary(policy, name);
    CheckName(boundary.name, "bucket value name");
    if (boundary.physical.size() != exact.logical.size()) {
      Invalid("bucket boundary has incompatible rank");
    }
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
  VerifyGuardForTemplate(policy.guard(), graph_template,
                         oracle.profile().bindings());
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
        !ScalarMatches(scalar, oracle.profile().bindings())) {
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
    if (actual.dimensions != expected.contract.logical().dimensions() ||
        actual.axis_names != expected.contract.logical().axis_names()) {
      Invalid("polymorphic symbolic boundary is incompatible");
    }
  }
}

std::string BucketPayload(const GraphTemplate& graph_template, const GuardedShapeProfile& profile,
                          const UnitSkeleton& unit, size_t index) {
  const BucketPolicy& policy = *profile.bucket_policy();
  std::string bytes("kxc.compiler.shape.bucket-specialization.v2");
  AppendField(&bytes, unit.semantic_key.canonical_bytes());
  AppendU64(&bytes, kShapeProfileAbiVersion);
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
  std::string bytes("kxc.compiler.shape.polymorphic-specialization.v2");
  AppendField(&bytes, unit.semantic_key.canonical_bytes());
  AppendU64(&bytes, kShapeProfileAbiVersion);
  AppendU64(&bytes, policy.policy_version());
  AppendField(&bytes, policy.guard().CanonicalString());
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
    if (std::any_of(boundary.physical.begin(), boundary.physical.end(),
                     [](int64_t extent) { return extent < 0; }) ||
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
                                     uint64_t workspace_upper_bound)
    : policy_version_(policy_version), guard_(std::move(guard)), allowlist_proofs_(std::move(allowlist_proofs)),
      runtime_extent_abi_(std::move(runtime_extent_abi)), boundaries_(std::move(boundaries)),
      workspace_upper_bound_(workspace_upper_bound) {
  if (policy_version_ == 0 || allowlist_proofs_.empty() || runtime_extent_abi_.empty() || boundaries_.empty()) {
    Invalid("polymorphic policy requires version, allowlist, runtime ABI, and boundaries");
  }
  std::sort(boundaries_.begin(), boundaries_.end(), [](const auto& left, const auto& right) { return left.name < right.name; });
  for (size_t i = 0; i < boundaries_.size(); ++i) {
    const auto& boundary = boundaries_[i];
    CheckName(boundary.name, "polymorphic boundary name");
    if ((!boundary.axis_names.empty() &&
         boundary.axis_names.size() != boundary.dimensions.size()) ||
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
uint64_t PolymorphicPolicy::workspace_upper_bound() const noexcept { return workspace_upper_bound_; }
std::string PolymorphicPolicy::CanonicalString() const {
  std::string bytes("kxc.shape.polymorphic-policy.v1");
  AppendU64(&bytes, policy_version_);
  AppendField(&bytes, guard_.CanonicalString());
  AppendU64(&bytes, allowlist_proofs_.size());
  for (const auto& proof : allowlist_proofs_) {
    AppendU64(&bytes, proof.ordered_unit_index);
    AppendField(&bytes, proof.unit_semantic_key.canonical_bytes());
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
  AppendU64(&bytes, workspace_upper_bound_);
  return "PolymorphicPolicy(" + Hex(bytes) + ")";
}

GuardedShapeProfile::GuardedShapeProfile(GuardedProfileKind kind, ShapeProfileKey key,
                                         ShapeProfileKey exact_oracle_key,
                                         BindingSet bindings,
                                         ApplicabilityGuard guard,
                                         std::vector<GuardedValueContract> values,
                                         std::optional<BucketPolicy> bucket_policy,
                                         std::optional<PolymorphicPolicy> polymorphic_policy)
    : kind_(kind), key_(std::move(key)),
      exact_oracle_key_(std::move(exact_oracle_key)),
      bindings_(std::move(bindings)), guard_(std::move(guard)),
      values_(std::move(values)), bucket_policy_(std::move(bucket_policy)),
      polymorphic_policy_(std::move(polymorphic_policy)) {}
GuardedProfileKind GuardedShapeProfile::kind() const noexcept { return kind_; }
const ShapeProfileKey& GuardedShapeProfile::key() const noexcept { return key_; }
const ShapeProfileKey& GuardedShapeProfile::exact_oracle_key() const noexcept { return exact_oracle_key_; }
const BindingSet& GuardedShapeProfile::bindings() const noexcept {
  return bindings_;
}
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
    values.push_back(GuardedValueContract{name, std::move(contract)});
  }
  return GuardedShapeProfile(
      GuardedProfileKind::kBucket,
      BuildShapeProfileKey(
          graph_template.key(), graph_template.CanonicalBytes(),
          oracle.profile().bindings().CanonicalString(),
          "bucket|" + policy.CanonicalString(), kShapeProfileAbiVersion),
      oracle.profile().key(), oracle.profile().bindings(), policy.guard(),
      std::move(values), policy, std::nullopt);
}

GuardedShapeProfile BuildPolymorphicProfile(const GraphTemplate& graph_template, const ExactOracle& oracle,
                                            const PolymorphicPolicy& policy) {
  VerifyOracleFirst(graph_template, oracle);
  VerifyPolymorphicPolicy(graph_template, oracle, policy);
  std::vector<GuardedValueContract> values;
  for (const std::string& name : AllValueNames(graph_template)) {
    values.push_back(GuardedValueContract{name, ExactValue(oracle, name).contract});
  }
  return GuardedShapeProfile(
      GuardedProfileKind::kPolymorphic,
      BuildShapeProfileKey(
          graph_template.key(), graph_template.CanonicalBytes(),
          oracle.profile().bindings().CanonicalString(),
          "polymorphic|" + policy.CanonicalString(), kShapeProfileAbiVersion),
      oracle.profile().key(), oracle.profile().bindings(), policy.guard(),
      std::move(values), std::nullopt, policy);
}

std::vector<GuardedUnitSpecializationRequest> MakeGuardedSpecializationRequests(
    const GraphTemplate& graph_template, const GuardedShapeProfile& profile) {
  graph_template.Verify();
  if (profile.key().graph_semantic_key() != graph_template.key() ||
      profile.exact_oracle_key().graph_semantic_key() != graph_template.key() ||
      !profile.guard().Matches(graph_template, profile.bindings())) {
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
        payload, unit.semantic_key, std::move(inputs), std::move(outputs)});
  }
  return requests;
}

}  // namespace kxc::api::experimental::shape_specialization::v1
