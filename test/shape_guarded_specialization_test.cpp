#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "kxc/shape/fakes/compiler_foundation_v1.h"

namespace {

#define CHECK(condition, message)                                                \
  do {                                                                           \
    if (!(condition)) {                                                          \
      std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n";    \
      return false;                                                              \
    }                                                                            \
  } while (0)

bool Throws(const std::function<void()>& action) {
  try {
    action();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

namespace shape = kxc::shape::experimental::v1;
using shape::ApplicabilityGuard;
using shape::BackendKind;
using shape::Binding;
using shape::BindingSet;
using shape::BucketPolicy;
using shape::BucketValueBoundary;
using shape::BuildBucketProfile;
using shape::BuildPolymorphicProfile;
using shape::Constraint;
using shape::DataType;
using shape::DeviceDescriptor;
using shape::DeviceKind;
using shape::DimExpr;
using shape::ExactOracle;
using shape::GraphLocalCallLocator;
using shape::GraphTemplate;
using shape::GraphTemplateKey;
using shape::InstantiateExactProfile;
using shape::LogicalShape;
using shape::MakeGuardedSpecializationRequests;
using shape::NamedTensorContract;
using shape::PhysicalShape;
using shape::PolymorphicPolicy;
using shape::PolymorphicUnitProof;
using shape::RuntimeExtentScalar;
using shape::ShapeProgram;
using shape::SymbolicBoundaryContract;
using shape::TailContract;
using shape::TargetBackendAbiDescriptor;
using shape::TargetKind;
using shape::TensorAbiDescriptor;
using shape::TensorShapeContract;
using shape::UnitSemanticKey;
using shape::UnitSkeleton;
using shape::ValidExtent;
namespace fake = shape::fakes::compiler_foundation_v1;

TensorAbiDescriptor DefaultAbi() {
  return TensorAbiDescriptor(
      DataType::kFloat32, DeviceDescriptor(DeviceKind::kCpu, 0),
      TargetBackendAbiDescriptor(TargetKind::kX86_64, BackendKind::kLlvm, 1));
}

TensorShapeContract ExactContract(const DimExpr& dimension) {
  return TensorShapeContract(LogicalShape({dimension}), PhysicalShape({dimension}),
                             ValidExtent({dimension}), DefaultAbi());
}

GraphTemplate MakeTemplate() {
  const DimExpr s = DimExpr::Symbol("s");
  const TensorShapeContract contract = ExactContract(s);
  const ShapeProgram program({"s"}, {NamedTensorContract{"source", contract}},
                             {NamedTensorContract{"middle", contract},
                              NamedTensorContract{"result", contract}});
  const GraphTemplateKey key(shape::kShapeContractVersion, "guarded.graph.v1", "pipeline.v1",
                             "capability.v1", "per-call.v1",
                             DefaultAbi().target_backend_abi());
  const UnitSemanticKey semantic(1, "elementwise.tail-safe.v1");
  return GraphTemplate(key, program,
                       {{GraphLocalCallLocator("call.0"), semantic, {"source"}, {"middle"}},
                        {GraphLocalCallLocator("call.1"), semantic, {"middle"}, {"result"}}});
}

ApplicabilityGuard Guard(int64_t lower, int64_t upper, int64_t divisor = 1) {
  std::vector<Constraint> constraints = {Constraint::Range(DimExpr::Symbol("s"), lower, upper)};
  if (divisor != 1) constraints.push_back(Constraint::DivisibleBy(DimExpr::Symbol("s"), divisor));
  return ApplicabilityGuard({"s"}, std::move(constraints));
}

BucketPolicy Bucket(const GraphTemplate& graph, ApplicabilityGuard guard, int64_t capacity,
                    bool pad = true, bool crop = true, bool predicate = true) {
  std::vector<BucketValueBoundary> boundaries;
  for (const char* name : {"source", "middle", "result"}) {
    boundaries.push_back(BucketValueBoundary{
        name, {capacity}, {1}, "contiguous.row_major", 1, "default", DefaultAbi()});
  }
  std::vector<TailContract> tails;
  for (size_t i = 0; i < graph.ordered_units().size(); ++i) {
    tails.push_back(TailContract{i, graph.ordered_units()[i].semantic_key, pad, true, predicate, crop});
  }
  return BucketPolicy("seq-128", 1, std::move(guard), std::move(boundaries), std::move(tails), 4096);
}

PolymorphicPolicy Polymorphic(const GraphTemplate& graph, ApplicabilityGuard guard,
                              RuntimeExtentScalar scalar) {
  const DimExpr s = DimExpr::Symbol("s");
  std::vector<PolymorphicUnitProof> proofs;
  for (size_t i = 0; i < graph.ordered_units().size(); ++i) {
    proofs.push_back(PolymorphicUnitProof{i, graph.ordered_units()[i].semantic_key, "elementwise-proof-v1"});
  }
  std::vector<SymbolicBoundaryContract> boundaries;
  for (const char* name : {"source", "middle", "result"}) {
    boundaries.push_back(SymbolicBoundaryContract{
        name, {s}, "contiguous.row_major", 1, "default", DefaultAbi()});
  }
  return PolymorphicPolicy(1, std::move(guard), std::move(proofs), {std::move(scalar)},
                           std::move(boundaries), graph.key().target_backend_abi(), 8192);
}

bool TestExactFirstGuardAndBucketContract() {
  const GraphTemplate graph = MakeTemplate();
  const ExactOracle s97 = InstantiateExactProfile(graph, BindingSet({Binding{"s", 97}}));
  const ExactOracle s128 = InstantiateExactProfile(graph, BindingSet({Binding{"s", 128}}));
  const ExactOracle s129 = InstantiateExactProfile(graph, BindingSet({Binding{"s", 129}}));
  CHECK(Throws([&] { (void)ApplicabilityGuard({"s"}, {}); }), "a policy cannot omit its guard domain");
  CHECK(!Guard(1, 128).Matches(graph, BindingSet()), "underbound guard requests must fail closed");
  CHECK(!Guard(1, 128).Matches(graph, BindingSet({Binding{"other", 97}})),
        "undeclared guard bindings must fail closed");

  const BucketPolicy policy = Bucket(graph, Guard(1, 128), 128);
  const auto p97 = BuildBucketProfile(graph, s97, policy);
  const auto p128 = BuildBucketProfile(graph, s128, policy);
  CHECK(p97.Value("source").contract.logical == std::vector<int64_t>({97}) &&
            p97.Value("source").contract.valid == std::vector<int64_t>({97}) &&
            p97.Value("source").contract.physical == std::vector<int64_t>({128}) &&
            p97.Value("source").contract.strides == std::vector<int64_t>({1}),
        "bucket profile must retain exact logical/valid while declaring physical capacity");
  CHECK(p97.exact_oracle_key() == s97.profile().key(), "the exact oracle request must remain in the profile");

  const auto r97 = MakeGuardedSpecializationRequests(graph, p97);
  const auto r128 = MakeGuardedSpecializationRequests(graph, p128);
  CHECK(r97[0].artifact_key == r128[0].artifact_key,
        "S=97 and S=128 may share only the identical explicit bucket artifact");
  const GraphTemplate same_key_different_template(
      graph.key(), graph.shape_program(),
      {{GraphLocalCallLocator("other.call.0"), graph.ordered_units()[0].semantic_key,
        {"source"}, {"middle"}},
       {GraphLocalCallLocator("other.call.1"), graph.ordered_units()[1].semantic_key,
        {"middle"}, {"result"}}});
  const ExactOracle other_oracle = InstantiateExactProfile(
      same_key_different_template, BindingSet({Binding{"s", 97}}));
  const auto other_profile = BuildBucketProfile(
      same_key_different_template, other_oracle,
      Bucket(same_key_different_template, Guard(1, 128), 128));
  const auto other_requests = MakeGuardedSpecializationRequests(
      same_key_different_template, other_profile);
  auto forged_content = r97;
  forged_content[0].exact_oracle_key = other_requests[0].exact_oracle_key;
  auto forged_abi_version = r97;
  forged_abi_version[0].exact_oracle_key = shape::ShapeProfileKey(
      graph.key(), graph.content_key(), BindingSet({Binding{"s", 97}}), "exact", 2);
  auto forged_call_index = r97;
  forged_call_index[0].ordered_call_index = 1;
  auto forged_locator = r97;
  forged_locator[0].call_locator = GraphLocalCallLocator("forged.call");
  auto forged_input_order = r97;
  forged_input_order[0].ordered_inputs.push_back(r97[0].ordered_inputs[0]);
  auto forged_output_order = r97;
  forged_output_order[0].ordered_outputs.clear();
  auto forged_rank = r97;
  forged_rank[0].ordered_inputs[0].logical.push_back(1);
  auto forged_stride = r97;
  forged_stride[0].ordered_inputs[0].strides = {2};
  auto forged_byte_overflow = r97;
  forged_byte_overflow[0].ordered_outputs[0].physical = {
      std::numeric_limits<int64_t>::max() / 4 + 1};
  forged_byte_overflow[0].ordered_outputs[0].strides = {1};
  auto forged_dtype = r97;
  forged_dtype[0].ordered_inputs[0].abi = TensorAbiDescriptor(
      DataType::kFloat16, DeviceDescriptor(DeviceKind::kCpu, 0),
      graph.key().target_backend_abi());
  auto forged_device_kind = r97;
  forged_device_kind[0].ordered_inputs[0].abi = TensorAbiDescriptor(
      DataType::kFloat32, DeviceDescriptor(DeviceKind::kCuda, 0),
      graph.key().target_backend_abi());
  auto forged_device_id = r97;
  forged_device_id[0].ordered_inputs[0].abi = TensorAbiDescriptor(
      DataType::kFloat32, DeviceDescriptor(DeviceKind::kCpu, 1),
      graph.key().target_backend_abi());
  auto forged_target = r97;
  forged_target[0].ordered_inputs[0].abi = TensorAbiDescriptor(
      DataType::kFloat32, DeviceDescriptor(DeviceKind::kCpu, 0),
      TargetBackendAbiDescriptor(TargetKind::kAArch64, BackendKind::kLlvm, 1));
  auto forged_backend = r97;
  forged_backend[0].ordered_inputs[0].abi = TensorAbiDescriptor(
      DataType::kFloat32, DeviceDescriptor(DeviceKind::kCpu, 0),
      TargetBackendAbiDescriptor(TargetKind::kX86_64, BackendKind::kNative, 1));
  auto forged_backend_abi = r97;
  forged_backend_abi[0].ordered_inputs[0].abi = TensorAbiDescriptor(
      DataType::kFloat32, DeviceDescriptor(DeviceKind::kCpu, 0),
      TargetBackendAbiDescriptor(TargetKind::kX86_64, BackendKind::kLlvm, 2));
  auto forged_guard = r97;
  forged_guard[0].guard_canonical = "ApplicabilityGuard(forged)";
  const auto alternate_profile = BuildBucketProfile(
      graph, s97, Bucket(graph, Guard(1, 128), 256));
  const auto alternate_requests = MakeGuardedSpecializationRequests(
      graph, alternate_profile);
  CHECK(alternate_requests[0].artifact_key.kind() == r97[0].artifact_key.kind() &&
            alternate_requests[0].artifact_key.unit_semantic_key() ==
                r97[0].artifact_key.unit_semantic_key() &&
            alternate_requests[0].artifact_key.CanonicalBytes() !=
                r97[0].artifact_key.CanonicalBytes(),
        "artifact-payload negative must differ only in canonical policy payload");
  auto forged_artifact_payload = r97;
  forged_artifact_payload[0].artifact_key = alternate_requests[0].artifact_key;
  auto missing_request = r97;
  missing_request.pop_back();

  fake::GuardedDeterministicMockCoordinator coordinator;
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_content); }),
        "guarded resolver must reject same key/different full template binding");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_abi_version); }),
        "guarded resolver must reject an unsupported exact-oracle shape ABI");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_call_index); }),
        "guarded resolver must reject a forged ordered call index");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_locator); }),
        "guarded resolver must reject a forged call locator");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_input_order); }),
        "guarded resolver must reject forged ordered input contracts");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_output_order); }),
        "guarded resolver must reject forged ordered output contracts");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_rank); }),
        "guarded resolver must reject a forged concrete rank");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_stride); }),
        "guarded resolver must reject a forged noncanonical stride");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_byte_overflow); }),
        "guarded resolver must reject a forged physical byte overflow");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_dtype); }),
        "guarded resolver must reject a forged dtype");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_device_kind); }),
        "guarded resolver must reject a forged device kind");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_device_id); }),
        "guarded resolver must reject a forged device id");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_target); }),
        "guarded resolver must reject a forged target");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_backend); }),
        "guarded resolver must reject a forged backend");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_backend_abi); }),
        "guarded resolver must reject a forged backend ABI version");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_guard); }),
        "guarded resolver must reject a forged canonical guard");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged_artifact_payload); }),
        "guarded resolver must reject a forged guarded artifact payload");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, missing_request); }),
        "guarded resolver must reject a missing ordered request");
  CHECK(coordinator.unique_resolve_count() == 0,
        "rejected guarded requests must not mutate coordinator state");
  const auto selected97 = coordinator.Resolve(graph, p97, r97);
  const auto selected128 = coordinator.Resolve(graph, p128, r128);
  CHECK(coordinator.unique_resolve_count() == 1, "full guarded artifact keys must deterministically reuse the bucket");
  CHECK(Throws([&] { (void)BuildBucketProfile(graph, s129, policy); }),
        "S=129 must reject at the guard before fake resolution");
  CHECK(coordinator.unique_resolve_count() == 1, "no fuzzy larger-capacity route may be resolved");

  CHECK(Throws([&] { (void)BuildBucketProfile(graph, s97, Bucket(graph, Guard(1, 128), 128, false)); }),
        "a padded bucket missing declared staging pad must reject");
  CHECK(Throws([&] { (void)BuildBucketProfile(graph, s97, Bucket(graph, Guard(1, 128), 128, true, false)); }),
        "a padded bucket missing output crop must reject");
  CHECK(Throws([&] { (void)BuildBucketProfile(graph, s97, Bucket(graph, Guard(1, 128), 128, true, true, false)); }),
        "a padded bucket missing tail predicate must reject");
  CHECK(Throws([&] {
          const BucketPolicy missing_tail("missing-tail", 1, Guard(1, 128),
              {{"source", {128}, {1}, "contiguous.row_major", 1, "default", DefaultAbi()},
               {"middle", {128}, {1}, "contiguous.row_major", 1, "default", DefaultAbi()},
               {"result", {128}, {1}, "contiguous.row_major", 1, "default", DefaultAbi()}},
              {{0, graph.ordered_units()[0].semantic_key, true, true, true, true}}, 0);
          (void)BuildBucketProfile(graph, s97, missing_tail);
        }), "a bucket policy missing an affected unit tail contract must reject");
  CHECK(Throws([&] { (void)BuildBucketProfile(graph, s97, Bucket(graph, Guard(1, 128), 96)); }),
        "physical capacity smaller than logical must reject");
  CHECK(Throws([&] {
          auto bad_boundaries = policy.boundaries();
          bad_boundaries[0].strides = {2};
          const BucketPolicy bad_stride(
              "bad-stride", 1, Guard(1, 128), std::move(bad_boundaries),
              policy.tail_contracts(), policy.workspace_bytes());
          (void)BuildBucketProfile(graph, s97, bad_stride);
        }), "bucket physical strides must be explicit and layout-compatible");
  CHECK(Throws([&] {
          auto huge_boundaries = policy.boundaries();
          huge_boundaries[0].physical = {
              std::numeric_limits<int64_t>::max() / 4 + 1};
          huge_boundaries[0].strides = {1};
          const BucketPolicy huge(
              "huge", 1, Guard(1, 128), std::move(huge_boundaries),
              policy.tail_contracts(), policy.workspace_bytes());
          (void)BuildBucketProfile(graph, s97, huge);
        }), "bucket physical byte extent overflow must fail closed");
  CHECK(Throws([&] {
          auto wrong_abi_boundaries = policy.boundaries();
          wrong_abi_boundaries[0].abi = TensorAbiDescriptor(
              DataType::kFloat16, DeviceDescriptor(DeviceKind::kCpu, 0),
              graph.key().target_backend_abi());
          const BucketPolicy wrong_abi(
              "wrong-abi", 1, Guard(1, 128), std::move(wrong_abi_boundaries),
              policy.tail_contracts(), policy.workspace_bytes());
          (void)BuildBucketProfile(graph, s97, wrong_abi);
        }), "bucket boundary dtype/device/ABI must match the exact contract");

  fake::GuardedDeterministicMockPlanAssembler assembler;
  const auto plan = assembler.Assemble(graph, p97, selected97);
  const auto again = assembler.Assemble(graph, p97, selected97);
  CHECK(plan.key() == again.key() && plan.ordered_calls().size() == 2,
        "guarded fake assembly must be deterministic");
  CHECK(plan.ordered_calls()[0].generation == 0 && plan.ordered_calls()[0].tail_contract.has_value() &&
            plan.ordered_calls()[0].inputs[0].contract.logical == std::vector<int64_t>({97}) &&
            plan.ordered_calls()[0].inputs[0].contract.physical == std::vector<int64_t>({128}) &&
            plan.ordered_calls()[0].inputs[0].contract.valid == std::vector<int64_t>({97}),
        "frozen fake plan must preserve logical, physical, valid, and tail data");
  CHECK(Throws([&] { (void)assembler.Assemble(graph, p97, selected128); }),
        "a selected artifact for another exact request/profile must reject despite shared bucket key");
  std::vector<fake::GuardedFakeSelectedArtifact> reversed = {selected97[1], selected97[0]};
  CHECK(Throws([&] { (void)assembler.Assemble(graph, p97, reversed); }), "reordered guarded artifacts must reject");
  return true;
}

bool TestPolymorphicGuardedContract() {
  const GraphTemplate graph = MakeTemplate();
  const ExactOracle s64 = InstantiateExactProfile(graph, BindingSet({Binding{"s", 64}}));
  const ExactOracle s96 = InstantiateExactProfile(graph, BindingSet({Binding{"s", 96}}));
  const ExactOracle s80 = InstantiateExactProfile(graph, BindingSet({Binding{"s", 80}}));
  const ExactOracle s160 = InstantiateExactProfile(graph, BindingSet({Binding{"s", 160}}));
  const PolymorphicPolicy policy = Polymorphic(graph, Guard(1, 128, 32),
                                               RuntimeExtentScalar{0, "extent_s", "s", 1, 128, 32});
  const auto p64 = BuildPolymorphicProfile(graph, s64, policy);
  const auto p96 = BuildPolymorphicProfile(graph, s96, policy);
  const auto r64 = MakeGuardedSpecializationRequests(graph, p64);
  const auto r96 = MakeGuardedSpecializationRequests(graph, p96);
  CHECK(r64[0].artifact_key == r96[0].artifact_key,
        "in-domain polymorphic exact requests share only a full symbolic contract artifact");
  CHECK(Throws([&] { (void)BuildPolymorphicProfile(graph, s80, policy); }),
        "divisibility guard must reject before the fake resolver");
  CHECK(Throws([&] { (void)BuildPolymorphicProfile(graph, s160, policy); }),
        "range guard must reject before the fake resolver");
  CHECK(Throws([&] {
          (void)PolymorphicPolicy(1, Guard(1, 128, 32), policy.allowlist_proofs(), {}, policy.boundaries(),
                                   graph.key().target_backend_abi(), 8192);
        }), "missing runtime scalar ABI must reject");
  CHECK(Throws([&] {
          const PolymorphicPolicy wrong_backend(
              1, Guard(1, 128, 32), policy.allowlist_proofs(),
              policy.runtime_extent_abi(), policy.boundaries(),
              TargetBackendAbiDescriptor(TargetKind::kX86_64,
                                         BackendKind::kNative, 1), 8192);
          (void)BuildPolymorphicProfile(graph, s64, wrong_backend);
        }), "polymorphic target/backend ABI mismatch must reject before fake resolution");

  CHECK(Throws([&] {
          (void)BuildPolymorphicProfile(graph, s64,
              Polymorphic(graph, Guard(1, 128, 32), RuntimeExtentScalar{1, "extent_s", "s", 1, 128, 32}));
        }), "reordered runtime scalar ordinal must reject");
  CHECK(Throws([&] {
          (void)PolymorphicPolicy(1, Guard(1, 128, 32), {},
                                   {RuntimeExtentScalar{0, "extent_s", "s", 1, 128, 32}},
                                   {SymbolicBoundaryContract{"source", {DimExpr::Symbol("s")},
                                                              "contiguous.row_major", 1, "default", DefaultAbi()}},
                                   graph.key().target_backend_abi(), 1);
        }), "missing allowlist/proof must reject");

  fake::GuardedDeterministicMockCoordinator coordinator;
  const auto selected = coordinator.Resolve(graph, p64, r64);
  (void)coordinator.Resolve(graph, p96, r96);
  CHECK(coordinator.unique_resolve_count() == 1,
        "in-domain polymorphic requests must share one full guarded artifact");
  fake::GuardedDeterministicMockPlanAssembler assembler;
  const auto plan = assembler.Assemble(graph, p64, selected);
  CHECK(!plan.ordered_calls()[0].tail_contract.has_value() &&
            plan.ordered_calls()[0].runtime_extent_abi.size() == 1 &&
            plan.ordered_calls()[0].runtime_extent_abi[0].ordinal == 0,
        "polymorphic fake plan retains ABI but executes nothing");
  return true;
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, bool (*)()>> tests = {
      {"exact_first_guard_and_bucket_contract", TestExactFirstGuardAndBucketContract},
      {"polymorphic_guarded_contract", TestPolymorphicGuardedContract},
  };
  int failures = 0;
  for (const auto& test : tests) {
    try {
      if (test.second()) std::cout << "[PASS] " << test.first << "\n";
      else ++failures;
    } catch (const std::exception& error) {
      std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
      ++failures;
    }
  }
  return failures == 0 ? 0 : 1;
}
