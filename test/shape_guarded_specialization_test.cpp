#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/guarded_shape_specialization.h"
#include "kxc/relay/op.h"
#include "support/shape_compiler_foundation_fakes.h"

namespace {

#define CHECK(condition, message)                                                \
  do {                                                                           \
    if (!(condition)) {                                                          \
      std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n";   \
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

namespace spec =
    kxc::api::experimental::shape_specialization::v1;
namespace fake = spec::fakes::compiler_foundation_v1;

kxc::api::GraphSemanticKey GraphKey() {
  const kxc::Var input("input", kxc::TensorType({4}, "float32"));
  const kxc::Expr first =
      kxc::Call(kxc::relay::Op::Get("nn_relu"), {input});
  return kxc::api::Compiler::BuildGraphSemanticKey(kxc::Function(
      {input}, kxc::Call(kxc::relay::Op::Get("nn_relu"), {first})));
}

spec::TensorShapeContract ExactContract(const spec::DimExpr& extent) {
  return spec::TensorShapeContract(
      spec::LogicalShape({extent}), spec::PhysicalCapacity({extent}),
      spec::ValidExtent({extent}));
}

spec::GraphTemplate MakeTemplate() {
  const spec::DimExpr s = spec::DimExpr::Symbol("s");
  const spec::TensorShapeContract contract = ExactContract(s);
  const kxc::api::UnitSemanticKey semantic("elementwise.tail-safe.v1");
  return spec::GraphTemplate(
      GraphKey(),
      spec::ShapeProgram(
          {"s"}, {{"source", contract}},
          {{"middle", contract}, {"result", contract}}),
      {{spec::GraphLocalCallLocator("call.0"), semantic, {"source"},
        {"middle"}},
       {spec::GraphLocalCallLocator("call.1"), semantic, {"middle"},
        {"result"}}});
}

spec::ApplicabilityGuard Guard(int64_t lower, int64_t upper,
                               int64_t divisor = 1) {
  std::vector<spec::Constraint> constraints = {
      spec::Constraint::Range(spec::DimExpr::Symbol("s"), lower, upper)};
  if (divisor != 1) {
    constraints.push_back(spec::Constraint::DivisibleBy(
        spec::DimExpr::Symbol("s"), divisor));
  }
  return spec::ApplicabilityGuard({"s"}, std::move(constraints));
}

spec::BucketPolicy Bucket(const spec::GraphTemplate& graph,
                          spec::ApplicabilityGuard guard,
                          int64_t capacity, bool pad = true,
                          bool crop = true, bool predicate = true) {
  std::vector<spec::BucketValueBoundary> boundaries;
  for (const char* name : {"source", "middle", "result"}) {
    boundaries.push_back({name, {capacity}});
  }
  std::vector<spec::TailContract> tails;
  for (size_t index = 0; index < graph.ordered_units().size(); ++index) {
    tails.push_back(
        {index, graph.ordered_units()[index].semantic_key, pad, true,
         predicate, crop});
  }
  return spec::BucketPolicy("seq-128", 1, std::move(guard),
                            std::move(boundaries), std::move(tails), 4096);
}

spec::PolymorphicPolicy Polymorphic(
    const spec::GraphTemplate& graph, spec::ApplicabilityGuard guard,
    spec::RuntimeExtentScalar scalar) {
  std::vector<spec::PolymorphicUnitProof> proofs;
  for (size_t index = 0; index < graph.ordered_units().size(); ++index) {
    proofs.push_back({index, graph.ordered_units()[index].semantic_key,
                      "elementwise-proof-v1"});
  }
  const spec::DimExpr s = spec::DimExpr::Symbol("s");
  std::vector<spec::SymbolicBoundaryContract> boundaries;
  for (const char* name : {"source", "middle", "result"}) {
    boundaries.push_back({name, {s}, {}});
  }
  return spec::PolymorphicPolicy(
      1, std::move(guard), std::move(proofs), {std::move(scalar)},
      std::move(boundaries), 8192);
}

bool TestBucketProfileAndFormalArtifactReuse() {
  const spec::GraphTemplate graph = MakeTemplate();
  const spec::ExactOracle s97 = spec::InstantiateExactProfile(
      graph, spec::BindingSet({spec::Binding{"s", 97}}));
  const spec::ExactOracle s128 = spec::InstantiateExactProfile(
      graph, spec::BindingSet({spec::Binding{"s", 128}}));
  const spec::ExactOracle s129 = spec::InstantiateExactProfile(
      graph, spec::BindingSet({spec::Binding{"s", 129}}));
  const spec::BucketPolicy policy = Bucket(graph, Guard(1, 128), 128);

  const spec::GuardedShapeProfile p97 =
      spec::BuildBucketProfile(graph, s97, policy);
  const spec::GuardedShapeProfile p128 =
      spec::BuildBucketProfile(graph, s128, policy);
  CHECK(p97.Value("source").contract.logical ==
                std::vector<int64_t>({97}) &&
            p97.Value("source").contract.physical ==
                std::vector<int64_t>({128}) &&
            p97.Value("source").contract.valid ==
                std::vector<int64_t>({97}),
        "bucket profile must contain only logical/capacity/valid shape data");
  CHECK(p97.key().graph_semantic_key() == graph.key() &&
            p97.key() != p128.key() &&
            p97.exact_oracle_key() == s97.profile().key(),
        "guarded profile must use formal graph/profile identities");

  const auto r97 =
      spec::MakeGuardedSpecializationRequests(graph, p97);
  const auto r128 =
      spec::MakeGuardedSpecializationRequests(graph, p128);
  CHECK(r97[0].unit_semantic_key == r128[0].unit_semantic_key &&
            r97[0].specialization_canonical ==
                r128[0].specialization_canonical,
        "an explicit bucket may reuse a primitive across exact requests");

  fake::GuardedDeterministicMockCoordinator coordinator;
  const auto selected97 = coordinator.Resolve(graph, p97, r97);
  const auto selected128 = coordinator.Resolve(graph, p128, r128);
  CHECK(coordinator.unique_resolve_count() == 1 &&
            selected97[0].artifact_key() ==
                selected128[0].artifact_key(),
        "bucket reuse must be represented by PrimitiveArtifactKey");

  const spec::GuardedShapeProfile p256 = spec::BuildBucketProfile(
      graph, s97, Bucket(graph, Guard(1, 256), 256));
  const auto r256 =
      spec::MakeGuardedSpecializationRequests(graph, p256);
  (void)coordinator.Resolve(graph, p256, r256);
  CHECK(coordinator.unique_resolve_count() == 2,
        "different specialization policy must miss the primitive key");

  fake::GuardedDeterministicMockPlanAssembler assembler;
  const auto plan97 = assembler.Assemble(graph, p97, selected97);
  const auto plan128 = assembler.Assemble(graph, p128, selected128);
  CHECK(plan97.key().defined() && plan128.key().defined() &&
            plan97.key() != plan128.key() &&
            plan97.key().shape_profile_key() == p97.key(),
        "whole-plan identity must be PlanVariantKey, not an artifact key");
  CHECK(plan97.ordered_calls()[0].tail_contract.has_value(),
        "bucket plan must retain its explicit tail contract");

  auto forged = r97;
  forged[0].specialization_canonical = r256[0].specialization_canonical;
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged); }),
        "resolver must reject a forged specialization payload");
  forged = r97;
  forged[0].unit_semantic_key =
      kxc::api::UnitSemanticKey("forged.unit");
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged); }),
        "resolver must reject a forged unit semantic key");
  forged = r97;
  forged[0].ordered_inputs[0].physical = {64};
  CHECK(Throws([&] { (void)coordinator.Resolve(graph, p97, forged); }),
        "resolver must reject forged concrete shape contracts");

  CHECK(Throws([&] { (void)spec::BuildBucketProfile(graph, s129, policy); }),
        "out-of-domain exact requests must fail the guard");
  CHECK(Throws([&] {
          (void)spec::BuildBucketProfile(
              graph, s97, Bucket(graph, Guard(1, 128), 96));
        }),
        "physical capacity smaller than logical extent must reject");
  CHECK(Throws([&] {
          (void)spec::BuildBucketProfile(
              graph, s97, Bucket(graph, Guard(1, 128), 128, false));
        }),
        "padded buckets require explicit tail padding");
  return true;
}

bool TestPolymorphicPolicyAndFormalPlanKey() {
  const spec::GraphTemplate graph = MakeTemplate();
  const spec::ExactOracle s64 = spec::InstantiateExactProfile(
      graph, spec::BindingSet({spec::Binding{"s", 64}}));
  const spec::ExactOracle s96 = spec::InstantiateExactProfile(
      graph, spec::BindingSet({spec::Binding{"s", 96}}));
  const spec::ExactOracle s80 = spec::InstantiateExactProfile(
      graph, spec::BindingSet({spec::Binding{"s", 80}}));
  const spec::ExactOracle s160 = spec::InstantiateExactProfile(
      graph, spec::BindingSet({spec::Binding{"s", 160}}));
  const spec::PolymorphicPolicy policy = Polymorphic(
      graph, Guard(1, 128, 32),
      {0, "extent_s", "s", 1, 128, 32});

  const auto p64 = spec::BuildPolymorphicProfile(graph, s64, policy);
  const auto p96 = spec::BuildPolymorphicProfile(graph, s96, policy);
  const auto r64 =
      spec::MakeGuardedSpecializationRequests(graph, p64);
  const auto r96 =
      spec::MakeGuardedSpecializationRequests(graph, p96);
  CHECK(r64[0].specialization_canonical ==
            r96[0].specialization_canonical,
        "polymorphic primitive specialization must exclude exact bindings");

  fake::GuardedDeterministicMockCoordinator coordinator;
  const auto selected = coordinator.Resolve(graph, p64, r64);
  (void)coordinator.Resolve(graph, p96, r96);
  CHECK(coordinator.unique_resolve_count() == 1,
        "in-domain polymorphic requests must reuse one primitive");

  fake::GuardedDeterministicMockPlanAssembler assembler;
  const auto plan = assembler.Assemble(graph, p64, selected);
  CHECK(plan.key().defined() &&
            plan.key().shape_profile_key() == p64.key() &&
            !plan.ordered_calls()[0].tail_contract.has_value() &&
            plan.ordered_calls()[0].runtime_extent_abi.size() == 1,
        "polymorphic plan must use a formal key and retain extent scalars");

  CHECK(Throws([&] {
          (void)spec::BuildPolymorphicProfile(graph, s80, policy);
        }),
        "divisibility guard must reject");
  CHECK(Throws([&] {
          (void)spec::BuildPolymorphicProfile(graph, s160, policy);
        }),
        "range guard must reject");
  CHECK(Throws([&] {
          (void)spec::PolymorphicPolicy(
              1, Guard(1, 128, 32), policy.allowlist_proofs(), {},
              policy.boundaries(), 8192);
        }),
        "missing runtime extent ABI must reject");
  CHECK(Throws([&] {
          (void)spec::PolymorphicPolicy(
              1, Guard(1, 128, 32), {},
              {{0, "extent_s", "s", 1, 128, 32}},
              policy.boundaries(), 8192);
        }),
        "missing operator proof must reject");
  return true;
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, bool (*)()>> tests = {
      {"bucket_profile_and_formal_artifact_reuse",
       TestBucketProfileAndFormalArtifactReuse},
      {"polymorphic_policy_and_formal_plan_key",
       TestPolymorphicPolicyAndFormalPlanKey},
  };
  int failures = 0;
  for (const auto& [name, test] : tests) {
    try {
      if (test()) {
        std::cout << "[PASS] " << name << "\n";
      } else {
        ++failures;
      }
    } catch (const std::exception& error) {
      std::cerr << "[FAIL] " << name << ": " << error.what() << "\n";
      ++failures;
    }
  }
  return failures == 0 ? 0 : 1;
}
