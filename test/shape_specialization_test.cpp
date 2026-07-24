#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/shape_specialization.h"
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

spec::GraphTemplate MakeTemplate(
    std::string first_locator = "call.0",
    std::string second_locator = "call.1",
    kxc::api::UnitSemanticKey semantic =
        kxc::api::UnitSemanticKey("elementwise.relu.v1")) {
  const spec::DimExpr n = spec::DimExpr::Symbol("n");
  const spec::TensorShapeContract contract = ExactContract(n);
  const spec::ShapeProgram program(
      {"n"}, {{"source", contract}},
      {{"middle", contract}, {"result", contract}});
  return spec::GraphTemplate(
      GraphKey(), program,
      {{spec::GraphLocalCallLocator(std::move(first_locator)), semantic,
        {"source"}, {"middle"}},
       {spec::GraphLocalCallLocator(std::move(second_locator)), semantic,
        {"middle"}, {"result"}}});
}

bool TestFormalProfileAndPrimitiveSelection() {
  using kxc::api::GraphSemanticKey;
  using kxc::api::PlanVariantKey;
  using kxc::api::PrimitiveArtifactKey;
  using kxc::api::ShapeProfileKey;

  static_assert(!std::is_same_v<GraphSemanticKey, ShapeProfileKey>);
  static_assert(!std::is_same_v<ShapeProfileKey, PrimitiveArtifactKey>);
  static_assert(!std::is_same_v<PrimitiveArtifactKey, PlanVariantKey>);

  const spec::GraphTemplate graph = MakeTemplate();
  const spec::ExactOracle small = spec::InstantiateExactProfile(
      graph, spec::BindingSet({spec::Binding{"n", 2}}));
  const spec::ExactOracle large = spec::InstantiateExactProfile(
      graph, spec::BindingSet({spec::Binding{"n", 8}}));

  CHECK(small.profile().key().graph_semantic_key() == graph.key(),
        "profile must reference the formal graph semantic key");
  CHECK(small.profile().policy_id() == "exact" &&
            small.profile().shape_abi_version() ==
                spec::kShapeProfileAbiVersion,
        "profile policy metadata must stay outside the opaque key");
  CHECK(small.profile().Value("source").contract.logical ==
            std::vector<int64_t>({2}),
        "shape binding must evaluate through ShapeProgram");
  CHECK(small.profile().key() != large.profile().key(),
        "different bindings must mint different formal shape profiles");

  const auto small_requests =
      spec::MakeExactSpecializationRequests(graph, small);
  const auto large_requests =
      spec::MakeExactSpecializationRequests(graph, large);
  CHECK(small_requests.size() == 2 && large_requests.size() == 2,
        "every ordered unit must produce one request");
  CHECK(small_requests[0].unit_semantic_key ==
            small_requests[1].unit_semantic_key,
        "graph-local routing must not enter unit semantics");
  CHECK(small_requests[0].signature_digest ==
            small_requests[1].signature_digest,
        "equal ordered shape boundaries must share a signature digest");

  fake::DeterministicMockCoordinator coordinator;
  const auto selected_small = coordinator.Resolve(small_requests);
  CHECK(coordinator.unique_resolve_count() == 1,
        "equal unit/profile/signature requests must reuse one primitive");
  (void)coordinator.Resolve(large_requests);
  CHECK(coordinator.unique_resolve_count() == 2,
        "a different shape profile must select another primitive");

  fake::DeterministicMockPlanAssembler assembler;
  const fake::FakeFrozenPlan plan =
      assembler.Assemble(graph, small, selected_small);
  const fake::FakeFrozenPlan again =
      assembler.Assemble(graph, small, selected_small);
  CHECK(plan.key().defined() && plan.key() == again.key() &&
            plan.key().graph_semantic_key() == graph.key() &&
            plan.key().shape_profile_key() == small.profile().key(),
        "frozen plan identity must be the formal component key");
  CHECK(plan.retained_artifacts().size() == 2 &&
            plan.retained_artifacts()[0].artifact_key().defined(),
        "fake plan must retain formal primitive artifact identities");
  return true;
}

bool TestTemplateContentAndRoutingFailClosed() {
  const spec::GraphTemplate graph = MakeTemplate();
  const spec::ExactOracle oracle = spec::InstantiateExactProfile(
      graph, spec::BindingSet({spec::Binding{"n", 4}}));
  CHECK(Throws([&] {
          (void)spec::InstantiateExactProfile(graph, spec::BindingSet());
        }),
        "unbound symbols must fail closed");

  const spec::GraphTemplate changed_routing =
      MakeTemplate("other.call.0", "other.call.1");
  CHECK(changed_routing.key() == graph.key() &&
            changed_routing.CanonicalBytes() != graph.CanonicalBytes(),
        "routing is template content, not graph semantic identity");
  CHECK(Throws([&] {
          (void)spec::MakeExactSpecializationRequests(changed_routing,
                                                       oracle);
        }),
        "an oracle cannot be replayed against changed template content");

  const spec::GraphTemplate changed_unit =
      MakeTemplate("call.0", "call.1",
                   kxc::api::UnitSemanticKey("different.unit.v1"));
  CHECK(Throws([&] {
          (void)spec::MakeExactSpecializationRequests(changed_unit, oracle);
        }),
        "an oracle cannot be replayed against changed unit semantics");

  fake::DeterministicMockCoordinator coordinator;
  auto requests = spec::MakeExactSpecializationRequests(graph, oracle);
  requests[0].ordered_call_index = 1;
  const auto selected = coordinator.Resolve(requests);
  fake::DeterministicMockPlanAssembler assembler;
  CHECK(Throws([&] { (void)assembler.Assemble(graph, oracle, selected); }),
        "plan assembly must reject forged ordered routing");
  return true;
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, bool (*)()>> tests = {
      {"formal_profile_and_primitive_selection",
       TestFormalProfileAndPrimitiveSelection},
      {"template_content_and_routing_fail_closed",
       TestTemplateContentAndRoutingFailClosed},
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
