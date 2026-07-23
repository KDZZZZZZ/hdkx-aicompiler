#include <exception>
#include <functional>
#include <iostream>
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
using shape::BackendKind;
using shape::Binding;
using shape::BindingSet;
using shape::ConcreteTensorShapeContract;
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
using shape::MakeExactSpecializationRequests;
using shape::NamedTensorContract;
using shape::PhysicalShape;
using shape::ShapeProgram;
using shape::TargetBackendAbiDescriptor;
using shape::TargetKind;
using shape::TensorAbiDescriptor;
using shape::TensorShapeContract;
using shape::UnitSemanticKey;
using shape::UnitSkeleton;
using shape::ValidExtent;
namespace fake = shape::fakes::compiler_foundation_v1;

TensorAbiDescriptor TensorAbi(
    DataType dtype = DataType::kFloat32,
    DeviceKind device_kind = DeviceKind::kCpu, uint32_t device_id = 0,
    TargetKind target = TargetKind::kX86_64,
    BackendKind backend = BackendKind::kLlvm, uint32_t backend_abi_version = 1) {
  return TensorAbiDescriptor(dtype, DeviceDescriptor(device_kind, device_id),
                             TargetBackendAbiDescriptor(target, backend,
                                                        backend_abi_version));
}

TensorShapeContract ExactContract(const DimExpr& dimension,
                                  TensorAbiDescriptor abi = TensorAbi()) {
  return TensorShapeContract(LogicalShape({dimension}), PhysicalShape({dimension}),
                             ValidExtent({dimension}), std::move(abi));
}

GraphTemplate MakeTemplate(std::string input_name = "source", std::string middle_name = "middle",
                           std::string result_name = "result", std::string first_locator = "call.0",
                           std::string second_locator = "call.1",
                           std::string graph_fingerprint = "graph.semantic.v1",
                           DataType dtype = DataType::kFloat32,
                           DeviceKind device_kind = DeviceKind::kCpu,
                           uint32_t device_id = 0,
                           TargetKind target = TargetKind::kX86_64,
                           BackendKind backend = BackendKind::kLlvm,
                           uint32_t backend_abi_version = 1) {
  const DimExpr n = DimExpr::Symbol("n");
  const TensorAbiDescriptor abi = TensorAbi(dtype, device_kind, device_id, target,
                                            backend, backend_abi_version);
  const TensorShapeContract contract = ExactContract(n, abi);
  const ShapeProgram program({"n"}, {NamedTensorContract{input_name, contract}},
                             {NamedTensorContract{middle_name, contract},
                              NamedTensorContract{result_name, contract}});
  const GraphTemplateKey key(shape::kShapeContractVersion,
                             std::move(graph_fingerprint), "pipeline.v1",
                             "capability.v1", "per-call.v1",
                             abi.target_backend_abi());
  const UnitSemanticKey semantic(1, "elementwise.relu.v1");
  return GraphTemplate(key, program,
                       {{GraphLocalCallLocator(std::move(first_locator)), semantic, {input_name}, {middle_name}},
                        {GraphLocalCallLocator(std::move(second_locator)), semantic, {middle_name}, {result_name}}});
}

bool TestExactProfilesAndRequests() {
  const GraphTemplate graph_template = MakeTemplate();
  const ExactOracle small = InstantiateExactProfile(graph_template, BindingSet({Binding{"n", 2}}));
  const ExactOracle large = InstantiateExactProfile(graph_template, BindingSet({Binding{"n", 8}}));
  CHECK(small.profile().key().policy_id() == "exact", "exact profile policy must be fixed by the oracle");
  CHECK(small.profile().key().graph_template() == graph_template.key(), "profile must retain template identity");
  CHECK(small.profile().Value("source").contract.logical == std::vector<int64_t>({2}),
        "the shape program must evaluate the input binding");

  const auto small_requests = MakeExactSpecializationRequests(graph_template, small);
  const auto large_requests = MakeExactSpecializationRequests(graph_template, large);
  CHECK(small_requests.size() == 2 && large_requests.size() == 2, "every ordered unit needs a request");
  CHECK(small_requests[0].artifact_key.unit_semantic_key() == small_requests[1].artifact_key.unit_semantic_key(),
        "equivalent unit semantics must stay unchanged across calls");
  CHECK(small_requests[0].artifact_key == small_requests[1].artifact_key,
        "repeated equivalent units must not include locator or value names in artifact identity");
  CHECK(small_requests[0].signature_digest == small_requests[1].signature_digest,
        "signature identity must use ordered contracts rather than graph-local names");
  CHECK(!(small_requests[0].artifact_key == large_requests[0].artifact_key),
        "different exact profiles need different artifact requests");

  const GraphTemplate renamed = MakeTemplate(
      "input.other", "middle.other", "result.other", "other.call.0",
      "other.call.1", "other.graph.semantic");
  const auto renamed_requests = MakeExactSpecializationRequests(
      renamed, InstantiateExactProfile(renamed, BindingSet({Binding{"n", 2}})));
  CHECK(!(renamed_requests[0].shape_profile_key ==
          small_requests[0].shape_profile_key),
        "different graph templates must retain distinct profile identity");
  CHECK(renamed_requests[0].artifact_key == small_requests[0].artifact_key,
        "graph/template locators and value names must not enter artifact identity");
  CHECK(Throws([&] { (void)InstantiateExactProfile(graph_template, BindingSet()); }),
        "all symbols must bind through ShapeProgram");

  const GraphTemplate same_key_different_routing = MakeTemplate(
      "source", "middle", "result", "forged.call.0", "forged.call.1",
      "graph.semantic.v1");
  CHECK(same_key_different_routing.key() == graph_template.key() &&
            !(same_key_different_routing.content_key() == graph_template.content_key()),
        "full template identity must include ordered skeleton and routing");
  CHECK(Throws([&] {
          (void)MakeExactSpecializationRequests(same_key_different_routing, small);
        }), "same caller key with different template content must reject the oracle");
  const GraphTemplate same_key_different_unit(
      graph_template.key(), graph_template.shape_program(),
      {{GraphLocalCallLocator("call.0"), UnitSemanticKey(1, "different.unit.v1"),
        {"source"}, {"middle"}},
       {GraphLocalCallLocator("call.1"), UnitSemanticKey(1, "elementwise.relu.v1"),
        {"middle"}, {"result"}}});
  CHECK(Throws([&] {
          (void)MakeExactSpecializationRequests(same_key_different_unit, small);
        }), "same caller key with different ordered unit skeleton must reject the oracle");

  const auto typed_requests = [](const GraphTemplate& graph) {
    return MakeExactSpecializationRequests(
        graph, InstantiateExactProfile(graph, BindingSet({Binding{"n", 2}})));
  };
  const GraphTemplate f16 = MakeTemplate("source", "middle", "result", "call.0", "call.1",
                                         "graph.semantic.v1", DataType::kFloat16);
  const GraphTemplate cpu1 = MakeTemplate("source", "middle", "result", "call.0", "call.1",
                                          "graph.semantic.v1", DataType::kFloat32,
                                          DeviceKind::kCpu, 1);
  const GraphTemplate cuda0 = MakeTemplate("source", "middle", "result", "call.0", "call.1",
                                           "graph.semantic.v1", DataType::kFloat32,
                                           DeviceKind::kCuda, 0);
  const GraphTemplate arm = MakeTemplate("source", "middle", "result", "call.0", "call.1",
                                         "graph.semantic.v1", DataType::kFloat32,
                                         DeviceKind::kCpu, 0, TargetKind::kAArch64);
  const GraphTemplate native = MakeTemplate("source", "middle", "result", "call.0", "call.1",
                                            "graph.semantic.v1", DataType::kFloat32,
                                            DeviceKind::kCpu, 0, TargetKind::kX86_64,
                                            BackendKind::kNative);
  const GraphTemplate abi2 = MakeTemplate("source", "middle", "result", "call.0", "call.1",
                                          "graph.semantic.v1", DataType::kFloat32,
                                          DeviceKind::kCpu, 0, TargetKind::kX86_64,
                                          BackendKind::kLlvm, 2);
  const auto f16_requests = typed_requests(f16);
  const auto cpu1_requests = typed_requests(cpu1);
  const auto cuda_requests = typed_requests(cuda0);
  const auto arm_requests = typed_requests(arm);
  const auto native_requests = typed_requests(native);
  const auto abi2_requests = typed_requests(abi2);
  CHECK(Throws([&] { (void)MakeExactSpecializationRequests(f16, small); }),
        "same caller key with different ShapeProgram ABI content must reject the oracle");
  CHECK(!(small.profile().key() == InstantiateExactProfile(
              f16, BindingSet({Binding{"n", 2}})).profile().key()) &&
            !(small_requests[0].artifact_key == f16_requests[0].artifact_key),
        "f32 and f16 must have distinct profile and artifact identity");
  CHECK(!(small_requests[0].shape_profile_key == cpu1_requests[0].shape_profile_key) &&
            !(small_requests[0].shape_profile_key == cuda_requests[0].shape_profile_key) &&
            !(small_requests[0].artifact_key == cpu1_requests[0].artifact_key) &&
            !(small_requests[0].artifact_key == cuda_requests[0].artifact_key),
        "device kind and id must participate in exact profile/artifact identity");
  CHECK(!(small_requests[0].shape_profile_key == arm_requests[0].shape_profile_key) &&
            !(small_requests[0].shape_profile_key == native_requests[0].shape_profile_key) &&
            !(small_requests[0].shape_profile_key == abi2_requests[0].shape_profile_key) &&
            !(small_requests[0].artifact_key == arm_requests[0].artifact_key) &&
            !(small_requests[0].artifact_key == native_requests[0].artifact_key) &&
            !(small_requests[0].artifact_key == abi2_requests[0].artifact_key),
        "target, backend, and backend ABI version must participate in exact profile/artifact identity");
  fake::DeterministicMockCoordinator typed_coordinator;
  for (const auto* requests : {&small_requests, &f16_requests, &cpu1_requests,
                               &cuda_requests, &arm_requests, &native_requests,
                               &abi2_requests}) {
    (void)typed_coordinator.Resolve(*requests);
  }
  CHECK(typed_coordinator.unique_resolve_count() == 7,
        "dtype/device/target/backend differences must be resolver misses");
  return true;
}

bool TestFakeCoordinatorAndFrozenPlan() {
  const GraphTemplate graph_template = MakeTemplate();
  const ExactOracle oracle = InstantiateExactProfile(graph_template, BindingSet({Binding{"n", 2}}));
  const auto requests = MakeExactSpecializationRequests(graph_template, oracle);
  fake::DeterministicMockCoordinator coordinator;
  const auto selected = coordinator.Resolve(requests);
  CHECK(selected.size() == 2 && selected[0].entry_symbol() == selected[1].entry_symbol(),
        "equivalent requests must resolve to the same deterministic fake entry");
  CHECK(selected[0].generation() == 0 && selected[1].generation() == 0,
        "the v1 fake generation must be fixed at zero");
  CHECK(coordinator.unique_resolve_count() == 1, "the coordinator must compare full keys and reuse one artifact");
  const auto selected_again = coordinator.Resolve(requests);
  CHECK(coordinator.unique_resolve_count() == 1 && selected_again[0].entry_symbol() == selected[0].entry_symbol(),
        "repeated resolution must be deterministic");
  CHECK(coordinator.resolve_records().size() == 4,
        "the fake must expose deterministic request records");

  fake::DeterministicMockPlanAssembler assembler;
  const fake::FakeFrozenPlan plan = assembler.Assemble(graph_template, oracle, selected);
  const fake::FakeFrozenPlan again = assembler.Assemble(graph_template, oracle, selected_again);
  CHECK(plan.key() == again.key(), "repeated assembly must produce the same plan key");
  CHECK(plan.ordered_calls().size() == 2 && plan.retained_artifacts().size() == 2,
        "frozen plan must retain calls and selected artifacts");
  CHECK(plan.ordered_calls()[0].call_locator.value() == "call.0" &&
            plan.ordered_calls()[1].call_locator.value() == "call.1",
        "frozen plan must preserve graph call order");
  for (const auto& call : plan.ordered_calls()) {
    CHECK(call.generation == 0 && call.inputs[0].contract.logical == std::vector<int64_t>({2}) &&
              call.inputs[0].contract.logical == call.inputs[0].contract.physical &&
              call.inputs[0].contract.logical == call.inputs[0].contract.valid,
          "frozen calls must retain exact concrete logical/physical/valid contracts");
  }

  std::vector<fake::FakeSelectedArtifact> reversed = {selected[1], selected[0]};
  CHECK(Throws([&] { (void)assembler.Assemble(graph_template, oracle, reversed); }),
        "reordered selections must fail even when their artifact keys are equal");
  CHECK(Throws([&] { (void)assembler.Assemble(graph_template, oracle, {selected[0]}); }),
        "missing selections must fail closed");

  const ExactOracle larger = InstantiateExactProfile(graph_template, BindingSet({Binding{"n", 8}}));
  const auto larger_selected = coordinator.Resolve(MakeExactSpecializationRequests(graph_template, larger));
  CHECK(Throws([&] { (void)assembler.Assemble(graph_template, oracle, larger_selected); }),
        "a larger exact-profile artifact must not satisfy a smaller request");
  return true;
}

bool TestMalformedTemplatesAndNonExactContracts() {
  const DimExpr n = DimExpr::Symbol("n");
  const TensorShapeContract exact = ExactContract(n);
  const TensorAbiDescriptor abi = TensorAbi();
  const GraphTemplateKey key(1, "bad.graph", "pipeline", "capability", "partition",
                             abi.target_backend_abi());
  const UnitSemanticKey semantic(1, "unit");
  const ShapeProgram routing_program({"n"}, {NamedTensorContract{"input", exact}},
                                     {NamedTensorContract{"first", exact}, NamedTensorContract{"second", exact}});
  CHECK(Throws([&] {
          (void)GraphTemplate(key, routing_program,
                              {{GraphLocalCallLocator("late"), semantic, {"first"}, {"second"}}});
        }), "routing must reject a consumer before its producer");
  CHECK(Throws([&] {
          (void)GraphTemplate(key, routing_program,
                              {{GraphLocalCallLocator("first"), semantic, {"input"}, {"first"}},
                               {GraphLocalCallLocator("second"), semantic, {"input"}, {"first"}}});
        }), "routing must reject duplicate producers");
  CHECK(Throws([&] {
          (void)GraphTemplate(
              key, routing_program,
              {{GraphLocalCallLocator("first"), semantic, {"input"}, {"first"}}});
        }), "routing must reject a declared output without a producer");

  const TensorShapeContract padded(LogicalShape({n}), PhysicalShape({DimExpr::Add({n, DimExpr::Const(1)})}),
                                   ValidExtent({n}), abi);
  const ShapeProgram padded_program({"n"}, {NamedTensorContract{"input", padded}},
                                    {NamedTensorContract{"output", padded}});
  const GraphTemplate padded_template(key, padded_program,
                                      {{GraphLocalCallLocator("only"), semantic, {"input"}, {"output"}}});
  CHECK(Throws([&] { (void)InstantiateExactProfile(padded_template, BindingSet({Binding{"n", 2}})); }),
        "non-exact physical contracts must be rejected before artifact selection");
  return true;
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, bool (*)()>> tests = {
      {"exact_profiles_and_requests", TestExactProfilesAndRequests},
      {"fake_coordinator_and_frozen_plan", TestFakeCoordinatorAndFrozenPlan},
      {"malformed_templates_and_nonexact_contracts", TestMalformedTemplatesAndNonExactContracts},
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
