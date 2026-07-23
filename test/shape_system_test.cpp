#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "kxc/shape/shape.h"

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
using shape::Constraint;
using shape::DataType;
using shape::DeferredConstraintKind;
using shape::DeviceDescriptor;
using shape::DeviceKind;
using shape::DimExpr;
using shape::ExactConstraintSolver;
using shape::GraphTemplateKey;
using shape::LogicalShape;
using shape::NamedTensorContract;
using shape::PhysicalShape;
using shape::ShapeProgram;
using shape::TargetBackendAbiDescriptor;
using shape::TargetKind;
using shape::TensorAbiDescriptor;
using shape::TensorShapeContract;
using shape::ValidExtent;

TargetBackendAbiDescriptor CpuAbi(BackendKind backend = BackendKind::kLlvm,
                                  TargetKind target = TargetKind::kX86_64) {
  return TargetBackendAbiDescriptor(target, backend, 1);
}

TensorAbiDescriptor F32Cpu(uint32_t device_id = 0) {
  return TensorAbiDescriptor(DataType::kFloat32,
                             DeviceDescriptor(DeviceKind::kCpu, device_id), CpuAbi());
}

TensorShapeContract Contract(std::vector<DimExpr> logical, std::vector<DimExpr> physical,
                             std::vector<DimExpr> valid) {
  return TensorShapeContract(LogicalShape(std::move(logical)), PhysicalShape(std::move(physical)),
                             ValidExtent(std::move(valid)), F32Cpu());
}

bool TestCanonicalizationAndErrors() {
  const DimExpr n = DimExpr::Symbol("n");
  const DimExpr m = DimExpr::Symbol("m");
  const DimExpr first = DimExpr::Add({n, DimExpr::Const(2), DimExpr::Add({m, DimExpr::Const(3)})});
  const DimExpr second = DimExpr::Add({DimExpr::Const(5), m, n});
  CHECK(first == second, "Add must flatten, fold, and sort deterministically");
  CHECK(DimExpr::Mul({DimExpr::Const(1), n}) == n, "Mul identity must fold");
  CHECK(DimExpr::Min({n, n, DimExpr::Const(8), DimExpr::Const(3)}) ==
            DimExpr::Min({DimExpr::Const(3), n}),
        "Min must deduplicate and fold constants");
  CHECK(DimExpr::Max({n, DimExpr::Max({n, m})}) == DimExpr::Max({m, n}),
        "Max must normalize associatively and commutatively");
  CHECK(DimExpr::Const(0).Evaluate(BindingSet()) == 0, "zero dimension must remain legal");
  CHECK(Throws([] { DimExpr::Const(-1); }), "-1 must not be accepted as a dimension");
  CHECK(Throws([] { DimExpr::FloorDiv(DimExpr::Const(1), 0); }), "division by zero must fail");
  CHECK(Throws([] { DimExpr::Add({DimExpr::Const(std::numeric_limits<int64_t>::max()), DimExpr::Const(1)}); }),
        "constant addition overflow must fail");
  CHECK(Throws([] { BindingSet({Binding{"n", -1}}); }), "negative binding values must fail");
  CHECK(Throws([&] {
          (void)DimExpr::Add({n, DimExpr::Const(1)}).Evaluate(
              BindingSet({Binding{"n", std::numeric_limits<int64_t>::max()}}));
        }), "binding-time expression overflow must fail");
  return true;
}

bool TestExactSolver() {
  const DimExpr n = DimExpr::Symbol("n");
  const DimExpr batch = DimExpr::Symbol("batch");
  const std::vector<Constraint> constraints = {
      Constraint::Range(batch, 0, 8),
      Constraint::Eq(n, DimExpr::Add({batch, DimExpr::Const(1)})),
      Constraint::DivisibleBy(DimExpr::Mul({batch, DimExpr::Const(2)}), 2),
      Constraint::BroadcastCompatible(DimExpr::Const(0), DimExpr::Const(1)),
  };
  const BindingSet solved = ExactConstraintSolver::Solve({"n", "batch"}, BindingSet({Binding{"batch", 3}}), constraints);
  CHECK(solved.Find("n") == 4 && solved.Find("batch") == 3, "exact equality inference failed");
  CHECK(Throws([&] { (void)ExactConstraintSolver::Solve({"n"}, BindingSet(), {Constraint::Range(n, 0, 8)}); }),
        "range constraints must not guess bindings");
  CHECK(Throws([&] { (void)ExactConstraintSolver::Solve({"n"}, BindingSet({Binding{"n", 2}}), {Constraint::Eq(n, DimExpr::Const(3))}); }),
        "contradictory equality must fail");
  CHECK(Throws([&] { (void)ExactConstraintSolver::Solve({"n"}, BindingSet({Binding{"n", 3}}), {Constraint::DivisibleBy(n, 2)}); }),
        "divisibility violation must fail");
  CHECK(Throws([&] { (void)ExactConstraintSolver::Solve({"n"}, BindingSet({Binding{"n", 2}}), {Constraint::BroadcastCompatible(n, DimExpr::Const(3))}); }),
        "broadcast violation must fail");
  CHECK(Throws([&] { (void)ExactConstraintSolver::Solve({"n"}, BindingSet(), {Constraint::Eq(n, DimExpr::Symbol("other"))}); }),
        "undeclared constraint symbols must fail");
  return true;
}

bool TestContractsAndProgram() {
  const DimExpr n = DimExpr::Symbol("n");
  const TensorShapeContract input = Contract({n, DimExpr::Const(2)}, {n, DimExpr::Const(2)},
                                             {n, DimExpr::Const(2)});
  const DimExpr output_extent = DimExpr::Add({n, DimExpr::Const(1)});
  const TensorShapeContract output = TensorShapeContract(
      LogicalShape({output_extent}, {std::optional<std::string>("features")} ),
      PhysicalShape({DimExpr::Const(8)}, std::nullopt, "contiguous.row_major", 16, "host"),
      ValidExtent({output_extent}), F32Cpu());
  const ShapeProgram program({"n"}, {NamedTensorContract{"input", input}},
                             {NamedTensorContract{"output", output}},
                             {Constraint::Range(n, 0, 7)});
  program.Verify();
  const auto evaluated = program.Evaluate(BindingSet({Binding{"n", 3}}));
  CHECK(evaluated.inputs.size() == 1 && evaluated.outputs.size() == 1, "program contracts missing");
  CHECK(evaluated.inputs[0].contract.logical == std::vector<int64_t>({3, 2}) &&
            evaluated.inputs[0].contract.physical == std::vector<int64_t>({3, 2}) &&
            evaluated.inputs[0].contract.valid == std::vector<int64_t>({3, 2}),
        "input logical/physical/valid contracts must remain separate");
  CHECK(evaluated.outputs[0].contract.logical == std::vector<int64_t>({4}) &&
            evaluated.outputs[0].contract.physical == std::vector<int64_t>({8}) &&
            evaluated.outputs[0].contract.valid == std::vector<int64_t>({4}) &&
            evaluated.outputs[0].contract.strides == std::vector<int64_t>({1}),
        "output contract or derived row-major stride is incorrect");

  CHECK(Throws([] {
          (void)Contract({DimExpr::Const(4)}, {DimExpr::Const(3)}, {DimExpr::Const(3)}).Evaluate(BindingSet());
        }), "logical capacity beyond physical capacity must fail");
  CHECK(Throws([] {
          (void)Contract({DimExpr::Const(4)}, {DimExpr::Const(4)}, {DimExpr::Const(5)}).Evaluate(BindingSet());
        }), "valid extent beyond logical extent must fail");
  CHECK(Throws([] {
          PhysicalShape({DimExpr::Const(4)}, std::nullopt, "contiguous", 3,
                        "default");
        }), "non-power-of-two alignment must fail");
  CHECK(Throws([] {
          PhysicalShape({DimExpr::Const(4)}, std::nullopt, "", 1, "default");
        }), "physical layout must be explicit");
  CHECK(Throws([] {
          PhysicalShape({DimExpr::Const(4)}, std::nullopt, "strided", 1, "default");
        }), "experimental v1 must gate unsupported layouts");
  CHECK(Throws([&] {
          const TensorShapeContract zero_stride(
              LogicalShape({DimExpr::Const(2), DimExpr::Const(2)}),
              PhysicalShape({DimExpr::Const(2), DimExpr::Const(2)},
                            std::vector<DimExpr>{DimExpr::Const(0), DimExpr::Const(1)}),
              ValidExtent({DimExpr::Const(2), DimExpr::Const(2)}), F32Cpu());
          (void)zero_stride.Evaluate(BindingSet());
        }), "zero writable stride must fail closed");
  CHECK(Throws([&] {
          const TensorShapeContract overlapping(
              LogicalShape({DimExpr::Const(2), DimExpr::Const(2)}),
              PhysicalShape({DimExpr::Const(2), DimExpr::Const(2)},
                            std::vector<DimExpr>{DimExpr::Const(1), DimExpr::Const(1)}),
              ValidExtent({DimExpr::Const(2), DimExpr::Const(2)}), F32Cpu());
          (void)overlapping.Evaluate(BindingSet());
        }), "overlapping writable strides must fail closed");
  CHECK(Throws([&] {
          const TensorShapeContract huge(
              LogicalShape({n, DimExpr::Const(2)}), PhysicalShape({n, DimExpr::Const(2)}),
              ValidExtent({n, DimExpr::Const(2)}), F32Cpu());
          (void)huge.Evaluate(BindingSet({Binding{"n", std::numeric_limits<int64_t>::max()}}));
        }), "binding-time derived stride overflow must fail closed");
  CHECK(Throws([&] {
          const TensorShapeContract huge_bytes(
              LogicalShape({n}), PhysicalShape({n}), ValidExtent({n}), F32Cpu());
          (void)huge_bytes.Evaluate(BindingSet({Binding{
              "n", std::numeric_limits<int64_t>::max() / 4 + 1}}));
        }), "binding-time physical byte extent overflow must fail closed");
  CHECK(Throws([&] { (void)program.Evaluate(BindingSet()); }), "program evaluation must reject underbound symbols");

  const DimExpr m = DimExpr::Symbol("m");
  const TensorShapeContract secondary = Contract({m}, {m}, {m});
  const ShapeProgram first_canonical({"n", "m"},
                                     {NamedTensorContract{"z", input}, NamedTensorContract{"a", secondary}},
                                     {NamedTensorContract{"output", output}},
                                     {Constraint::Range(n, 0, 7), Constraint::Range(m, 0, 7)});
  const ShapeProgram reordered({"m", "n"},
                               {NamedTensorContract{"a", secondary}, NamedTensorContract{"z", input}},
                               {NamedTensorContract{"output", output}},
                               {Constraint::Range(m, 0, 7), Constraint::Range(n, 0, 7)});
  CHECK(first_canonical == reordered && first_canonical.CanonicalString() == reordered.CanonicalString(),
        "program canonicalization must ignore set-like declaration ordering");
  return true;
}

bool TestKeysAndNoFuzzyCapacity() {
  const GraphTemplateKey first(shape::kShapeContractVersion, "a", "bc",
                               "cpu", "per-call", CpuAbi());
  const GraphTemplateKey second(shape::kShapeContractVersion, "ab", "c",
                                "cpu", "per-call", CpuAbi());
  const GraphTemplateKey other_capability(shape::kShapeContractVersion,
                                          "a", "bc", "cuda", "per-call", CpuAbi());
  const GraphTemplateKey other_target(shape::kShapeContractVersion,
                                      "a", "bc", "cpu", "per-call",
                                      CpuAbi(BackendKind::kLlvm, TargetKind::kAArch64));
  const GraphTemplateKey other_backend(shape::kShapeContractVersion,
                                       "a", "bc", "cpu", "per-call",
                                       CpuAbi(BackendKind::kNative));
  CHECK(first.CanonicalBytes() != second.CanonicalBytes(), "key fields need collision-safe boundaries");
  CHECK(first.CanonicalBytes() != other_capability.CanonicalBytes(),
        "capability fingerprint must participate in template identity");
  CHECK(first.CanonicalBytes() != other_target.CanonicalBytes(),
        "strong target kind must participate in template identity");
  CHECK(first.CanonicalBytes() != other_backend.CanonicalBytes(),
        "strong backend kind must participate in template identity");
  CHECK(Throws([] {
          GraphTemplateKey(0, "graph", "pipeline", "capability", "partition", CpuAbi());
        }), "key version zero must fail");
  CHECK(Throws([] {
          (void)TargetBackendAbiDescriptor(TargetKind::kX86_64, BackendKind::kLlvm, 0);
        }), "backend ABI version zero must fail");
  CHECK(!shape::SupportsConstraint(DeferredConstraintKind::kSameRank) &&
            !shape::SupportsConstraint(DeferredConstraintKind::kLayoutCompatible),
        "deferred rank/layout constraints must report a closed gate");
  CHECK(Throws([] { shape::RequireConstraintSupport(DeferredConstraintKind::kSameRank); }) &&
            Throws([] { shape::RequireConstraintSupport(DeferredConstraintKind::kLayoutCompatible); }),
        "SameRank/LayoutCompatible must reject until implemented");

  const DimExpr n = DimExpr::Symbol("n");
  const auto bounded = Contract({n}, {DimExpr::Const(5)}, {n});
  CHECK(Throws([&] { (void)bounded.Evaluate(BindingSet({Binding{"n", 6}})); }),
        "larger physical capacity is not a cached_dims>=query compatibility rule");
  return true;
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, bool (*)()>> tests = {
      {"canonicalization_and_errors", TestCanonicalizationAndErrors},
      {"exact_solver", TestExactSolver},
      {"contracts_and_program", TestContractsAndProgram},
      {"keys_and_no_fuzzy_capacity", TestKeysAndNoFuzzyCapacity},
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
