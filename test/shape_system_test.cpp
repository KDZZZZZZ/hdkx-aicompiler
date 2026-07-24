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
using shape::Binding;
using shape::BindingSet;
using shape::Constraint;
using shape::DeferredConstraintKind;
using shape::DimExpr;
using shape::ExactConstraintSolver;
using shape::LogicalShape;
using shape::NamedTensorContract;
using shape::PhysicalCapacity;
using shape::ShapeProgram;
using shape::TensorShapeContract;
using shape::ValidExtent;

TensorShapeContract Contract(std::vector<DimExpr> logical, std::vector<DimExpr> physical,
                             std::vector<DimExpr> valid) {
  return TensorShapeContract(
      LogicalShape(std::move(logical)),
      PhysicalCapacity(std::move(physical)),
      ValidExtent(std::move(valid)));
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
      PhysicalCapacity({DimExpr::Const(8)}),
      ValidExtent({output_extent}));
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
            evaluated.outputs[0].contract.valid == std::vector<int64_t>({4}),
        "output logical/physical/valid contract is incorrect");

  CHECK(Throws([] {
          (void)Contract({DimExpr::Const(4)}, {DimExpr::Const(3)}, {DimExpr::Const(3)}).Evaluate(BindingSet());
        }), "logical capacity beyond physical capacity must fail");
  CHECK(Throws([] {
          (void)Contract({DimExpr::Const(4)}, {DimExpr::Const(4)}, {DimExpr::Const(5)}).Evaluate(BindingSet());
        }), "valid extent beyond logical extent must fail");
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

bool TestPolicyGatesAndNoFuzzyCapacity() {
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
      {"policy_gates_and_no_fuzzy_capacity",
       TestPolicyGatesAndNoFuzzyCapacity},
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
