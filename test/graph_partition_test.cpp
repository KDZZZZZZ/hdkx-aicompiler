/*! \file test/graph_partition_test.cpp
 * \brief Verifies stable value ids and one-compute-Call compilation units.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/compilation_unit.h"
#include "kxc/relay/transforms/infer_type.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

bool Throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

kxc::Call Add(const kxc::Expr& lhs, const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get("add"), {lhs, rhs});
}

kxc::Call Multiply(const kxc::Expr& lhs, const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get("mul"), {lhs, rhs});
}

kxc::Call Softmax(const kxc::Expr& input, int axis) {
    return kxc::Call(kxc::relay::Op::Get("softmax"), {input},
                     kxc::relay::SoftmaxAttrs::Create(axis));
}

kxc::Function MakeChain() {
    using namespace kxc;
    TensorType type({4}, "float32");
    Var x("x", type);
    Var y("y", type);
    Var z("z", type);
    return relay::InferTypePass(Function({x, y, z}, Multiply(Add(x, y), z)));
}

kxc::Function MakeBranchTuple() {
    using namespace kxc;
    TensorType type({4}, "float32");
    Var x("x", type);
    Var y("y", type);
    Var z("z", type);
    Call root = Add(x, y);
    return relay::InferTypePass(
        Function({x, y, z}, Tuple({Multiply(root, z), Add(root, z)})));
}

bool SameIds(const kxc::Array<int64_t>& actual,
             std::initializer_list<int64_t> expected) {
    if (actual.size() != expected.size()) return false;
    size_t index = 0;
    for (int64_t value : expected) {
        if (actual[index++] != value) return false;
    }
    return true;
}

bool TestStableChainIdsAndBoundaries() {
    using namespace kxc::api::internal;
    const PartitionedGraph graph = PartitionValueGraph(BuildValueGraph(MakeChain()));
    TEST_CHECK(graph.value_graph.values.size() == 5,
               "three parameters and two Call outputs require five values");
    TEST_CHECK(graph.units.size() == 2 && graph.calls.size() == 2,
               "chain requires one unit and KernelCall per compute Call");
    TEST_CHECK(SameIds(graph.input_value_ids, {0, 1, 2}) &&
                   SameIds(graph.output_value_ids, {4}),
               "chain graph boundary ids changed");
    TEST_CHECK(SameIds(graph.units[0].input_value_ids, {0, 1}) &&
                   SameIds(graph.units[0].output_value_ids, {3}) &&
                   SameIds(graph.units[1].input_value_ids, {3, 2}) &&
                   SameIds(graph.units[1].output_value_ids, {4}),
               "chain unit boundaries must expose producer outputs as values");
    return true;
}

bool TestEquivalentGraphsAreDeterministic() {
    using namespace kxc::api::internal;
    const PartitionedGraph first =
        PartitionValueGraph(BuildValueGraph(MakeBranchTuple()));
    const PartitionedGraph second =
        PartitionValueGraph(BuildValueGraph(MakeBranchTuple()));
    TEST_CHECK(first.value_graph.values.size() == second.value_graph.values.size() &&
                   first.units.size() == second.units.size(),
               "equivalent graphs changed cardinality");
    TEST_CHECK(first.units.size() == 3 &&
                   SameIds(first.output_value_ids, {4, 5}),
               "branch tuple topology must assign root, left, right in post-order");
    for (size_t i = 0; i < first.units.size(); ++i) {
        TEST_CHECK(first.units[i].unit_id == second.units[i].unit_id &&
                       first.units[i].symbol == second.units[i].symbol &&
                       first.units[i].structural_hash ==
                           second.units[i].structural_hash,
                   "stable unit identity must not depend on Object addresses");
    }
    return true;
}

bool TestPartitionRejectsMissingDuplicateAndNonCallOwnership() {
    using namespace kxc;
    using namespace kxc::api::internal;
    PartitionedGraph missing = PartitionValueGraph(BuildValueGraph(MakeChain()));
    missing.units.pop_back();
    TEST_CHECK(Throws([&] { ValidatePartition(missing); }),
               "missing ordinary Call ownership must fail");

    PartitionedGraph duplicate = PartitionValueGraph(BuildValueGraph(MakeChain()));
    duplicate.units.push_back(duplicate.units[0]);
    duplicate.calls.push_back(duplicate.calls[0]);
    TEST_CHECK(Throws([&] { ValidatePartition(duplicate); }),
               "duplicate ordinary Call ownership must fail");

    PartitionedGraph non_call = PartitionValueGraph(BuildValueGraph(MakeChain()));
    non_call.units[0].call = Tuple({non_call.units[0].call,
                                    non_call.units[1].call});
    TEST_CHECK(Throws([&] { ValidatePartition(non_call); }),
               "a unit containing a non-Call aggregate must fail");
    return true;
}

bool TestUncheckedGraphFailsBeforePartition() {
    using namespace kxc;
    using namespace kxc::api::internal;
    TensorType type({4}, "float32");
    Var x("x", type);
    Var y("y", type);
    Function unchecked({x, y}, Add(x, y));
    TEST_CHECK(Throws([&] { (void)BuildValueGraph(unchecked); }),
               "partition input must have complete checked types");
    return true;
}

bool TestRepeatedLogicalArgumentUsesOneBoundaryValue() {
    using namespace kxc;
    using namespace kxc::api::internal;
    TensorType type({4}, "float32");
    Var x("x", type);
    Function function = relay::InferTypePass(Function({x}, Add(x, x)));
    const PartitionedGraph graph =
        PartitionValueGraph(BuildValueGraph(function));
    TEST_CHECK(graph.units.size() == 1 &&
                   SameIds(graph.units[0].input_value_ids, {0}),
               "repeated logical operands must share one external boundary value");
    TEST_CHECK(graph.value_graph.calls[0].argument_value_ids.size() == 2 &&
                   graph.value_graph.calls[0].argument_value_ids[0] == 0 &&
                   graph.value_graph.calls[0].argument_value_ids[1] == 0,
               "logical argument order and duplicates must remain available to lowering");
    return true;
}

bool TestAttrsValuesParticipateInStructuralHash() {
    using namespace kxc;
    using namespace kxc::api::internal;
    TensorType type({2, 2}, "float32");
    Var first_input("input", type);
    Var second_input("input", type);
    const PartitionedGraph first = PartitionValueGraph(BuildValueGraph(
        relay::InferTypePass(Function({first_input}, Softmax(first_input, 0)))));
    const PartitionedGraph second = PartitionValueGraph(BuildValueGraph(
        relay::InferTypePass(Function({second_input}, Softmax(second_input, 1)))));
    TEST_CHECK(first.units.size() == 1 && second.units.size() == 1,
               "single-call attrs fixtures must each produce one unit");
    TEST_CHECK(!(first.units[0].structural_hash ==
                 second.units[0].structural_hash),
               "different attrs values must produce different unit hashes");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"stable_chain_ids_and_boundaries", TestStableChainIdsAndBoundaries},
        {"equivalent_graphs_are_deterministic", TestEquivalentGraphsAreDeterministic},
        {"partition_rejects_invalid_ownership",
         TestPartitionRejectsMissingDuplicateAndNonCallOwnership},
        {"unchecked_graph_fails_before_partition", TestUncheckedGraphFailsBeforePartition},
        {"repeated_argument_uses_one_boundary",
         TestRepeatedLogicalArgumentUsesOneBoundaryValue},
        {"attrs_values_participate_in_hash",
         TestAttrsValuesParticipateInStructuralHash},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
                continue;
            }
            std::cout << "[PASS] " << test.first << "\n";
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
