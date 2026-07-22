/*! \file test/operator_compilation_test.cpp
 * \brief Characterizes whole-graph lowering and locks per-operator unit counts.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kxc/compiler/lowering/relay_to_tir.h"
#include "kxc/relay/op.h"
#include "kxc/relay/relay.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

struct GraphFixture {
    const char* name;
    kxc::Function function;
    size_t expected_compute_calls;
    int64_t expected_graph_outputs;
};

kxc::Call Add(const kxc::Expr& lhs, const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get("add"), {lhs, rhs});
}

kxc::Call Multiply(const kxc::Expr& lhs, const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get("mul"), {lhs, rhs});
}

void CollectUniqueCalls(const kxc::Expr& expr,
                        std::unordered_set<const kxc::Object*>* calls) {
    if (!expr.defined()) return;
    if (const auto* call = expr.As<kxc::CallNode>()) {
        if (!calls->insert(expr.get()).second) return;
        for (const auto& arg : call->args) CollectUniqueCalls(arg, calls);
        return;
    }
    if (const auto* function = expr.As<kxc::FunctionNode>()) {
        CollectUniqueCalls(function->body, calls);
        return;
    }
    if (const auto* tuple = expr.As<kxc::TupleNode>()) {
        for (const auto& field : tuple->fields) CollectUniqueCalls(field, calls);
        return;
    }
    if (const auto* get_item = expr.As<kxc::TupleGetItemNode>()) {
        CollectUniqueCalls(get_item->tuple, calls);
        return;
    }
    if (const auto* let = expr.As<kxc::LetNode>()) {
        CollectUniqueCalls(let->value, calls);
        CollectUniqueCalls(let->body, calls);
        return;
    }
    if (const auto* branch = expr.As<kxc::IfNode>()) {
        CollectUniqueCalls(branch->cond, calls);
        CollectUniqueCalls(branch->true_branch, calls);
        CollectUniqueCalls(branch->false_branch, calls);
    }
}

size_t CountUniqueCalls(const kxc::Function& function) {
    std::unordered_set<const kxc::Object*> calls;
    CollectUniqueCalls(function, &calls);
    return calls.size();
}

bool ReadIntAttr(const kxc::tir::PrimFunc& function, const char* key,
                 int64_t* value) {
    const kxc::String attr_key(key);
    if (!function.defined() || !function->attrs.count(attr_key)) return false;
    const auto* integer = function->attrs.at(attr_key).As<kxc::tir::IntImmNode>();
    if (!integer) return false;
    *value = integer->value;
    return true;
}

bool HasGlobalSymbol(const kxc::tir::PrimFunc& function, const char* expected) {
    const kxc::String key("global_symbol");
    if (!function.defined() || !function->attrs.count(key)) return false;
    try {
        return kxc::String(function->attrs.at(key)) == expected;
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<GraphFixture> MakeFixtures() {
    using namespace kxc;
    const TensorType tensor_type({4}, "float32");

    Var chain_x("chain_x", tensor_type);
    Var chain_y("chain_y", tensor_type);
    Var chain_z("chain_z", tensor_type);
    Call chain_first = Add(chain_x, chain_y);
    Call chain_output = Multiply(chain_first, chain_z);

    Var branch_x("branch_x", tensor_type);
    Var branch_y("branch_y", tensor_type);
    Var branch_z("branch_z", tensor_type);
    Call branch_root = Add(branch_x, branch_y);
    Call branch_left = Multiply(branch_root, branch_z);
    Call branch_right = Add(branch_root, branch_z);

    Var diamond_x("diamond_x", tensor_type);
    Var diamond_y("diamond_y", tensor_type);
    Var diamond_z("diamond_z", tensor_type);
    Call diamond_root = Add(diamond_x, diamond_y);
    Call diamond_left = Multiply(diamond_root, diamond_z);
    Call diamond_right = Add(diamond_root, diamond_z);
    Call diamond_output = Add(diamond_left, diamond_right);

    Var tuple_x("tuple_x", tensor_type);
    Var tuple_y("tuple_y", tensor_type);
    Call tuple_left = Add(tuple_x, tuple_y);
    Call tuple_right = Multiply(tuple_x, tuple_y);

    return {
        {"chain", Function({chain_x, chain_y, chain_z}, chain_output), 2, 1},
        {"branch",
         Function({branch_x, branch_y, branch_z},
                  Tuple({branch_left, branch_right})),
         3, 2},
        {"diamond",
         Function({diamond_x, diamond_y, diamond_z}, diamond_output), 4, 1},
        {"tuple_output", Function({tuple_x, tuple_y}, Tuple({tuple_left, tuple_right})),
         2, 2},
    };
}

bool TestWholeGraphSinglePrimFuncBaseline() {
    for (const auto& fixture : MakeFixtures()) {
        const kxc::relay::LoweredFunction lowered =
            kxc::relay::LowerToTIR(fixture.function);
        TEST_CHECK(lowered.defined() && lowered->prim_func.defined(),
                   std::string(fixture.name) + " did not produce a PrimFunc");
        TEST_CHECK(CountUniqueCalls(fixture.function) == fixture.expected_compute_calls,
                   std::string(fixture.name) + " fixture call count changed");
        TEST_CHECK(HasGlobalSymbol(lowered->prim_func, "main"),
                   std::string(fixture.name) +
                       " no longer uses the current whole-graph main symbol");
        int64_t output_count = -1;
        TEST_CHECK(ReadIntAttr(lowered->prim_func, "kxc.output_count", &output_count) &&
                       output_count == fixture.expected_graph_outputs,
                   std::string(fixture.name) + " graph output ABI changed");
    }
    return true;
}

bool TestPerOperatorTargetCardinality() {
    for (const auto& fixture : MakeFixtures()) {
        const size_t expected_units = CountUniqueCalls(fixture.function);
        const kxc::relay::LoweredFunction lowered =
            kxc::relay::LowerToTIR(fixture.function);
        const size_t current_primfuncs =
            lowered.defined() && lowered->prim_func.defined() ? 1U : 0U;
        TEST_CHECK(expected_units == fixture.expected_compute_calls,
                   std::string(fixture.name) +
                       " must allocate exactly one target unit per compute Call");
        TEST_CHECK(expected_units > 1,
                   std::string(fixture.name) +
                       " must remain a multi-operator migration fixture");
        TEST_CHECK(current_primfuncs != expected_units,
                   std::string(fixture.name) +
                       " unexpectedly satisfies the future per-operator PrimFunc contract; "
                       "update this characterization when unit lowering is implemented");
    }
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"whole_graph_single_primfunc_baseline", TestWholeGraphSinglePrimFuncBaseline},
        {"per_operator_target_cardinality", TestPerOperatorTargetCardinality},
    };
    bool ok = true;
    for (const auto& test : tests) {
        const bool passed = test.second();
        std::cout << (passed ? "[PASS] " : "[FAIL] ") << test.first << "\n";
        ok = passed && ok;
    }
    return ok ? 0 : 1;
}
