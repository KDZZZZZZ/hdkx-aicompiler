/*! \file test/executable_capability_test.cpp
 * \brief Standalone static-exact executable capability and ValueGraph checks.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/executable_capability.h"
#include "../src/compiler/internal/value_graph.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                       \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

std::string ErrorText(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception& error) {
        return error.what();
    }
    return "";
}

kxc::Call Add(const kxc::Expr& lhs, const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get("add"), {lhs, rhs});
}

bool SameIds(const kxc::Array<int64_t>& lhs, const kxc::Array<int64_t>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

bool TestStaticExactDiagnosticsAndIfGate() {
    using namespace kxc;
    using namespace kxc::api::internal;
    const TensorType type({4}, "float32");

    Var untyped("untyped");
    const std::string untyped_error = ErrorText([&] {
        VerifyExecutableCapability(Function({untyped}, untyped),
                                    StaticDataflowExecutableCapabilities());
    });
    TEST_CHECK(untyped_error.find("required capability=defined_typed_relay") !=
                   std::string::npos,
               "untyped Relay must report the typed capability");

    Var free_var("free", type);
    Function free_function({}, free_var);
    free_function = relay::InferTypePass(free_function);
    const std::string free_error = ErrorText([&] {
        VerifyExecutableCapability(free_function, StaticDataflowExecutableCapabilities());
    });
    TEST_CHECK(free_error.find("path=function.body") != std::string::npos &&
                   free_error.find("node=Var") != std::string::npos &&
                   free_error.find("required capability=lexically_bound_var") !=
                       std::string::npos,
               "free Var diagnostic must carry deterministic path, kind, and capability");

    Var dynamic("dynamic", TensorType({-1, 4}, "float32"));
    Function dynamic_function({dynamic}, dynamic);
    dynamic_function = relay::InferTypePass(dynamic_function);
    const std::string dynamic_error = ErrorText([&] {
        VerifyExecutableCapability(dynamic_function,
                                    StaticDataflowExecutableCapabilities());
    });
    TEST_CHECK(dynamic_error.find("shape[0]") != std::string::npos &&
                   dynamic_error.find("static_exact_shape") != std::string::npos,
               "every negative TensorType dimension must be rejected");

    Var predicate("predicate", TensorType({}, "bool"));
    Var x("x", type);
    Var y("y", type);
    Function conditional({predicate, x, y}, If(predicate, x, y));
    conditional = relay::InferTypePass(conditional);
    const std::string if_error = ErrorText([&] {
        VerifyExecutableCapability(conditional, StaticDataflowExecutableCapabilities());
    });
    TEST_CHECK(if_error.find("path=function.body") != std::string::npos &&
                   if_error.find("node=If") != std::string::npos &&
                   if_error.find("required capability=if") != std::string::npos,
               "default static-dataflow capability must reject If with a clear gate");

    ExecutableCapabilityOptions with_if = StaticDataflowExecutableCapabilities();
    with_if.allow_if = true;
    VerifyExecutableCapability(conditional, with_if);
    return true;
}

bool TestFunctionValueIsRejected() {
    using namespace kxc;
    using namespace kxc::api::internal;
    Var x("x", TensorType({4}, "float32"));
    Function inner({}, x);
    Function outer({x}, inner);
    outer = relay::InferTypePass(outer);
    const std::string error = ErrorText([&] {
        VerifyExecutableCapability(outer, StaticDataflowExecutableCapabilities());
    });
    TEST_CHECK(error.find("node=Function") != std::string::npos &&
                   error.find("first_order_relay") != std::string::npos,
               "function values and closures must be rejected");
    return true;
}

bool TestValueGraphLetMatchesNestedTopology() {
    using namespace kxc;
    using namespace kxc::api::internal;
    const TensorType type({4}, "float32");
    Var x("x", type);
    Var y("y", type);
    Function nested({x, y}, Add(Add(x, y), y));
    nested = relay::InferTypePass(nested);

    Var lx("x", type);
    Var ly("y", type);
    Var shared("shared", type);
    Function with_let({lx, ly}, Let(shared, Add(lx, ly), Add(shared, ly)));
    with_let = relay::InferTypePass(with_let);

    const ValueGraph nested_graph = BuildValueGraph(nested);
    const ValueGraph let_graph = BuildValueGraph(with_let);
    TEST_CHECK(nested_graph.values.size() == let_graph.values.size(),
               "lexical Let changed value count");
    TEST_CHECK(nested_graph.calls.size() == let_graph.calls.size(),
               "lexical Let changed Call count");
    TEST_CHECK(SameIds(nested_graph.output_value_ids, let_graph.output_value_ids),
               "lexical Let changed stable output value ids");
    TEST_CHECK(let_graph.calls.size() == 2 &&
                   let_graph.calls[1].argument_value_ids.size() == 2 &&
                   let_graph.calls[1].argument_value_ids[0] == 2 &&
                   let_graph.calls[1].argument_value_ids[1] == 1,
               "bound Var must resolve exactly to the value ids of its Let value");

    Var tx("tx", type);
    Var ty("ty", type);
    Var tuple_value("tuple_value", TupleType({type, type}));
    Function tuple_let({tx, ty},
                       Let(tuple_value, Tuple({Add(tx, ty), Add(ty, tx)}),
                           Tuple({TupleGetItem(tuple_value, 1),
                                  TupleGetItem(tuple_value, 0)})));
    tuple_let = relay::InferTypePass(tuple_let);
    const ValueGraph tuple_graph = BuildValueGraph(tuple_let);
    TEST_CHECK(tuple_graph.calls.size() == 2 && tuple_graph.output_value_ids.size() == 2 &&
                   tuple_graph.output_value_ids[0] == 3 && tuple_graph.output_value_ids[1] == 2,
               "Tuple Let and TupleGetItem must retain their resolved leaf ids");
    return true;
}

bool TestValueGraphFreeVarRejects() {
    using namespace kxc;
    using namespace kxc::api::internal;
    Var free_var("free", TensorType({4}, "float32"));
    Function function({}, free_var);
    function = relay::InferTypePass(function);
    const std::string error = ErrorText([&] { (void)BuildValueGraph(function); });
    TEST_CHECK(error.find("lexically_bound_var") != std::string::npos,
               "ValueGraph must capability-reject free Vars before graph construction");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"static_exact_diagnostics_and_if_gate", TestStaticExactDiagnosticsAndIfGate},
        {"function_value_rejected", TestFunctionValueIsRejected},
        {"value_graph_let_matches_nested", TestValueGraphLetMatchesNestedTopology},
        {"value_graph_free_var_rejected", TestValueGraphFreeVarRejects},
    };
    for (const auto& test : tests) {
        try {
            if (!test.second()) return 1;
            std::cout << "[PASS] " << test.first << "\n";
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            return 1;
        }
    }
    return 0;
}
