/*! \file test/compiler_capability_test.cpp
 * \brief Verifies the fail-closed executable capability contract and boundaries.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/capability.h"
#include "kxc/compiler/compiler.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "../src/compiler/internal/lowered_graph.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

bool ThrowsWith(const std::function<void()>& fn, const std::string& expected) {
    try {
        fn();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

kxc::Function TypedAdd() {
    using namespace kxc;
    TensorType type({4}, "float32");
    Var lhs("lhs", type);
    Var rhs("rhs", type);
    return relay::InferTypePass(
        Function({lhs, rhs}, Call(relay::Op::Get("add"), {lhs, rhs})));
}

bool TestStaticExactAcceptedAtEveryBoundary() {
    using namespace kxc;
    using namespace kxc::api;
    const Function function = TypedAdd();
    const Target target = BuildTarget(Device::CPU());
    for (CapabilityBoundary boundary : {
             CapabilityBoundary::kCompilerEntry,
             CapabilityBoundary::kPostGraphPass,
             CapabilityBoundary::kPrePartition}) {
        const CapabilityResult result = CapabilityVerifier::Verify(
            CapabilityRequest{function, target, "model/main", "pipeline-v1",
                              boundary, CapabilityMode::kStaticExact, true});
        TEST_CHECK(result.supported && result.issues.empty() &&
                       result.normalized_requirements.size() == 4 &&
                       result.target_identity.find("llvm") != std::string::npos &&
                       result.pipeline_fingerprint == "pipeline-v1" &&
                       result.requested_mode == CapabilityMode::kStaticExact,
                   "current static-exact add graph should expose normalized context");
    }
    return true;
}

bool TestRepresentableControlFlowRejectedWithLocator() {
    using namespace kxc;
    using namespace kxc::api;
    Var cond("cond", TensorType({}, "bool"));
    Var lhs("lhs", TensorType({4}, "float32"));
    Var rhs("rhs", TensorType({4}, "float32"));
    Function function({cond, lhs, rhs}, If(cond, lhs, rhs));
    const CapabilityResult result = CapabilityVerifier::Verify(
        CapabilityRequest{function, BuildTarget(Device::CPU()), "model/control",
                          "", CapabilityBoundary::kCompilerEntry,
                          CapabilityMode::kStaticExact, false});
    TEST_CHECK(!result.supported && !result.issues.empty() &&
                   result.issues[0].missing_capability == "control_flow.if" &&
                   result.issues[0].diagnostic_locator == "model/control/body",
               "If rejection must identify the node and missing capability");
    TEST_CHECK(ThrowsWith(
                   [&] {
                       (void)api::Compiler::Compile(
                           function, api::CompileConfig::Create(
                                         BuildTarget(Device::CPU()), 0));
                   },
                   "CapabilityVerifier[compiler_entry]"),
               "Compiler entry must run the shared verifier before lowering");
    return true;
}

bool TestLetAndFutureModesFailClosed() {
    using namespace kxc;
    using namespace kxc::api;
    TensorType type({4}, "float32");
    Var input("input", type);
    Var local("local", type);
    Function function({input}, Let(local, input, local));
    const Target target = BuildTarget(Device::CPU());
    CapabilityResult let_result = CapabilityVerifier::Verify(
        CapabilityRequest{function, target, "model/let", "",
                          CapabilityBoundary::kCompilerEntry,
                          CapabilityMode::kStaticExact, false});
    TEST_CHECK(!let_result.supported &&
                   let_result.missing_capabilities[0] == "control_flow.let",
               "Let must not pass because InferType can represent it");

    CapabilityResult mode_result = CapabilityVerifier::Verify(
        CapabilityRequest{TypedAdd(), target, "model/add", "",
                          CapabilityBoundary::kCompilerEntry,
                          CapabilityMode::kShapeSpecialization, false});
    TEST_CHECK(!mode_result.supported &&
                   mode_result.missing_capabilities[0] == "execution_mode" &&
                   mode_result.Diagnostic().find("shape_specialization") !=
                       std::string::npos,
               "future execution modes must not be inferred from static support");
    return true;
}

bool TestDynamicSentinelAndUnspecifiedOpRejected() {
    using namespace kxc;
    using namespace kxc::api;
    Var dynamic("dynamic", TensorType({-1, 4}, "float32"));
    Function dynamic_function({dynamic}, dynamic);
    CapabilityResult dynamic_result = CapabilityVerifier::Verify(
        CapabilityRequest{dynamic_function, BuildTarget(Device::CPU()),
                          "model/dynamic", "",
                          CapabilityBoundary::kCompilerEntry,
                          CapabilityMode::kStaticExact, false});
    TEST_CHECK(!dynamic_result.supported &&
                   dynamic_result.missing_capabilities[0] ==
                       "static_exact_shape",
               "legacy -1 input acceptance is not dynamic-shape capability");

    TensorType type({4}, "float32");
    Var input("input", type);
    relay::Op unspecified("test.unspecified");
    Function unspecified_function({input}, Call(unspecified, {input}));
    CapabilityResult op_result = CapabilityVerifier::Verify(
        CapabilityRequest{unspecified_function, BuildTarget(Device::CPU()),
                          "model/unknown", "",
                          CapabilityBoundary::kCompilerEntry,
                          CapabilityMode::kStaticExact, false});
    TEST_CHECK(!op_result.supported &&
                   op_result.missing_capabilities[0] ==
                       "registered_operator",
               "an IR Call without an explicit operator contract must fail closed");
    return true;
}

bool TestPrePartitionBoundaryCannotBeBypassed() {
    using namespace kxc;
    Var cond("cond", TensorType({}, "bool"));
    Var lhs("lhs", TensorType({4}, "float32"));
    Var rhs("rhs", TensorType({4}, "float32"));
    Function function({cond, lhs, rhs}, If(cond, lhs, rhs));
    TEST_CHECK(ThrowsWith(
                   [&] {
                       (void)api::internal::LowerGraph(function, Device::CPU());
                   },
                   "CapabilityVerifier[pre_partition]"),
               "direct per-unit lowering must run the pre-partition gate");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"static_exact_all_boundaries", TestStaticExactAcceptedAtEveryBoundary},
        {"control_flow_locator", TestRepresentableControlFlowRejectedWithLocator},
        {"let_and_future_modes", TestLetAndFutureModesFailClosed},
        {"dynamic_and_unspecified_op", TestDynamicSentinelAndUnspecifiedOpRejected},
        {"pre_partition_not_bypassed", TestPrePartitionBoundaryCannotBeBypassed},
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
