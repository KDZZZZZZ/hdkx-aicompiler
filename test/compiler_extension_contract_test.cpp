/*! \file test/compiler_extension_contract_test.cpp
 * \brief Verifies generic operator/pass extension and layer-boundary contracts.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/pass/pass.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/tir/transforms/pipeline.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

bool Throws(const std::function<void()>& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

bool SamePassIdentity(const kxc::PassSpec& lhs, const kxc::PassSpec& rhs) {
    return lhs.dialect == rhs.dialect && lhs.name == rhs.name &&
           lhs.schema_version == rhs.schema_version && lhs.scope == rhs.scope &&
           lhs.phase == rhs.phase && lhs.implementation_key == rhs.implementation_key;
}

bool TestOperatorRegistryContract() {
    using namespace kxc;
    relay::OperatorSpec incomplete;
    incomplete.name = "test.incomplete";
    TEST_CHECK(Throws([&] { relay::ValidateOperatorSpec(incomplete); }),
               "operator without category and arity must fail validation");

    relay::OperatorSpec missing_lowering;
    missing_lowering.name = "test.missing_lowering";
    missing_lowering.category = "test";
    missing_lowering.input_arity.num_inputs = 1;
    missing_lowering.type_relation_key = "test.type";
    missing_lowering.lowering_kind = relay::OperatorLoweringKind::kSingleTE;
    TEST_CHECK(Throws([&] { relay::ValidateOperatorSpec(missing_lowering); }),
               "compute operator without a lowering key must fail validation");

    relay::CheckOperatorRegistry();
    const std::vector<relay::OperatorSpec> first = relay::ListOperatorSpecs();
    const std::vector<relay::OperatorSpec> second = relay::ListOperatorSpecs();
    TEST_CHECK(!first.empty() && first.size() == second.size(),
               "operator registry must expose a stable non-empty spec list");
    for (size_t i = 0; i < first.size(); ++i) {
        TEST_CHECK(first[i].name == second[i].name &&
                       relay::SerializeOperatorSpec(first[i]) ==
                           relay::SerializeOperatorSpec(second[i]),
                   "operator spec enumeration or serialization is not deterministic");
        TEST_CHECK(first[i].schema_version > 0,
                   "registered operator has an invalid schema version");
        const bool is_compute =
            first[i].lowering_kind == relay::OperatorLoweringKind::kSingleTE ||
            first[i].lowering_kind == relay::OperatorLoweringKind::kMultiTE;
        const bool has_complete_arity =
            first[i].input_arity.num_inputs >= 0 ||
            (first[i].input_arity.num_inputs == -1 &&
             first[i].input_arity.min_inputs >= 0 &&
             first[i].input_arity.max_inputs >= first[i].input_arity.min_inputs);
        TEST_CHECK(!is_compute ||
                       (has_complete_arity &&
                        !first[i].type_relation_key.empty() &&
                        !first[i].lowering_key.empty()),
                   std::string("registered compute operator has an incomplete specification: ") +
                       relay::SerializeOperatorSpec(first[i]));
    }
    TEST_CHECK(relay::Op::TryGet("__kxc_missing_compute_operator__") == nullptr,
               "TryGet must not create unknown operators");
    TEST_CHECK(Throws([] { relay::Op::Get("__kxc_missing_compute_operator__"); }),
               "strict operator lookup must reject unknown compute operators");
    return true;
}

bool TestPassRegistryContract() {
    using namespace kxc;
    const Array<PassSpec> relay_specs = relay::RelayRegisteredPassSpecs();
    const Array<PassSpec> tir_specs = tir::TIRRegisteredPassSpecs();
    TEST_CHECK(!relay_specs.empty() && !tir_specs.empty(),
               "Relay and TIR pass registries must both be populated");

    Array<PassSpec> combined;
    for (const auto& spec : relay_specs) combined.push_back(spec);
    for (const auto& spec : tir_specs) combined.push_back(spec);
    ValidatePassSpecs(combined);

    const Array<PassSpec> registry_specs = PassRegistry::Global().List();
    const Array<PassSpec> repeated = PassRegistry::Global().List();
    TEST_CHECK(registry_specs.size() == combined.size() &&
                   repeated.size() == registry_specs.size(),
               "pass registry lost or duplicated registered specifications");
    for (size_t i = 0; i < registry_specs.size(); ++i) {
        TEST_CHECK(SamePassIdentity(registry_specs[i], repeated[i]),
                   "pass registry enumeration is not deterministic");
    }
    PassRegistry::Global().FreezeAndValidate();
    PassSpec after_freeze;
    after_freeze.name = String("test.after_freeze");
    after_freeze.dialect = IRDialect::kRelay;
    after_freeze.scope = PassScope::kGraph;
    after_freeze.phase = String("relay_optimize");
    after_freeze.implementation_key = String("kxc.test.after_freeze");
    TEST_CHECK(Throws([&] { PassRegistry::Global().Register(after_freeze); }),
               "pass registration after registry freeze must fail");
    return true;
}

bool TestPassScopeAndDialectRejectedBeforeExecution() {
    using namespace kxc;
    const Array<PassSpec> relay_specs = relay::RelayRegisteredPassSpecs();
    const Array<PassSpec> tir_specs = tir::TIRRegisteredPassSpecs();
    TEST_CHECK(!relay_specs.empty() && !tir_specs.empty(),
               "registered pass fixtures are required");
    TEST_CHECK(Throws([&] {
                   ValidatePassSpecForPipeline(relay_specs[0], IRDialect::kTIR,
                                               PassScope::kPrimFunc,
                                               "tir_optimize");
               }),
               "Relay pass must not enter a TIR PrimFunc pipeline");
    TEST_CHECK(Throws([&] {
                   ValidatePassSpecForPipeline(tir_specs[0], IRDialect::kRelay,
                                               PassScope::kGraph,
                                               "relay_optimize");
               }),
               "TIR pass must not enter a Relay graph pipeline");

    PassSpec incomplete;
    incomplete.name = String("test.incomplete");
    incomplete.phase = String("relay_optimize");
    TEST_CHECK(Throws([&] { ValidatePassSpec(incomplete); }),
               "pass without implementation key must fail validation");

    PassSpec duplicate = relay_specs[0];
    TEST_CHECK(Throws([&] { ValidatePassSpecs({duplicate, duplicate}); }),
               "duplicate pass identity must fail validation");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"operator_registry_contract", TestOperatorRegistryContract},
        {"pass_registry_contract", TestPassRegistryContract},
        {"pass_scope_and_dialect_rejected", TestPassScopeAndDialectRejectedBeforeExecution},
    };
    bool ok = true;
    for (const auto& test : tests) {
        const bool passed = test.second();
        std::cout << (passed ? "[PASS] " : "[FAIL] ") << test.first << "\n";
        ok = passed && ok;
    }
    return ok ? 0 : 1;
}
