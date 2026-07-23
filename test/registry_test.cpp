/*! \file test/registry_test.cpp
 * \brief Characterize global PackedFunc registry behavior.
 */

#include "kxc/ffi/registry.h"
#include "kxc/relay/op.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define TEST_CHECK(cond, message)                         \
    do {                                                  \
        if (!(cond)) {                                    \
            std::cerr << "[FAIL] " << (message) << '\n'; \
            return false;                                 \
        }                                                 \
    } while (false)

bool TestSetGetAndDuplicateFailure() {
    constexpr const char* kName = "kxc.test.registry.increment";
    kxc::Registry& registry = kxc::Registry::Global();
    registry.Set(kName, kxc::ToPackedFunc([](int value) { return value + 1; }));

    kxc::PackedFunc increment = registry.Get(kName);
    TEST_CHECK(increment.defined(), "registered function should be discoverable");
    TEST_CHECK(increment(41).As<int64_t>() == 42,
               "registered function should preserve its body");
    TEST_CHECK(!registry.Get("kxc.test.registry.missing").defined(),
               "missing registry entry should return an empty function");

    try {
        registry.Set(kName, kxc::ToPackedFunc([](int value) { return value; }));
        TEST_CHECK(false, "duplicate registration should throw");
    } catch (const std::runtime_error&) {
    }
    return true;
}

template <typename Fn>
bool ExpectThrow(Fn&& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

bool TestRelayOperatorRegistryLookupAndSpecs() {
    const kxc::relay::Op& add = kxc::relay::Op::Get("add");
    TEST_CHECK(add->name == "add", "registered op lookup should return canonical op");
    TEST_CHECK(add.has_spec(), "registered op should expose OperatorSpec");
    TEST_CHECK(add.spec().type_relation_key == "FInferType",
               "fluent registration should derive type relation key");
    TEST_CHECK(add.spec().lowering_kind == kxc::relay::OperatorLoweringKind::kSingleTE,
               "fluent registration should derive single TE lowering kind");
    TEST_CHECK(add.spec().lowering_key == "FRelayToTE",
               "fluent registration should derive lowering key");

    TEST_CHECK(kxc::relay::Op::TryGet("definitely_missing_op") == nullptr,
               "TryGet should not create missing operators");
    TEST_CHECK(ExpectThrow([&] { kxc::relay::Op::Get("definitely_missing_op"); }),
               "plain missing op lookup should throw");
    TEST_CHECK(ExpectThrow([&] { kxc::relay::Op::Get("unknown.namespaced_op"); }),
               "namespaced missing op lookup should also throw");

    const kxc::relay::Op& gather = kxc::relay::Op::Get("gather");
    TEST_CHECK(gather->name == "gather" &&
                   gather.spec().attrs_type_key == "GatherAttrs",
               "gather should expose its canonical attrs contract");
    TEST_CHECK(kxc::Registry::Global().Get("kxc.relay.op._make.gather").defined(),
               "gather canonical FFI helper should be registered");
    const std::string gather_attrs = kxc::relay::SerializeAttrs(
        kxc::relay::GatherAttrs::Create(-1));
    TEST_CHECK(Contains(gather_attrs, "GatherAttrsNode") &&
                   Contains(gather_attrs, "axis"),
               "gather attrs serialization should include the canonical axis");

    const kxc::relay::Op& copy = kxc::relay::Op::Get("device.copy");
    TEST_CHECK(copy.has_spec(), "explicitly registered control op should expose metadata");
    TEST_CHECK(copy.spec().effect == kxc::relay::OperatorEffectKind::kDeviceCommunication,
               "device external op should be marked as device communication");
    TEST_CHECK(copy.spec().lowering_kind == kxc::relay::OperatorLoweringKind::kExecPlan,
               "device external op should be routed through exec_plan lowering");

    kxc::relay::CheckOperatorRegistry();
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::Op::Register("kxc.test.registry.after_freeze");
               }),
               "operator registration after registry freeze should throw");
    std::vector<kxc::relay::OperatorSpec> specs = kxc::relay::ListOperatorSpecs();
    TEST_CHECK(!specs.empty(), "registry should list operator specs");

    bool found_add = false;
    for (const auto& spec : specs) {
        if (spec.name == "add") {
            found_add = true;
            std::string serialized = kxc::relay::SerializeOperatorSpec(spec);
            TEST_CHECK(Contains(serialized, "name=add;schema_version=1;category=tensor.math"),
                       "operator spec serialization should be stable and ordered");
            TEST_CHECK(Contains(serialized, "type_relation_key=FInferType"),
                       "operator spec serialization should include type key");
            TEST_CHECK(Contains(serialized, "lowering_kind=single;lowering_key=FRelayToTE"),
                       "operator spec serialization should include lowering key");
        }
    }
    TEST_CHECK(found_add, "ListOperatorSpecs should include add");
    return true;
}

bool TestRelayOperatorDuplicateRegistrationFailure() {
    kxc::relay::OperatorSpec spec;
    spec.name = "kxc.test.registry.duplicate_op";
    spec.category = "test";
    spec.input_arity.num_inputs = 0;
    spec.output_arity = 0;
    kxc::relay::Op::Register(spec);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::Op::Register(spec); }),
               "duplicate Relay op registration should throw");
    return true;
}

}  // namespace

int main() {
    if (!TestSetGetAndDuplicateFailure()) return EXIT_FAILURE;
    if (!TestRelayOperatorDuplicateRegistrationFailure()) return EXIT_FAILURE;
    if (!TestRelayOperatorRegistryLookupAndSpecs()) return EXIT_FAILURE;
    std::cout << "All registry tests passed.\n";
    return EXIT_SUCCESS;
}
