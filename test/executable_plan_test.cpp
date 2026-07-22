/*! \file test/executable_plan_test.cpp
 * \brief Verifies runtime-neutral ExecutablePlan invariants and call ordering.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/runtime/executable_plan.h"
#include "../src/runtime/internal/memory_plan.h"

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

DLDataType Float32() {
    return DLDataType{kDLFloat, 32, 1};
}

kxc::Array<kxc::runtime::ValueSpec> ValidValues() {
    using namespace kxc;
    using namespace kxc::runtime;
    return {
        ValueSpec(0, 0, {2, 3}, Float32(), Device::CPU(), true),
        ValueSpec(1, 1, {3}, Float32(), Device::CPU(), false, true),
        ValueSpec(2, 2, {2, 3}, Float32(), Device::CPU()),
        ValueSpec(3, 3, {2, 3}, Float32(), Device::CPU(), false, false, true),
    };
}

bool TestValidPlan() {
    using namespace kxc;
    using namespace kxc::runtime;
    Array<ValueSpec> source_values = ValidValues();
    Array<KernelCall> source_calls = {
        KernelCall("unit_0", {0, 1}, {2}),
        KernelCall("unit_1", {2}, {3}),
    };
    ExecutablePlan plan(source_values, source_calls, {0}, {1}, {3});
    source_values.push_back(source_values[0]);
    source_calls.push_back(KernelCall("unrelated", {0}, {2}));

    TEST_CHECK(plan.values().size() == 4 && plan.calls().size() == 2,
               "plan must deep-copy constructor arrays");
    TEST_CHECK(plan.calls()[0]->symbol == "unit_0" &&
                   plan.calls()[1]->symbol == "unit_1",
               "kernel call order must be stable");
    TEST_CHECK(plan.values()[1]->is_constant,
               "constant role must be preserved");
    TEST_CHECK(plan.constant_value_ids().size() == 1 &&
                   plan.constant_value_ids()[0] == 1,
               "constant value order must be explicit and stable");
    TEST_CHECK(plan.get()->GetTypeKey() == "kxc.runtime.ExecutablePlanNode",
               "executable plan type key must be stable");
    return true;
}

bool TestInvalidValueIdsAndMetadata() {
    using namespace kxc;
    using namespace kxc::runtime;
    TEST_CHECK(Throws([] {
                   ValueSpec(-1, -1, {1}, Float32(), Device::CPU(), true);
               }),
               "negative value id must fail");
    TEST_CHECK(Throws([] {
                   ValueSpec(0, -1, {1}, Float32(), Device::CPU(), true);
               }),
               "negative storage id must fail");
    TEST_CHECK(Throws([] {
                   ValueSpec(0, 0, {1}, DLDataType{}, Device::CPU(), true);
               }),
               "missing dtype metadata must fail");
    TEST_CHECK(Throws([] {
                   ValueSpec(0, 0, {1}, Float32(), Device(), true);
               }),
               "missing device metadata must fail");

    auto* missing_shape = new ValueSpecNode();
    missing_shape->value_id = 0;
    missing_shape->storage_id = 0;
    missing_shape->dtype = Float32();
    missing_shape->device = Device::CPU();
    missing_shape->is_input = true;
    ObjectRef raw(missing_shape);
    TEST_CHECK(Throws([&] { ValueSpec value(raw); }),
               "missing shape metadata must fail without rejecting scalar shape");
    return true;
}

bool TestMemoryReuseAndSharingGuards() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Array<ValueSpec> values = {
        ValueSpec(0, 0, {4}, Float32(), Device::CPU(), true),
        ValueSpec(1, 1, {4}, Float32(), Device::CPU()),
        ValueSpec(2, 2, {4}, Float32(), Device::CPU()),
        ValueSpec(3, 3, {4}, Float32(), Device::CPU()),
        ValueSpec(4, 4, {4}, Float32(), Device::CPU(), false, false, true),
    };
    const Array<KernelCall> calls = {
        KernelCall("unit_0", {0}, {1}),
        KernelCall("unit_1", {1}, {2}),
        KernelCall("unit_2", {2}, {3}),
        KernelCall("unit_3", {3}, {4}),
    };
    const ExecutablePlan planned = internal::PlanMemory(
        ExecutablePlan(values, calls, {0}, {}, {4}));
    TEST_CHECK(planned.values()[1]->storage_id ==
                   planned.values()[3]->storage_id &&
                   planned.values()[2]->storage_id !=
                       planned.values()[3]->storage_id &&
                   planned.values()[4]->storage_id == 4,
               "last-use planning should reuse only non-overlapping intermediates");

    const ExecutablePlan guarded = internal::PlanMemory(ExecutablePlan(
        {ValueSpec(0, 0, {4}, Float32(), Device::CPU(), true),
         ValueSpec(1, 1, {4}, Float32(), Device::CPU(), false, false,
                   false, true, false),
         ValueSpec(2, 2, {4}, Float32(), Device::CPU()),
         ValueSpec(3, 3, {4}, Float32(), Device::CPU(), false, false,
                   false, false, true),
         ValueSpec(4, 4, {4}, Float32(), Device::CPU()),
         ValueSpec(5, 5, {4}, Float32(), Device::CPU(), false, false, true)},
        {KernelCall("g0", {0}, {1}), KernelCall("g1", {1}, {2}),
         KernelCall("g2", {2}, {3}), KernelCall("g3", {3}, {4}),
         KernelCall("g4", {4}, {5})},
        {0}, {}, {5}));
    TEST_CHECK(guarded.values()[1]->storage_id == 1 &&
                   guarded.values()[3]->storage_id == 3,
               "alias and async-live intermediates must keep dedicated storage");

    TEST_CHECK(Throws([&] {
                   ExecutablePlan invalid(
                       {ValueSpec(0, 0, {4}, Float32(), Device::CPU(), true),
                        ValueSpec(1, 7, {4}, Float32(), Device::CPU()),
                        ValueSpec(2, 7, {4}, Float32(), Device::CPU()),
                        ValueSpec(3, 3, {4}, Float32(), Device::CPU(), false,
                                  false, true)},
                       {KernelCall("a", {0}, {1}),
                        KernelCall("b", {1}, {2}),
                        KernelCall("c", {2}, {3})},
                       {0}, {}, {3});
               }),
               "values live in the same call must not share storage");
    TEST_CHECK(Throws([&] {
                   ExecutablePlan invalid(
                       {ValueSpec(0, 9, {4}, Float32(), Device::CPU(), true),
                        ValueSpec(1, 9, {4}, Float32(), Device::CPU()),
                        ValueSpec(2, 2, {4}, Float32(), Device::CPU(), false,
                                  false, true)},
                       {KernelCall("a", {0}, {1}),
                        KernelCall("b", {1}, {2})},
                       {0}, {}, {2});
               }),
               "graph inputs must not share a storage slot");
    return true;
}

bool TestDuplicateProducerAndUndefinedInput() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Array<ValueSpec> values = ValidValues();
    TEST_CHECK(Throws([&] {
                   ExecutablePlan plan(
                       values,
                       {KernelCall("unit_0", {0, 1}, {2}),
                        KernelCall("unit_1", {2}, {2}),
                        KernelCall("unit_2", {2}, {3})},
                       {0}, {1}, {3});
               }),
               "duplicate producers must fail");
    TEST_CHECK(Throws([&] {
                   ExecutablePlan plan(
                       values,
                       {KernelCall("unit_0", {99}, {2}),
                        KernelCall("unit_1", {2}, {3})},
                       {0}, {1}, {3});
               }),
               "undefined call input must fail");
    return true;
}

bool TestCallOrderAndMissingOutput() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Array<ValueSpec> values = ValidValues();
    TEST_CHECK(Throws([&] {
                   ExecutablePlan plan(
                       values,
                       {KernelCall("unit_1", {2}, {3}),
                        KernelCall("unit_0", {0, 1}, {2})},
                       {0}, {1}, {3});
               }),
               "consumer before producer must fail");
    TEST_CHECK(Throws([&] {
                   ExecutablePlan plan(
                       values, {KernelCall("unit_0", {0, 1}, {2})}, {0}, {1},
                       {3});
               }),
               "missing graph-output producer must fail");

    Array<ValueSpec> passthrough_values = {
        ValueSpec(0, 0, {2, 3}, Float32(), Device::CPU(), true, false, true),
    };
    ExecutablePlan passthrough(passthrough_values, {}, {0}, {}, {0});
    TEST_CHECK(passthrough.calls().empty() && passthrough.output_value_ids()[0] == 0,
               "a graph input may also be a direct graph output");
    return true;
}

bool TestRoleListsAndObjectTypeChecks() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Array<ValueSpec> values = ValidValues();
    const Array<KernelCall> calls = {
        KernelCall("unit_0", {0, 1}, {2}),
        KernelCall("unit_1", {2}, {3}),
    };
    TEST_CHECK(Throws([&] {
                   ExecutablePlan plan(values, calls, {}, {1}, {3});
               }),
               "ordered input list must cover all input values");
    TEST_CHECK(Throws([&] {
                   ExecutablePlan plan(values, calls, {0}, {1}, {2});
               }),
               "ordered output list must contain graph outputs only");
    TEST_CHECK(Throws([&] {
                   ExecutablePlan plan(values, calls, {0}, {}, {3});
               }),
               "ordered constant list must cover all constant values");
    TEST_CHECK(Throws([&] { ValueSpec wrong(ObjectRef(Device::CPU())); }),
               "wrong ObjectRef type must fail safely");
    TEST_CHECK(Throws([&] {
                   ExecutablePlan undefined{ObjectRef()};
                   (void)undefined.values();
               }),
               "undefined ExecutablePlan must fail when accessed");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"valid_plan", TestValidPlan},
        {"invalid_value_ids_and_metadata", TestInvalidValueIdsAndMetadata},
        {"duplicate_producer_and_undefined_input",
         TestDuplicateProducerAndUndefinedInput},
        {"call_order_and_missing_output", TestCallOrderAndMissingOutput},
        {"role_lists_and_object_type_checks", TestRoleListsAndObjectTypeChecks},
        {"memory_reuse_and_sharing_guards", TestMemoryReuseAndSharingGuards},
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
