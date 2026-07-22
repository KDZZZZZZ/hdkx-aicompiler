/*! \file test/registry_test.cpp
 * \brief Characterize global PackedFunc registry behavior.
 */

#include "kxc/ffi/registry.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

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

}  // namespace

int main() {
    if (!TestSetGetAndDuplicateFailure()) return EXIT_FAILURE;
    std::cout << "All registry tests passed.\n";
    return EXIT_SUCCESS;
}
