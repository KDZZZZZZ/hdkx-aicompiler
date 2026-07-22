/*! \file test/packed_func_test.cpp
 * \brief Characterize PackedFunc argument conversion, return values, and failures.
 */

#include "kxc/ffi/packed_func.h"

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

bool TestTypedCall() {
    kxc::PackedFunc add = kxc::ToPackedFunc(
        [](int lhs, int rhs) { return lhs + rhs; });
    TEST_CHECK(add(19, 23).As<int64_t>() == 42,
               "typed integer arguments should round-trip through PackedFunc");

    kxc::PackedFunc join = kxc::ToPackedFunc(
        [](std::string lhs, std::string rhs) { return lhs + rhs; });
    TEST_CHECK(join(std::string("k"), std::string("xc")).As<std::string>() == "kxc",
               "string arguments and results should retain their storage");
    return true;
}

bool TestFailureContracts() {
    kxc::PackedFunc add = kxc::ToPackedFunc(
        [](int lhs, int rhs) { return lhs + rhs; });
    try {
        (void)add(1);
        TEST_CHECK(false, "arity mismatch should throw");
    } catch (const std::runtime_error&) {
    }

    try {
        (void)add(std::string("wrong"), 1);
        TEST_CHECK(false, "type mismatch should throw");
    } catch (const std::runtime_error&) {
    }

    try {
        kxc::PackedFunc empty;
        (void)empty();
        TEST_CHECK(false, "empty PackedFunc invocation should throw");
    } catch (const std::runtime_error&) {
    }
    return true;
}

}  // namespace

int main() {
    if (!TestTypedCall()) return EXIT_FAILURE;
    if (!TestFailureContracts()) return EXIT_FAILURE;
    std::cout << "All packed func tests passed.\n";
    return EXIT_SUCCESS;
}
