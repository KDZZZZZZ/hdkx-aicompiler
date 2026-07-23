/*! \file test/compiler_identity_test.cpp
 * \brief Verifies canonical compiler identity separation and full equality.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/identity.h"

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

bool TestGraphLocatorIsNotUnitSemantics() {
    using namespace kxc::api;
    const GraphValueLocator first{"graph-a", 7, 0};
    const GraphValueLocator second{"graph-a", 8, 0};
    TEST_CHECK(first.CanonicalBytes() != second.CanonicalBytes(),
               "graph routing locators must retain value ids");

    const UnitSemanticKey semantic("op=add;inputs=f32[4],f32[4]");
    TEST_CHECK(semantic.defined() &&
                   semantic.canonical_bytes().find("graph-a") ==
                       std::string::npos,
               "unit semantic key must be constructed without graph locator data");
    TEST_CHECK(Throws([] {
                   (void)GraphValueLocator{"", -1, 0}.CanonicalBytes();
               }),
               "invalid plan-local locators must fail closed");
    return true;
}

bool TestDigestCollisionUsesCanonicalEquality() {
    using namespace kxc::api;
    const UnitSemanticKey first("canonical-unit-a", "forced-collision");
    const UnitSemanticKey second("canonical-unit-b", "forced-collision");
    TEST_CHECK(first.digest() == second.digest() && first != second,
               "digest is only an index; full canonical bytes decide equality");

    const ArtifactKey first_artifact(first, "cpu-v1", "relay-tir-o2", 1,
                                     "schedule-v1", "llvm-v1",
                                     "artifact-collision");
    const ArtifactKey second_artifact(second, "cpu-v1", "relay-tir-o2", 1,
                                      "schedule-v1", "llvm-v1",
                                      "artifact-collision");
    TEST_CHECK(first_artifact.digest() == second_artifact.digest() &&
                   first_artifact != second_artifact,
               "artifact lookup must compare complete canonical keys");
    return true;
}

bool TestEveryArtifactSemanticFieldCausesSafeMiss() {
    using namespace kxc::api;
    const UnitSemanticKey unit("unit-a");
    const ArtifactKey baseline(unit, "cpu-v1", "pipeline-a", 1,
                               "schedule-a", "backend-a");
    const std::vector<ArtifactKey> changed = {
        ArtifactKey(UnitSemanticKey("unit-b"), "cpu-v1", "pipeline-a", 1,
                    "schedule-a", "backend-a"),
        ArtifactKey(unit, "cuda-v1", "pipeline-a", 1, "schedule-a",
                    "backend-a"),
        ArtifactKey(unit, "cpu-v1", "pipeline-b", 1, "schedule-a",
                    "backend-a"),
        ArtifactKey(unit, "cpu-v1", "pipeline-a", 2, "schedule-a",
                    "backend-a"),
        ArtifactKey(unit, "cpu-v1", "pipeline-a", 1, "schedule-b",
                    "backend-a"),
        ArtifactKey(unit, "cpu-v1", "pipeline-a", 1, "schedule-a",
                    "backend-b"),
    };
    for (const ArtifactKey& candidate : changed) {
        TEST_CHECK(candidate != baseline,
                   "unit/target/pipeline/ABI/schedule/backend change must miss");
    }
    const ArtifactKey same(unit, "cpu-v1", "pipeline-a", 1,
                           "schedule-a", "backend-a");
    TEST_CHECK(same == baseline && same.digest() == baseline.digest(),
               "identical canonical artifact fields must be deterministic");
    return true;
}

bool TestDispatchAndPlanVariantRemainSeparate() {
    using namespace kxc::api;
    const DispatchKey exact("unit-a/cpu", "shape=[4];layout=contiguous",
                            "exact-v1");
    const DispatchKey bucket("unit-a/cpu", "shape=[1..8];valid=[4]",
                             "bucket-v1");
    TEST_CHECK(exact.defined() && bucket.defined() && !(exact == bucket),
               "dispatch applicability must not collapse into primitive semantics");

    const PlanVariantKey first("graph-template-a", {{"artifact-a", 0}},
                               "shape=[4]", "memory-plan-v1");
    const PlanVariantKey next_generation(
        "graph-template-a", {{"artifact-a", 1}}, "shape=[4]",
        "memory-plan-v1");
    TEST_CHECK(first.defined() && next_generation.defined() &&
                   !(first == next_generation),
               "selected immutable generation is part of plan variant identity");
    TEST_CHECK(Throws([] {
                   (void)PlanVariantKey("graph", {}, "shape=[4]", "memory-v1");
               }),
               "a frozen plan variant requires selected artifacts");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"locator_is_not_semantics", TestGraphLocatorIsNotUnitSemantics},
        {"digest_collision_full_equality", TestDigestCollisionUsesCanonicalEquality},
        {"artifact_field_safe_miss", TestEveryArtifactSemanticFieldCausesSafeMiss},
        {"dispatch_and_plan_are_separate", TestDispatchAndPlanVariantRemainSeparate},
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
