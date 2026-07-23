/*! \file test/adaptive_contract_test.cpp
 * \brief Validates strong exact keys and immutable adaptive artifacts.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kxc/compiler/adaptive.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << '\n'; \
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

class FakeExecutable final
    : public kxc::api::adaptive::ArtifactExecutable {
public:
    FakeExecutable(std::string name, bool ready = true)
        : name_(std::move(name)), ready_(ready) {}

    bool IsReady() const noexcept override { return ready_; }
    std::string DebugName() const override { return name_; }

private:
    std::string name_;
    bool ready_{true};
};

using kxc::api::adaptive::CompileRequest;
using kxc::api::adaptive::DispatchKey;
using kxc::api::adaptive::KernelArtifact;
using kxc::api::adaptive::KernelArtifactKey;
using kxc::api::adaptive::KernelSlotKey;
using kxc::api::adaptive::PlanAbiFingerprint;
using kxc::api::adaptive::RequestKind;

KernelArtifactKey ArtifactKey(std::string slot = "unit:add|target:cpu",
                              std::string artifact = "pipeline:p0|backend:fake-v1") {
    return KernelArtifactKey(KernelSlotKey(std::move(slot)),
                             std::move(artifact));
}

bool TestStrongCanonicalKeys() {
    static_assert(!std::is_convertible_v<std::string, KernelSlotKey>);
    static_assert(!std::is_same_v<KernelSlotKey, PlanAbiFingerprint>);
    static_assert(!std::is_same_v<DispatchKey, PlanAbiFingerprint>);

    const KernelSlotKey first("unit:add|target:cpu");
    const KernelSlotKey same("unit:add|target:cpu");
    const KernelSlotKey other("unit:mul|target:cpu");
    TEST_CHECK(first == same && first != other,
               "canonical key equality must compare complete bytes");
    TEST_CHECK(Throws([] { KernelSlotKey invalid(""); }),
               "empty slot key should fail");
    TEST_CHECK(Throws([] { DispatchKey::Exact(""); }),
               "empty dispatch key should fail");
    TEST_CHECK(Throws([] {
                   KernelArtifactKey invalid(KernelSlotKey("slot"), "");
               }),
               "empty artifact key should fail");
    return true;
}

bool TestFullStructuredArtifactEquality() {
    const KernelArtifactKey first = ArtifactKey();
    const KernelArtifactKey same = ArtifactKey();
    const KernelArtifactKey different_slot =
        ArtifactKey("unit:mul|target:cpu", "pipeline:p0|backend:fake-v1");
    const KernelArtifactKey different_pipeline =
        ArtifactKey("unit:add|target:cpu", "pipeline:p1|backend:fake-v1");
    TEST_CHECK(first == same && first != different_slot &&
                   first != different_pipeline,
               "artifact equality must include slot and canonical compiler fields");

    const DispatchKey shape_a = DispatchKey::Exact("f32[2,3]|contiguous");
    const DispatchKey shape_a_copy =
        DispatchKey::Exact("f32[2,3]|contiguous");
    const DispatchKey shape_b = DispatchKey::Exact("f32[4,3]|contiguous");
    TEST_CHECK(shape_a == shape_a_copy && shape_a != shape_b,
               "dispatch equality must be exact rather than fuzzy");
    return true;
}

bool TestCompileRequestValidationAndSchedulingSeparation() {
    const KernelArtifactKey key = ArtifactKey();
    const DispatchKey dispatch = DispatchKey::Exact("f32[2,3]|contiguous");
    const PlanAbiFingerprint abi("args:input-f32[2,3],output-f32[2,3]");
    const CompileRequest demand(key, dispatch, abi, "model@1",
                                RequestKind::kDemand, 100);
    const CompileRequest prewarm(key, dispatch, abi, "model@1",
                                 RequestKind::kPrewarm, -10);
    TEST_CHECK(demand.artifact_key() == prewarm.artifact_key() &&
                   demand.dispatch_key() == prewarm.dispatch_key() &&
                   demand.required_abi() == prewarm.required_abi() &&
                   demand.kind() != prewarm.kind() &&
                   demand.priority() != prewarm.priority(),
               "scheduling fields must stay separate from compile identity");
    TEST_CHECK(Throws([&] {
                   CompileRequest invalid(key, dispatch, abi, "",
                                          RequestKind::kDemand);
               }),
               "empty model revision should fail");
    return true;
}

bool TestImmutableArtifactOwnsTypedExecutable() {
    auto executable = std::make_shared<const FakeExecutable>("fake-kernel-v1");
    auto artifact = std::make_shared<const KernelArtifact>(
        ArtifactKey(), DispatchKey::Exact("f32[2,3]|contiguous"),
        PlanAbiFingerprint("args:input-f32[2,3],output-f32[2,3]"),
        executable, 4096, "fake-compiler:attempt-1");
    executable.reset();

    TEST_CHECK(artifact->executable() && artifact->executable()->IsReady() &&
                   artifact->executable()->DebugName() == "fake-kernel-v1",
               "artifact must strongly retain its typed executable");
    TEST_CHECK(artifact->byte_size() == 4096 &&
                   artifact->provenance() == "fake-compiler:attempt-1",
               "artifact metadata should be frozen at construction");

    TEST_CHECK(Throws([] {
                   KernelArtifact invalid(
                       ArtifactKey(), DispatchKey::Exact("exact"),
                       PlanAbiFingerprint("abi"), nullptr, 1, "fake");
               }),
               "null executable should fail");
    TEST_CHECK(Throws([] {
                   KernelArtifact invalid(
                       ArtifactKey(), DispatchKey::Exact("exact"),
                       PlanAbiFingerprint("abi"),
                       std::make_shared<const FakeExecutable>("bad", false),
                       1, "fake");
               }),
               "non-ready executable should fail");
    TEST_CHECK(Throws([] {
                   KernelArtifact invalid(
                       ArtifactKey(), DispatchKey::Exact("exact"),
                       PlanAbiFingerprint("abi"),
                       std::make_shared<const FakeExecutable>("ready"),
                       0, "fake");
               }),
               "zero byte size should fail");
    TEST_CHECK(Throws([] {
                   KernelArtifact invalid(
                       ArtifactKey(), DispatchKey::Exact("exact"),
                       PlanAbiFingerprint("abi"),
                       std::make_shared<const FakeExecutable>("ready"),
                       1, "");
               }),
               "missing provenance should fail");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"strong_canonical_keys", TestStrongCanonicalKeys},
        {"full_structured_artifact_equality",
         TestFullStructuredArtifactEquality},
        {"compile_request_validation_and_scheduling_separation",
         TestCompileRequestValidationAndSchedulingSeparation},
        {"immutable_artifact_owns_typed_executable",
         TestImmutableArtifactOwnsTypedExecutable},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
                continue;
            }
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what()
                      << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
