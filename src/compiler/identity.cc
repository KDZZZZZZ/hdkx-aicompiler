/*! \file src/compiler/identity.cc
 * \brief Implements length-delimited compiler identity canonicalization.
 */

#include "kxc/compiler/identity.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "kxc/profiling/profiling.h"

namespace kxc::api {
namespace {

void AppendField(std::string* out, const std::string& name,
                 const std::string& value) {
    *out += std::to_string(name.size()) + ":" + name + "=" +
            std::to_string(value.size()) + ":" + value + ";";
}

void RequireNonEmpty(const std::string& value, const char* field) {
    if (value.empty()) {
        throw std::invalid_argument(std::string("canonical identity requires ") +
                                    field);
    }
}

std::string Digest(const std::string& canonical,
                   std::string index_digest) {
    return index_digest.empty() ? profiling::HashText(canonical)
                                : std::move(index_digest);
}

}  // namespace

std::string GraphValueLocator::CanonicalBytes() const {
    RequireNonEmpty(graph_revision, "graph revision");
    if (value_id < 0 || output_index < 0) {
        throw std::invalid_argument(
            "graph value locator requires non-negative value and output ids");
    }
    std::string canonical;
    AppendField(&canonical, "kind", "graph-value-locator-v1");
    AppendField(&canonical, "graph_revision", graph_revision);
    AppendField(&canonical, "value_id", std::to_string(value_id));
    AppendField(&canonical, "output_index", std::to_string(output_index));
    return canonical;
}

UnitSemanticKey::UnitSemanticKey(std::string canonical_bytes,
                                 std::string index_digest)
    : canonical_bytes_(std::move(canonical_bytes)) {
    RequireNonEmpty(canonical_bytes_, "unit semantic canonical bytes");
    digest_ = Digest(canonical_bytes_, std::move(index_digest));
}

bool UnitSemanticKey::defined() const noexcept {
    return !canonical_bytes_.empty() && !digest_.empty();
}

const std::string& UnitSemanticKey::canonical_bytes() const noexcept {
    return canonical_bytes_;
}

const std::string& UnitSemanticKey::digest() const noexcept { return digest_; }

bool UnitSemanticKey::operator==(const UnitSemanticKey& other) const noexcept {
    return canonical_bytes_ == other.canonical_bytes_;
}

bool UnitSemanticKey::operator!=(const UnitSemanticKey& other) const noexcept {
    return !(*this == other);
}

ArtifactKey::ArtifactKey(UnitSemanticKey unit_semantic_key,
                         std::string target_capability_fingerprint,
                         std::string pipeline_fingerprint, int abi_version,
                         std::string schedule_version,
                         std::string backend_version,
                         std::string index_digest)
    : unit_semantic_key_(std::move(unit_semantic_key)) {
    if (!unit_semantic_key_.defined()) {
        throw std::invalid_argument(
            "artifact identity requires a unit semantic key");
    }
    RequireNonEmpty(target_capability_fingerprint, "target fingerprint");
    RequireNonEmpty(pipeline_fingerprint, "pipeline fingerprint");
    RequireNonEmpty(schedule_version, "schedule version");
    RequireNonEmpty(backend_version, "backend version");
    if (abi_version <= 0) {
        throw std::invalid_argument(
            "artifact identity requires a positive ABI version");
    }
    AppendField(&canonical_bytes_, "kind", "artifact-key-v1");
    AppendField(&canonical_bytes_, "unit_semantic",
                unit_semantic_key_.canonical_bytes());
    AppendField(&canonical_bytes_, "target",
                target_capability_fingerprint);
    AppendField(&canonical_bytes_, "pipeline", pipeline_fingerprint);
    AppendField(&canonical_bytes_, "abi", std::to_string(abi_version));
    AppendField(&canonical_bytes_, "schedule", schedule_version);
    AppendField(&canonical_bytes_, "backend", backend_version);
    digest_ = Digest(canonical_bytes_, std::move(index_digest));
}

bool ArtifactKey::defined() const noexcept {
    return unit_semantic_key_.defined() && !canonical_bytes_.empty() &&
           !digest_.empty();
}

const UnitSemanticKey& ArtifactKey::unit_semantic_key() const noexcept {
    return unit_semantic_key_;
}

const std::string& ArtifactKey::canonical_bytes() const noexcept {
    return canonical_bytes_;
}

const std::string& ArtifactKey::digest() const noexcept { return digest_; }

bool ArtifactKey::operator==(const ArtifactKey& other) const noexcept {
    return canonical_bytes_ == other.canonical_bytes_;
}

bool ArtifactKey::operator!=(const ArtifactKey& other) const noexcept {
    return !(*this == other);
}

DispatchKey::DispatchKey(std::string artifact_family,
                         std::string shape_layout_valid_extent,
                         std::string variant_policy_version,
                         std::string index_digest) {
    RequireNonEmpty(artifact_family, "artifact family");
    RequireNonEmpty(shape_layout_valid_extent,
                    "shape/layout/valid-extent contract");
    RequireNonEmpty(variant_policy_version, "variant policy version");
    AppendField(&canonical_bytes_, "kind", "dispatch-key-v1");
    AppendField(&canonical_bytes_, "artifact_family", artifact_family);
    AppendField(&canonical_bytes_, "applicability",
                shape_layout_valid_extent);
    AppendField(&canonical_bytes_, "variant_policy",
                variant_policy_version);
    digest_ = Digest(canonical_bytes_, std::move(index_digest));
}

bool DispatchKey::defined() const noexcept {
    return !canonical_bytes_.empty() && !digest_.empty();
}

const std::string& DispatchKey::canonical_bytes() const noexcept {
    return canonical_bytes_;
}

const std::string& DispatchKey::digest() const noexcept { return digest_; }

bool DispatchKey::operator==(const DispatchKey& other) const noexcept {
    return canonical_bytes_ == other.canonical_bytes_;
}

PlanVariantKey::PlanVariantKey(
    std::string graph_template_revision,
    std::vector<std::pair<std::string, uint64_t>> artifact_generations,
    std::string concrete_profile, std::string memory_plan_version,
    std::string index_digest) {
    RequireNonEmpty(graph_template_revision, "graph template revision");
    RequireNonEmpty(concrete_profile, "concrete profile");
    RequireNonEmpty(memory_plan_version, "memory plan version");
    if (artifact_generations.empty()) {
        throw std::invalid_argument(
            "plan variant identity requires selected artifact generations");
    }
    AppendField(&canonical_bytes_, "kind", "plan-variant-key-v1");
    AppendField(&canonical_bytes_, "graph_template",
                graph_template_revision);
    for (const auto& artifact : artifact_generations) {
        RequireNonEmpty(artifact.first, "selected artifact identity");
        AppendField(&canonical_bytes_, "artifact", artifact.first);
        AppendField(&canonical_bytes_, "generation",
                    std::to_string(artifact.second));
    }
    AppendField(&canonical_bytes_, "profile", concrete_profile);
    AppendField(&canonical_bytes_, "memory_plan", memory_plan_version);
    digest_ = Digest(canonical_bytes_, std::move(index_digest));
}

bool PlanVariantKey::defined() const noexcept {
    return !canonical_bytes_.empty() && !digest_.empty();
}

const std::string& PlanVariantKey::canonical_bytes() const noexcept {
    return canonical_bytes_;
}

const std::string& PlanVariantKey::digest() const noexcept { return digest_; }

bool PlanVariantKey::operator==(const PlanVariantKey& other) const noexcept {
    return canonical_bytes_ == other.canonical_bytes_;
}

}  // namespace kxc::api
