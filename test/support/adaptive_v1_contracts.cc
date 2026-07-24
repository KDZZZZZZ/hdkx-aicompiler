/*! \file test/support/adaptive_v1_contracts.cc
 * \brief Implements static-exact adaptive identity and artifact contracts.
 */

#include "adaptive_v1.h"

#include <stdexcept>
#include <utility>

namespace kxc::api::experimental::adaptive::v1 {

DispatchKey DispatchKey::Exact(std::string canonical) {
    return DispatchKey(std::move(canonical));
}

DispatchKey::DispatchKey(std::string canonical)
    : canonical_(std::move(canonical)) {
    if (canonical_.empty()) {
        throw std::invalid_argument(
            "exact dispatch canonical key must not be empty");
    }
}

KernelArtifactKey::KernelArtifactKey(KernelSlotKey slot_key,
                                     std::string canonical)
    : slot_key_(std::move(slot_key)), canonical_(std::move(canonical)) {
    if (canonical_.empty()) {
        throw std::invalid_argument(
            "artifact canonical key must not be empty");
    }
}

CompileRequest::CompileRequest(
    KernelArtifactKey artifact_key, DispatchKey dispatch_key,
    PlanAbiFingerprint required_abi, std::string model_revision,
    RequestKind kind, int priority,
    std::chrono::steady_clock::time_point deadline)
    : artifact_key_(std::move(artifact_key)),
      dispatch_key_(std::move(dispatch_key)),
      required_abi_(std::move(required_abi)),
      model_revision_(std::move(model_revision)),
      kind_(kind),
      priority_(priority),
      deadline_(deadline) {
    if (model_revision_.empty()) {
        throw std::invalid_argument(
            "compile request model revision must not be empty");
    }
    switch (kind_) {
        case RequestKind::kDemand:
        case RequestKind::kCanary:
        case RequestKind::kPrewarm:
            return;
    }
    throw std::invalid_argument("compile request kind is invalid");
}

KernelArtifact::KernelArtifact(
    KernelArtifactKey key, DispatchKey applicability,
    PlanAbiFingerprint compatible_abi,
    std::shared_ptr<const ArtifactExecutable> executable,
    std::size_t byte_size, std::string provenance)
    : key_(std::move(key)),
      applicability_(std::move(applicability)),
      compatible_abi_(std::move(compatible_abi)),
      executable_(std::move(executable)),
      byte_size_(byte_size),
      provenance_(std::move(provenance)) {
    if (!executable_ || !executable_->IsReady()) {
        throw std::invalid_argument(
            "kernel artifact requires a ready typed executable");
    }
    if (byte_size_ == 0) {
        throw std::invalid_argument(
            "kernel artifact byte size must be non-zero");
    }
    if (provenance_.empty()) {
        throw std::invalid_argument(
            "kernel artifact provenance must not be empty");
    }
}

ArtifactValidationToken::ArtifactValidationToken(std::string opaque)
    : opaque_(std::move(opaque)) {
    if (opaque_.empty()) {
        throw std::invalid_argument(
            "artifact validation token must not be empty");
    }
}

ArtifactValidationRecord::ArtifactValidationRecord(
    std::string record_id, KernelArtifactKey artifact_key,
    DispatchKey dispatch_key, PlanAbiFingerprint compatible_abi,
    std::size_t artifact_bytes, ArtifactValidationToken token)
    : record_id_(std::move(record_id)),
      artifact_key_(std::move(artifact_key)),
      dispatch_key_(std::move(dispatch_key)),
      compatible_abi_(std::move(compatible_abi)),
      artifact_bytes_(artifact_bytes),
      token_(std::move(token)) {
    if (record_id_.empty() || artifact_bytes_ == 0) {
        throw std::invalid_argument(
            "artifact validation record requires an id and byte size");
    }
}

GenerationHealthToken::GenerationHealthToken(std::string opaque)
    : opaque_(std::move(opaque)) {
    if (opaque_.empty()) {
        throw std::invalid_argument(
            "generation health token must not be empty");
    }
}

GenerationHealthRecord::GenerationHealthRecord(
    std::string record_id, KernelSlotKey slot_key,
    KernelArtifactKey artifact_key, DispatchKey dispatch_key,
    PlanAbiFingerprint compatible_abi, Generation generation,
    GenerationHealthDisposition disposition, std::string evidence_id,
    GenerationHealthToken token)
    : record_id_(std::move(record_id)),
      slot_key_(std::move(slot_key)),
      artifact_key_(std::move(artifact_key)),
      dispatch_key_(std::move(dispatch_key)),
      compatible_abi_(std::move(compatible_abi)),
      generation_(generation),
      disposition_(disposition),
      evidence_id_(std::move(evidence_id)),
      token_(std::move(token)) {
    if (record_id_.empty() || generation_ == 0 || evidence_id_.empty()) {
        throw std::invalid_argument(
            "generation health record requires ids and a generation");
    }
    switch (disposition_) {
        case GenerationHealthDisposition::kHealthy:
        case GenerationHealthDisposition::kQuarantined:
            return;
    }
    throw std::invalid_argument(
        "generation health record disposition is invalid");
}

CompileAttempt CompileAttempt::Ready(
    std::shared_ptr<const KernelArtifact> artifact) {
    if (!artifact) {
        throw std::invalid_argument(
            "ready compile attempt requires an artifact");
    }
    return CompileAttempt(std::move(artifact),
                          CompileFailureCategory::kDeterministic, "");
}

CompileAttempt CompileAttempt::Failed(CompileFailureCategory category,
                                      std::string diagnostic) {
    if (diagnostic.empty()) {
        throw std::invalid_argument(
            "failed compile attempt requires a diagnostic");
    }
    return CompileAttempt(nullptr, category, std::move(diagnostic));
}

CompileAttempt::CompileAttempt(
    std::shared_ptr<const KernelArtifact> artifact,
    CompileFailureCategory category, std::string diagnostic)
    : artifact_(std::move(artifact)),
      failure_category_(category),
      diagnostic_(std::move(diagnostic)) {}

CompileResult CompileResult::Ready(
    std::shared_ptr<const KernelArtifact> artifact,
    std::shared_ptr<const ArtifactValidationRecord> validation,
    std::uint32_t attempt) {
    if (!artifact || !validation || attempt == 0) {
        throw std::invalid_argument(
            "ready compile result requires an artifact, validation, and attempt");
    }
    return CompileResult(CompileStatus::kReady, std::move(artifact),
                         std::move(validation),
                         CompileFailureCategory::kDeterministic, "", attempt,
                         false, {});
}

CompileResult CompileResult::Failure(
    CompileStatus status, CompileFailureCategory category,
    std::string diagnostic, std::uint32_t attempt, bool retryable,
    std::chrono::steady_clock::time_point retry_after) {
    if (status == CompileStatus::kReady || diagnostic.empty()) {
        throw std::invalid_argument(
            "failure result requires non-ready status and diagnostic");
    }
    return CompileResult(status, nullptr, nullptr, category,
                         std::move(diagnostic), attempt, retryable,
                         retry_after);
}

CompileResult::CompileResult(
    CompileStatus status, std::shared_ptr<const KernelArtifact> artifact,
    std::shared_ptr<const ArtifactValidationRecord> validation,
    CompileFailureCategory failure_category, std::string diagnostic,
    std::uint32_t attempt, bool retryable,
    std::chrono::steady_clock::time_point retry_after)
    : status_(status),
      artifact_(std::move(artifact)),
      validation_(std::move(validation)),
      failure_category_(failure_category),
      diagnostic_(std::move(diagnostic)),
      attempt_(attempt),
      retryable_(retryable),
      retry_after_(retry_after) {}

}  // namespace kxc::api::experimental::adaptive::v1
