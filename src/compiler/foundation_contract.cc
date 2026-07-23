/*! \file src/compiler/foundation_contract.cc
 * \brief Implements compiler-foundation DTO validation and canonical requests.
 */

#include "kxc/compiler/foundation_contract.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace kxc::api {
namespace {

void AppendField(std::string* canonical, const std::string& name,
                 const std::string& value) {
    *canonical += std::to_string(name.size()) + ":" + name + "=" +
                  std::to_string(value.size()) + ":" + value + ";";
}

}  // namespace

ArtifactHandle::ArtifactHandle(ArtifactRecord record) {
    if (!record.artifact_key.defined() || record.executable_token.empty() ||
        record.signature_digest.empty() ||
        record.launch_metadata_digest.empty() || record.provenance.empty() ||
        record.byte_size == 0 || record.validation_record.empty()) {
        throw std::invalid_argument(
            "ArtifactHandle requires a complete validated ready artifact");
    }
    record_ = std::make_shared<const ArtifactRecord>(std::move(record));
}

bool ArtifactHandle::defined() const noexcept { return record_ != nullptr; }

const ArtifactRecord& ArtifactHandle::record() const {
    if (!record_) throw std::logic_error("ArtifactHandle is undefined");
    return *record_;
}

ArtifactPin::ArtifactPin(ArtifactHandle handle) : handle_(std::move(handle)) {
    if (!handle_.defined()) {
        throw std::invalid_argument("ArtifactPin requires an ArtifactHandle");
    }
}

bool ArtifactPin::defined() const noexcept { return handle_.defined(); }

const ArtifactHandle& ArtifactPin::handle() const {
    if (!handle_.defined()) throw std::logic_error("ArtifactPin is undefined");
    return handle_;
}

std::string CompileRequest::CanonicalSingleflightKey() const {
    if (!artifact_key.defined() || request_origin.empty() ||
        cancellation.id.empty()) {
        throw std::invalid_argument(
            "CompileRequest requires artifact identity, origin, and cancellation id");
    }
    std::string canonical;
    AppendField(&canonical, "kind", "compile-request-v1");
    AppendField(&canonical, "artifact", artifact_key.canonical_bytes());
    AppendField(&canonical, "dispatch",
                dispatch_key ? dispatch_key->canonical_bytes() : "static-exact");
    return canonical;
}

void FrozenPlanInput::Validate() const {
    if (graph_template_locator.empty() || selected_artifacts.empty() ||
        logical_physical_valid_extent_contract.empty() ||
        memory_plan_version.empty()) {
        throw std::invalid_argument(
            "FrozenPlanInput requires graph, artifacts, value contract, and memory version");
    }
    if (logical_physical_valid_extent_contract.find("-1") !=
        std::string::npos) {
        throw std::invalid_argument(
            "FrozenPlanInput cannot use legacy -1 as a shape contract");
    }
    for (const GraphValueLocator& locator : value_routing) {
        (void)locator.CanonicalBytes();
    }
    for (const SelectedArtifact& selected : selected_artifacts) {
        if (!selected.artifact_pin.defined() ||
            selected.applicability_proof.empty() ||
            selected.signature_digest.empty() ||
            selected.signature_digest !=
                selected.artifact_pin.handle().record().signature_digest) {
            throw std::invalid_argument(
                "FrozenPlanInput selected artifact proof/signature is invalid");
        }
    }
}

}  // namespace kxc::api
