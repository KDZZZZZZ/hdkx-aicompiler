/*! \file src/compiler/identity.cc
 * \brief Implements length-delimited compiler identity canonicalization.
 */

#include "kxc/compiler/identity.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../runtime/internal/compiled_module_node.h"
#include "kxc/profiling/profiling.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"

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

template <typename T>
void AppendInteger(std::string* out, const std::string& name, T value) {
    AppendField(out, name, std::to_string(static_cast<int64_t>(value)));
}

void AppendShape(std::string* out, const Array<int64_t>& shape,
                 const char* context) {
    AppendInteger(out, "rank", shape.size());
    for (int64_t dimension : shape) {
        if (dimension < 0) {
            throw std::invalid_argument(std::string(context) +
                                        " requires static exact dimensions");
        }
        AppendInteger(out, "dimension", dimension);
    }
}

void AppendValueContract(std::string* out,
                         const runtime::ValueSpec& value) {
    // Graph-local value/storage ids are locators, not reusable ABI identity.
    AppendInteger(out, "dtype_code", value->dtype.code);
    AppendInteger(out, "dtype_bits", value->dtype.bits);
    AppendInteger(out, "dtype_lanes", value->dtype.lanes);
    AppendInteger(out, "device_type", value->device.device_type());
    AppendInteger(out, "device_id", value->device.device_id());
    AppendInteger(out, "is_input", value->is_input);
    AppendInteger(out, "is_constant", value->is_constant);
    AppendInteger(out, "is_output", value->is_output);
    AppendInteger(out, "is_alias", value->is_alias);
    AppendInteger(out, "is_async_live", value->is_async_live);
    AppendShape(out, value.shape(), "plan ABI");
}

const Target& ModuleTarget(const CompiledModule& module) {
    const auto* node = module.As<CompiledModuleNode>();
    if (!node) {
        throw std::invalid_argument(
            "plan ABI requires a valid CompiledModule");
    }
    return node->target_;
}

void AppendTargetContract(std::string* out, const Target& target) {
    if (!target.defined() || !target.As<TargetNode>()) {
        throw std::invalid_argument("plan ABI requires a defined Target");
    }
    const TargetNode* node = target.operator->();
    AppendField(out, "target_kind", node->kind);
    AppendInteger(out, "target_device_type", node->device_type);
    AppendInteger(out, "target_device_id", node->device_id);
    AppendInteger(out, "target_exists", node->attrs.exists);
    AppendInteger(out, "target_max_threads_block",
                  node->attrs.max_threads_per_block);
    AppendInteger(out, "target_warp", node->attrs.warp_size);
    AppendInteger(out, "target_shared_mem_block",
                  node->attrs.max_shared_memory_per_block);
    AppendField(out, "target_compute_version",
                node->attrs.compute_version);
    AppendField(out, "target_device_name", node->attrs.device_name);
    AppendInteger(out, "target_max_clock_khz",
                  node->attrs.max_clock_rate_khz);
    AppendInteger(out, "target_max_registers_block",
                  node->attrs.max_registers_per_block);
    AppendInteger(out, "target_api_version", node->attrs.api_version);
    AppendInteger(out, "target_driver_version",
                  node->attrs.driver_version);
    AppendInteger(out, "target_l2_bytes", node->attrs.l2_cache_size_bytes);
    AppendInteger(out, "target_global_bytes",
                  node->attrs.total_global_memory);
    AppendInteger(out, "target_shared_mem_sm",
                  node->attrs.max_shared_memory_per_multiprocessor);
    AppendInteger(out, "target_registers_sm",
                  node->attrs.max_registers_per_multiprocessor);
    AppendInteger(out, "target_threads_sm",
                  node->attrs.max_threads_per_multiprocessor);
    AppendInteger(out, "target_compute_major",
                  node->attrs.compute_version_major);
    AppendInteger(out, "target_compute_minor",
                  node->attrs.compute_version_minor);
    AppendInteger(out, "target_multiprocessors",
                  node->attrs.multi_processor_count);
    AppendField(out, "target_arch", node->attrs.arch);
}

void AppendArrayContract(std::string* out, const runtime::NDArray& value) {
    if (!value.defined()) {
        throw std::invalid_argument("plan ABI constant must be defined");
    }
    const DLDataType dtype = value.dtype();
    AppendInteger(out, "constant_dtype_code", dtype.code);
    AppendInteger(out, "constant_dtype_bits", dtype.bits);
    AppendInteger(out, "constant_dtype_lanes", dtype.lanes);
    AppendInteger(out, "constant_device_type", value.device().device_type());
    AppendInteger(out, "constant_device_id", value.device().device_id());
    AppendShape(out, value.shape(), "plan ABI constant");
    std::string bytes(value.NBytes(), '\0');
    if (!bytes.empty()) value.CopyToBytes(bytes.data(), bytes.size());
    AppendField(out, "constant_bytes", bytes);
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

std::string LinkSymbol::CanonicalBytes() const {
    RequireNonEmpty(value, "link symbol");
    std::string canonical;
    AppendField(&canonical, "kind", "link-symbol-v1");
    AppendField(&canonical, "symbol", value);
    return canonical;
}

std::string StorageId::CanonicalBytes() const {
    if (value < 0) {
        throw std::invalid_argument("storage id must be non-negative");
    }
    std::string canonical;
    AppendField(&canonical, "kind", "plan-storage-id-v1");
    AppendField(&canonical, "storage_id", std::to_string(value));
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

std::string OrderedArtifactIdentity::CanonicalBytes() const {
    if (!artifact_key.defined()) {
        throw std::invalid_argument(
            "ordered artifact identity requires a primitive ArtifactKey");
    }
    RequireNonEmpty(link_symbol, "ordered artifact link symbol");
    std::string canonical;
    AppendField(&canonical, "kind", "ordered-primitive-artifact-v1");
    AppendInteger(&canonical, "call_index", call_index);
    AppendField(&canonical, "link_symbol", link_symbol);
    AppendField(&canonical, "artifact_key", artifact_key.canonical_bytes());
    return canonical;
}

bool OrderedArtifactIdentity::operator==(
    const OrderedArtifactIdentity& other) const noexcept {
    return call_index == other.call_index && link_symbol == other.link_symbol &&
           artifact_key == other.artifact_key;
}

bool OrderedArtifactIdentity::operator!=(
    const OrderedArtifactIdentity& other) const noexcept {
    return !(*this == other);
}

DispatchKey::DispatchKey(std::string artifact_family,
                         std::string shape_layout_valid_extent,
                         std::string variant_policy_version,
                         std::string index_digest) {
    RequireNonEmpty(artifact_family, "artifact family");
    RequireNonEmpty(shape_layout_valid_extent,
                    "shape/layout/valid-extent contract");
    if (shape_layout_valid_extent.find("-1") != std::string::npos) {
        throw std::invalid_argument(
            "dispatch identity cannot use legacy -1 as shape applicability");
    }
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

bool DispatchKey::operator!=(const DispatchKey& other) const noexcept {
    return !(*this == other);
}

PlanAbiFingerprint::PlanAbiFingerprint(std::string canonical_bytes,
                                       std::string index_digest)
    : canonical_bytes_(std::move(canonical_bytes)) {
    RequireNonEmpty(canonical_bytes_, "plan ABI canonical bytes");
    digest_ = Digest(canonical_bytes_, std::move(index_digest));
}

bool PlanAbiFingerprint::defined() const noexcept {
    return !canonical_bytes_.empty() && !digest_.empty();
}

const std::string& PlanAbiFingerprint::canonical_bytes() const noexcept {
    return canonical_bytes_;
}

const std::string& PlanAbiFingerprint::digest() const noexcept {
    return digest_;
}

bool PlanAbiFingerprint::operator==(
    const PlanAbiFingerprint& other) const noexcept {
    return canonical_bytes_ == other.canonical_bytes_;
}

bool PlanAbiFingerprint::operator!=(
    const PlanAbiFingerprint& other) const noexcept {
    return !(*this == other);
}

PlanAbiFingerprint BuildPlanAbiFingerprint(
    const CompiledModule& module,
    const runtime::ExecutablePlan& plan,
    const std::vector<OrderedArtifactIdentity>& ordered_artifacts) {
    if (!module.defined() || !module.IsReady()) {
        throw std::invalid_argument(
            "plan ABI requires a ready CompiledModule");
    }
    plan.Validate();
    std::string canonical;
    // v4 is intentionally a new byte contract.  It covers callable/runtime
    // ABI only; selected artifacts and their generations/receipts are PlanVariant
    // selection identity and must never affect compatibility.
    AppendField(&canonical, "kind", "static-exact-plan-abi-v4-callable-runtime");
    AppendTargetContract(&canonical, ModuleTarget(module));
    const Array<runtime::KernelCall> calls = plan.calls();
    if (ordered_artifacts.size() != calls.size()) {
        throw std::invalid_argument(
            "plan ABI requires one ordered primitive artifact per call");
    }
    for (size_t index = 0; index < ordered_artifacts.size(); ++index) {
        const OrderedArtifactIdentity& artifact = ordered_artifacts[index];
        if (artifact.call_index != index ||
            artifact.link_symbol != std::string(calls[index]->symbol)) {
            throw std::invalid_argument(
                "plan ABI ordered artifact mapping differs from its call");
        }
    }
    for (const auto& value : plan.values()) {
        AppendField(&canonical, "value_begin", "v1");
        AppendValueContract(&canonical, value);
    }
    std::unordered_map<int64_t, size_t> value_ordinals;
    for (size_t ordinal = 0; ordinal < plan.values().size(); ++ordinal) {
        value_ordinals.emplace(plan.values()[ordinal]->value_id, ordinal);
    }
    for (const auto& call : calls) {
        AppendField(&canonical, "call_symbol", std::string(call->symbol));
        for (int64_t id : call.input_value_ids()) {
            const auto found = value_ordinals.find(id);
            if (found == value_ordinals.end()) throw std::invalid_argument("plan ABI call input is absent");
            AppendInteger(&canonical, "call_input_ordinal", found->second);
        }
        AppendField(&canonical, "call_inputs_end", "v2");
        for (int64_t id : call.output_value_ids()) {
            const auto found = value_ordinals.find(id);
            if (found == value_ordinals.end()) throw std::invalid_argument("plan ABI call output is absent");
            AppendInteger(&canonical, "call_output_ordinal", found->second);
        }
        AppendField(&canonical, "call_outputs_end", "v2");
        const codegen::KernelSignature signature =
            module.signature(call->symbol);
        const codegen::KernelLaunchMetadata metadata =
            module.launch_metadata(call->symbol);
        signature.Validate();
        metadata.Validate();
        AppendField(&canonical, "kernel_signature", signature.ToString());
        AppendField(&canonical, "launch_metadata", metadata.ToString());
    }
    for (int64_t id : plan.input_value_ids()) {
        AppendInteger(&canonical, "graph_input_ordinal", value_ordinals.at(id));
    }
    AppendField(&canonical, "graph_inputs_end", "v2");
    for (int64_t id : plan.constant_value_ids()) {
        AppendInteger(&canonical, "graph_constant_ordinal", value_ordinals.at(id));
    }
    AppendField(&canonical, "graph_constants_end", "v2");
    for (int64_t id : plan.output_value_ids()) {
        AppendInteger(&canonical, "graph_output_ordinal", value_ordinals.at(id));
    }
    AppendField(&canonical, "graph_outputs_end", "v2");
    std::vector<std::pair<std::string, runtime::NDArray>> constants;
    for (const auto& item : module.constants()) {
        constants.emplace_back(std::string(item.first), item.second);
    }
    std::sort(constants.begin(), constants.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.first < rhs.first;
              });
    for (const auto& item : constants) {
        AppendField(&canonical, "constant_key", item.first);
        AppendArrayContract(&canonical, item.second);
    }
    AppendField(&canonical, "constants_end", "v1");
    return PlanAbiFingerprint(std::move(canonical));
}

DispatchKey BuildStaticExactDispatchKey(
    const ArtifactKey& graph_artifact_key,
    const runtime::ExecutablePlan& plan) {
    if (!graph_artifact_key.defined()) {
        throw std::invalid_argument(
            "static-exact dispatch requires a graph ArtifactKey");
    }
    plan.Validate();
    std::unordered_map<int64_t, runtime::ValueSpec> values;
    for (const auto& value : plan.values()) {
        values.emplace(value->value_id, value);
    }
    std::string profile;
    AppendField(&profile, "kind", "static-exact-input-profile-v1");
    for (int64_t value_id : plan.input_value_ids()) {
        const auto found = values.find(value_id);
        if (found == values.end()) {
            throw std::invalid_argument(
                "static-exact dispatch references an unknown graph input");
        }
        AppendValueContract(&profile, found->second);
        AppendField(&profile, "layout", "contiguous");
        AppendField(&profile, "valid_extent", "equals-logical-shape");
    }
    return DispatchKey(
        graph_artifact_key.unit_semantic_key().canonical_bytes(),
        std::move(profile), "static-exact-plan-v1");
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
