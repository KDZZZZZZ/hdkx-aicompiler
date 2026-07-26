/*! \file src/compiler/experimental_identity.cc
 * \brief Implements non-installed Shape/adaptive identity canonicalization.
 */

#include "kxc/compiler/experimental_identity.h"

#include "../internal/identity_canonical.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "runtime/internal/compiled_module_node.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"

namespace kxc::api {
namespace {

using internal::AppendField;
using internal::AppendInteger;
using internal::Digest;
using internal::RequireNonEmpty;

bool ContainsLegacyDynamicDimension(const std::string& applicability) {
    bool inside_shape = false;
    for (size_t index = 0; index < applicability.size(); ++index) {
        const char current = applicability[index];
        if (current == '[') {
            inside_shape = true;
            continue;
        }
        if (current == ']') {
            inside_shape = false;
            continue;
        }
        if (!inside_shape || current != '-' ||
            index + 1 >= applicability.size() ||
            applicability[index + 1] != '1') {
            continue;
        }

        const bool begins_dimension =
            index == 0 || applicability[index - 1] == '[' ||
            applicability[index - 1] == ',' ||
            applicability[index - 1] == ' ';
        const size_t end = index + 2;
        const bool ends_dimension =
            end == applicability.size() || applicability[end] == ']' ||
            applicability[end] == ',' || applicability[end] == ' ';
        if (begins_dimension && ends_dimension) {
            return true;
        }
    }
    return applicability.find("dimension=2:-1;") != std::string::npos;
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
    AppendInteger(out, "is_state", value->is_state);
    AppendInteger(out, "write_mode", static_cast<uint8_t>(value->write_mode));
    AppendInteger(out, "valid_bytes", value->valid_bytes);
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

std::string StaticExactInputProfileCanonical(
    const runtime::ExecutablePlan& plan) {
    plan.Validate();
    std::unordered_map<int64_t, runtime::ValueSpec> values;
    for (const auto& value : plan.values()) {
        values.emplace(value->value_id, value);
    }
    std::string profile;
    AppendField(&profile, "kind", "static-exact-input-profile-v3-state-contract");
    for (int64_t value_id : plan.input_value_ids()) {
        const auto found = values.find(value_id);
        if (found == values.end()) {
            throw std::invalid_argument(
                "static-exact profile references an unknown graph input");
        }
        AppendValueContract(&profile, found->second);
        AppendField(&profile, "layout", "contiguous");
        AppendField(&profile, "valid_extent", "equals-logical-shape");
    }
    return profile;
}

}  // namespace

std::string OrderedArtifactIdentity::CanonicalBytes() const {
    if (!artifact_key.defined()) {
        throw std::invalid_argument(
            "ordered artifact identity requires a PrimitiveArtifactKey");
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

ShapeProfileKey::ShapeProfileKey(GraphSemanticKey graph_semantic_key,
                                 std::string canonical_bytes)
    : graph_semantic_key_(std::move(graph_semantic_key)),
      canonical_bytes_(std::move(canonical_bytes)) {
    if (!graph_semantic_key_.defined()) {
        throw std::invalid_argument(
            "shape profile identity requires graph semantics");
    }
    RequireNonEmpty(canonical_bytes_, "shape profile canonical bytes");
    digest_ = Digest(canonical_bytes_, {});
}

bool ShapeProfileKey::defined() const noexcept {
    return graph_semantic_key_.defined() && !canonical_bytes_.empty() &&
           !digest_.empty();
}

const GraphSemanticKey& ShapeProfileKey::graph_semantic_key() const noexcept {
    return graph_semantic_key_;
}

const std::string& ShapeProfileKey::canonical_bytes() const noexcept {
    return canonical_bytes_;
}

const std::string& ShapeProfileKey::digest() const noexcept {
    return digest_;
}

bool ShapeProfileKey::operator==(
    const ShapeProfileKey& other) const noexcept {
    return canonical_bytes_ == other.canonical_bytes_;
}

bool ShapeProfileKey::operator!=(
    const ShapeProfileKey& other) const noexcept {
    return !(*this == other);
}

bool ShapeProfileKey::operator<(
    const ShapeProfileKey& other) const noexcept {
    return canonical_bytes_ < other.canonical_bytes_;
}

ShapeProfileKey BuildShapeProfileKey(
    const GraphSemanticKey& graph_semantic_key,
    const std::string& shape_program_canonical,
    const std::string& bindings_canonical,
    const std::string& specialization_policy,
    uint32_t shape_abi_version) {
    if (!graph_semantic_key.defined()) {
        throw std::invalid_argument(
            "shape profile requires graph semantics");
    }
    RequireNonEmpty(shape_program_canonical, "shape program canonical bytes");
    RequireNonEmpty(bindings_canonical, "shape bindings canonical bytes");
    RequireNonEmpty(specialization_policy, "shape specialization policy");
    if (shape_abi_version == 0) {
        throw std::invalid_argument(
            "shape profile requires a positive ABI version");
    }
    std::string canonical;
    AppendField(&canonical, "kind", "shape-profile-key-v2");
    AppendField(&canonical, "graph_semantic",
                graph_semantic_key.canonical_bytes());
    AppendField(&canonical, "shape_program", shape_program_canonical);
    AppendField(&canonical, "bindings", bindings_canonical);
    AppendField(&canonical, "policy", specialization_policy);
    AppendInteger(&canonical, "shape_abi", shape_abi_version);
    return ShapeProfileKey(graph_semantic_key, std::move(canonical));
}

ShapeProfileKey BuildStaticExactShapeProfileKey(
    const GraphSemanticKey& graph_semantic_key,
    const runtime::ExecutablePlan& plan) {
    const std::string profile = StaticExactInputProfileCanonical(plan);
    return BuildShapeProfileKey(
        graph_semantic_key, profile, "static-exact-no-symbolic-bindings",
        "static-exact-plan-v2", 1);
}

DispatchKey::DispatchKey(std::string artifact_family,
                         std::string shape_layout_valid_extent,
                         std::string variant_policy_version) {
    RequireNonEmpty(artifact_family, "artifact family");
    RequireNonEmpty(shape_layout_valid_extent,
                    "shape/layout/valid-extent contract");
    if (ContainsLegacyDynamicDimension(shape_layout_valid_extent)) {
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
    digest_ = Digest(canonical_bytes_, {});
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

bool DispatchKey::operator<(const DispatchKey& other) const noexcept {
    return canonical_bytes_ < other.canonical_bytes_;
}

PlanAbiFingerprint::PlanAbiFingerprint(std::string canonical_bytes)
    : canonical_bytes_(std::move(canonical_bytes)) {
    RequireNonEmpty(canonical_bytes_, "plan ABI canonical bytes");
    digest_ = Digest(canonical_bytes_, {});
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
    // v6 covers callable/runtime ABI, including state and explicit donation.
    // Selected artifacts and their generations/receipts are PlanVariant
    // selection identity and must never affect compatibility.
    AppendField(&canonical, "kind", "static-exact-plan-abi-v6-state-alias");
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
    const Array<runtime::ValueSpec> values = plan.values();
    std::unordered_map<int64_t, size_t> value_ordinals;
    for (size_t ordinal = 0; ordinal < values.size(); ++ordinal) {
        value_ordinals.emplace(values[ordinal]->value_id, ordinal);
    }
    for (const auto& value : values) {
        AppendField(&canonical, "value_begin", "v2");
        AppendValueContract(&canonical, value);
        const auto source = value_ordinals.find(value->alias_source_value_id);
        AppendInteger(&canonical, "alias_source_ordinal",
                      source == value_ordinals.end()
                          ? int64_t{-1}
                          : static_cast<int64_t>(source->second));
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
        AppendField(&canonical, "kernel_signature", signature.CanonicalBytes());
        AppendField(&canonical, "launch_metadata", metadata.CanonicalBytes());
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
    for (int64_t id : plan.state_value_ids()) {
        AppendInteger(&canonical, "state_ordinal", value_ordinals.at(id));
    }
    AppendField(&canonical, "states_end", "v1");
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
    const GraphSemanticKey& graph_semantic_key,
    const ShapeProfileKey& shape_profile_key) {
    if (!graph_semantic_key.defined()) {
        throw std::invalid_argument(
            "static-exact dispatch requires graph semantics");
    }
    if (!shape_profile_key.defined() ||
        shape_profile_key.graph_semantic_key() != graph_semantic_key) {
        throw std::invalid_argument(
            "static-exact dispatch requires a profile for its graph");
    }
    return DispatchKey(
        graph_semantic_key.canonical_bytes(),
        shape_profile_key.canonical_bytes(), "static-exact-plan-v2");
}

std::string OrderedArtifactSelectionIdentity::CanonicalBytes() const {
    if (!artifact_key.defined()) {
        throw std::invalid_argument(
            "ordered selection requires a primitive artifact key");
    }
    RequireNonEmpty(link_symbol, "ordered selection link symbol");
    std::string canonical;
    AppendField(&canonical, "kind", "ordered-primitive-selection-v1");
    AppendInteger(&canonical, "call_index", call_index);
    AppendField(&canonical, "link_symbol", link_symbol);
    AppendField(&canonical, "primitive_artifact",
                artifact_key.canonical_bytes());
    AppendInteger(&canonical, "generation", generation);
    return canonical;
}

PlanVariantKey::PlanVariantKey(GraphSemanticKey graph_semantic_key,
                               ShapeProfileKey shape_profile_key,
                               std::string canonical_bytes)
    : graph_semantic_key_(std::move(graph_semantic_key)),
      shape_profile_key_(std::move(shape_profile_key)),
      canonical_bytes_(std::move(canonical_bytes)) {
    if (!graph_semantic_key_.defined() || !shape_profile_key_.defined() ||
        shape_profile_key_.graph_semantic_key() != graph_semantic_key_) {
        throw std::invalid_argument(
            "plan variant requires matching graph and shape identities");
    }
    RequireNonEmpty(canonical_bytes_, "plan variant canonical bytes");
    digest_ = Digest(canonical_bytes_, {});
}

PlanVariantKey BuildPlanVariantKey(
    const GraphSemanticKey& graph_semantic_key,
    const ShapeProfileKey& shape_profile_key,
    const std::vector<OrderedArtifactSelectionIdentity>& ordered_artifacts,
    const std::string& memory_plan_version) {
    if (!graph_semantic_key.defined() || !shape_profile_key.defined() ||
        shape_profile_key.graph_semantic_key() != graph_semantic_key) {
        throw std::invalid_argument(
            "plan variant requires matching graph and shape identities");
    }
    RequireNonEmpty(memory_plan_version, "memory plan version");
    if (ordered_artifacts.empty()) {
        throw std::invalid_argument(
            "plan variant identity requires selected artifact generations");
    }
    std::string canonical;
    AppendField(&canonical, "kind", "plan-variant-key-v2");
    AppendField(&canonical, "graph_semantic",
                graph_semantic_key.canonical_bytes());
    AppendField(&canonical, "shape_profile",
                shape_profile_key.canonical_bytes());
    for (size_t index = 0; index < ordered_artifacts.size(); ++index) {
        if (ordered_artifacts[index].call_index != index) {
            throw std::invalid_argument(
                "plan variant artifact selections must be ordered");
        }
        AppendField(&canonical, "selection",
                    ordered_artifacts[index].CanonicalBytes());
    }
    AppendField(&canonical, "memory_plan", memory_plan_version);
    return PlanVariantKey(graph_semantic_key, shape_profile_key,
                          std::move(canonical));
}

bool PlanVariantKey::defined() const noexcept {
    return graph_semantic_key_.defined() && shape_profile_key_.defined() &&
           !canonical_bytes_.empty() && !digest_.empty();
}

const GraphSemanticKey& PlanVariantKey::graph_semantic_key() const noexcept {
    return graph_semantic_key_;
}

const ShapeProfileKey& PlanVariantKey::shape_profile_key() const noexcept {
    return shape_profile_key_;
}

const std::string& PlanVariantKey::canonical_bytes() const noexcept {
    return canonical_bytes_;
}

const std::string& PlanVariantKey::digest() const noexcept { return digest_; }

bool PlanVariantKey::operator==(const PlanVariantKey& other) const noexcept {
    return canonical_bytes_ == other.canonical_bytes_;
}

bool PlanVariantKey::operator!=(const PlanVariantKey& other) const noexcept {
    return !(*this == other);
}

bool PlanVariantKey::operator<(const PlanVariantKey& other) const noexcept {
    return canonical_bytes_ < other.canonical_bytes_;
}

}  // namespace kxc::api
