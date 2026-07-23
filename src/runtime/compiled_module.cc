/*! \file src/runtime/compiled_module.cc
 * \brief 实现 opaque CompiledModule 的组装、校验与启动。
 */

#include "kxc/runtime/compiled_module.h"
#include "kxc/support/object_registration.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "internal/compiled_module_node.h"
#include "internal/kernel_argument_validation.h"

namespace kxc::api {

KXC_OBJECT_DEFINE_WITH_KEY(CompiledModuleNode, "kxc.api.CompiledModuleNode")

namespace {

Device TargetDevice(const Target& target) {
    return Device(target->device_type, target->device_id);
}

Map<String, runtime::NDArray> CopyConstantHandles(
    const Map<String, runtime::NDArray>& source) {
    Map<String, runtime::NDArray> result;
    for (const auto& item : source) result.Set(item.first, item.second);
    return result;
}

Map<String, runtime::NDArray> CloneConstantPayloads(
    const Map<String, runtime::NDArray>& source) {
    Map<String, runtime::NDArray> result;
    for (const auto& item : source) {
        result.Set(item.first, item.second.CopyTo(item.second.device()));
    }
    return result;
}

void ValidateConstantValue(const String& constant_key,
                           const codegen::KernelArgSpec& spec,
                           const runtime::NDArray& value) {
    if (!value.defined()) {
        throw std::invalid_argument(
            "CompiledModule constant '" + std::string(constant_key) +
            "' is undefined");
    }
    if (!SameDType(value.dtype(), spec->dtype) ||
        value.device() != spec->device) {
        throw std::invalid_argument(
            "CompiledModule constant '" + std::string(constant_key) +
            "' dtype or device does not match its signature");
    }
    const Array<int64_t> expected_shape = spec.shape();
    const Array<int64_t> actual_shape = value.shape();
    if (expected_shape.size() != actual_shape.size()) {
        throw std::invalid_argument(
            "CompiledModule constant '" + std::string(constant_key) +
            "' rank does not match its signature");
    }
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (expected_shape[i] != actual_shape[i]) {
            throw std::invalid_argument(
                "CompiledModule constant '" + std::string(constant_key) +
                "' shape does not match its signature");
        }
    }
}

bool SameConstantContract(const codegen::KernelArgSpec& lhs,
                          const codegen::KernelArgSpec& rhs) {
    if (!SameDType(lhs->dtype, rhs->dtype) || lhs->device != rhs->device) {
        return false;
    }
    const Array<int64_t> lhs_shape = lhs.shape();
    const Array<int64_t> rhs_shape = rhs.shape();
    if (lhs_shape.size() != rhs_shape.size()) return false;
    for (size_t i = 0; i < lhs_shape.size(); ++i) {
        if (lhs_shape[i] != rhs_shape[i]) return false;
    }
    return true;
}

void ValidateConstants(const std::vector<internal::CompiledModuleEntry>& entries,
                       const Map<String, runtime::NDArray>& constants) {
    std::unordered_map<std::string, codegen::KernelArgSpec> required_specs;
    for (const auto& entry : entries) {
        for (const auto& spec : entry.signature.arguments()) {
            if (spec->role != codegen::KernelArgRole::kConstant) continue;
            const std::string key = std::string(spec->constant_key);
            auto inserted = required_specs.emplace(key, spec);
            if (!inserted.second &&
                !SameConstantContract(inserted.first->second, spec)) {
                throw std::invalid_argument(
                    "CompiledModule constant '" + key +
                    "' has conflicting signatures across entries");
            }
        }
    }
    for (const auto& item : required_specs) {
        const String key(item.first);
        if (!constants.count(key)) {
            throw std::invalid_argument(
                "CompiledModule is missing constant '" + item.first + "'");
        }
        ValidateConstantValue(key, item.second, constants.at(key));
    }
    if (constants.size() != required_specs.size()) {
        throw std::invalid_argument(
            "CompiledModule constant table contains unexpected keys");
    }
}

const CompiledModuleNode* CheckedNode(const CompiledModule& module) {
    const auto* node = module.As<CompiledModuleNode>();
    if (!node) throw std::runtime_error("undefined or invalid CompiledModule");
    return node;
}

const internal::CompiledModuleEntry& FindEntry(
    const CompiledModuleNode* node, const String& symbol) {
    const std::string wanted = std::string(symbol);
    const auto entry = node->entries_.find(wanted);
    if (entry != node->entries_.end()) return entry->second;
    throw std::out_of_range("CompiledModule has no function '" + wanted + "'");
}

void ValidateEntry(const internal::CompiledModuleEntry& entry,
                   const Device& target_device) {
    if (!entry.signature.defined() || !entry.launch_metadata.defined()) {
        throw std::invalid_argument(
            "CompiledModule entry signature and launch metadata must be defined");
    }
    entry.signature.Validate();
    entry.launch_metadata.Validate();
    if (entry.launch_metadata->device != target_device) {
        throw std::invalid_argument(
            "CompiledModule target and launch metadata devices do not match");
    }
    if (!entry.executable.defined() || !entry.executable.IsReady()) {
        throw std::invalid_argument("CompiledModule executable must be ready");
    }
    if (entry.executable.signature().get() != entry.signature.get() ||
        entry.executable.launch_metadata().get() != entry.launch_metadata.get()) {
        throw std::invalid_argument(
            "CompiledModule executable contract does not match module metadata");
    }
    for (const auto& spec : entry.signature.arguments()) {
        if (spec->device != target_device) {
            throw std::invalid_argument(
                "CompiledModule signature device does not match target");
        }
    }
}

AsyncOperation LaunchEntry(
    const internal::CompiledModuleEntry& entry,
    const Map<String, runtime::NDArray>& constants,
    const Array<runtime::NDArray>& ordered_arguments,
    const DeviceStream& stream) {
    if (!entry.executable.IsReady()) {
        throw std::runtime_error(
            "kernel '" + std::string(entry.signature->symbol) +
            "' executable is not ready");
    }
    if (!stream.defined()) {
        throw std::invalid_argument(
            "kernel '" + std::string(entry.signature->symbol) +
            "' requires a defined DeviceStream");
    }
    if (stream.device() != entry.launch_metadata->device) {
        throw std::invalid_argument(
            "kernel '" + std::string(entry.signature->symbol) +
            "' stream device expected " +
            entry.launch_metadata->device.ToString() + ", actual " +
            stream.device().ToString());
    }

    const Array<codegen::KernelArgSpec> specs = entry.signature.arguments();
    if (ordered_arguments.size() != specs.size()) {
        throw std::invalid_argument(
            "kernel '" + std::string(entry.signature->symbol) +
            "' argument count expected " + std::to_string(specs.size()) +
            ", actual " + std::to_string(ordered_arguments.size()));
    }
    Array<runtime::NDArray> launch_arguments;
    for (size_t i = 0; i < specs.size(); ++i) {
        ValidateKernelArgument(entry.signature, i, specs[i],
                               ordered_arguments[i], constants);
        launch_arguments.push_back(
            specs[i]->role == codegen::KernelArgRole::kConstant
                ? constants.at(specs[i]->constant_key)
                : ordered_arguments[i]);
    }
    return entry.executable.Launch(launch_arguments, stream);
}

}  // namespace

CompiledModule internal::BuildCompiledModule(
    Target target,
    std::vector<CompiledModuleEntry> entries,
    Map<String, runtime::NDArray> constants,
    std::shared_ptr<profiling::ProfileContext> profile_context) {
    if (!target.defined()) {
        throw std::invalid_argument("CompiledModule target must be defined");
    }
    if (entries.empty()) {
        throw std::invalid_argument(
            "CompiledModule must contain at least one entry");
    }
    const Device target_device = TargetDevice(target);
    std::unordered_set<std::string> symbols;
    for (const auto& entry : entries) {
        ValidateEntry(entry, target_device);
        const std::string symbol = std::string(entry.signature->symbol);
        if (!symbols.insert(symbol).second) {
            throw std::invalid_argument(
                "CompiledModule contains duplicate symbol '" + symbol + "'");
        }
    }
    ValidateConstants(entries, constants);

    return CompiledModule(ObjectRef(new CompiledModuleNode(
        std::move(target), std::move(entries), CopyConstantHandles(constants),
        std::move(profile_context))));
}

CompiledModule::CompiledModule(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<CompiledModuleNode>()) {
        SetData(nullptr);
        throw std::invalid_argument(
            "ObjectRef does not contain CompiledModuleNode");
    }
}

AsyncOperation CompiledModule::Launch(
    const String& symbol,
    const Array<runtime::NDArray>& ordered_arguments,
    const DeviceStream& stream) const {
    const auto* node = CheckedNode(*this);
    return LaunchEntry(FindEntry(node, symbol), node->constants_,
                       ordered_arguments, stream);
}

codegen::KernelSignature CompiledModule::signature(const String& symbol) const {
    return FindEntry(CheckedNode(*this), symbol).signature;
}

codegen::KernelLaunchMetadata CompiledModule::launch_metadata(
    const String& symbol) const {
    return FindEntry(CheckedNode(*this), symbol).launch_metadata;
}

Map<String, runtime::NDArray> CompiledModule::constants() const {
    return CloneConstantPayloads(CheckedNode(*this)->constants_);
}

const Map<String, runtime::NDArray>&
internal::BorrowCompiledModuleConstants(const CompiledModule& module) {
    return CheckedNode(module)->constants_;
}

bool CompiledModule::HasFunction(const String& symbol) const {
    const auto* node = CheckedNode(*this);
    return node->entries_.count(std::string(symbol)) != 0;
}

size_t CompiledModule::entry_count() const {
    return CheckedNode(*this)->entries_.size();
}

Array<String> CompiledModule::symbols() const {
    const auto* node = CheckedNode(*this);
    Array<String> result;
    for (const auto& item : node->entries_) {
        result.push_back(item.second.signature->symbol);
    }
    return result;
}

bool CompiledModule::IsReady() const noexcept {
    const auto* node = As<CompiledModuleNode>();
    if (!node || node->entries_.empty()) return false;
    for (const auto& item : node->entries_) {
        if (!item.second.executable.IsReady()) return false;
    }
    return true;
}

String CompiledModule::GetStatus() const {
    return String(IsReady() ? "ready" : "not_ready");
}

String CompiledModule::GetProfileBundlePath() const {
    const auto* node = CheckedNode(*this);
    return String(node->profile_context_ ? node->profile_context_->bundle_dir() : "");
}

}  // namespace kxc::api
