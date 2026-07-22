/*! \file src/runtime/compiled_module.cc
 * \brief 实现 opaque CompiledModule 的组装、校验与启动。
 */

#include "kxc/runtime/compiled_module.h"
#include "kxc/support/object_registration.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "internal/compiled_module_node.h"
#include "internal/kernel_argument_validation.h"

namespace kxc::api {

KXC_OBJECT_DEFINE_WITH_KEY(CompiledModuleNode, "kxc.api.CompiledModuleNode")

namespace {

Device TargetDevice(const Target& target) {
    return Device(target->device_type, target->device_id);
}

Map<String, runtime::NDArray> CopyConstants(
    const Map<String, runtime::NDArray>& source) {
    Map<String, runtime::NDArray> result;
    for (const auto& item : source) result.Set(item.first, item.second);
    return result;
}

void ValidateConstants(const codegen::KernelSignature& signature,
                       const Map<String, runtime::NDArray>& constants) {
    size_t expected_count = 0;
    for (const auto& spec : signature.arguments()) {
        if (spec->role != codegen::KernelArgRole::kConstant) continue;
        ++expected_count;
        if (!constants.count(spec->constant_key)) {
            throw std::invalid_argument(
                "CompiledModule is missing constant '" +
                std::string(spec->constant_key) + "'");
        }
        const runtime::NDArray& value = constants.at(spec->constant_key);
        if (!value.defined()) {
            throw std::invalid_argument(
                "CompiledModule constant '" + std::string(spec->constant_key) +
                "' is undefined");
        }
        if (!SameDType(value.dtype(), spec->dtype) ||
            value.device() != spec->device) {
            throw std::invalid_argument(
                "CompiledModule constant '" + std::string(spec->constant_key) +
                "' dtype or device does not match its signature");
        }
        const Array<int64_t> expected_shape = spec.shape();
        const Array<int64_t> actual_shape = value.shape();
        if (expected_shape.size() != actual_shape.size()) {
            throw std::invalid_argument(
                "CompiledModule constant '" + std::string(spec->constant_key) +
                "' rank does not match its signature");
        }
        for (size_t i = 0; i < expected_shape.size(); ++i) {
            if (expected_shape[i] != actual_shape[i]) {
                throw std::invalid_argument(
                    "CompiledModule constant '" +
                    std::string(spec->constant_key) +
                    "' shape does not match its signature");
            }
        }
    }
    if (constants.size() != expected_count) {
        throw std::invalid_argument(
            "CompiledModule constant table contains unexpected keys");
    }
}

const CompiledModuleNode* CheckedNode(const CompiledModule& module) {
    const auto* node = module.As<CompiledModuleNode>();
    if (!node) throw std::runtime_error("undefined or invalid CompiledModule");
    return node;
}

}  // namespace

CompiledModule internal::BuildCompiledModule(
    Target target,
    tir::PrimFunc prim_func,
    codegen::KernelSignature signature,
    codegen::KernelLaunchMetadata launch_metadata,
    Map<String, runtime::NDArray> constants,
    codegen::CompiledKernel executable,
    std::shared_ptr<profiling::ProfileContext> profile_context) {
    if (!target.defined()) {
        throw std::invalid_argument("CompiledModule target must be defined");
    }
    if (!signature.defined() || !launch_metadata.defined()) {
        throw std::invalid_argument(
            "CompiledModule signature and launch metadata must be defined");
    }
    signature.Validate();
    launch_metadata.Validate();
    const Device target_device = TargetDevice(target);
    if (launch_metadata->device != target_device) {
        throw std::invalid_argument(
            "CompiledModule target and launch metadata devices do not match");
    }
    if (!executable.defined() || !executable.IsReady()) {
        throw std::invalid_argument("CompiledModule executable must be ready");
    }
    if (executable.signature().get() != signature.get() ||
        executable.launch_metadata().get() != launch_metadata.get()) {
        throw std::invalid_argument(
            "CompiledModule executable contract does not match module metadata");
    }
    for (const auto& spec : signature.arguments()) {
        if (spec->device != target_device) {
            throw std::invalid_argument(
                "CompiledModule signature device does not match target");
        }
    }
    ValidateConstants(signature, constants);

    return CompiledModule(ObjectRef(new CompiledModuleNode(
        std::move(target), std::move(prim_func), std::move(signature),
        std::move(launch_metadata), CopyConstants(constants),
        std::move(executable), std::move(profile_context))));
}

CompiledModule::CompiledModule(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<CompiledModuleNode>()) {
        SetData(nullptr);
        throw std::invalid_argument(
            "ObjectRef does not contain CompiledModuleNode");
    }
}

AsyncOperation CompiledModule::Launch(
    const Array<runtime::NDArray>& ordered_arguments,
    const DeviceStream& stream) const {
    const auto* node = CheckedNode(*this);
    if (!node->executable_.IsReady()) {
        throw std::runtime_error(
            "kernel '" + std::string(node->signature_->symbol) +
            "' executable is not ready");
    }
    if (!stream.defined()) {
        throw std::invalid_argument(
            "kernel '" + std::string(node->signature_->symbol) +
            "' requires a defined DeviceStream");
    }
    if (stream.device() != node->launch_metadata_->device) {
        throw std::invalid_argument(
            "kernel '" + std::string(node->signature_->symbol) +
            "' stream device expected " +
            node->launch_metadata_->device.ToString() + ", actual " +
            stream.device().ToString());
    }

    const Array<codegen::KernelArgSpec> specs = node->signature_.arguments();
    if (ordered_arguments.size() != specs.size()) {
        throw std::invalid_argument(
            "kernel '" + std::string(node->signature_->symbol) +
            "' argument count expected " + std::to_string(specs.size()) +
            ", actual " + std::to_string(ordered_arguments.size()));
    }
    for (size_t i = 0; i < specs.size(); ++i) {
        ValidateKernelArgument(node->signature_, i, specs[i],
                               ordered_arguments[i], node->constants_);
    }
    return node->executable_.Launch(ordered_arguments, stream);
}

codegen::KernelSignature CompiledModule::signature() const {
    return CheckedNode(*this)->signature_;
}

codegen::KernelLaunchMetadata CompiledModule::launch_metadata() const {
    return CheckedNode(*this)->launch_metadata_;
}

Map<String, runtime::NDArray> CompiledModule::constants() const {
    return CopyConstants(CheckedNode(*this)->constants_);
}

bool CompiledModule::IsReady() const noexcept {
    const auto* node = As<CompiledModuleNode>();
    return node != nullptr && node->executable_.IsReady();
}

String CompiledModule::GetStatus() const {
    return String(IsReady() ? "ready" : "not_ready");
}

String CompiledModule::GetProfileBundlePath() const {
    const auto* node = CheckedNode(*this);
    return String(node->profile_context_ ? node->profile_context_->bundle_dir() : "");
}

}  // namespace kxc::api
