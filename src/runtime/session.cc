/*! \file src/runtime/session.cc
 * \brief 实现 RuntimeSession 的签名驱动参数装配和同步/异步执行。
 */

#include "kxc/runtime/session.h"
#include "kxc/support/object_registration.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "internal/kernel_argument_validation.h"
#include "internal/session_node.h"

namespace kxc::runtime {

KXC_OBJECT_DEFINE_WITH_KEY(RuntimeSessionNode, "kxc.runtime.RuntimeSessionNode")

namespace {

/*! \brief 保存一次调用按签名组装的完整参数和其中的输出子序列。 */
struct PreparedArguments final {
    /*! \brief 传给 CompiledModule::Launch 的完整有序参数。 */
    Array<NDArray> ordered;
    /*! \brief 返回调用方的输出张量，元素与 ordered 中对应对象共享身份。 */
    Array<NDArray> outputs;
};

/*! \brief 重新验证 ObjectRef 恢复出的 module，禁止半初始化 session。 */
void ValidateModule(const api::CompiledModule& module) {
    if (!module.defined()) {
        throw std::invalid_argument(
            "RuntimeSession requires a defined CompiledModule");
    }
    if (!module.IsReady()) {
        throw std::invalid_argument(
            "RuntimeSession requires a ready CompiledModule");
    }
    module.signature().Validate();
    module.launch_metadata().Validate();
}

/*! \brief 按 Signature 唯一顺序组合 inputs、module constants 和新 outputs。 */
PreparedArguments PrepareArguments(const api::CompiledModule& module,
                                   const Array<NDArray>& inputs) {
    const codegen::KernelSignature signature = module.signature();
    const Array<codegen::KernelArgSpec> specs = signature.arguments();
    size_t expected_inputs = 0;
    for (const auto& spec : specs) {
        if (spec->role == codegen::KernelArgRole::kInput) ++expected_inputs;
    }
    if (inputs.size() != expected_inputs) {
        throw std::invalid_argument(
            "RuntimeSession kernel '" + std::string(signature->symbol) +
            "' input count expected " + std::to_string(expected_inputs) +
            ", actual " + std::to_string(inputs.size()));
    }

    size_t input_index = 0;
    for (size_t signature_index = 0; signature_index < specs.size();
         ++signature_index) {
        const auto& spec = specs[signature_index];
        if (spec->role != codegen::KernelArgRole::kInput) continue;
        api::ValidateKernelArgument(signature, signature_index, spec,
                                    inputs[input_index]);
        ++input_index;
    }

    const Map<String, NDArray> constants = module.constants();
    PreparedArguments prepared;
    input_index = 0;
    for (const auto& spec : specs) {
        switch (spec->role) {
            case codegen::KernelArgRole::kInput:
                prepared.ordered.push_back(inputs[input_index++]);
                break;
            case codegen::KernelArgRole::kConstant:
                if (!constants.count(spec->constant_key)) {
                    throw std::runtime_error(
                        "RuntimeSession module is missing constant '" +
                        std::string(spec->constant_key) + "'");
                }
                // 必须使用模块持有的同一对象，CompiledModule 会再次验证常量身份。
                prepared.ordered.push_back(constants.at(spec->constant_key));
                break;
            case codegen::KernelArgRole::kOutput: {
                const Array<int64_t> shape = spec.shape();
                for (int64_t dimension : shape) {
                    if (dimension == codegen::kDynamicDimension) {
                        throw std::runtime_error(
                            "RuntimeSession cannot allocate dynamic output without "
                            "a shape function");
                    }
                }
                NDArray output = NDArray::Empty(
                    shape, spec->dtype, spec->device, spec->alignment);
                prepared.outputs.push_back(output);
                prepared.ordered.push_back(std::move(output));
                break;
            }
        }
    }
    return prepared;
}

}  // namespace

/*! \brief 创建只持有一个 ready CompiledModule 的 RuntimeSession 节点。 */
RuntimeSession::RuntimeSession(api::CompiledModule module) {
    ValidateModule(module);
    SetData(new RuntimeSessionNode(std::move(module)));
}

/*! \brief 从 ObjectRef 恢复 RuntimeSession，并重新验证节点内容。 */
RuntimeSession::RuntimeSession(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<RuntimeSessionNode>()) {
        SetData(nullptr);
        throw std::invalid_argument(
            "ObjectRef does not contain RuntimeSessionNode");
    }
    if (defined()) ValidateModule(operator->()->module);
}

/*! \brief 在模块设备默认 stream 上执行，等待后返回自动分配输出。 */
Array<NDArray> RuntimeSession::Run(const Array<NDArray>& inputs) const {
    const DeviceStream stream =
        DeviceStream::Default(operator->()->module.launch_metadata()->device);
    RunAsyncResult result = RunAsync(inputs, stream);
    result.completion.Wait();
    return result.outputs;
}

/*! \brief 校验显式 stream，组装参数并转交唯一的 CompiledModule launch。 */
RunAsyncResult RuntimeSession::RunAsync(const Array<NDArray>& inputs,
                                        const DeviceStream& stream) const {
    const api::CompiledModule& module = operator->()->module;
    if (!stream.defined() || !stream.As<DeviceStreamNode>()) {
        throw std::invalid_argument(
            "RuntimeSession RunAsync requires a defined DeviceStream");
    }
    if (stream.device() != module.launch_metadata()->device) {
        throw std::invalid_argument(
            "RuntimeSession stream device expected " +
            module.launch_metadata()->device.ToString() + ", actual " +
            stream.device().ToString());
    }
    PreparedArguments prepared = PrepareArguments(module, inputs);
    AsyncOperation completion = module.Launch(prepared.ordered, stream);
    return RunAsyncResult{std::move(prepared.outputs), std::move(completion)};
}

/*! \brief 返回经过 ObjectRef 动态类型检查的 RuntimeSession 节点。 */
const RuntimeSessionNode* RuntimeSession::operator->() const {
    const auto* node = As<RuntimeSessionNode>();
    if (!node) throw std::runtime_error("undefined or invalid RuntimeSession");
    return node;
}

}  // namespace kxc::runtime
