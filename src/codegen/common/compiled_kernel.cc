/*! \file src/codegen/common/compiled_kernel.cc
 * \brief 实现已编译内核的强类型组装、状态查询与后端分发。
 */

#include "../internal/compiled_kernel.h"
#include "kxc/support/object_registration.h"

#include <stdexcept>
#include <utility>

namespace kxc::codegen {

KXC_OBJECT_DEFINE_WITH_KEY(CompiledKernelNode, "kxc.codegen.CompiledKernelNode")

// 组装时立即冻结签名、启动元数据和 launcher 的共同生命周期。
CompiledKernel::CompiledKernel(
    KernelSignature signature,
    KernelLaunchMetadata launch_metadata,
    std::shared_ptr<const KernelLauncher> launcher) {
    if (!signature.defined()) {
        throw std::invalid_argument("CompiledKernel signature must be defined");
    }
    if (!launch_metadata.defined()) {
        throw std::invalid_argument("CompiledKernel launch metadata must be defined");
    }
    signature.Validate();
    launch_metadata.Validate();
    if (!launcher) {
        throw std::invalid_argument("CompiledKernel launcher must be defined");
    }

    SetData(new CompiledKernelNode(std::move(signature),
                                   std::move(launch_metadata),
                                   std::move(launcher)));
}

// ObjectRef 恢复只接受真实 CompiledKernelNode，禁止错误静态转换穿透 codegen 边界。
CompiledKernel::CompiledKernel(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<CompiledKernelNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain CompiledKernelNode");
    }
}

// 参数已经由 CompiledModule 校验；此处只检查资源状态并转交后端 ABI 层。
AsyncOperation CompiledKernel::Launch(
    const Array<runtime::NDArray>& arguments,
    const DeviceStream& stream) const {
    const auto* node = operator->();
    if (!node->launcher || !node->launcher->IsReady()) {
        throw std::runtime_error("CompiledKernel launcher is not ready");
    }
    return node->launcher->Launch(arguments, stream, ObjectRef(*this));
}

// undefined 句柄和未就绪 launcher 都不是可执行产物。
bool CompiledKernel::IsReady() const noexcept {
    const auto* node = As<CompiledKernelNode>();
    return node != nullptr && node->launcher != nullptr && node->launcher->IsReady();
}

// 返回 ObjectRef 句柄会共享不可变节点，不复制或暴露后端原生资源。
KernelSignature CompiledKernel::signature() const { return operator->()->signature; }

// 启动元数据与 launcher 由同一 CompiledKernelNode 保持一致生命周期。
KernelLaunchMetadata CompiledKernel::launch_metadata() const {
    return operator->()->launch_metadata;
}

// 所有成员访问都经过动态类型检查，undefined 句柄会给出确定错误。
const CompiledKernelNode* CompiledKernel::operator->() const {
    const auto* node = As<CompiledKernelNode>();
    if (!node) throw std::runtime_error("undefined or invalid CompiledKernel");
    return node;
}

}  // namespace kxc::codegen
