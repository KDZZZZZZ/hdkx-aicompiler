/*! \file src/compiler/compile_config.cc
 * \brief 实现编译配置默认值、编译入口和 CompiledModule 执行逻辑。
 */

#include "kxc/compiler/compile_config.h"
#include "kxc/support/object_registration.h"

#include <stdexcept>
#include <utility>

namespace kxc {
namespace api {

KXC_OBJECT_DEFINE(CompileConfigNode)

namespace {

Target SnapshotTarget(const Target& target) {
    const auto* source = target.As<TargetNode>();
    if (!source) {
        throw std::invalid_argument(
            "CompileConfig target must contain a valid TargetNode");
    }
    auto* snapshot = new TargetNode();
    snapshot->kind = source->kind;
    snapshot->device_type = source->device_type;
    snapshot->device_id = source->device_id;
    snapshot->attrs = source->attrs;
    return Target(ObjectRef(snapshot));
}

}  // namespace

// 从 ObjectRef 恢复配置时验证真实节点类型和完整字段，禁止静态误解释对象布局。
CompileConfig::CompileConfig(const ObjectRef& ref) : ObjectRef(ref) {
    if (!defined() || !As<CompileConfigNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain CompileConfigNode");
    }
    try {
        Validate();
    } catch (...) {
        SetData(nullptr);
        throw;
    }
}

// 创建完整快照后立即验证，使非法 Target 或优化等级不能进入 Compiler。
CompileConfig CompileConfig::Create(
    Target target, int opt_level, profiling::ProfileOptions profile_options) {
    auto* node = new CompileConfigNode();
    node->target = SnapshotTarget(target);
    node->opt_level = opt_level;
    node->profile_options = profiling::ApplyEnvironmentOverrides(
        std::move(profile_options));
    return CompileConfig(ObjectRef(node));
}

// Target 同时携带 backend kind、设备身份和能力快照，三者必须互相一致。
void CompileConfig::Validate() const {
    const auto* node = As<CompileConfigNode>();
    if (!node) {
        throw std::invalid_argument("CompileConfig must be defined");
    }
    if (node->opt_level < 0 || node->opt_level > 3) {
        throw std::invalid_argument("CompileConfig opt_level must be in [0, 3]");
    }
    if (!node->target.defined()) {
        throw std::invalid_argument("CompileConfig target must be defined");
    }
    const TargetNode* target = node->target.As<TargetNode>();
    if (!target) {
        throw std::invalid_argument("CompileConfig target has an invalid object type");
    }
    const bool llvm_cpu = target->kind == "llvm" && target->device_type == kCPU;
    const bool cuda_gpu = target->kind == "cuda" && target->device_type == kCUDA;
    if (!llvm_cpu && !cuda_gpu) {
        throw std::invalid_argument(
            "CompileConfig target kind and device type are inconsistent");
    }
    if (target->device_id < 0 || (target->device_type == kCPU && target->device_id != 0)) {
        throw std::invalid_argument("CompileConfig target device id is invalid");
    }
    if (target->attrs.exists != 1) {
        throw std::invalid_argument("CompileConfig target device is unavailable");
    }
    if (target->attrs.device_name.empty() || target->attrs.arch.empty() ||
        target->attrs.max_threads_per_block <= 0 || target->attrs.warp_size <= 0 ||
        target->attrs.multi_processor_count <= 0) {
        throw std::invalid_argument("CompileConfig target capability snapshot is incomplete");
    }
}

}  // namespace api
}  // namespace kxc
