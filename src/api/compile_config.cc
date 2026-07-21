/*! \file src/api/compile_config.cc
 * \brief 实现编译配置默认值、编译入口和 CompiledModule 执行逻辑。
 */

#include "api/compile_config.h"

#include <stdexcept>
#include <utility>

namespace kxc {
namespace api {

namespace {

// 将环境变量覆盖集中应用一次，避免 Compiler 各阶段重复解释 profiling 配置。
void ApplyProfileDefaults(CompileConfigNode* node) {
    node->profile_options = profiling::ApplyEnvironmentOverrides(node->profile_options);
}

}  // namespace

// 从 ObjectRef 恢复配置时先检查真实节点类型，禁止 operator-> 静态误解释对象布局。
CompileConfig::CompileConfig(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<CompileConfigNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain CompileConfigNode");
    }
}

// 创建完整配置后立即验证，使非法 Target 或优化等级不能进入 Compiler。
CompileConfig CompileConfig::Create(Target target, int opt_level) {
    auto* node = new CompileConfigNode();
    node->target = std::move(target);
    node->opt_level = opt_level;
    ApplyProfileDefaults(node);
    CompileConfig config{ObjectRef(node)};
    config.Validate();
    return config;
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
