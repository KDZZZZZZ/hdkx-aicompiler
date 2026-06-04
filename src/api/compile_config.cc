/*! \file src/api/compile_config.cc
 * \brief 实现编译配置默认值、编译入口和 CompiledModule 执行逻辑。
 */

#include "api/compile_config.h"

namespace kxc {
namespace api {

namespace {

void ApplyProfileDefaults(CompileConfigNode* node) {
    node->profile_options = profiling::ApplyEnvironmentOverrides(node->profile_options);
}

}  // namespace

CompileConfig CompileConfig::AOT(Target target, int opt_level) {
    auto* node = new CompileConfigNode();
    node->mode = CompileMode::kAOT;
    node->target = target;
    node->opt_level = opt_level;
    node->enable_hot_swap = false;
    ApplyProfileDefaults(node);
    return CompileConfig(node);
}

CompileConfig CompileConfig::JIT(Target target) {
    auto* node = new CompileConfigNode();
    node->mode = CompileMode::kJIT;
    node->target = target;
    node->opt_level = 1;  // 快速编译
    node->enable_hot_swap = false;
    ApplyProfileDefaults(node);
    return CompileConfig(node);
}

CompileConfig CompileConfig::Adaptive(Target target,
                                      Array<Array<int64_t>> suggested_shapes) {
    auto* node = new CompileConfigNode();
    node->mode = CompileMode::kAdaptive;
    node->target = target;
    node->opt_level = 2;
    node->suggested_input_shapes = suggested_shapes;
    node->enable_hot_swap = true;
    ApplyProfileDefaults(node);
    return CompileConfig(node);
}

}  // namespace api
}  // namespace kxc
