#include "api/compile_config.h"

namespace kxc {
namespace api {

CompileConfig CompileConfig::AOT(Target target, int opt_level) {
    auto* node = new CompileConfigNode();
    node->mode = CompileMode::kAOT;
    node->target = target;
    node->opt_level = opt_level;
    node->enable_hot_swap = false;
    return CompileConfig(node);
}

CompileConfig CompileConfig::JIT(Target target) {
    auto* node = new CompileConfigNode();
    node->mode = CompileMode::kJIT;
    node->target = target;
    node->opt_level = 1;  // 快速编译
    node->enable_hot_swap = false;
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
    return CompileConfig(node);
}

}  // namespace api
}  // namespace kxc
