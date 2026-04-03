#pragma once

#include <string>
#include <vector>

#include "base/container.h"
#include "base/object.h"
#include "base/target.h"
#include "codegen/codegen.h"

namespace kxc {
namespace api {

enum class CompileMode {
    kAOT,       // 离线深度优化，一次编译生成最优代码
    kJIT,       // 快速编译，运行时profiling
    kAdaptive,  // 先快速执行，后台编译，热替换
};

class CompileConfigNode : public Object {
public:
    CompileMode mode{CompileMode::kAOT};
    int opt_level{2};                               // 0-3
    Target target;
    codegen::CodeGenBackend backend{codegen::CodeGenBackend::kLLVM};

    // Adaptive模式
    Array<Array<int64_t>> suggested_input_shapes;    // 建议shape
    int background_threads{0};                       // 0=auto
    bool enable_hot_swap{true};
    std::string cache_dir;                           // 空=内存缓存

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(CompileConfigNode)

class CompileConfig : public ObjectRef {
public:
    using ObjectRef::ObjectRef;

    static CompileConfig AOT(Target target, int opt_level = 3);
    static CompileConfig JIT(Target target);
    static CompileConfig Adaptive(Target target,
                                  Array<Array<int64_t>> suggested_shapes = {});

    const CompileConfigNode* operator->() const {
        return static_cast<const CompileConfigNode*>(object_);
    }
};

}  // namespace api
}  // namespace kxc
