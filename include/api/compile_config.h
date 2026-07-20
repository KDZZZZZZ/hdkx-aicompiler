/*! \file include/api/compile_config.h
 * \brief 定义面向调用方的编译配置、Compiler 和 CompiledModule 公共 API。
 */

#pragma once

#include <string>
#include <vector>

#include "base/profiling.h"
#include "base/container.h"
#include "base/object.h"
#include "base/target.h"
#include "codegen/backend.h"

namespace kxc {
namespace api {

/*!
 * \brief 编译模式，决定 Compiler::Compile 的优化深度和运行时形态。
 */
enum class CompileMode {
    kAOT,       // 离线深度优化，一次编译生成最优代码
    kJIT,       // 快速编译，运行时profiling
    kAdaptive,  // 先快速执行，后台编译，热替换
};

/*!
 * \brief 编译配置的对象节点。
 *
 * 该节点集中描述目标设备、codegen backend、优化等级、自适应运行时参数和
 * profiling 选项。CompileConfig 是它的引用类型句柄。
 */
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

    profiling::ProfileOptions profile_options;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(CompileConfigNode)

/*!
 * \brief 编译配置句柄，提供 AOT/JIT/Adaptive 三种常用配置工厂。
 */
class CompileConfig : public ObjectRef {
public:
    using ObjectRef::ObjectRef;

    /*! \brief 创建离线 AOT 配置，默认走较深优化。 */
    static CompileConfig AOT(Target target, int opt_level = 3);

    /*! \brief 创建快速 JIT 配置，偏向降低首次编译延迟。 */
    static CompileConfig JIT(Target target);

    /*! \brief 创建自适应配置，运行时按 shape 缓存和后台优化 kernel。 */
    static CompileConfig Adaptive(Target target,
                                  Array<Array<int64_t>> suggested_shapes = {});

    CompileConfigNode* operator->() {
        return static_cast<CompileConfigNode*>(const_cast<Object*>(object_));
    }
    const CompileConfigNode* operator->() const {
        return static_cast<const CompileConfigNode*>(object_);
    }
};

}  // namespace api
}  // namespace kxc
