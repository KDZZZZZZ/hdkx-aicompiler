/*! \file include/kxc/compiler/compile_config.h
 * \brief 定义面向调用方的编译配置、Compiler 和 CompiledModule 公共 API。
 */

#pragma once

#include "kxc/profiling/profiling.h"
#include "kxc/support/object.h"
#include "kxc/target/target.h"

namespace kxc {
namespace api {

/*!
 * \brief 编译配置的对象节点。
 *
 * 该节点只描述目标、优化等级和 profiling 选项。实际 codegen backend 必须
 * 由 Target 推导，运行时会话也不得混入编译配置。
 */
class CompileConfigNode : public Object {
public:
    /*! \brief 编译优化等级，合法范围为 0 到 3。 */
    int opt_level{2};
    /*! \brief 编译目标及其设备能力快照，是 backend 选择的唯一事实来源。 */
    Target target;
    /*! \brief 编译阶段 profiling 和 artifact 捕获选项。 */
    profiling::ProfileOptions profile_options;

    KXC_OBJECT_DECLARE
};


/*!
 * \brief 类型安全的编译配置句柄，由 Target 和优化等级统一创建。
 */
class CompileConfig : public ObjectRef {
public:
    /*! \brief 从通用对象引用恢复配置，并验证节点类型和全部字段。 */
    explicit CompileConfig(const ObjectRef& ref);
    /*! \brief 创建 target 快照、应用一次 profiling 环境覆盖并立即验证全部字段。 */
    static CompileConfig Create(
        Target target, int opt_level = 2,
        profiling::ProfileOptions profile_options = {});
    /*! \brief 校验优化等级以及 Target kind、设备身份和能力的一致性。 */
    void Validate() const;

    /*! \brief 返回经过类型检查的只读配置节点。 */
    const CompileConfigNode* operator->() const {
        return As<CompileConfigNode>();
    }
};

}  // namespace api
}  // namespace kxc
