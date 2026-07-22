/*! \file include/kxc/compiler/compiler.h
 * \brief 定义面向调用方的编译配置、Compiler 和 CompiledModule 公共 API。
 */

#pragma once

#include "kxc/compiler/compile_config.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/support/container.h"
#include "kxc/relay/relay.h"

namespace kxc {
namespace api {

/*!
 * \brief Relay Function 到可执行模块的统一编译入口。
 *
 * Compiler 负责串起 Relay pass、Relay->TIR lowering、TIR pass 和真实 codegen。
 * backend 完全由 CompileConfig.target 决定，编译入口不再创建运行时会话。
 */
class Compiler {
public:
    /*!
     * \brief 从 Relay Function 编译出 CompiledModule。
     * \param func 输入 Relay 函数。
     * \param config 编译配置。
     * \return 可运行的编译模块。
     */
    static CompiledModule Compile(Function func, CompileConfig config);

    /*! \brief 返回 opt_level 对应的确定性 Relay pass 顺序。 */
    static Array<String> RelayPassPolicy(int opt_level);
    /*! \brief 返回 opt_level 和 Target 对应的确定性 TIR pass 顺序。 */
    static Array<String> TIRPassPolicy(int opt_level, const Target& target);
};

}  // namespace api
}  // namespace kxc
