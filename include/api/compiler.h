/*! \file include/api/compiler.h
 * \brief 定义面向调用方的编译配置、Compiler 和 CompiledModule 公共 API。
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "api/compile_config.h"
#include "base/ndarray.h"
#include "codegen/compiled_kernel.h"
#include "relay/relay.h"
#include "relay/transforms/lower.h"
#include "tir/stmt.h"

namespace kxc {
namespace api {

class CompiledModule;

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
};

/*!
 * \brief 同步编译产物句柄，封装 PrimFunc、常量和可执行 kernel。
 */
class CompiledModule {
public:
    CompiledModule() = default;

    /*! \brief 构造真实编译产物，并保活 PrimFunc、kernel 和常量 payload。 */
    CompiledModule(CompileConfig config,
                   tir::PrimFunc prim_func,
                   codegen::CompiledKernel kernel,
                   Array<relay::ConstantBinding> constants,
                   std::shared_ptr<profiling::ProfileContext> profile_context = nullptr);

    /*! \brief 使用现有内部 packed ABI 同步运行 kernel；Task 6 将迁移为 NDArray。 */
    void Run(const std::vector<void*>& packed_args);

    /*! \brief 返回模块是否持有可运行 kernel。 */
    bool IsReady() const;

    /*! \brief 返回 compiled 或 not_ready 状态。 */
    std::string GetStatus() const;

    /*! \brief 获取生成的 TIR PrimFunc，主要用于调试和保存 IR。 */
    const tir::PrimFunc& GetPrimFunc() const { return prim_func_; }

    /*! \brief 返回常量绑定的独立数组，保证编译模块持有 payload 生命周期。 */
    Array<relay::ConstantBinding> GetConstants() const;

    /*! \brief 将 PrimFunc 对应的 C 源码保存到指定路径。 */
    void SaveCSource(const std::string& path) const;

    /*! \brief 返回 profiling bundle 路径；未开启 profiling 时为空字符串。 */
    std::string GetProfileBundlePath() const;

private:
    CompileConfig config_;

    // 当前同步模块的编译配置和真实后端产物。
    tir::PrimFunc prim_func_;
    codegen::CompiledKernel kernel_;
    /*! \brief 与 kernel 参数段同序的常量绑定，由模块保活到最后一次引用释放。 */
    Array<relay::ConstantBinding> constants_;
    std::string c_source_;

    std::shared_ptr<profiling::ProfileContext> profile_context_;
};

}  // namespace api
}  // namespace kxc
