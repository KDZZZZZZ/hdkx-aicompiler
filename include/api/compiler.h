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
namespace runtime {
class RuntimeSession;
}

namespace api {

class CompiledModule;

/*!
 * \brief Relay Function 到可执行模块的统一编译入口。
 *
 * Compiler 负责串起 Relay pass、Relay->TIR lowering、TIR pass 和 codegen。
 * Adaptive 模式会返回带 RuntimeSession 的模块，AOT/JIT 模式会返回已编译 kernel。
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
 * \brief 编译产物句柄，封装 AOT/JIT kernel 或 Adaptive RuntimeSession。
 */
class CompiledModule {
public:
    CompiledModule() = default;

    /*! \brief AOT/JIT 模式构造函数，直接持有 PrimFunc 和 CompiledKernel。 */
    CompiledModule(CompileConfig config,
                   tir::PrimFunc prim_func,
                   codegen::CompiledKernel kernel,
                   Array<relay::ConstantBinding> constants,
                   std::shared_ptr<profiling::ProfileContext> profile_context = nullptr);

    /*! \brief Adaptive 模式构造函数，持有 Relay 函数和 RuntimeSession。 */
    CompiledModule(CompileConfig config,
                   Function relay_func,
                   std::shared_ptr<runtime::RuntimeSession> session,
                   std::shared_ptr<profiling::ProfileContext> profile_context = nullptr);

    /*! \brief AOT/JIT 主运行接口，参数按 packed buffer 指针传入。 */
    void Run(const std::vector<void*>& packed_args);

    /*!
     * \brief Adaptive 运行接口。
     * \param packed_args packed buffer 指针。
     * \param input_shapes 每个输入的 shape，用于缓存索引和 shape 预测。
     */
    void Run(const std::vector<void*>& packed_args,
             const std::vector<std::vector<int64_t>>& input_shapes);

    /*! \brief 提前为指定 shape 编译并写入缓存。 */
    void WarmUp(const std::vector<std::vector<int64_t>>& input_shapes);

    /*! \brief 等待所有后台编译任务完成。 */
    void WaitAll();

    /*! \brief 返回模块是否具备可运行 kernel 或 session。 */
    bool IsReady() const;

    /*! \brief 返回模块或 runtime session 的人类可读状态。 */
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

    // AOT/JIT模式
    tir::PrimFunc prim_func_;
    codegen::CompiledKernel kernel_;
    /*! \brief 与 kernel 参数段同序的常量绑定，由模块保活到最后一次引用释放。 */
    Array<relay::ConstantBinding> constants_;
    std::string c_source_;

    // Adaptive模式
    Function relay_func_;
    std::shared_ptr<runtime::RuntimeSession> session_;
    std::shared_ptr<profiling::ProfileContext> profile_context_;
};

}  // namespace api
}  // namespace kxc
