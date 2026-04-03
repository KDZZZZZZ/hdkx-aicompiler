#pragma once

#include <memory>
#include <string>
#include <vector>

#include "api/compile_config.h"
#include "base/ndarray.h"
#include "codegen/compiled_kernel.h"
#include "relay/relay.h"
#include "tir/stmt.h"

namespace kxc {
namespace runtime {
class RuntimeSession;
}

namespace api {

class CompiledModule;

// ==================== Compiler ====================

class Compiler {
public:
    // 从Relay Function编译
    // AOT模式：同步编译，返回立即可用的module
    // Adaptive模式：返回带RuntimeSession的module，首次Run时触发编译
    static CompiledModule Compile(Function func, CompileConfig config);
};

// ==================== CompiledModule ====================

class CompiledModule {
public:
    CompiledModule() = default;

    // AOT/JIT模式：直接持有编译好的kernel
    CompiledModule(CompileConfig config,
                   tir::PrimFunc prim_func,
                   codegen::CompiledKernel kernel);

    // Adaptive模式：持有RuntimeSession
    CompiledModule(CompileConfig config,
                   Function relay_func,
                   std::shared_ptr<runtime::RuntimeSession> session);

    // 主运行接口（AOT/JIT）
    void Run(const std::vector<void*>& packed_args);

    // 自适应运行接口（Adaptive）
    // input_shapes: 每个输入的shape，用于缓存索引和shape预测
    void Run(const std::vector<void*>& packed_args,
             const std::vector<std::vector<int64_t>>& input_shapes);

    // 预热：提前为指定shape编译
    void WarmUp(const std::vector<std::vector<int64_t>>& input_shapes);

    // 等待所有后台编译完成
    void WaitAll();

    // 查询
    bool IsReady() const;
    std::string GetStatus() const;

    // 获取生成的IR（调试用）
    const tir::PrimFunc& GetPrimFunc() const { return prim_func_; }

    // 序列化
    void SaveCSource(const std::string& path) const;

private:
    CompileConfig config_;

    // AOT/JIT模式
    tir::PrimFunc prim_func_;
    codegen::CompiledKernel kernel_;
    std::string c_source_;

    // Adaptive模式
    Function relay_func_;
    std::shared_ptr<runtime::RuntimeSession> session_;
};

}  // namespace api
}  // namespace kxc
