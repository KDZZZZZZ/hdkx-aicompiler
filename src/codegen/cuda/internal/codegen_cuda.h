/*! \file src/codegen/cuda/internal/codegen_cuda.h
 * \brief 定义从已完成 CUDA 线程绑定的 TIR 生成 CUDA C 源码的严格发射器。
 */

#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kxc/tir/stmt.h"

namespace kxc::codegen {

/*!
 * \brief 把单个已调度 PrimFunc 发射为 extern "C" __global__ CUDA 内核。
 *
 * 该类只负责源码生成，不隐式执行调度、NVRTC 编译或 module 加载。输入必须已经由
 * CUDA schedule 显式插入 ThreadBinding；任何未知 TIR 节点都会抛出异常，避免生成
 * 带占位注释但行为不完整的可执行源码。
 */
class CodeGenCUDA {
public:
    /*! \brief 重置单次生成状态，并按 PrimFunc.params 的稳定顺序生成 CUDA 源码。 */
    std::string Generate(const tir::PrimFunc& function,
                         const std::string& symbol);

    /*! \brief Emits several independent kernels into one NVRTC translation unit. */
    std::string GenerateModule(
        const std::vector<std::pair<tir::PrimFunc, std::string>>& functions);

private:
    /*! \brief 递归生成一个标量表达式；undefined 和未知节点均视为 codegen 错误。 */
    std::string GenExpr(const tir::PrimExpr& expression);
    /*! \brief 递归生成语句和作用域，并保持稳定的两空格缩进。 */
    void GenStmt(const tir::Stmt& statement);
    /*! \brief 将受支持的标量 TIR dtype 映射为 CUDA C++ 类型。 */
    std::string DTypeName(tir::DataType dtype) const;
    /*! \brief 为 TIR Var 分配本次 Generate 内唯一且合法的 CUDA 标识符。 */
    std::string VarName(const tir::Var& variable);
    /*! \brief 将结构化 ThreadIndexKind 映射为唯一 CUDA builtin。 */
    static const char* ThreadBuiltin(tir::ThreadIndexKind kind);
    /*! \brief 输出当前语句层级的缩进。 */
    void Indent();

    std::ostringstream output_;
    int indent_{0};
    size_t next_variable_id_{0};
    /*! \brief 确认输入确实经过 CUDA 调度，禁止把串行 TIR 发射为空并行内核。 */
    bool saw_thread_binding_{false};
    std::unordered_map<const Object*, std::string> variable_names_;
    std::unordered_set<std::string> used_names_;
};

}  // namespace kxc::codegen
