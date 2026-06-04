/*! \file include/codegen/compiled_kernel.h
 * \brief 定义 codegen 后端、C/LLVM codegen、JIT 和 compiled kernel 抽象。
 */

#pragma once

#include <string>
#include <vector>

#include "base/container.h"
#include "base/object.h"
#include "codegen/codegen.h"
#include "tir/stmt.h"

namespace kxc {
namespace codegen {

/*! \brief 已编译 kernel 节点，统一保存 LLVM JIT 或 C 动态库后端资源。 */
class CompiledKernelNode : public Object {
public:
    std::string kernel_name;
    CodeGenBackend backend;

    /*! \brief 统一调用入口函数指针，LLVM JIT 和 C 编译后端都会填充。 */
    void* func_ptr{nullptr};

    /*! \brief LLVM JIT 路径关联资源，生命周期由 JIT 引擎管理。 */
    void* jit_resource{nullptr};

    /*! \brief C 后端动态库路径和句柄。 */
    std::string so_path;
    void* lib_handle{nullptr};

    /*! \brief kernel 参数 buffer 签名信息。 */
    Array<tir::Buffer> param_buffers;

    ~CompiledKernelNode() override;
    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(CompiledKernelNode)

/*! \brief 已编译 kernel 的引用类型，提供统一执行和就绪状态查询。 */
class CompiledKernel : public ObjectRef {
public:
    using ObjectRef::ObjectRef;

    /*! \brief 以 void* 参数数组调用底层 kernel。 */
    void operator()(const std::vector<void*>& args) const;

    /*! \brief 判断底层函数指针是否已可用。 */
    bool IsReady() const;
    const CompiledKernelNode* operator->() const {
        return static_cast<const CompiledKernelNode*>(object_);
    }
};

}  // namespace codegen
}  // namespace kxc
