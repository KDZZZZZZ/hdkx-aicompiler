#pragma once

#include <string>
#include <vector>

#include "base/container.h"
#include "base/object.h"
#include "codegen/codegen.h"
#include "tir/stmt.h"

namespace kxc {
namespace codegen {

class CompiledKernelNode : public Object {
public:
    std::string kernel_name;
    CodeGenBackend backend;

    // 函数指针（LLVM JIT和C编译最终都得到这个）
    void* func_ptr{nullptr};

    // LLVM JIT路径资源（生命周期由JIT引擎管理）
    void* jit_resource{nullptr};

    // C路径资源
    std::string so_path;
    void* lib_handle{nullptr};

    // 函数签名信息
    Array<tir::Buffer> param_buffers;

    ~CompiledKernelNode() override;
    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(CompiledKernelNode)

class CompiledKernel : public ObjectRef {
public:
    using ObjectRef::ObjectRef;

    // 调用kernel（统一接口）
    void operator()(const std::vector<void*>& args) const;

    bool IsReady() const;
    const CompiledKernelNode* operator->() const {
        return static_cast<const CompiledKernelNode*>(object_);
    }
};

}  // namespace codegen
}  // namespace kxc
