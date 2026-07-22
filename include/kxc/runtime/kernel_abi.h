/*! \file include/kxc/runtime/kernel_abi.h
 * \brief 定义后端无关的内核参数契约和启动元数据。
 */

#pragma once

#include <cstdint>
#include <string>

#include <dlpack/dlpack.h>

#include "kxc/support/container.h"
#include "kxc/runtime/device.h"
#include "kxc/runtime/ndarray.h"

namespace kxc::codegen {

/*! \brief 代码生成路径标识；实际可执行能力由具体产物契约决定。 */
enum class CodeGenBackend : int {
    kLLVM = 0,
    kCUDA = 1,
};

/*! \brief 保存 lowered TIR 常量参数段对应的稳定 key 顺序。 */
class KernelConstantKeysNode final : public Object {
public:
    KXC_OBJECT_DECLARE

private:
    friend class KernelConstantKeys;
    /*! \brief 私有 key 数组，防止外部通过容器别名改写 ABI 参数顺序。 */
    Array<String> keys_;
};


/*! \brief 常量 key 列表的类型安全、不可变对象句柄。 */
class KernelConstantKeys : public ObjectRef {
public:
    /*! \brief 深拷贝并校验 key 列表。 */
    explicit KernelConstantKeys(Array<String> keys);
    /*! \brief 从 PrimFunc attr 恢复列表，并验证节点类型和内容。 */
    explicit KernelConstantKeys(const ObjectRef& ref);

    /*! \brief 返回独立 Array，调用方修改不会影响 PrimFunc metadata。 */
    Array<String> keys() const;
    /*! \brief 校验每个 key 非空且在列表内唯一。 */
    void Validate() const;
    /*! \brief 返回经过运行时类型检查的只读节点。 */
    const KernelConstantKeysNode* operator->() const;
};

/*! \brief 动态输入维度的唯一哨兵；当前编译器仍只生成静态 shape。 */
constexpr int64_t kDynamicDimension = -1;

/*! \brief 参数在内核调用契约中的数据流角色。 */
enum class KernelArgRole : int {
    /*! \brief 由调用方提供且内核只读的输入张量。 */
    kInput = 0,
    /*! \brief 由编译结果的稳定 key 解析且内核只读的常量张量。 */
    kConstant = 1,
    /*! \brief 由调用方或后续强类型运行时分配且允许内核写入的输出张量。 */
    kOutput = 2,
};

/*! \brief 保存单个有序内核参数的完整张量契约。 */
class KernelArgSpecNode final : public Object {
public:
    /*! \brief 参数的诊断名称；同一签名内必须唯一且非空。 */
    String name;
    /*! \brief 参数的数据流角色，决定顺序、可变性和常量 key 规则。 */
    KernelArgRole role{KernelArgRole::kInput};
    /*! \brief 与 NDArray 直接比较的 DLPack dtype。 */
    DLDataType dtype{};
    /*! \brief 参数必须位于的物理设备。 */
    Device device;
    /*! \brief 数据首地址要求的字节对齐，使用固定位宽便于跨平台持久化。 */
    uint64_t alignment{1};
    /*! \brief 内核是否允许修改该参数指向的数据。 */
    bool mutable_data{false};
    /*! \brief 常量参数的稳定绑定 key；非定值参数必须为空。 */
    String constant_key;

    KXC_OBJECT_DECLARE

private:
    friend class KernelArgSpec;
    /*! \brief 私有 shape 存储，防止 Array 共享别名绕过构造期校验。 */
    Array<int64_t> shape_;
};


/*! \brief 单个有序内核参数契约的类型安全对象句柄。 */
class KernelArgSpec : public ObjectRef {
public:
    /*! \brief 构造并立即校验参数契约，禁止公开 API 产生非法节点。 */
    KernelArgSpec(String name, KernelArgRole role, DLDataType dtype,
                  Array<int64_t> shape, Device device, uint64_t alignment = 1,
                  bool mutable_data = false, String constant_key = String());
    /*! \brief 从通用对象引用恢复参数契约，并校验运行时节点类型。 */
    explicit KernelArgSpec(const ObjectRef& ref);

    /*! \brief 返回张量 rank；它只从 shape 计算，不保存重复状态。 */
    int rank() const;
    /*! \brief 返回 shape 的独立 Array，调用方修改副本不会改变已验证契约。 */
    Array<int64_t> shape() const;
    /*! \brief 校验 dtype、shape、角色、对齐和常量 key 的全部不变量。 */
    void Validate() const;
    /*! \brief 生成确定性诊断文本，不包含进程内地址或类型索引。 */
    std::string ToString() const;
    /*! \brief 返回经过类型检查的只读节点。 */
    const KernelArgSpecNode* operator->() const;
};

/*! \brief 保存单个内核入口的有序参数契约。 */
class KernelSignatureNode final : public Object {
public:
    /*! \brief 后端模块中的稳定入口符号。 */
    String symbol;
    KXC_OBJECT_DECLARE

private:
    friend class KernelSignature;
    /*! \brief 私有参数序列，防止 Array 共享别名删除或重排已验证 ABI。 */
    Array<KernelArgSpec> arguments_;
};


/*! \brief 后端无关的内核入口签名句柄。 */
class KernelSignature : public ObjectRef {
public:
    /*! \brief 构造并立即验证入口符号和完整参数序列。 */
    KernelSignature(String symbol, Array<KernelArgSpec> arguments);
    /*! \brief 从通用对象引用恢复签名，并校验运行时节点类型。 */
    explicit KernelSignature(const ObjectRef& ref);

    /*! \brief 校验参数分段、名称/key 唯一性以及输出契约。 */
    void Validate() const;
    /*! \brief 返回参数序列的独立 Array，保持节点内 ABI 顺序不可变。 */
    Array<KernelArgSpec> arguments() const;
    /*! \brief 判断签名是否含动态输入；动态输出会在 Validate 中被拒绝。 */
    bool has_dynamic_input_shape() const;
    /*! \brief 生成参数顺序稳定的诊断文本。 */
    std::string ToString() const;
    /*! \brief 返回经过类型检查的只读节点。 */
    const KernelSignatureNode* operator->() const;
};

/*! \brief 三维 CUDA grid/block 尺寸；CPU 后端固定使用 1x1x1。 */
struct Dim3 {
    uint32_t x{1};
    uint32_t y{1};
    uint32_t z{1};
};

/*! \brief 保存某个后端产物的设备和启动参数，不描述张量调用契约。 */
class KernelLaunchMetadataNode final : public Object {
public:
    /*! \brief 内核实际执行的物理设备。 */
    Device device;
    /*! \brief 生成该可执行产物的真实后端。 */
    CodeGenBackend backend{CodeGenBackend::kLLVM};
    /*! \brief CUDA grid 尺寸；LLVM 后端必须保持 1x1x1。 */
    Dim3 grid;
    /*! \brief CUDA block 尺寸；LLVM 后端必须保持 1x1x1。 */
    Dim3 block;
    /*! \brief 每次 CUDA launch 请求的动态 shared memory 字节数。 */
    uint64_t dynamic_shared_memory_bytes{0};

    KXC_OBJECT_DECLARE
};


/*! \brief 后端启动元数据的类型安全对象句柄。 */
class KernelLaunchMetadata : public ObjectRef {
public:
    /*! \brief 构造并校验 backend、device 和三维启动尺寸。 */
    KernelLaunchMetadata(Device device, CodeGenBackend backend,
                         Dim3 grid = {}, Dim3 block = {},
                         uint64_t dynamic_shared_memory_bytes = 0);
    /*! \brief 从通用对象引用恢复元数据，并校验运行时节点类型。 */
    explicit KernelLaunchMetadata(const ObjectRef& ref);

    /*! \brief 校验 CPU/LLVM 与 CUDA 各自允许的启动元数据。 */
    void Validate() const;
    /*! \brief 生成确定性的后端、设备和 launch 尺寸文本。 */
    std::string ToString() const;
    /*! \brief 返回经过类型检查的只读节点。 */
    const KernelLaunchMetadataNode* operator->() const;
};

/*!
 * \brief 从 lowered PrimFunc 的参数、Buffer 和结构化 attrs 构建内核签名。
 * \param function 已完成 TIR pass 的函数。
 * \param constants 由 lowering 保活的 key 到 NDArray payload 映射。
 * \param target 决定每个参数物理设备的编译目标。
 * \param symbol 后端模块中的入口符号；显式值是最终签名的事实来源。
 */
}  // namespace kxc::codegen
