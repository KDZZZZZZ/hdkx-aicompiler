/*! \file include/base/ndarray.h
 * \brief 定义由 Storage 支撑的连续张量对象及其复制、视图接口。
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include "base/container.h"
#include "base/device_stream.h"

namespace kxc::runtime {

/*! \brief 将受支持的 dtype 名称解析为 DLPack 数据类型。 */
DLDataType DataTypeFromString(const std::string& dtype);

/*! \brief 保存张量元数据，并维护指向同一 Storage 的 DLTensor 借用视图。 */
class NDArrayNode final : public Object {
public:
    /*! \brief 共享持有张量数据所在的物理存储。 */
    Storage storage;
    /*! \brief 张量元素类型；与 dl_tensor.dtype 保持一致。 */
    DLDataType dtype{};
    /*! \brief 自有 shape 缓冲区，供 dl_tensor.shape 借用。 */
    std::vector<int64_t> shape_storage;
    /*! \brief 自有元素步长缓冲区，供 dl_tensor.strides 借用。 */
    std::vector<int64_t> strides_storage;
    /*! \brief 相对 Storage 起始地址的视图偏移，单位为字节。 */
    size_t byte_offset{0};
    /*! \brief 非拥有型 DLPack 视图，其指针生命周期受本节点约束。 */
    DLTensor dl_tensor{};

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(NDArrayNode)

/*! \brief 共享持有 Storage 的连续张量值对象。 */
class NDArray : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    /*! \brief 从对象系统引用恢复 NDArray，并校验节点类型。 */
    explicit NDArray(const ObjectRef& ref);

    /*! \brief 在指定 Device 上创建满足 alignment 的未初始化连续张量。 */
    static NDArray Empty(Array<int64_t> shape, DLDataType dtype, Device device,
                         size_t alignment = 0);
    /*! \brief 创建满足 alignment 的连续张量并由设备后端清零。 */
    static NDArray Zeros(Array<int64_t> shape, DLDataType dtype, Device device,
                         size_t alignment = 0);

    /*! \brief 返回张量 Storage 所在的物理设备。 */
    Device device() const;
    /*! \brief 返回张量逻辑维度的副本。 */
    Array<int64_t> shape() const;
    /*! \brief 返回 DLPack 元素类型。 */
    DLDataType dtype() const;
    /*! \brief 返回共享持有的底层 Storage。 */
    Storage storage() const;
    /*! \brief 返回逻辑张量覆盖的字节数；标量包含一个元素。 */
    size_t NBytes() const;
    /*! \brief 判断 strides 是否与行主序连续布局一致。 */
    bool IsContiguous() const;

    /*! \brief 在 CPU 字节缓冲区与本张量之间执行同步复制。 */
    void CopyFromBytes(const void* source, size_t nbytes) const;
    /*! \brief 将张量同步复制到 CPU 字节缓冲区。 */
    void CopyToBytes(void* destination, size_t nbytes) const;
    /*! \brief 从 shape、dtype 相同的连续 NDArray 同步复制。 */
    void CopyFrom(const NDArray& source) const;
    /*! \brief 在目标 Device 分配独立 Storage 并复制数据。 */
    NDArray CopyTo(const Device& destination) const;
    /*! \brief 异步复制并返回保活源、目标 Storage 的完成句柄。 */
    AsyncOperation CopyFromAsync(const NDArray& source,
                                 const DeviceStream& stream) const;
    /*! \brief 创建共享 Storage 的连续视图；byte_offset 单位为字节。 */
    NDArray CreateView(Array<int64_t> shape, Array<int64_t> strides,
                       size_t byte_offset) const;

    /*! \brief 返回由当前 NDArray 共享持有的只读节点。 */
    const NDArrayNode* operator->() const;
    /*! \brief 返回生命周期受 NDArray 约束的只读 DLTensor 借用指针。 */
    const DLTensor* operator*() const;
};

}  // namespace kxc::runtime
