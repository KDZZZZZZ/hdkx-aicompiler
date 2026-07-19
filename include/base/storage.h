/*! \file include/base/storage.h
 * \brief 定义设备内存的 RAII 所有权对象和字节范围校验。
 */

#pragma once

#include <cstddef>

#include "base/device.h"
#include "base/object.h"

namespace kxc {

/*! \brief Storage 释放底层地址时采用的所有权策略。 */
enum class StorageOwnership {
    kOwned,     /*!< 由 DeviceAPI 分配并在析构时释放。 */
    kExternal,  /*!< 外部地址，析构时恰好调用一次外部 deleter。 */
    kWorkspace, /*!< 短期工作区，由上层池管理，不在此处释放。 */
};

/*! \brief 外部内存释放回调；context 由导入方原样保存。 */
using ExternalStorageDeleter = void (*)(void* data, void* context);

/*! \brief 只描述物理存储，不保存张量 shape 或 dtype。 */
class StorageNode final : public Object {
public:
    /*! \brief 底层地址所属的物理设备。 */
    Device device;
    /*! \brief 后端地址；仅 CPU Storage 可由主机直接解引用。 */
    void* data{nullptr};
    /*! \brief 可访问容量，单位为字节。 */
    size_t capacity_bytes{0};
    /*! \brief 分配地址的对齐要求，单位为字节。 */
    size_t alignment{0};
    /*! \brief 决定析构时由 DeviceAPI、外部 deleter 或上层工作区负责释放。 */
    StorageOwnership ownership{StorageOwnership::kOwned};
    /*! \brief kExternal Storage 析构时恰好调用一次的释放函数。 */
    ExternalStorageDeleter external_deleter{nullptr};
    /*! \brief 原样传递给 external_deleter 的非拥有型上下文。 */
    void* deleter_context{nullptr};

    /*! \brief 按 ownership 执行无异常析构，并确保外部 deleter 至多调用一次。 */
    ~StorageNode() override;
    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(StorageNode)

/*! \brief 共享持有一段设备内存的引用计数句柄。 */
class Storage : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    /*! \brief 从对象系统引用恢复 Storage，并校验节点类型。 */
    explicit Storage(const ObjectRef& ref);

    /*! \brief 在 device 上分配 nbytes；零容量 Storage 的 data 为 nullptr。 */
    static Storage Alloc(const Device& device, size_t nbytes, size_t alignment = 0);
    /*! \brief 接管外部地址及其 deleter，不复制数据。 */
    static Storage FromExternal(const Device& device, void* data,
                                size_t capacity_bytes,
                                ExternalStorageDeleter deleter,
                                void* context);

    /*! \brief 返回底层地址所属的物理设备。 */
    Device device() const;
    /*! \brief 返回可访问容量，单位为字节。 */
    size_t capacity_bytes() const;
    /*! \brief 返回不透明后端地址；不转移所有权。 */
    void* data() const;
    /*! \brief 校验 [offset, offset+nbytes) 未超出容量，单位均为字节。 */
    void ValidateRange(size_t offset, size_t nbytes) const;
    /*! \brief 返回由当前 Storage 共享持有的只读节点。 */
    const StorageNode* operator->() const;
};

}  // namespace kxc
