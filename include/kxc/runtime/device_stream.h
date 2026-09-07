/*! \file include/kxc/runtime/device_stream.h
 * \brief 定义设备执行流、异步完成句柄和带生命周期保活的复制接口。
 */

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "kxc/support/container.h"
#include "kxc/runtime/storage.h"

namespace kxc {

/*! \brief 完成观测回调；at_registration 表示注册时操作已经完成。 */
using AsyncCompletionCallback = std::function<void(bool at_registration)>;

/*! \brief 包装后端 stream 句柄，并记录其所属 Device 和所有权。 */
class DeviceStreamNode final : public Object {
public:
    /*! \brief stream 所属的物理设备。 */
    Device device;
    /*! \brief 非拥有型后端句柄；是否销毁由 owns_handle 决定。 */
    void* backend_handle{nullptr};
    /*! \brief 为 true 时析构函数负责释放 backend_handle。 */
    bool owns_handle{false};

    /*! \brief 仅为拥有型 stream 释放后端句柄，析构错误不向外抛出。 */
    ~DeviceStreamNode() override;
    KXC_OBJECT_DECLARE
};


/*! \brief 绑定到单个物理 Device 的执行流。 */
class DeviceStream : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    /*! \brief 从对象系统引用恢复 DeviceStream，并校验节点类型。 */
    explicit DeviceStream(const ObjectRef& ref);

    /*! \brief 创建拥有后端句柄的非默认 stream。 */
    static DeviceStream Create(const Device& device);
    /*! \brief 创建引用默认 stream 的非拥有型对象。 */
    static DeviceStream Default(const Device& device);

    /*! \brief 返回 stream 所属的物理设备。 */
    Device device() const;
    /*! \brief 判断该对象是否引用后端默认 stream。 */
    bool is_default() const;
    /*! \brief 等待该 stream 上此前排入的工作完成。 */
    void Sync() const;
    /*! \brief 返回由当前句柄共享持有的只读节点。 */
    const DeviceStreamNode* operator->() const;
};

/*! \brief 用 event 表示完成状态，并保活异步工作依赖的对象。 */
class AsyncOperationNode final : public Object {
public:
    /*! \brief 提交异步工作的 stream，并间接保活其后端句柄。 */
    DeviceStream stream;
    /*! \brief 在 event 完成前不得释放的输入、输出物理存储。 */
    Array<Storage> retained_storage;
    /*! \brief 异步后端执行期必须保活的可执行对象；同步后端可以为空。 */
    ObjectRef retained_executable;
    /*! \brief Append-only owners retained until the completion handle is destroyed. */
    std::vector<std::shared_ptr<void>> retained_contexts;
    /*! \brief 完成事件的拥有型后端句柄；完成或析构时释放。 */
    void* backend_event{nullptr};
    /*! \brief event 已完成且资源已回收时为 true。 */
    mutable bool completed{false};
    /*! \brief 已注册但尚未结算的完成观测；结算时恰好各触发一次。 */
    mutable std::vector<AsyncCompletionCallback> completion_callbacks;
    /*! \brief 完成观测是否已全部结算；保证恰好结算一次。 */
    mutable bool completion_settled{false};
    /*! \brief 串行化完成查询、等待和 event 释放。 */
    mutable std::mutex mutex;

    /*! \brief 最后一个引用释放时等待 event，避免异步依赖被提前析构。 */
    ~AsyncOperationNode() override;
    KXC_OBJECT_DECLARE
};


/*! \brief 可等待、可轮询且析构时保证资源不被提前释放的完成句柄。 */
class AsyncOperation : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    /*! \brief 从对象系统引用恢复 AsyncOperation，并校验节点类型。 */
    explicit AsyncOperation(const ObjectRef& ref);

    /*! \brief 构造已经完成的操作，常用于同步 CPU 路径。 */
    static AsyncOperation Completed(const DeviceStream& stream,
                                    Array<Storage> retained = {});
    /*! \brief 接管 event，并由完成句柄持续保活 storage/executable。 */
    static AsyncOperation Pending(const DeviceStream& stream, void* event,
                                  Array<Storage> retained,
                                  ObjectRef executable = ObjectRef());

    /*! \brief Appends storage and one owner, retained until this handle is destroyed. */
    void RetainDependencies(Array<Storage> retained,
                            std::shared_ptr<void> context) const;

    /*! \brief 阻塞等待并释放 event；可重复调用。 */
    void Wait() const;
    /*! \brief 非阻塞查询；就绪后释放 event 并转入 completed 状态。 */
    bool IsReady() const;
    /*! \brief 注册一次性完成观测。观测点为 Wait、IsReady 和析构，注册时
     *  已完成则立即触发；每个回调在三者之间恰好结算一次，回调内部不得再
     *  等待同一句柄（此时 completed 已为 true，嵌套等待会直接返回），
     *  也不得抛出异常。已完成且已结算后再注册的回调不会触发。 */
    void ObserveCompletion(AsyncCompletionCallback callback) const;
    /*! \brief 返回执行该异步操作的物理设备。 */
    Device device() const;
    /*! \brief 返回由当前句柄共享持有的只读节点。 */
    const AsyncOperationNode* operator->() const;
};

/*! \brief 校验容量后执行同步 Storage 字节复制。 */
void StorageCopySync(const Storage& from, size_t from_offset,
                     const Storage& to, size_t to_offset, size_t nbytes);
/*! \brief 将复制排入 stream，并通过返回对象保活两端 Storage。 */
AsyncOperation StorageCopyAsync(const Storage& from, size_t from_offset,
                                const Storage& to, size_t to_offset,
                                size_t nbytes, const DeviceStream& stream);

}  // namespace kxc
