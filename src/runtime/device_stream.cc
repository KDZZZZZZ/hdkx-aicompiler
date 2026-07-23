/*! \file src/runtime/device_stream.cc
 * \brief 实现 stream/event 生命周期和异步复制的资源保活协议。
 */

#include "kxc/runtime/device_stream.h"
#include "kxc/support/object_registration.h"

#include <iostream>
#include <exception>
#include <stdexcept>

#include "kxc/runtime/device_api.h"

namespace kxc {

KXC_OBJECT_DEFINE(DeviceStreamNode)
KXC_OBJECT_DEFINE(AsyncOperationNode)

namespace {

// 承载无法确认完成的异步依赖，避免后端仍访问时发生释放后使用。
struct FailedAsyncRetention {
    DeviceStream stream;
    Array<Storage> storage;
    ObjectRef executable;
    std::vector<std::shared_ptr<void>> contexts;
    void* event;
};

// 无法证明后端工作已结束时把依赖保留到进程结束；连保活都失败时只能终止。
void RetainFailedNode(const AsyncOperationNode& node) noexcept {
    try {
        (void)new FailedAsyncRetention{node.stream, node.retained_storage,
                                       node.retained_executable,
                                       node.retained_contexts,
                                       node.backend_event};
    } catch (...) {
        std::terminate();
    }
}

// 保留完整异步操作，使其 event、Storage 和可执行对象都不被提前释放。
void RetainFailedOperation(const AsyncOperation& operation) noexcept {
    try {
        (void)new AsyncOperation(operation);
    } catch (...) {
        std::terminate();
    }
}

// 判断两个等长半开字节区间是否重叠；调用前已经完成容量校验。
bool RangesOverlap(size_t lhs_offset, size_t rhs_offset, size_t nbytes) {
    return nbytes != 0 && lhs_offset < rhs_offset + nbytes &&
           rhs_offset < lhs_offset + nbytes;
}

}  // namespace

// 仅销毁显式拥有的后端 stream，析构期错误降级为诊断。
DeviceStreamNode::~DeviceStreamNode() {
    if (!owns_handle) return;
    try {
        GetDeviceAPI(device.device_type())->FreeStream(device, backend_handle);
    } catch (const std::exception& error) {
        std::cerr << "DeviceStream release failed for " << device.ToString()
                  << ": " << error.what() << '\n';
    }
}

// 从通用对象引用恢复 DeviceStream，并执行运行时类型检查。
DeviceStream::DeviceStream(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<DeviceStreamNode>()) {
        throw std::runtime_error("ObjectRef does not contain DeviceStreamNode");
    }
}

// 创建并独占一个后端 stream 句柄。
DeviceStream DeviceStream::Create(const Device& device) {
    auto* node = new DeviceStreamNode();
    DeviceStream stream(node);
    node->device = device;
    node->backend_handle = GetDeviceAPI(device.device_type())->CreateStream(device);
    node->owns_handle = true;
    return stream;
}

// 创建不拥有后端资源的默认 stream 视图，空句柄语义由后端解释。
DeviceStream DeviceStream::Default(const Device& device) {
    auto* node = new DeviceStreamNode();
    DeviceStream stream(node);
    node->device = device;
    node->backend_handle = nullptr;
    node->owns_handle = false;
    return stream;
}

// 返回 stream 所属的物理设备。
Device DeviceStream::device() const { return operator->()->device; }
// 判断该对象是否仅引用后端默认 stream。
bool DeviceStream::is_default() const {
    return !operator->()->owns_handle && operator->()->backend_handle == nullptr;
}

// 等待当前 stream 中此前提交的全部工作完成。
void DeviceStream::Sync() const {
    GetDeviceAPI(device().device_type())
        ->StreamSync(device(), operator->()->backend_handle);
}

// 返回经过 DeviceStream 类型约束的底层节点。
const DeviceStreamNode* DeviceStream::operator->() const {
    const auto* node = As<DeviceStreamNode>();
    if (!node) throw std::runtime_error("undefined or invalid DeviceStream");
    return node;
}

// 最后一个引用释放时完成并回收 event，同时保证被保活资源可安全析构。
AsyncOperationNode::~AsyncOperationNode() {
    if (completed || backend_event == nullptr || !stream.defined()) return;
    // 析构不能抛；等待失败时把全部依赖转入永久保活区。
    try {
        DeviceAPI* api = GetDeviceAPI(stream.device().device_type());
        api->WaitEvent(stream.device(), backend_event);
        api->FreeEvent(stream.device(), backend_event);
        backend_event = nullptr;
        completed = true;
    } catch (const std::exception& error) {
        RetainFailedNode(*this);
        std::cerr << "AsyncOperation completion failed: " << error.what() << '\n';
    }
}

// 从通用对象引用恢复 AsyncOperation，并执行运行时类型检查。
AsyncOperation::AsyncOperation(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<AsyncOperationNode>()) {
        throw std::runtime_error("ObjectRef does not contain AsyncOperationNode");
    }
}

// 构造已完成操作，仍保留依赖直至句柄释放以统一生命周期语义。
AsyncOperation AsyncOperation::Completed(const DeviceStream& stream,
                                         Array<Storage> retained) {
    auto* node = new AsyncOperationNode();
    AsyncOperation operation(node);
    node->stream = stream;
    node->retained_storage = std::move(retained);
    node->completed = true;
    return operation;
}

// 构造待完成操作并接管 event、Storage 与可执行对象的生命周期。
AsyncOperation AsyncOperation::Pending(const DeviceStream& stream, void* event,
                                       Array<Storage> retained,
                                       ObjectRef executable) {
    AsyncOperationNode* node = nullptr;
    // 节点构造失败时回收尚未绑定的 event，并保留原始构造异常。
    try {
        node = new AsyncOperationNode();
    } catch (...) {
        std::exception_ptr original = std::current_exception();
        try {
            GetDeviceAPI(stream.device().device_type())
                ->FreeEvent(stream.device(), event);
        } catch (...) {
        }
        std::rethrow_exception(original);
    }
    AsyncOperation operation(node);
    node->stream = stream;
    node->backend_event = event;
    node->retained_storage = std::move(retained);
    node->retained_executable = std::move(executable);
    return operation;
}

void AsyncOperation::RetainDependencies(Array<Storage> retained,
                                        std::shared_ptr<void> context) const {
    if (!context) {
        throw std::invalid_argument(
            "AsyncOperation retained context must not be null");
    }
    const auto* node = operator->();
    std::lock_guard<std::mutex> lock(node->mutex);
    auto* mutable_node = const_cast<AsyncOperationNode*>(node);
    for (const auto& storage : retained) {
        if (!storage.defined()) {
            throw std::invalid_argument(
                "AsyncOperation cannot retain undefined Storage");
        }
        bool already_retained = false;
        for (const auto& existing : mutable_node->retained_storage) {
            if (existing.get() == storage.get()) {
                already_retained = true;
                break;
            }
        }
        if (!already_retained) {
            mutable_node->retained_storage.push_back(storage);
        }
    }
    mutable_node->retained_contexts.push_back(std::move(context));
}

// 阻塞等待并恰好释放一次完成 event。
void AsyncOperation::Wait() const {
    const auto* node = operator->();
    // Wait 与 IsReady 可能并发，互斥锁保证 event 恰好释放一次。
    std::lock_guard<std::mutex> lock(node->mutex);
    if (node->completed) return;
    DeviceAPI* api = GetDeviceAPI(node->stream.device().device_type());
    api->WaitEvent(node->stream.device(), node->backend_event);
    api->FreeEvent(node->stream.device(), node->backend_event);
    const_cast<AsyncOperationNode*>(node)->backend_event = nullptr;
    node->completed = true;
}

// 非阻塞查询完成状态；首次观察到完成时回收 event。
bool AsyncOperation::IsReady() const {
    const auto* node = operator->();
    std::lock_guard<std::mutex> lock(node->mutex);
    if (node->completed) return true;
    DeviceAPI* api = GetDeviceAPI(node->stream.device().device_type());
    if (!api->QueryEvent(node->stream.device(), node->backend_event)) return false;
    api->FreeEvent(node->stream.device(), node->backend_event);
    const_cast<AsyncOperationNode*>(node)->backend_event = nullptr;
    node->completed = true;
    return true;
}

// 返回承载该异步操作的 stream 设备。
Device AsyncOperation::device() const { return operator->()->stream.device(); }

// 返回经过 AsyncOperation 类型约束的底层节点。
const AsyncOperationNode* AsyncOperation::operator->() const {
    const auto* node = As<AsyncOperationNode>();
    if (!node) throw std::runtime_error("undefined or invalid AsyncOperation");
    return node;
}

// 校验两个 Storage 区间后执行同步复制，并限制非 CPU 重叠语义。
void StorageCopySync(const Storage& from, size_t from_offset,
                     const Storage& to, size_t to_offset, size_t nbytes) {
    from.ValidateRange(from_offset, nbytes);
    to.ValidateRange(to_offset, nbytes);
    // CPU 后端支持重叠 memmove；其他后端暂不接受重叠区间。
    if (from.get() == to.get()) {
        if (from_offset == to_offset || nbytes == 0) return;
        if (RangesOverlap(from_offset, to_offset, nbytes) &&
            from.device().device_type() != kCPU) {
            throw std::invalid_argument("overlapping non-CPU Storage copy is unsupported");
        }
    }
    DeviceCopySync(from.device(), from.data(), from_offset, to.device(), to.data(),
                   to_offset, nbytes);
}

// 按复制方向选择 stream 设备，提交复制并保活所有异步依赖。
AsyncOperation StorageCopyAsync(const Storage& from, size_t from_offset,
                                const Storage& to, size_t to_offset,
                                size_t nbytes, const DeviceStream& stream) {
    from.ValidateRange(from_offset, nbytes);
    to.ValidateRange(to_offset, nbytes);
    const Device from_device = from.device();
    const Device to_device = to.device();
    // H2D 使用目标 CUDA 流，D2H/D2D 使用源 CUDA 流，H2H 立即完成。
    Device expected = from_device.device_type() == kCPU ? to_device : from_device;
    if (from_device.device_type() == kCPU && to_device.device_type() == kCPU) {
        expected = from_device;
    }
    if (stream.device() != expected) {
        throw std::invalid_argument("stream device does not match copy direction");
    }
    if (from.get() == to.get()) {
        if (from_offset == to_offset || nbytes == 0) {
            return AsyncOperation::Completed(stream, Array<Storage>{from, to});
        }
        if (RangesOverlap(from_offset, to_offset, nbytes) &&
            from_device.device_type() != kCPU) {
            throw std::invalid_argument("overlapping non-CPU Storage copy is unsupported");
        }
    }
    DeviceAPI* api = GetDeviceAPI(expected.device_type());
    Array<Storage> retained{from, to};
    if (expected.device_type() == kCPU) {
        api->CopyDataAsync(from_device, from.data(), from_offset, to_device,
                           to.data(), to_offset, nbytes, stream->backend_handle);
        return AsyncOperation::Completed(stream, std::move(retained));
    }
    // 先建立 event 和保活对象再入队，确保任何已提交 DMA 都有所有权承载者。
    void* event = api->CreateEvent(expected);
    AsyncOperation operation =
        AsyncOperation::Pending(stream, event, std::move(retained));
    try {
        api->CopyDataAsync(from_device, from.data(), from_offset, to_device,
                           to.data(), to_offset, nbytes, stream->backend_handle);
        api->RecordEvent(expected, event, stream->backend_handle);
    } catch (...) {
        // 入队后失败先同步确认完成；同步也失败时永久保活并重抛原始错误。
        std::exception_ptr original = std::current_exception();
        try {
            stream.Sync();
        } catch (...) {
            RetainFailedOperation(operation);
            std::rethrow_exception(original);
        }
        try {
            operation.Wait();
        } catch (...) {
            RetainFailedOperation(operation);
        }
        std::rethrow_exception(original);
    }
    return operation;
}

}  // namespace kxc
