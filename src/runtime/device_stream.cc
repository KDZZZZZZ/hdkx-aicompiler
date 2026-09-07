/*! \file src/runtime/device_stream.cc
 * \brief 实现 stream/event 生命周期和异步复制的资源保活协议。
 */

#include "kxc/runtime/device_stream.h"
#include "kxc/support/object_registration.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "kxc/runtime/device_api.h"
#include "kxc/runtime/execution_observer.h"

namespace kxc {

// 执行观测钩子来自 runtime 子命名空间；本文件只消费其纯数据接口。
using runtime::AllocationInfo;
using runtime::CopyInfo;
using runtime::CurrentExecutionObserver;
using runtime::CurrentExecutionRunCorrelation;
using runtime::DispatchExecutionObservation;
using runtime::ExecutionObserver;

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

// 每个完成回调在 Wait/IsReady/析构之间恰好结算一次。结算时 completed 已经
// 为 true，因此回调内再等待同一句柄会立即返回；回调异常被吞掉，观测不能
// 改变执行结果。
void SettleCompletionCallbacks(AsyncOperationNode& node, bool at_registration) noexcept {
    std::vector<AsyncCompletionCallback> callbacks;
    {
        std::lock_guard<std::mutex> lock(node.mutex);
        if (node.completion_settled) return;
        node.completion_settled = true;
        callbacks = std::move(node.completion_callbacks);
        node.completion_callbacks.clear();
    }
    for (auto& callback : callbacks) {
        if (!callback) continue;
        try {
            callback(at_registration);
        } catch (...) {
            // 完成观测不得影响执行结果。
        }
    }
}

// 上报一次同步拷贝记账；异常文本取自原始异常，不改变抛出行为。
void DispatchCopyFailure(ExecutionObserver* observer, const CopyInfo& copy) noexcept {
    if (observer == nullptr) return;
    DispatchExecutionObservation(observer, [&copy](ExecutionObserver& sink) {
        sink.OnCopy(copy, CurrentExecutionRunCorrelation());
    });
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
// 析构是完成观测的最后一个结算点：真正观测到完成（等待 event 成功，或
// 无后端 event 的 CPU 完成态句柄）时结算回调；无法确认完成时不结算，
// 也不虚构完成。
AsyncOperationNode::~AsyncOperationNode() {
    if (!completed && backend_event == nullptr) {
        // 无后端 event 的句柄没有可等待的异步工作（CPU 语义下视为完成）。
        completed = true;
    } else if (!completed && stream.defined()) {
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
    if (completed) SettleCompletionCallbacks(*this, false);
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

// 阻塞等待并恰好释放一次完成事件。
void AsyncOperation::Wait() const {
    const auto* node = operator->();
    // Wait 与 IsReady 可能并发，互斥锁保证 event 恰好释放一次。
    std::unique_lock<std::mutex> lock(node->mutex);
    if (!node->completed) {
        DeviceAPI* api = GetDeviceAPI(node->stream.device().device_type());
        api->WaitEvent(node->stream.device(), node->backend_event);
        api->FreeEvent(node->stream.device(), node->backend_event);
        const_cast<AsyncOperationNode*>(node)->backend_event = nullptr;
        node->completed = true;
    }
    lock.unlock();
    // 完成观测在锁外结算，回调内部仍可安全进入同一句柄的非等待路径。
    SettleCompletionCallbacks(*const_cast<AsyncOperationNode*>(node), false);
}

// 非阻塞查询完成状态；首次观察到完成时回收 event。
bool AsyncOperation::IsReady() const {
    const auto* node = operator->();
    std::unique_lock<std::mutex> lock(node->mutex);
    if (!node->completed) {
        DeviceAPI* api = GetDeviceAPI(node->stream.device().device_type());
        if (!api->QueryEvent(node->stream.device(), node->backend_event)) return false;
        api->FreeEvent(node->stream.device(), node->backend_event);
        const_cast<AsyncOperationNode*>(node)->backend_event = nullptr;
        node->completed = true;
    }
    lock.unlock();
    SettleCompletionCallbacks(*const_cast<AsyncOperationNode*>(node), false);
    return true;
}

// 注册一次性完成观测；注册时已完成则立即触发（每次注册独立触发一次），
// 否则存储后在 Wait/IsReady/析构首次观测到完成时对所有已注册回调恰好
// 结算一次。同一句柄可以被多方观测：先注册者随结算触发，后注册者若
// 遇到已完成句柄则注册即触发。
void AsyncOperation::ObserveCompletion(AsyncCompletionCallback callback) const {
    if (!callback) return;
    const auto* node = operator->();
    bool fire_now = false;
    {
        std::lock_guard<std::mutex> lock(node->mutex);
        if (!node->completed) {
            node->completion_callbacks.push_back(std::move(callback));
        } else {
            fire_now = true;
        }
    }
    if (fire_now) {
        // 注册时已完成：立即触发一次；completed 已为 true，回调内的嵌套
        // 等待会直接返回。回调异常被吞掉，观测不能改变执行结果。
        try {
            callback(true);
        } catch (...) {
        }
    }
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
    // 未装配观测器时只有一次空判断；没有发生的拷贝不虚构事件。
    ExecutionObserver* observer = CurrentExecutionObserver();
    const std::chrono::steady_clock::time_point begin =
        observer == nullptr ? std::chrono::steady_clock::time_point{}
                            : std::chrono::steady_clock::now();
    try {
        DeviceCopySync(from.device(), from.data(), from_offset, to.device(), to.data(),
                       to_offset, nbytes);
    } catch (...) {
        if (observer != nullptr) {
            CopyInfo info;
            info.from_device = from.device();
            info.to_device = to.device();
            info.bytes = nbytes;
            info.duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
            try {
                throw;
            } catch (const std::exception& error) {
                info.error_message = error.what();
            } catch (...) {
                info.error_message = "unknown copy failure";
            }
            DispatchCopyFailure(observer, info);
        }
        throw;
    }
    if (observer != nullptr) {
        CopyInfo info;
        info.from_device = from.device();
        info.to_device = to.device();
        info.bytes = nbytes;
        info.duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - begin)
                               .count();
        DispatchExecutionObservation(observer, [&info](ExecutionObserver& sink) {
            sink.OnCopy(info, CurrentExecutionRunCorrelation());
        });
    }
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
        // CPU 异步拷贝在主机上同步完成；提交动作的耗时即主机执行耗时。
        ExecutionObserver* observer = CurrentExecutionObserver();
        const std::chrono::steady_clock::time_point begin =
            observer == nullptr ? std::chrono::steady_clock::time_point{}
                                : std::chrono::steady_clock::now();
        try {
            api->CopyDataAsync(from_device, from.data(), from_offset, to_device,
                               to.data(), to_offset, nbytes, stream->backend_handle);
        } catch (...) {
            if (observer != nullptr) {
                CopyInfo info;
                info.from_device = from_device;
                info.to_device = to_device;
                info.bytes = nbytes;
                info.submitted_async = true;
                info.duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - begin)
                                       .count();
                try {
                    throw;
                } catch (const std::exception& error) {
                    info.error_message = error.what();
                } catch (...) {
                    info.error_message = "unknown copy failure";
                }
                DispatchCopyFailure(observer, info);
            }
            throw;
        }
        AsyncOperation operation =
            AsyncOperation::Completed(stream, std::move(retained));
        if (observer != nullptr) {
            CopyInfo info;
            info.from_device = from_device;
            info.to_device = to_device;
            info.bytes = nbytes;
            info.submitted_async = true;
            info.duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
            AsyncCompletionCallback completion;
            DispatchExecutionObservation(observer, [&info, &completion](ExecutionObserver& sink) {
                completion = sink.OnCopySubmitted(info, CurrentExecutionRunCorrelation());
            });
            // CPU 句柄已完成：注册即触发，两个观测点在同一主机时刻。
            if (completion) operation.ObserveCompletion(std::move(completion));
        }
        return operation;
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
        // 观测提交失败不改变既有失败处理顺序：先记录，再同步确认完成。
        ExecutionObserver* observer = CurrentExecutionObserver();
        if (observer != nullptr) {
            CopyInfo info;
            info.from_device = from_device;
            info.to_device = to_device;
            info.bytes = nbytes;
            info.submitted_async = true;
            try {
                throw;
            } catch (const std::exception& error) {
                info.error_message = error.what();
            } catch (...) {
                info.error_message = "unknown copy failure";
            }
            DispatchCopyFailure(observer, info);
        }
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
    ExecutionObserver* observer = CurrentExecutionObserver();
    if (observer != nullptr) {
        CopyInfo info;
        info.from_device = from_device;
        info.to_device = to_device;
        info.bytes = nbytes;
        info.submitted_async = true;
        AsyncCompletionCallback completion;
        DispatchExecutionObservation(observer, [&info, &completion](ExecutionObserver& sink) {
            completion = sink.OnCopySubmitted(info, CurrentExecutionRunCorrelation());
        });
        if (completion) operation.ObserveCompletion(std::move(completion));
    }
    return operation;
}

}  // namespace kxc
