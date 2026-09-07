/*! \file src/runtime/storage.cc
 * \brief 实现设备存储的 RAII 释放、外部所有权接管和范围校验。
 */

#include "kxc/runtime/storage.h"
#include "kxc/support/object_registration.h"

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "kxc/runtime/device_api.h"
#include "kxc/runtime/execution_observer.h"

namespace kxc {

// 执行观测钩子来自 runtime 子命名空间；本文件只消费其纯数据接口。
using runtime::AllocationInfo;
using runtime::AllocationKind;
using runtime::CurrentExecutionObserver;
using runtime::CurrentExecutionRunCorrelation;
using runtime::DispatchExecutionObservation;
using runtime::ExecutionObserver;

KXC_OBJECT_DEFINE(StorageNode)

// 按所有权策略释放底层内存，并把析构期错误降级为诊断。
StorageNode::~StorageNode() {
    // Object 析构不得抛异常；后端释放失败只能记录诊断。
    try {
        if (ownership == StorageOwnership::kOwned) {
            DeviceFree(device, data);
        } else if (ownership == StorageOwnership::kExternal && external_deleter) {
            external_deleter(data, deleter_context);
        }
    } catch (const std::exception& error) {
        std::cerr << "Storage release failed for " << device.ToString() << ": "
                  << error.what() << '\n';
    }
    data = nullptr;
}

// 从通用对象引用恢复 Storage，并执行运行时类型检查。
Storage::Storage(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<StorageNode>()) {
        throw std::runtime_error("ObjectRef does not contain StorageNode");
    }
}

// 在指定设备分配独占 Storage；ObjectRef 先接管节点以覆盖分配异常。
// 未装配观测器时只有一次空判断；记账路径被外层 hold 抑制时不重复上报。
Storage Storage::Alloc(const Device& device, size_t nbytes, size_t alignment) {
    auto* node = new StorageNode();
    Storage storage(node);
    node->device = device;
    node->capacity_bytes = nbytes;
    node->alignment = alignment;
    node->ownership = StorageOwnership::kOwned;
    ExecutionObserver* observer = CurrentExecutionObserver();
    const std::chrono::steady_clock::time_point begin =
        observer == nullptr ? std::chrono::steady_clock::time_point{}
                            : std::chrono::steady_clock::now();
    try {
        node->data = DeviceAlloc(device, nbytes, alignment);
    } catch (...) {
        if (observer != nullptr) {
            AllocationInfo info;
            info.device = device;
            info.bytes = nbytes;
            info.alignment = alignment;
            info.kind = AllocationKind::kFresh;
            info.duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
            try {
                throw;
            } catch (const std::exception& error) {
                info.error_message = error.what();
            } catch (...) {
                info.error_message = "unknown allocation failure";
            }
            const AllocationInfo recorded = std::move(info);
            DispatchExecutionObservation(observer, [&recorded](ExecutionObserver& sink) {
                sink.OnAllocation(recorded, CurrentExecutionRunCorrelation());
            });
        }
        throw;
    }
    if (observer != nullptr) {
        AllocationInfo info;
        info.device = device;
        info.bytes = nbytes;
        info.alignment = alignment;
        info.kind = AllocationKind::kFresh;
        info.duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - begin)
                               .count();
        DispatchExecutionObservation(observer, [&info](ExecutionObserver& sink) {
            sink.OnAllocation(info, CurrentExecutionRunCorrelation());
        });
    }
    return storage;
}

// 包装外部内存及其可选 deleter，不复制底层数据。
Storage Storage::FromExternal(const Device& device, void* data,
                              size_t capacity_bytes,
                              ExternalStorageDeleter deleter, void* context) {
    if (capacity_bytes != 0 && data == nullptr) {
        throw std::invalid_argument("non-empty external Storage requires data");
    }
    auto* node = new StorageNode();
    Storage storage(node);
    node->device = device;
    node->data = data;
    node->capacity_bytes = capacity_bytes;
    node->ownership = StorageOwnership::kExternal;
    // deleter 可空，此时 Storage 仅保活元数据而不释放外部地址。
    node->external_deleter = deleter;
    node->deleter_context = context;
    return storage;
}

// 返回底层地址所属的物理设备。
Device Storage::device() const {
    if (!defined()) throw std::runtime_error("undefined Storage has no device");
    return operator->()->device;
}

// 返回可访问容量；未定义对象的只读容量按零处理。
size_t Storage::capacity_bytes() const {
    return defined() ? operator->()->capacity_bytes : 0;
}

// 返回后端地址；零容量 Storage 可以合法持有空地址。
void* Storage::data() const {
    return defined() ? operator->()->data : nullptr;
}

// 验证 [offset, offset+nbytes) 是否完整落在 Storage 容量内。
void Storage::ValidateRange(size_t offset, size_t nbytes) const {
    if (!defined()) throw std::runtime_error("Storage range on undefined object");
    // 先减后比较，避免计算 offset + nbytes 时发生 size_t 溢出。
    if (offset > capacity_bytes() || nbytes > capacity_bytes() - offset) {
        throw std::out_of_range("Storage byte range exceeds capacity");
    }
}

// 返回经过 Storage 类型约束的底层节点。
const StorageNode* Storage::operator->() const {
    const auto* node = As<StorageNode>();
    if (!node) throw std::runtime_error("undefined or invalid Storage");
    return node;
}

}  // namespace kxc
