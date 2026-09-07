/*! \file src/runtime/internal/value_table.h
 * \brief Per-run ownership table for stable executable-plan value ids.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kxc/runtime/executable_plan.h"
#include "kxc/runtime/execution_observer.h"
#include "kxc/runtime/ndarray.h"

namespace kxc::runtime::internal {

class ValueTable final {
public:
    /*! \brief 安装分配记账观测；observer 为空时保持原有零开销路径。
     *  复用与别名由本表上报，新分配让 Storage::Alloc 的内层记录被
     *  hold 抑制后由本表统一上报，同一分配决策恰好产生一个事件。 */
    void ObserveAllocations(ExecutionObserver* observer,
                            ExecutionRunCorrelation correlation) {
        observer_ = observer;
        correlation_ = std::move(correlation);
    }

    bool Contains(int64_t value_id) const {
        return storage_by_value_.count(value_id) != 0;
    }

    void Bind(const ValueSpec& spec, NDArray value) {
        if (!spec.defined() || !value.defined()) {
            throw std::invalid_argument(
                "ValueTable bindings require a ValueSpec and NDArray");
        }
        ValidateValidBytes(spec, value);
        if (!storage_by_value_
                 .emplace(spec->value_id, spec->storage_id)
                 .second) {
            throw std::logic_error("ValueTable value id is already bound");
        }
        const auto existing = values_by_storage_.find(spec->storage_id);
        if (existing != values_by_storage_.end()) {
            retired_storage_.push_back(existing->second.storage());
            existing->second = std::move(value);
        } else {
            values_by_storage_.emplace(spec->storage_id, std::move(value));
        }
    }

    void Alias(const ValueSpec& spec, int64_t source_value_id) {
        if (!spec.defined() || Contains(spec->value_id) ||
            !Contains(source_value_id) || !spec->is_alias ||
            spec->write_mode != ValueWriteMode::kInPlace ||
            spec->alias_source_value_id != source_value_id ||
            spec->storage_id != storage_by_value_.at(source_value_id)) {
            throw std::logic_error("ValueTable alias contract is not bound");
        }
        ExecutionObserver* observer = observer_;
        const std::chrono::steady_clock::time_point begin =
            observer == nullptr ? std::chrono::steady_clock::time_point{}
                                : std::chrono::steady_clock::now();
        ValidateValidBytes(spec, Get(source_value_id));
        storage_by_value_.emplace(spec->value_id, spec->storage_id);
        if (observer != nullptr) {
            AllocationInfo info;
            info.device = spec->device;
            info.bytes = Get(source_value_id).NBytes();
            info.alignment =
                values_by_storage_.at(spec->storage_id).storage()->alignment;
            info.kind = AllocationKind::kAlias;
            info.duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
            DispatchExecutionObservation(observer, [&info, this](ExecutionObserver& sink) {
                sink.OnAllocation(info, correlation_);
            });
        }
    }

    NDArray Allocate(const ValueSpec& spec, uint64_t alignment) {
        if (!spec.defined() || alignment == 0) {
            throw std::invalid_argument(
                "ValueTable allocation requires a ValueSpec and alignment");
        }
        if (Contains(spec->value_id)) {
            throw std::logic_error("ValueTable value id is already bound");
        }
        ExecutionObserver* observer = observer_;
        const bool observing = observer != nullptr;
        const std::chrono::steady_clock::time_point begin =
            observing ? std::chrono::steady_clock::now()
                      : std::chrono::steady_clock::time_point{};
        NDArray value;
        bool reused = false;
        {
            // 记账期间抑制内层 Storage::Alloc 的上报，一次分配决策只产生
            // 一个事件；未装配观测器时没有 hold，路径与原来逐位一致。
            const ExecutionObservationHold hold(observing);
            const auto slot = values_by_storage_.find(spec->storage_id);
            if (slot != values_by_storage_.end() &&
                Compatible(slot->second, spec, alignment)) {
                value = slot->second;
                reused = true;
            } else {
                if (slot != values_by_storage_.end()) {
                    retired_storage_.push_back(slot->second.storage());
                }
                value = NDArray::Empty(spec.shape(), spec->dtype, spec->device,
                                       alignment);
            }
        }
        ValidateValidBytes(spec, value);
        storage_by_value_.emplace(spec->value_id, spec->storage_id);
        values_by_storage_.insert_or_assign(spec->storage_id, value);
        if (observing) {
            AllocationInfo info;
            info.device = spec->device;
            info.bytes = value.NBytes();
            info.alignment = alignment;
            info.kind = reused ? AllocationKind::kReuse : AllocationKind::kFresh;
            info.duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
            DispatchExecutionObservation(observer, [&info, this](ExecutionObserver& sink) {
                sink.OnAllocation(info, correlation_);
            });
        }
        return value;
    }

    NDArray Get(int64_t value_id) const {
        const auto binding = storage_by_value_.find(value_id);
        if (binding == storage_by_value_.end()) {
            throw std::out_of_range("ValueTable value id is not bound");
        }
        return values_by_storage_.at(binding->second);
    }

    Array<Storage> RetainedStorage() const {
        Array<Storage> retained;
        std::unordered_set<const Object*> seen;
        for (const auto& item : values_by_storage_) {
            Storage storage = item.second.storage();
            if (seen.insert(storage.get()).second) retained.push_back(storage);
        }
        for (const auto& storage : retired_storage_) {
            if (seen.insert(storage.get()).second) retained.push_back(storage);
        }
        return retained;
    }

private:
    static void ValidateValidBytes(const ValueSpec& spec,
                                   const NDArray& array) {
        if (spec->valid_bytes != -1 &&
            static_cast<uint64_t>(spec->valid_bytes) > array.NBytes()) {
            throw std::invalid_argument(
                "ValueTable valid bytes exceed the bound tensor");
        }
    }

    static bool Compatible(const NDArray& array, const ValueSpec& spec,
                           uint64_t alignment) {
        if (array.dtype().code != spec->dtype.code ||
            array.dtype().bits != spec->dtype.bits ||
            array.dtype().lanes != spec->dtype.lanes ||
            array.device() != spec->device ||
            array.storage()->alignment < alignment) {
            return false;
        }
        const Array<int64_t> actual = array.shape();
        const Array<int64_t> expected = spec.shape();
        if (actual.size() != expected.size()) return false;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != expected[i]) return false;
        }
        return true;
    }

    std::unordered_map<int64_t, int64_t> storage_by_value_;
    std::unordered_map<int64_t, NDArray> values_by_storage_;
    Array<Storage> retired_storage_;
    ExecutionObserver* observer_{nullptr};
    ExecutionRunCorrelation correlation_;
};

}  // namespace kxc::runtime::internal
