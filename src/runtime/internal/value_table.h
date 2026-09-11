/*! \file src/runtime/internal/value_table.h
 * \brief Per-run ownership table for stable executable-plan value ids.
 *
 * The table is a stack of frames. A linear execution uses a single base frame
 * and behaves exactly as a flat map. A structured execution pushes a fresh
 * frame for each region activation (notably one per loop iteration) so a
 * static value id may be re-bound to a new tensor in a different frame while
 * still forbidding a double binding inside one frame.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
        for (auto frame = frames_.rbegin(); frame != frames_.rend(); ++frame) {
            if (frame->storage_by_value.count(value_id) != 0) return true;
        }
        return false;
    }

    /*! \brief True when `value_id` is bound in the innermost frame. */
    bool ContainsInCurrentFrame(int64_t value_id) const {
        return frames_.back().storage_by_value.count(value_id) != 0;
    }

    /*! \brief Push a fresh frame; existing bindings remain visible as parents. */
    void PushFrame() {
        frames_.emplace_back();
    }

    /*! \brief Discard the innermost frame. The base frame is never popped. */
    void PopFrame() {
        if (frames_.size() <= 1) {
            throw std::logic_error("ValueTable cannot pop its base frame");
        }
        frames_.pop_back();
    }

    void Bind(const ValueSpec& spec, NDArray value) {
        if (!spec.defined() || !value.defined()) {
            throw std::invalid_argument(
                "ValueTable bindings require a ValueSpec and NDArray");
        }
        ValidateValidBytes(spec, value);
        Frame& frame = frames_.back();
        if (!frame.storage_by_value
                 .emplace(spec->value_id, spec->storage_id)
                 .second) {
            throw std::logic_error("ValueTable value id is already bound");
        }
        lifetime_storage_.push_back(value.storage());
        const auto existing = frame.values_by_storage.find(spec->storage_id);
        if (existing != frame.values_by_storage.end()) {
            existing->second = std::move(value);
        } else {
            frame.values_by_storage.emplace(spec->storage_id, std::move(value));
        }
    }

    void Alias(const ValueSpec& spec, int64_t source_value_id) {
        const int64_t source_storage = SourceStorageId(source_value_id);
        if (!spec.defined() || ContainsInCurrentFrame(spec->value_id) ||
            !spec->is_alias ||
            spec->write_mode != ValueWriteMode::kInPlace ||
            spec->alias_source_value_id != source_value_id ||
            spec->storage_id != source_storage) {
            throw std::logic_error("ValueTable alias contract is not bound");
        }
        ExecutionObserver* observer = observer_;
        const std::chrono::steady_clock::time_point begin =
            observer == nullptr ? std::chrono::steady_clock::time_point{}
                                : std::chrono::steady_clock::now();
        ValidateValidBytes(spec, Get(source_value_id));
        frames_.back().storage_by_value.emplace(spec->value_id, spec->storage_id);
        lifetime_storage_.push_back(Get(source_value_id).storage());
        if (observer != nullptr) {
            const NDArray source = Get(source_value_id);
            AllocationInfo info;
            info.device = spec->device;
            info.bytes = source.NBytes();
            info.alignment = source.storage()->alignment;
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
        Frame& frame = frames_.back();
        if (frame.storage_by_value.count(spec->value_id) != 0) {
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
            const auto slot = frame.values_by_storage.find(spec->storage_id);
            if (slot != frame.values_by_storage.end() &&
                Compatible(slot->second, spec, alignment)) {
                value = slot->second;
                reused = true;
            } else {
                value = NDArray::Empty(spec.shape(), spec->dtype, spec->device,
                                       alignment);
            }
        }
        ValidateValidBytes(spec, value);
        frame.storage_by_value.emplace(spec->value_id, spec->storage_id);
        frame.values_by_storage.insert_or_assign(spec->storage_id, value);
        lifetime_storage_.push_back(value.storage());
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
        for (auto frame = frames_.rbegin(); frame != frames_.rend(); ++frame) {
            const auto binding = frame->storage_by_value.find(value_id);
            if (binding == frame->storage_by_value.end()) continue;
            return frame->values_by_storage.at(binding->second);
        }
        throw std::out_of_range("ValueTable value id is not bound");
    }

    Array<Storage> RetainedStorage() const {
        Array<Storage> retained;
        std::unordered_set<const Object*> seen;
        for (const auto& storage : lifetime_storage_) {
            if (storage.defined() && seen.insert(storage.get()).second) {
                retained.push_back(storage);
            }
        }
        return retained;
    }

private:
    struct Frame final {
        std::unordered_map<int64_t, int64_t> storage_by_value;
        std::unordered_map<int64_t, NDArray> values_by_storage;
    };

    int64_t SourceStorageId(int64_t source_value_id) const {
        for (auto frame = frames_.rbegin(); frame != frames_.rend(); ++frame) {
            const auto binding = frame->storage_by_value.find(source_value_id);
            if (binding != frame->storage_by_value.end()) return binding->second;
        }
        throw std::logic_error("ValueTable alias source is not bound");
    }

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

    std::vector<Frame> frames_{Frame{}};
    /*! \brief Every distinct Storage ever bound. Kept alive until the run's
     *  completion so pending operations never observe freed storage; frames
     *  only govern value-id lookup and single-bind-per-frame. */
    Array<Storage> lifetime_storage_;
    ExecutionObserver* observer_{nullptr};
    ExecutionRunCorrelation correlation_;
};

}  // namespace kxc::runtime::internal
