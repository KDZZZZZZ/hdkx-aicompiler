/*! \file src/runtime/internal/value_table.h
 * \brief Per-run ownership table for stable executable-plan value ids.
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "kxc/runtime/executable_plan.h"
#include "kxc/runtime/ndarray.h"

namespace kxc::runtime::internal {

class ValueTable final {
public:
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
        ValidateValidBytes(spec, Get(source_value_id));
        storage_by_value_.emplace(spec->value_id, spec->storage_id);
    }

    NDArray Allocate(const ValueSpec& spec, uint64_t alignment) {
        if (!spec.defined() || alignment == 0) {
            throw std::invalid_argument(
                "ValueTable allocation requires a ValueSpec and alignment");
        }
        if (Contains(spec->value_id)) {
            throw std::logic_error("ValueTable value id is already bound");
        }
        NDArray value;
        const auto slot = values_by_storage_.find(spec->storage_id);
        if (slot != values_by_storage_.end() &&
            Compatible(slot->second, spec, alignment)) {
            value = slot->second;
        } else {
            if (slot != values_by_storage_.end()) {
                retired_storage_.push_back(slot->second.storage());
            }
            value = NDArray::Empty(spec.shape(), spec->dtype, spec->device,
                                   alignment);
        }
        ValidateValidBytes(spec, value);
        storage_by_value_.emplace(spec->value_id, spec->storage_id);
        values_by_storage_.insert_or_assign(spec->storage_id, value);
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
};

}  // namespace kxc::runtime::internal
