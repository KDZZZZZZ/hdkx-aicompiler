/*! \file src/support/type_registry.cc
 * \brief Implements process-local Object type metadata registration.
 */

#include "kxc/support/object.h"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace kxc {
namespace {

struct TypeRegistryState {
    std::mutex mutex;
    std::unordered_map<std::string, std::unique_ptr<TypeInfo>> by_key;
    std::unordered_map<TypeIndex, const TypeInfo*> by_index;
    TypeIndex next_index{1000};
};

TypeRegistryState& GlobalTypeRegistryState() {
    static TypeRegistryState state;
    return state;
}

}  // namespace

TypeInfo::TypeInfo(TypeIndex runtime_index, std::string type_key)
    : runtime_index_(runtime_index), type_key_(std::move(type_key)) {}

const TypeInfo& TypeRegistry::Register(const std::string& type_key) {
    TypeRegistryState& state = GlobalTypeRegistryState();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.by_key.find(type_key);
    if (it != state.by_key.end()) return *it->second;

    const TypeIndex runtime_index =
        type_key == "Object" ? kKXC_OBJECT_TYPE : state.next_index;
    if (state.by_index.count(runtime_index) != 0) {
        throw std::logic_error("duplicate runtime type index");
    }

    auto info = std::unique_ptr<TypeInfo>(new TypeInfo(runtime_index, type_key));
    auto key_result = state.by_key.emplace(type_key, std::move(info));
    try {
        auto index_result =
            state.by_index.emplace(runtime_index, key_result.first->second.get());
        if (!index_result.second) {
            throw std::logic_error("duplicate runtime type index");
        }
    } catch (...) {
        state.by_key.erase(key_result.first);
        throw;
    }
    if (type_key != "Object") ++state.next_index;
    return *key_result.first->second;
}

const TypeInfo* TypeRegistry::FindByKey(std::string_view type_key) {
    TypeRegistryState& state = GlobalTypeRegistryState();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.by_key.find(std::string(type_key));
    return it == state.by_key.end() ? nullptr : it->second.get();
}

const TypeInfo* TypeRegistry::FindByIndex(TypeIndex runtime_index) {
    TypeRegistryState& state = GlobalTypeRegistryState();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.by_index.find(runtime_index);
    return it == state.by_index.end() ? nullptr : it->second;
}

}  // namespace kxc
