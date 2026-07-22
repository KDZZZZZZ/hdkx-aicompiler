/*! \file src/ffi/registry.cc
 * \brief Implements the process-global PackedFunc registry.
 */

#include "kxc/ffi/registry.h"
#include "kxc/ffi/registration.h"

#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace kxc {

namespace {

std::unordered_map<std::string, PackedFunc>& GlobalFunctions() {
    static std::unordered_map<std::string, PackedFunc> functions;
    return functions;
}

}  // namespace

Registry::Registry() = default;

Registry::~Registry() = default;

Registry& Registry::Global() {
    RegisterBuiltins();
    static Registry instance;
    return instance;
}

void Registry::Set(const std::string& name, PackedFunc func) {
    auto& functions = GlobalFunctions();
    if (functions.count(name)) {
        throw std::runtime_error("Function " + name + " is already registered.");
    }
    functions[name] = std::move(func);
}

PackedFunc Registry::Get(const std::string& name) const {
    const auto& functions = GlobalFunctions();
    auto it = functions.find(name);
    return it == functions.end() ? PackedFunc() : it->second;
}

RegistryEntry::RegistryEntry(const std::string& name) : name_(name) {}

RegistryEntry& RegistryEntry::set_body(PackedFunc f) {
    Registry::Global().Set(name_, std::move(f));
    return *this;
}

}  // namespace kxc
