#pragma once
#include "packedfunc.h"
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace kxc {

class Registry {
public:
    static Registry& Global() {
        static Registry instance;
        return instance;
    }

    void Set(const std::string& name, PackedFunc func) {
        if (functions_.count(name)) {
            // Or just overwrite, depending on desired behavior
            throw std::runtime_error("Function " + name + " is already registered.");
        }
        functions_[name] = std::move(func);
    }

    PackedFunc Get(const std::string& name) const {
        auto it = functions_.find(name);
        if (it == functions_.end()) {
            return PackedFunc();
        }
        return it->second;
    }

private:
    Registry() = default;
    ~Registry() = default;
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;

    std::unordered_map<std::string, PackedFunc> functions_;
};

class RegistryEntry {
public:
    RegistryEntry(const std::string& name) : name_(name) {}
    RegistryEntry& set_body(PackedFunc f) {
        Registry::Global().Set(name_, f);
        return *this;
    }
private:
    std::string name_;
};

// 辅助宏，简化注册流程
#define KXC_STR_CONCAT_(__x, __y) __x##__y
#define KXC_STR_CONCAT(__x, __y) KXC_STR_CONCAT_(__x, __y)

#define KXC_REGISTER_GLOBAL(Name) \
    static ::kxc::RegistryEntry KXC_STR_CONCAT(__kxc_reg_entry_, __COUNTER__) = ::kxc::RegistryEntry(Name)

} // namespace kxc
