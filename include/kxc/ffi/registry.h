/*! \file include/kxc/ffi/registry.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once
#include "kxc/ffi/packed_func.h"
#include <string>

namespace kxc {

class Registry {
public:
    static Registry& Global();

    void Set(const std::string& name, PackedFunc func);
    PackedFunc Get(const std::string& name) const;

private:
    Registry();
    ~Registry();
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;
};

class RegistryEntry {
public:
    explicit RegistryEntry(const std::string& name);
    RegistryEntry& set_body(PackedFunc f);
private:
    std::string name_;
};

} // namespace kxc
