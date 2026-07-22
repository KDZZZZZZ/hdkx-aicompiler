#pragma once

#include <unordered_map>
#include <utility>

#include "kxc/runtime/session.h"

namespace kxc::runtime {

class RuntimeSessionNode final : public Object {
public:
    RuntimeSessionNode(api::CompiledModule compiled_module,
                       ExecutablePlan executable_plan, Device execution_device,
                       std::unordered_map<int64_t, String> constant_keys)
        : module(std::move(compiled_module)),
          plan(std::move(executable_plan)),
          device(std::move(execution_device)),
          constant_keys_by_value(std::move(constant_keys)) {}

    api::CompiledModule module;
    ExecutablePlan plan;
    Device device;
    std::unordered_map<int64_t, String> constant_keys_by_value;
    KXC_OBJECT_DECLARE
};

}  // namespace kxc::runtime
