#pragma once

#include <utility>

#include "kxc/runtime/session.h"

namespace kxc::runtime {

class RuntimeSessionNode final : public Object {
public:
    explicit RuntimeSessionNode(api::CompiledModule compiled_module)
        : module(std::move(compiled_module)) {}

    api::CompiledModule module;
    KXC_OBJECT_DECLARE
};

}  // namespace kxc::runtime
