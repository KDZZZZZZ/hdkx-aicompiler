/*! \file src/compiler/internal/lowered_function.h
 * \brief Private validated result of one primitive TE-to-TIR lowering.
 */

#pragma once

#include <cstdint>

#include "kxc/relay/relay.h"
#include "kxc/tir/stmt.h"

namespace kxc::relay {

namespace internal {
inline constexpr const char* kTEScheduleContractAttr =
    "kxc.te.schedule_contract";
}  // namespace internal

class ConstantBindingNode final : public Object {
public:
    String key;
    runtime::NDArray value;
    int64_t param_index{-1};

    KXC_OBJECT_DECLARE
};

class ConstantBinding : public ObjectRef {
public:
    ConstantBinding(String key, runtime::NDArray value, int64_t param_index);
    explicit ConstantBinding(const ObjectRef& ref);

    void Validate() const;
    const ConstantBindingNode* operator->() const;
};

class LoweredFunctionNode final : public Object {
public:
    tir::PrimFunc prim_func;

    KXC_OBJECT_DECLARE

private:
    friend class LoweredFunction;
    Array<ConstantBinding> constants_;
};

class LoweredFunction : public ObjectRef {
public:
    LoweredFunction(tir::PrimFunc prim_func,
                    Array<ConstantBinding> constants);
    explicit LoweredFunction(const ObjectRef& ref);

    Array<ConstantBinding> constants() const;
    void Validate() const;
    const LoweredFunctionNode* operator->() const;
};

}  // namespace kxc::relay
