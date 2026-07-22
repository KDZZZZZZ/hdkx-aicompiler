/*! \file include/kxc/ffi/registration.h
 * \brief Opt-in macros for registering packed functions in implementation files.
 */

#pragma once

#include "kxc/ffi/registry.h"

namespace kxc {

/*! \brief Forces every built-in registration translation unit into static links. */
void RegisterBuiltins();

}  // namespace kxc

#define KXC_STR_CONCAT_(__x, __y) __x##__y
#define KXC_STR_CONCAT(__x, __y) KXC_STR_CONCAT_(__x, __y)

#define KXC_REGISTER_GLOBAL(Name)                                               \
    static ::kxc::RegistryEntry KXC_STR_CONCAT(__kxc_reg_entry_, __COUNTER__) = \
        ::kxc::RegistryEntry(Name)
