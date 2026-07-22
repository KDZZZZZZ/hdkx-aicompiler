/*! \file include/kxc/support/object_registration.h
 * \brief Opt-in definitions for Object type metadata; include from one .cc only.
 */

#pragma once

#include "kxc/support/object.h"

#undef KXC_OBJECT_DEFINE_WITH_KEY
#undef KXC_OBJECT_DEFINE

#define KXC_OBJECT_DEFINE_WITH_KEY(TypeName, TypeKey) \
    const kxc::TypeInfo& TypeName::_type_info =        \
        kxc::TypeRegistry::Register(TypeKey);          \
    const uint32_t TypeName::_type_index =             \
        TypeName::_type_info.runtime_index();

#define KXC_OBJECT_DEFINE(TypeName) \
    KXC_OBJECT_DEFINE_WITH_KEY(TypeName, #TypeName)
