/*! \file src/support/string.cc
 * \brief Implements the non-template String object family.
 */

#include "kxc/support/container.h"

#include "kxc/support/object_registration.h"

namespace kxc {

StringObj::StringObj() = default;

StringObj::StringObj(std::string s) : data(std::move(s)) {}

KXC_OBJECT_DEFINE_WITH_KEY(StringObj, "String")

String::String() : ObjectRef(new StringObj()) {}

String::String(const char* s) : ObjectRef(new StringObj(s)) {}

String::String(std::string s) : ObjectRef(new StringObj(std::move(s))) {}

String::String(const ObjectRef& n) : ObjectRef(n) {}

const StringObj* String::operator->() const {
    return static_cast<const StringObj*>(object_);
}

String::operator std::string() const {
    return defined() ? operator->()->data : std::string();
}

bool String::operator==(const String& other) const {
    if (object_ == other.object_) return true;
    if (!defined() || !other.defined()) return false;
    return operator->()->data == other->data;
}

bool String::operator==(const std::string& other) const {
    return defined() && operator->()->data == other;
}

bool String::operator==(const char* other) const {
    if (!defined()) return other == nullptr;
    return operator->()->data == other;
}

}  // namespace kxc
