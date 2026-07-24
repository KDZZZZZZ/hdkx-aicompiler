#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "support/canonical.h"
#include "support/hash.h"

namespace kxc::api::internal {

inline void AppendField(std::string* out, const std::string& name,
                        const std::string& value) {
    support::CanonicalBytesEncoder field;
    field.Field(name, value);
    *out += std::move(field).Take();
}

inline void RequireNonEmpty(const std::string& value, const char* field) {
    if (value.empty()) {
        throw std::invalid_argument(std::string("canonical identity requires ") +
                                    field);
    }
}

inline std::string Digest(const std::string& canonical,
                          std::string index_digest) {
    return index_digest.empty() ? support::HashText(canonical)
                                : std::move(index_digest);
}

template <typename T>
inline void AppendInteger(std::string* out, const std::string& name,
                          T value) {
    AppendField(out, name, std::to_string(static_cast<int64_t>(value)));
}

}  // namespace kxc::api::internal
