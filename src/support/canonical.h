/*! \file src/support/canonical.h
 * \brief Private deterministic length-delimited canonical byte encoder.
 */

#pragma once

#include <string>
#include <string_view>
#include <utility>

namespace kxc::support {

class CanonicalBytesEncoder final {
public:
    CanonicalBytesEncoder() = default;

    explicit CanonicalBytesEncoder(std::string_view format) {
        Field("format", format);
    }

    void Field(std::string_view name, std::string_view value) {
        bytes_ += std::to_string(name.size()) + ":";
        bytes_.append(name.data(), name.size());
        bytes_ += "=" + std::to_string(value.size()) + ":";
        bytes_.append(value.data(), value.size());
        bytes_ += ";";
    }

    template <typename Integer>
    void IntegerField(std::string_view name, Integer value) {
        Field(name, std::to_string(value));
    }

    void BoolField(std::string_view name, bool value) {
        Field(name, value ? "1" : "0");
    }

    std::string Take() && { return std::move(bytes_); }

private:
    std::string bytes_;
};

}  // namespace kxc::support
