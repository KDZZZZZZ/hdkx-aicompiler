/*! \file include/kxc/support/hash.h
 * \brief Deterministic text hashing for compiler and runtime identity paths.
 */

#pragma once

#include <string>
#include <string_view>

namespace kxc::support {

std::string HashText(std::string_view text);

}  // namespace kxc::support
