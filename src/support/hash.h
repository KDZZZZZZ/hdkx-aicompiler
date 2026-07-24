/*! \file src/support/hash.h
 * \brief Private deterministic text hashing for compiler and runtime identity paths.
 */

#pragma once

#include <string>
#include <string_view>

namespace kxc::support {

std::string HashText(std::string_view text);

}  // namespace kxc::support
