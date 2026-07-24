/*! \file src/support/hash.cc
 * \brief Implements deterministic text hashing shared outside profiling.
 */

#include "support/hash.h"

#include <cstdint>
#include <sstream>

namespace kxc::support {

std::string HashText(std::string_view text) {
    std::uint64_t hash = 5381;
    for (unsigned char byte : text) hash = ((hash << 5) + hash) + byte;
    std::ostringstream os;
    os << std::hex << hash;
    return os.str();
}

}  // namespace kxc::support
