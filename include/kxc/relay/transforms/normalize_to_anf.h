/*! \file include/kxc/relay/transforms/normalize_to_anf.h
 * \brief Deterministic administrative-normal-form normalization for Relay.
 */

#pragma once

#include <string>

#include "kxc/relay/relay.h"

namespace kxc {
namespace relay {

/*! \brief Returns whether a typed Function is in the supported Relay ANF. */
bool IsANF(const Function& function, std::string* diagnostic = nullptr);

/*! \brief Throws std::invalid_argument with an ANF diagnostic when verification fails. */
void VerifyANF(const Function& function);

/*! \brief Normalizes a typed Function into deterministic, idempotent Relay ANF. */
Function NormalizeToANF(const Function& function);

}  // namespace relay
}  // namespace kxc
