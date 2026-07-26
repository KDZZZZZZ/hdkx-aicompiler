/*! \file src/compiler/internal/relay_snapshot.h
 * \brief Deep snapshot of prepared Relay operator descriptors.
 */

#pragma once

#include "execution_contract.h"

namespace kxc::api::internal {

// Deep-copies a caller Function (types, attrs, constants, metadata) so the
// compiler can mutate its working copy without touching caller-visible IR.
Function CloneRelaySnapshot(const Function& function);

// Detaches every prepared Call's operator descriptor from the shared
// registry object by deep-cloning it, so later compiler passes cannot
// mutate caller-visible IR through the prepared graph.
void FreezePreparedOperators(PreparedCompilerGraph* prepared);

}  // namespace kxc::api::internal
