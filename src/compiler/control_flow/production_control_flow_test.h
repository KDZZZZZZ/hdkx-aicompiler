/*! \file src/compiler/control_flow/production_control_flow_test.h
 * \brief Private deterministic seam for control-flow lease generation tests.
 */
#pragma once

#include <cstdint>

namespace kxc::api::internal {

// These functions share the production authority.  They are intentionally not
// installed and exist only so tests can cover the terminal generation state.
std::uint64_t MintControlFlowLeaseGenerationForTest();
void SetControlFlowLeaseGenerationForTest(std::uint64_t last_generation);

}  // namespace kxc::api::internal
