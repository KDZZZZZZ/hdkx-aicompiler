#pragma once
#include "relay/relay.h"
#include "tir/stmt.h"

namespace kxc {
namespace relay {

// Lowers a Relay Function to a TIR PrimFunc.
// This involves:
// 1. converting Relay Graph to TE Compute Graph
// 2. Creating a default Schedule (or just traversing the TE graph)
// 3. Lowering to TIR PrimFunc
tir::PrimFunc LowerToTIR(Function func);

}
}
