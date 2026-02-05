#pragma once
#include "relay/relay.h"
#include "tir/stmt.h"

namespace kxc {
namespace relay {

// Lowers a Relay Function to a TIR Stmt (representing the body of a PrimFunc).
// This involves:
// 1. converting Relay Graph to TE Compute Graph
// 2. Creating a default Schedule (or just traversing the TE graph)
// 3. Lowering to TIR Stmt
tir::Stmt LowerToTIR(Function func);

}
}
