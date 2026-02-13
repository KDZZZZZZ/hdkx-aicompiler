#pragma once

#include "tir/stmt.h"

#include <ostream>

namespace kxc {
namespace tir {
namespace pass {

// A read-only pass that traverses TIR and prints a structured PrimFunc dump.
class IRPrinterPass {
public:
    explicit IRPrinterPass(int indent_spaces = 2);

    void Run(const PrimFunc& func, std::ostream& os) const;

private:
    int indent_spaces_;
};

// Convenience helper.
void DumpPrimFunc(const PrimFunc& func, std::ostream& os, int indent_spaces = 2);

}  // namespace pass
}  // namespace tir
}  // namespace kxc

