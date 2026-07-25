/*! \file include/kxc/tir/printer/print_ir.h
 * \brief 声明 TIR IR 文本打印工具。
 */

#pragma once

#include "kxc/tir/stmt.h"

#include <ostream>

namespace kxc {
namespace tir {
namespace printer {

// A read-only printer that traverses TIR and emits a structured PrimFunc dump.
class IRPrinter {
public:
    explicit IRPrinter(int indent_spaces = 2);

    void Run(const PrimFunc& func, std::ostream& os) const;

private:
    int indent_spaces_;
};

// Convenience helper.
void DumpPrimFunc(const PrimFunc& func, std::ostream& os, int indent_spaces = 2);

}  // namespace printer
}  // namespace tir
}  // namespace kxc
