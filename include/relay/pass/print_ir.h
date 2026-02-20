#pragma once

#include "relay/relay.h"

#include <ostream>
#include <string>

namespace kxc {
namespace relay {
namespace pass {

class IRPrinterPass {
public:
    explicit IRPrinterPass(int indent_spaces = 2);

    void Run(const Expr& expr, std::ostream& os) const;
    void Run(const Function& func, std::ostream& os) const;

private:
    int indent_spaces_;
};

void DumpExpr(const Expr& expr, std::ostream& os, int indent_spaces = 2);
void DumpFunction(const Function& func, std::ostream& os, int indent_spaces = 2);
std::string ToText(const Expr& expr, int indent_spaces = 2);
std::string ToText(const Function& func, int indent_spaces = 2);

}  // namespace pass
}  // namespace relay
}  // namespace kxc
