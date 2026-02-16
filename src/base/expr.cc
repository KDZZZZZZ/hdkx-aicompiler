#include "base/expr.h"

namespace kxc {

Span::Span(std::string source_name, int line, int column) {
    auto* node = new SpanNode();
    node->source_name = std::move(source_name);
    node->line = line;
    node->column = column;
    SetData(node);
}

}  // namespace kxc

