/*! \file src/base/expr.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

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

