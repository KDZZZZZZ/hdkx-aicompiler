/*! \file src/base/expr.cc
 * \brief 实现共享表达式辅助函数。
 */

#include "base/expr.h"

#include <stdexcept>
#include <utility>

namespace kxc {

Span::Span(std::string source_name, int line, int column) {
    auto* node = new SpanNode();
    node->source_name = std::move(source_name);
    node->line = line;
    node->column = column;
    SetData(node);
}

void SetCheckedType(const Expr& expr, Type checked_type) {
    if (!expr.defined()) {
        throw std::runtime_error("SetCheckedType expects a defined expression");
    }
    // checked_type_ 是类型推导阶段补写的元数据；这里只更新该字段，不改变表达式结构。
    auto* node = const_cast<ExprNode*>(static_cast<const ExprNode*>(expr.get()));
    node->checked_type_ = std::move(checked_type);
}

}  // namespace kxc
