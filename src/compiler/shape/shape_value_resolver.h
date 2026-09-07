/*! \file src/compiler/shape/shape_value_resolver.h
 * \brief M3 受限形状值解析：链式证明、折叠与目标表达式投影。
 *
 * 输入是受限代表图的 Relay 快照（参数已注类型、树形调用体）。解析器：
 * 1) 逐节点推导符号维表达式（模板符号或常量）与形状值元素表达式；
 * 2) 校验 Gather/Concat 链的常量索引、越界、唯一来源，校验
 *    reshape_dynamic/expand 的元素总数与广播证明（DimExpr canonical 等值）；
 * 3) 把长度 ≥ 2 的链折叠为单个 shape_expr 单元，把控制目标投影为
 *    reshape_dynamic/expand 的 canonical attrs —— 链仍是唯一表达式来源，
 *    attrs 是验证过的投影并进入版本化身份。
 */

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"
#include "kxc/shape/shape.h"

namespace kxc::api::experimental::restricted_symbolic_shape::v1 {

struct InputAxisSymbol;

namespace shape_resolution {
// 与 relay::kShapeExprKind* 对齐的本地编码（避免扩散 relay 头依赖）。
inline constexpr int64_t kExprKindConst = 0;
inline constexpr int64_t kExprKindInputAxis = 1;

struct EncodedExpr final {
    Array<int64_t> kinds;
    Array<int64_t> values;
    Array<int64_t> axes;
};

struct Resolution final {
    // 折叠并投影后的代表图快照（attrs 已附着；链单元已归并）。
    Function rewritten;
    // 折叠后调用树的 post-order 算子名序列（注册表原名）。
    std::vector<std::string> registry_operations;
    // 与 registry_operations 平行的调用 attrs；undefined 表示无属性。
    std::vector<kxc::relay::Attrs> unit_attrs;
    // 折叠后不再存在 attr-free 算子携带属性的情况（解析器已强制）。
    bool has_operation_attrs{false};
    // 与 registry_operations 平行：形状值单元的 value 表达式覆盖
    //（相对单元局部输入 0 = 源张量）；nullopt 表示由合同推导。
    std::vector<std::optional<EncodedExpr>> unit_value_expressions;
    // 折叠后树中每个值的符号维表达式（"value.<id>" → dims）。
    std::map<std::string, std::vector<kxc::shape::experimental::v1::DimExpr>>
        value_dimensions;
};

/*! \brief 解析并重写受限形状值子图。
 *
 * parameter_symbols: (parameter_index, axis) → 模板符号名（范围/整除约束
 * 由调用方维护）。任何无法证明的形状（数据相关、未知 rank、多源链、
 * 越界索引、元素总数不等、非法广播、-1 推断）都抛错拒绝。
 */
[[nodiscard]] Resolution ResolveShapeValues(
    const Function& snapshot,
    const std::vector<InputAxisSymbol>& parameter_symbols);

}  // namespace shape_resolution
}  // namespace kxc::api::experimental::restricted_symbolic_shape::v1
