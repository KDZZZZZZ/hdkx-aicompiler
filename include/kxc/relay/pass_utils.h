/*! \file include/kxc/relay/pass_utils.h
 * \brief 定义 Relay IR 节点、算子注册、attrs 和 Relay 到 TE lowering 属性。
 */

#pragma once

#include <cstddef>
#include <string>

#include "kxc/target/virtual_device.h"
#include "kxc/relay/relay.h"

namespace kxc {
namespace relay {
namespace pass_utils {

// Returns false if expr is not a scalar constant.
bool TryGetScalarConstantValue(const Expr& expr, double* out_value);
bool TryGetScalarConstantValueWithDType(const Expr& expr, double* out_value,
                                        DLDataType* out_dtype);
bool IsConstZero(const Expr& expr);
bool IsConstOne(const Expr& expr);

// Creates a scalar Constant with requested dtype.
Constant MakeScalarConstant(double value, const DLDataType& dtype);

// Conservative purity check for Relay expressions.
bool HasSideEffect(const Expr& expr);

// Counts uses of `var` in `expr`, excluding binder definitions.
size_t CountVarUses(const Expr& expr, const Var& var);

// 将 source 的元数据复制到 dest，并返回 dest。
Expr CopyVirtualDevice(const Expr& source, const Expr& dest);
std::string GetCallOpName(const CallNode* call);

// Creates a text key suitable for local CSE.
std::string ExprStructuralKey(const Expr& expr);

// Replaces all uses of `target` in `expr` with `replacement`.
Expr SubstituteVar(const Expr& expr, const Var& target, const Expr& replacement);

// 一次性把若干变量替换为对应表达式。与逐个 SubstituteVar 相比：
// 单次 DAG 安全遍历（共享子表达式只重写一次），并且替换是同时的——
// 某个 replacement 内出现的其它 target 不会被级联替换。
// targets 与 replacements 必须等长；任一未定义即抛错。
Expr SubstituteVars(const Expr& expr, const Array<Var>& targets,
                    const Array<Expr>& replacements);

// 把 `function` 的全部形参替换为 `arguments`（按位对应），返回去掉形参的
// 函数体表达式。这是把导入的独立 Function 内联进控制流 region（如 While
// body）的前提：While body 必须是同一 Function 内的表达式，不能是另一个
// Function 的调用。arity 不匹配或实参未定义时 fail closed。
Expr InlineFunctionParameters(const Function& function,
                              const Array<Expr>& arguments);

// Returns a copy of `virtual_device` with updated memory scope.
VirtualDevice WithMemoryScope(const VirtualDevice& virtual_device,
                              const std::string& memory_scope);

}  // namespace pass_utils
}  // namespace relay
}  // namespace kxc
