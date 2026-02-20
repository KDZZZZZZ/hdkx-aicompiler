#pragma once

#include <cstddef>
#include <string>

#include "base/virtual_device.h"
#include "relay/relay.h"

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

// Copies virtual_device_ from source to dest and returns dest.
Expr CopyVirtualDevice(const Expr& source, const Expr& dest);
std::string GetCallOpName(const CallNode* call);

// Creates a text key suitable for local CSE.
std::string ExprStructuralKey(const Expr& expr);

// Replaces all uses of `target` in `expr` with `replacement`.
Expr SubstituteVar(const Expr& expr, const Var& target, const Expr& replacement);

// Returns a copy of `virtual_device` with updated memory scope.
VirtualDevice WithMemoryScope(const VirtualDevice& virtual_device,
                              const std::string& memory_scope);

}  // namespace pass_utils
}  // namespace relay
}  // namespace kxc
