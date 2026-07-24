#pragma once

#include "kxc/runtime/kernel_abi.h"

namespace kxc::api::internal {

inline bool SamePhysicalKernelAbi(const codegen::KernelSignature& left,
                                  const codegen::KernelSignature& right) {
    left.Validate();
    right.Validate();
    const Array<codegen::KernelArgSpec> left_args = left.arguments();
    const Array<codegen::KernelArgSpec> right_args = right.arguments();
    if (left_args.size() != right_args.size()) return false;
    for (size_t index = 0; index < left_args.size(); ++index) {
        const auto& lhs = left_args[index];
        const auto& rhs = right_args[index];
        const DLDataType left_dtype = lhs->dtype;
        const DLDataType right_dtype = rhs->dtype;
        if (lhs->role != rhs->role ||
            left_dtype.code != right_dtype.code ||
            left_dtype.bits != right_dtype.bits ||
            left_dtype.lanes != right_dtype.lanes ||
            lhs->device != rhs->device ||
            lhs->alignment != rhs->alignment ||
            lhs->mutable_data != rhs->mutable_data) {
            return false;
        }
        const Array<int64_t> left_shape = lhs.shape();
        const Array<int64_t> right_shape = rhs.shape();
        if (left_shape.size() != right_shape.size()) return false;
        for (size_t axis = 0; axis < left_shape.size(); ++axis) {
            if (left_shape[axis] != right_shape[axis]) return false;
        }
    }
    return true;
}

}  // namespace kxc::api::internal
