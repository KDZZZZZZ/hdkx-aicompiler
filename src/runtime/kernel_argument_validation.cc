/*! \file src/runtime/kernel_argument_validation.cc
 * \brief 实现公共 NDArray 内核参数的完整元数据、范围和身份校验。
 */

#include "internal/kernel_argument_validation.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kxc::api {
namespace {

/*! \brief 把 dtype 格式化为稳定诊断文本，避免只显示枚举 code。 */
std::string DTypeText(DLDataType dtype) {
    std::ostringstream os;
    os << static_cast<int>(dtype.code) << ':' << static_cast<int>(dtype.bits)
       << 'x' << dtype.lanes;
    return os.str();
}

/*! \brief 抛出同时包含 symbol、参数序号和名称的统一诊断。 */
[[noreturn]] void ThrowArgumentError(
    const String& owner, size_t index,
    const codegen::KernelArgSpec& spec, const std::string& detail) {
    throw std::invalid_argument(
        "kernel '" + std::string(owner) + "' argument[" +
        std::to_string(index) + "] '" + std::string(spec->name) + "': " +
        detail);
}

bool SamePayload(const runtime::NDArray& left,
                 const runtime::NDArray& right) {
    if (left.get() == right.get()) return true;
    if (left.NBytes() != right.NBytes()) return false;
    std::vector<unsigned char> left_bytes(left.NBytes());
    std::vector<unsigned char> right_bytes(right.NBytes());
    left.CopyToBytes(left_bytes.empty() ? nullptr : left_bytes.data(),
                     left_bytes.size());
    right.CopyToBytes(right_bytes.empty() ? nullptr : right_bytes.data(),
                      right_bytes.size());
    return left_bytes == right_bytes;
}

}  // namespace

/*! \brief 按 DLPack 三元组比较 dtype，不忽略 vector lanes。 */
bool SameDType(DLDataType lhs, DLDataType rhs) {
    return lhs.code == rhs.code && lhs.bits == rhs.bits &&
           lhs.lanes == rhs.lanes;
}

/*! \brief 执行不依赖其他参数的安全检查，并验证 constant 完整 payload。 */
void ValidateTensorArgument(
    const String& owner, size_t index,
    const codegen::KernelArgSpec& spec, const runtime::NDArray& argument,
    const Map<String, runtime::NDArray>& constants) {
    if (!argument.defined()) {
        ThrowArgumentError(owner, index, spec, "NDArray is undefined");
    }
    if (!argument.storage().defined()) {
        ThrowArgumentError(owner, index, spec, "Storage is undefined");
    }
    if (!SameDType(argument.dtype(), spec->dtype)) {
        ThrowArgumentError(owner, index, spec,
                           "dtype expected " + DTypeText(spec->dtype) +
                               ", actual " + DTypeText(argument.dtype()));
    }
    if (argument.device() != spec->device) {
        ThrowArgumentError(owner, index, spec,
                           "device expected " + spec->device.ToString() +
                               ", actual " + argument.device().ToString());
    }
    if (!argument.IsContiguous()) {
        ThrowArgumentError(owner, index, spec,
                           "layout must be contiguous");
    }

    const Array<int64_t> expected_shape = spec.shape();
    const Array<int64_t> actual_shape = argument.shape();
    if (actual_shape.size() != expected_shape.size()) {
        ThrowArgumentError(owner, index, spec,
                           "rank expected " +
                               std::to_string(expected_shape.size()) +
                               ", actual " +
                               std::to_string(actual_shape.size()));
    }
    for (size_t dimension = 0; dimension < expected_shape.size(); ++dimension) {
        if (actual_shape[dimension] < 0) {
            ThrowArgumentError(owner, index, spec,
                               "actual shape contains a negative dimension");
        }
        if (expected_shape[dimension] != codegen::kDynamicDimension &&
            expected_shape[dimension] != actual_shape[dimension]) {
            ThrowArgumentError(
                owner, index, spec,
                "shape dimension " + std::to_string(dimension) +
                    " expected " + std::to_string(expected_shape[dimension]) +
                    ", actual " + std::to_string(actual_shape[dimension]));
        }
    }

    size_t nbytes = 0;
    try {
        nbytes = argument.NBytes();
        argument.storage().ValidateRange(argument->byte_offset, nbytes);
    } catch (const std::exception& error) {
        ThrowArgumentError(owner, index, spec,
                           std::string("storage range is invalid: ") +
                               error.what());
    }
    // 零元素张量不会被内核解引用，因此允许其 Storage data 为 nullptr。
    if (nbytes != 0) {
        void* base = argument.storage().data();
        if (!base) {
            ThrowArgumentError(owner, index, spec,
                               "non-empty tensor has a null data pointer");
        }
        const uintptr_t base_address = reinterpret_cast<uintptr_t>(base);
        if (argument->byte_offset >
            std::numeric_limits<uintptr_t>::max() - base_address) {
            ThrowArgumentError(owner, index, spec,
                               "effective data address overflows uintptr_t");
        }
        const uintptr_t effective_address =
            base_address + argument->byte_offset;
        if (effective_address % spec->alignment != 0) {
            ThrowArgumentError(
                owner, index, spec,
                "effective data address does not satisfy alignment " +
                    std::to_string(spec->alignment));
        }
    }

    if (spec->role == codegen::KernelArgRole::kConstant) {
        if (!constants.count(spec->constant_key)) {
            ThrowArgumentError(owner, index, spec,
                               "bound constant is missing");
        }
        const runtime::NDArray& bound = constants.at(spec->constant_key);
        bool matches = false;
        try {
            matches = SamePayload(argument, bound);
        } catch (const std::exception& error) {
            ThrowArgumentError(owner, index, spec,
                               std::string("constant payload comparison failed: ") +
                                   error.what());
        }
        if (!matches) {
            ThrowArgumentError(
                owner, index, spec,
                "constant argument does not match the bound payload bytes");
        }
    }
}

void ValidateKernelArgument(
    const codegen::KernelSignature& signature, size_t index,
    const codegen::KernelArgSpec& spec, const runtime::NDArray& argument,
    const Map<String, runtime::NDArray>& constants) {
    ValidateTensorArgument(signature->symbol, index, spec, argument, constants);
}

}  // namespace kxc::api
