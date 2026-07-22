/*! \file src/runtime/internal/kernel_argument_validation.h
 * \brief 声明 CompiledModule 与 RuntimeSession 共用的 NDArray 参数校验。
 */

#pragma once

#include <cstddef>

#include "kxc/runtime/ndarray.h"
#include "kxc/runtime/kernel_abi.h"

namespace kxc::api {

/*! \brief 按 DLPack code、bits 和 lanes 完整比较两个 dtype。 */
bool SameDType(DLDataType lhs, DLDataType rhs);

/*!
 * \brief 校验单个 NDArray 是否满足指定 KernelArgSpec。
 *
 * constants 仅在 spec 为 constant 时用于验证稳定 key 对应的对象身份；input
 * 和 output 调用可以传空 Map。错误统一包含 kernel symbol、参数槽位和名称。
 */
void ValidateKernelArgument(
    const codegen::KernelSignature& signature, size_t index,
    const codegen::KernelArgSpec& spec, const runtime::NDArray& argument,
    const Map<String, runtime::NDArray>& constants = {});

}  // namespace kxc::api
