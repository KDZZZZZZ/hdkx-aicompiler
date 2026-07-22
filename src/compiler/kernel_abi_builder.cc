/*! \file src/compiler/kernel_abi_builder.cc
 * \brief Builds the runtime kernel ABI from compiler-owned TIR metadata.
 */

#include "internal/kernel_abi_builder.h"

#include <limits>
#include <stdexcept>

namespace kxc::codegen {
namespace {

int64_t ReadIntAttr(const tir::PrimFunc& function, const char* key) {
  const String attr_key(key);
  if (!function->attrs.count(attr_key)) {
    throw std::invalid_argument(std::string("PrimFunc is missing attr ") + key);
  }
  const auto* value = function->attrs.at(attr_key).As<tir::IntImmNode>();
  if (!value) {
    throw std::invalid_argument(std::string("PrimFunc attr is not integer: ") + key);
  }
  return value->value;
}

DLDataType DTypeFromTIR(tir::DataType dtype) {
  if (dtype.code == 1 && dtype.bits == 1) {
    return DLDataType{kDLBool, 8, dtype.lanes};
  }
  uint8_t code = 0;
  switch (dtype.code) {
    case 0:
      code = kDLInt;
      break;
    case 1:
      code = kDLUInt;
      break;
    case 2:
      code = kDLFloat;
      break;
    default:
      throw std::invalid_argument("PrimFunc parameter uses unsupported TIR dtype");
  }
  if (dtype.bits == 0 || dtype.bits % 8 != 0 || dtype.lanes == 0) {
    throw std::invalid_argument("PrimFunc parameter dtype is not byte-addressable");
  }
  return DLDataType{code, dtype.bits, dtype.lanes};
}

bool ConstantDTypeMatchesTIR(DLDataType payload, tir::DataType dtype) {
  if (payload.lanes != dtype.lanes) return false;
  if (payload.code == kDLBool) {
    return payload.bits == 8 && dtype.code == 1 &&
           (dtype.bits == 1 || dtype.bits == 8);
  }
  const DLDataType lowered = DTypeFromTIR(dtype);
  return payload.code == lowered.code && payload.bits == lowered.bits &&
         payload.lanes == lowered.lanes;
}

uint64_t NaturalAlignment(DLDataType dtype) {
  const uint64_t bytes =
      (static_cast<uint64_t>(dtype.bits) / 8) * static_cast<uint64_t>(dtype.lanes);
  if (bytes == 0) return 1;
  return bytes & (~bytes + 1);
}

Array<int64_t> ShapeFromBuffer(const tir::Buffer& buffer, KernelArgRole role) {
  Array<int64_t> shape;
  for (const auto& extent : buffer->shape) {
    if (const auto* integer = extent.As<tir::IntImmNode>()) {
      if (integer->value < 0) {
        throw std::invalid_argument("PrimFunc Buffer shape contains a negative extent");
      }
      shape.push_back(integer->value);
      continue;
    }
    const auto* variable = extent.As<tir::VarNode>();
    if (role != KernelArgRole::kInput || !variable || variable->dtype.lanes != 1 ||
        (variable->dtype.code != 0 && variable->dtype.code != 1)) {
      throw std::invalid_argument(
          "Only an integer symbolic input extent may be dynamic");
    }
    shape.push_back(kDynamicDimension);
  }
  return shape;
}

}  // namespace

KernelSignature BuildKernelSignature(const tir::PrimFunc& function,
                                     const Map<String, runtime::NDArray>& constants,
                                     const Target& target, String symbol) {
  if (!function.defined()) {
    throw std::invalid_argument("BuildKernelSignature requires a defined PrimFunc");
  }
  if (!target.defined() || !target.As<TargetNode>()) {
    throw std::invalid_argument("BuildKernelSignature requires a defined Target");
  }
  if (std::string(symbol).empty()) {
    throw std::invalid_argument("BuildKernelSignature requires a non-empty symbol");
  }
  if (target->device_id < 0 || target->device_type == kUnknown ||
      !((target->device_type == kCPU && target->kind == "llvm") ||
        (target->device_type == kCUDA && target->kind == "cuda"))) {
    throw std::invalid_argument("Target kind and physical device type are inconsistent");
  }
  if (function->params.size() >
      static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("PrimFunc parameter count exceeds int64 range");
  }

  const int64_t input_count = ReadIntAttr(function, "kxc.input_count");
  const int64_t constant_count = ReadIntAttr(function, "kxc.constant_count");
  const int64_t output_count = ReadIntAttr(function, "kxc.output_count");
  const int64_t output_start = ReadIntAttr(function, "kxc.output_param_start");
  if (input_count < 0 || constant_count < 0 || output_count <= 0 ||
      input_count > std::numeric_limits<int64_t>::max() - constant_count ||
      output_start != input_count + constant_count ||
      output_start > std::numeric_limits<int64_t>::max() - output_count ||
      output_start + output_count != static_cast<int64_t>(function->params.size())) {
    throw std::invalid_argument("PrimFunc parameter count attrs are inconsistent");
  }

  const String constant_keys_attr("kxc.constant_keys");
  if (!function->attrs.count(constant_keys_attr)) {
    throw std::invalid_argument("PrimFunc is missing constant key metadata");
  }
  const KernelConstantKeys key_list(function->attrs.at(constant_keys_attr));
  const Array<String> constant_keys = key_list.keys();
  if (constant_keys.size() != static_cast<size_t>(constant_count)) {
    throw std::invalid_argument("PrimFunc constant key count is inconsistent");
  }

  Device device(target->device_type, target->device_id);
  Array<KernelArgSpec> arguments;
  for (size_t i = 0; i < function->params.size(); ++i) {
    const tir::Var& parameter = function->params[i];
    if (!function->buffer_map.count(parameter)) {
      throw std::invalid_argument("PrimFunc parameter has no Buffer metadata");
    }
    const tir::Buffer& buffer = function->buffer_map.at(parameter);
    if (!buffer.defined() || buffer->data.get() != parameter.get()) {
      throw std::invalid_argument(
          "PrimFunc Buffer data variable does not match its parameter");
    }
    if (!buffer->strides.empty() || buffer->offset_factor != 0) {
      throw std::invalid_argument(
          "PrimFunc Buffer is incompatible with the contiguous NDArray ABI");
    }
    const auto* elem_offset = buffer->elem_offset.As<tir::IntImmNode>();
    if (!elem_offset || elem_offset->value != 0) {
      throw std::invalid_argument("PrimFunc Buffer elem_offset must be zero");
    }
    if (buffer->data_alignment < 0) {
      throw std::invalid_argument("PrimFunc Buffer data_alignment must be non-negative");
    }

    KernelArgRole role = KernelArgRole::kOutput;
    String constant_key;
    bool mutable_data = true;
    if (i < static_cast<size_t>(input_count)) {
      role = KernelArgRole::kInput;
      mutable_data = false;
    } else if (i < static_cast<size_t>(output_start)) {
      role = KernelArgRole::kConstant;
      mutable_data = false;
      constant_key = constant_keys[i - static_cast<size_t>(input_count)];
    }

    DLDataType dtype = DTypeFromTIR(buffer->dtype);
    if (role == KernelArgRole::kConstant) {
      if (!constants.count(constant_key)) {
        throw std::invalid_argument("Constant payload is missing for signature key");
      }
      const runtime::NDArray& payload = constants.at(constant_key);
      if (!payload.defined() || !ConstantDTypeMatchesTIR(payload.dtype(), buffer->dtype)) {
        throw std::invalid_argument(
            "Constant payload dtype does not match its TIR Buffer");
      }
      const Array<int64_t> buffer_shape = ShapeFromBuffer(buffer, role);
      const Array<int64_t> payload_shape = payload.shape();
      if (buffer_shape.size() != payload_shape.size()) {
        throw std::invalid_argument(
            "Constant payload rank does not match its TIR Buffer");
      }
      for (size_t dim = 0; dim < buffer_shape.size(); ++dim) {
        if (buffer_shape[dim] != payload_shape[dim]) {
          throw std::invalid_argument(
              "Constant payload shape does not match its TIR Buffer");
        }
      }
      dtype = payload.dtype();
    }
    const uint64_t alignment =
        buffer->data_alignment > 0
            ? static_cast<uint64_t>(buffer->data_alignment)
            : NaturalAlignment(dtype);
    String name(buffer->name.empty() ? parameter->name_hint : buffer->name);
    arguments.push_back(KernelArgSpec(name, role, dtype, ShapeFromBuffer(buffer, role),
                                      device, alignment, mutable_data, constant_key));
  }
  return KernelSignature(std::move(symbol), std::move(arguments));
}

}  // namespace kxc::codegen
