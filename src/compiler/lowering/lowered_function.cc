/*! \file src/compiler/lowering/lowered_function.cc
 * \brief Owns validated private primitive TE-to-TIR result objects.
 */

#include "../internal/lowered_function.h"

#include "kxc/runtime/kernel_abi.h"
#include "kxc/support/object_registration.h"

#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace kxc::relay {
namespace {

tir::DataType DTypeFromDL(const DLDataType& dtype) {
  if (dtype.code == kDLBool && dtype.bits == 8) return tir::DataType::Bool();
  if (dtype.code == kDLInt) return tir::DataType::Int(dtype.bits, dtype.lanes);
  if (dtype.code == kDLUInt) return tir::DataType::UInt(dtype.bits, dtype.lanes);
  if (dtype.code == kDLFloat) return tir::DataType::Float(dtype.bits, dtype.lanes);
  throw std::invalid_argument("Unsupported DLPack dtype in LoweredFunction");
}

}  // namespace

KXC_OBJECT_DEFINE_WITH_KEY(
    ConstantBindingNode, "kxc.compiler.internal.ConstantBindingNode")
KXC_OBJECT_DEFINE_WITH_KEY(
    LoweredFunctionNode, "kxc.compiler.internal.LoweredFunctionNode")

ConstantBinding::ConstantBinding(String key, runtime::NDArray value,
                                 int64_t param_index) {
  auto* node = new ConstantBindingNode();
  node->key = std::move(key);
  node->value = std::move(value);
  node->param_index = param_index;
  SetData(node);
  Validate();
}

ConstantBinding::ConstantBinding(const ObjectRef& ref) : ObjectRef(ref) {
  if (defined() && !As<ConstantBindingNode>()) {
    SetData(nullptr);
    throw std::invalid_argument("ObjectRef does not contain ConstantBindingNode");
  }
  if (defined()) Validate();
}

void ConstantBinding::Validate() const {
  const auto* node = operator->();
  if (std::string(node->key).empty()) {
    throw std::invalid_argument("ConstantBinding key must not be empty");
  }
  if (!node->value.defined()) {
    throw std::invalid_argument("ConstantBinding value must be defined");
  }
  if (node->param_index < 0) {
    throw std::invalid_argument("ConstantBinding param_index must be non-negative");
  }
}

const ConstantBindingNode* ConstantBinding::operator->() const {
  const auto* node = As<ConstantBindingNode>();
  if (!node) throw std::runtime_error("undefined or invalid ConstantBinding");
  return node;
}

LoweredFunction::LoweredFunction(tir::PrimFunc prim_func,
                                 Array<ConstantBinding> constants) {
  auto* node = new LoweredFunctionNode();
  node->prim_func = std::move(prim_func);
  for (const auto& binding : constants) node->constants_.push_back(binding);
  SetData(node);
  Validate();
}

LoweredFunction::LoweredFunction(const ObjectRef& ref) : ObjectRef(ref) {
  if (defined() && !As<LoweredFunctionNode>()) {
    SetData(nullptr);
    throw std::invalid_argument("ObjectRef does not contain LoweredFunctionNode");
  }
  if (defined()) Validate();
}

Array<ConstantBinding> LoweredFunction::constants() const {
  Array<ConstantBinding> result;
  for (const auto& binding : operator->()->constants_) result.push_back(binding);
  return result;
}

void LoweredFunction::Validate() const {
  const auto* node = operator->();
  const tir::PrimFunc& prim_func = node->prim_func;
  const Array<ConstantBinding>& constants = node->constants_;
  if (!prim_func.defined()) {
    throw std::invalid_argument("LoweredFunction prim_func must be defined");
  }
  const String schedule_key(internal::kTEScheduleContractAttr);
  if (!prim_func->attrs.count(schedule_key)) {
    throw std::invalid_argument(
        "LoweredFunction requires a TE schedule contract");
  }
  const auto* schedule_contract =
      prim_func->attrs.at(schedule_key).As<StringObj>();
  if (!schedule_contract || schedule_contract->data.empty()) {
    throw std::invalid_argument(
        "LoweredFunction TE schedule contract is malformed");
  }

  int64_t input_count = -1;
  int64_t constant_count = -1;
  int64_t output_count = -1;
  int64_t output_param_start = -1;
  const auto read_count = [&](const char* key, int64_t* value) {
    const String attr_key(key);
    if (!prim_func->attrs.count(attr_key)) return false;
    const auto* integer = prim_func->attrs.at(attr_key).As<tir::IntImmNode>();
    if (!integer) return false;
    *value = integer->value;
    return true;
  };
  if (!read_count("kxc.input_count", &input_count) ||
      !read_count("kxc.constant_count", &constant_count) ||
      !read_count("kxc.output_count", &output_count) ||
      !read_count("kxc.output_param_start", &output_param_start)) {
    throw std::invalid_argument(
        "LoweredFunction requires integer parameter count attrs");
  }
  if (input_count < 0 || constant_count < 0 || output_count <= 0 ||
      static_cast<size_t>(constant_count) != constants.size()) {
    throw std::invalid_argument("LoweredFunction parameter count mismatch");
  }
  if (output_param_start != input_count + constant_count ||
      output_param_start + output_count !=
          static_cast<int64_t>(prim_func->params.size())) {
    throw std::invalid_argument("LoweredFunction output parameter range mismatch");
  }

  const String constant_keys_attr("kxc.constant_keys");
  if (!prim_func->attrs.count(constant_keys_attr)) {
    throw std::invalid_argument("LoweredFunction requires constant key metadata");
  }
  const codegen::KernelConstantKeys constant_key_list(
      prim_func->attrs.at(constant_keys_attr));
  const Array<String> constant_keys = constant_key_list.keys();
  if (constant_keys.size() != constants.size()) {
    throw std::invalid_argument("LoweredFunction constant key count mismatch");
  }

  std::unordered_set<std::string> keys;
  for (size_t i = 0; i < constants.size(); ++i) {
    const ConstantBinding& binding = constants[i];
    const int64_t expected_index = input_count + static_cast<int64_t>(i);
    if (binding->param_index != expected_index ||
        static_cast<size_t>(binding->param_index) >= prim_func->params.size()) {
      throw std::invalid_argument(
          "LoweredFunction constant binding parameter index mismatch");
    }
    if (!keys.insert(std::string(binding->key)).second) {
      throw std::invalid_argument("LoweredFunction constant keys must be unique");
    }
    if (!(constant_keys[i] == binding->key)) {
      throw std::invalid_argument(
          "LoweredFunction constant key metadata does not match bindings");
    }
    const tir::Var& parameter =
        prim_func->params[static_cast<size_t>(binding->param_index)];
    if (!prim_func->buffer_map.count(parameter)) {
      throw std::invalid_argument("LoweredFunction constant parameter has no Buffer");
    }
    const tir::Buffer& buffer = prim_func->buffer_map.at(parameter);
    if (DTypeFromDL(binding->value.dtype()) != buffer->dtype ||
        buffer->shape.size() != binding->value->shape_storage.size()) {
      throw std::invalid_argument("LoweredFunction constant dtype or rank mismatch");
    }
    for (size_t dim = 0; dim < buffer->shape.size(); ++dim) {
      const auto* extent = buffer->shape[dim].As<tir::IntImmNode>();
      if (!extent || extent->value != binding->value->shape_storage[dim]) {
        throw std::invalid_argument("LoweredFunction constant shape mismatch");
      }
    }
  }
}

const LoweredFunctionNode* LoweredFunction::operator->() const {
  const auto* node = As<LoweredFunctionNode>();
  if (!node) throw std::runtime_error("undefined or invalid LoweredFunction");
  return node;
}

}  // namespace kxc::relay
