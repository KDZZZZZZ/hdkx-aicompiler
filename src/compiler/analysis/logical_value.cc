/*! \file src/compiler/analysis/logical_value.cc
 * \brief Shared construction and validation for compiler logical values.
 */

#include "../internal/logical_value.h"

#include <stdexcept>
#include <utility>

namespace kxc::api::internal {
namespace {

[[noreturn]] void Fail(const std::string& path, const std::string& detail) {
    throw std::invalid_argument("LogicalValueContract: path=" + path + "; " +
                                detail);
}

void Flatten(const Type& type, const std::string& path,
             std::vector<Type>* leaves) {
    if (const auto* tensor = type.As<TensorTypeNode>()) {
        if (tensor->dtype.empty()) {
            Fail(path, "capability=tensor_dtype; tensor dtype must be explicit");
        }
        for (std::size_t index = 0; index < tensor->shape.size(); ++index) {
            if (tensor->shape[index] < 0) {
                Fail(path + ".shape[" + std::to_string(index) + "]",
                     "capability=static_exact_shape; dimension is dynamic");
            }
        }
        leaves->push_back(type);
        return;
    }
    if (const auto* tuple = type.As<TupleTypeNode>()) {
        for (std::size_t index = 0; index < tuple->fields.size(); ++index) {
            Flatten(tuple->fields[index],
                    path + ".fields[" + std::to_string(index) + "]", leaves);
        }
        return;
    }
    Fail(path, "requires TensorType or nested TupleType tensor leaves");
}

}  // namespace

std::vector<Type> FlattenLogicalTensorTypes(const Type& type,
                                            const std::string& path) {
    if (!type.defined()) Fail(path, "requires a defined checked Type");
    std::vector<Type> leaves;
    Flatten(type, path, &leaves);
    if (leaves.empty()) Fail(path, "requires at least one tensor leaf");
    return leaves;
}

Device ResolveLogicalValueDevice(const Expr& source, Device default_device,
                                 const std::string& path) {
    if (!source.defined()) Fail(path, "requires a defined Relay source");
    if (!default_device.defined()) default_device = Device::CPU();
    const auto* relay = dynamic_cast<const RelayNode*>(source.get());
    if (!relay || !relay->virtual_device_.defined()) return default_device;
    const Device explicit_device = relay->virtual_device_->device;
    if (!explicit_device.defined()) {
        Fail(path, "explicit Relay VirtualDevice must define a device");
    }
    if (explicit_device.device_type() != kCPU &&
        explicit_device.device_type() != kCUDA) {
        Fail(path, "Relay VirtualDevice must be CPU or CUDA");
    }
    return explicit_device;
}

std::vector<LogicalValueContract> MakeLogicalValueLeaves(
    const Expr& source, const Type& checked_type, LogicalValueOrigin origin,
    ValueId first_id, Device default_device, const std::string& source_locator) {
    if (first_id < 0) Fail(source_locator, "first value id must be non-negative");
    const std::vector<Type> leaf_types =
        FlattenLogicalTensorTypes(checked_type, source_locator + ".checked_type");
    const Device device =
        ResolveLogicalValueDevice(source, std::move(default_device), source_locator);
    std::vector<LogicalValueContract> values;
    values.reserve(leaf_types.size());
    for (std::size_t index = 0; index < leaf_types.size(); ++index) {
        values.push_back(LogicalValueContract{
            first_id + static_cast<ValueId>(index), leaf_types[index], device,
            origin, source,
            source_locator + ".leaf[" + std::to_string(index) + "]"});
    }
    return values;
}

const TensorTypeNode& RequireLogicalTensorType(
    const LogicalValueContract& value, const std::string& context) {
    const auto* tensor = value.checked_type.As<TensorTypeNode>();
    if (!tensor) {
        throw std::invalid_argument(context +
                                    " requires a logical TensorType leaf");
    }
    return *tensor;
}

bool SameLogicalValueContract(const LogicalValueContract& left,
                              const LogicalValueContract& right) {
    return TypeEqual(left.checked_type, right.checked_type) &&
           left.device == right.device;
}

bool IsCpuScalarBool(const LogicalValueContract& value) {
    const auto* tensor = value.checked_type.As<TensorTypeNode>();
    return tensor && tensor->dtype == "bool" && tensor->shape.empty() &&
           value.device == Device::CPU();
}

}  // namespace kxc::api::internal
