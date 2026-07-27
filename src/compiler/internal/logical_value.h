/*! \file src/compiler/internal/logical_value.h
 * \brief Compiler-authoritative logical tensor value contracts.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kxc/relay/relay.h"
#include "kxc/runtime/device.h"

namespace kxc::api::internal {

using ValueId = std::int64_t;

// Capability token for the one bounded request preparation path.  It cannot be
// minted from a bare Relay Function; static analysis overloads remain the
// default everywhere else.
class BoundedLogicalShapeAdmission final {
private:
    BoundedLogicalShapeAdmission() = default;
    friend class BoundedCompilePreparationAccess;
};

enum class LogicalValueOrigin : int {
    kParameter = 0,
    kConstant = 1,
    kPrimitiveOutput = 2,
    kPhi = 3,
    kLoopCarried = 4,
};

/*! \brief One compiler-side tensor leaf. Type is the sole dtype/shape authority. */
struct LogicalValueContract {
    ValueId id{-1};
    Type checked_type;
    Device device{Device::CPU()};
    LogicalValueOrigin origin{LogicalValueOrigin::kPrimitiveOutput};
    Expr source;
    std::string source_locator;
};

/*! \brief Recursively flattens TensorType leaves in deterministic field order. */
std::vector<Type> FlattenLogicalTensorTypes(const Type& type,
                                            const std::string& path);
std::vector<Type> FlattenLogicalTensorTypes(
    const Type& type, const std::string& path,
    const BoundedLogicalShapeAdmission& admission);

/*! \brief Resolves explicit Relay placement or uses the supplied default device. */
Device ResolveLogicalValueDevice(const Expr& source, Device default_device,
                                 const std::string& path);

/*! \brief Creates dense logical tensor leaves beginning at first_id. */
std::vector<LogicalValueContract> MakeLogicalValueLeaves(
    const Expr& source, const Type& checked_type, LogicalValueOrigin origin,
    ValueId first_id, Device default_device, const std::string& source_locator);
std::vector<LogicalValueContract> MakeLogicalValueLeaves(
    const Expr& source, const Type& checked_type, LogicalValueOrigin origin,
    ValueId first_id, Device default_device, const std::string& source_locator,
    const BoundedLogicalShapeAdmission& admission);

/*! \throws std::invalid_argument unless value has a TensorType. */
const TensorTypeNode& RequireLogicalTensorType(
    const LogicalValueContract& value, const std::string& context);

bool SameLogicalValueContract(const LogicalValueContract& left,
                              const LogicalValueContract& right);
bool IsCpuScalarBool(const LogicalValueContract& value);

}  // namespace kxc::api::internal
