/*! \file include/kxc/runtime/module_invocation.h
 * \brief Versioned, immutable logical invocation ABI for CompiledModule entries.
 *
 * Physical KernelSignature remains the backend ABI.  This contract is the
 * logical layer binding input guards and output allocation to that signature.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <stdexcept>
#include <vector>

#include "kxc/runtime/device_stream.h"
#include "kxc/runtime/kernel_abi.h"

namespace kxc::api {

using ModuleExtent = std::uint64_t;

class ModuleShapeExpr final {
public:
    enum class Kind { kConst, kInputAxis, kAdd, kMul, kFloorDiv, kMin, kMax };
    static ModuleShapeExpr Const(ModuleExtent value);
    static ModuleShapeExpr InputAxis(std::size_t input, std::size_t axis);
    static ModuleShapeExpr Add(ModuleShapeExpr lhs, ModuleShapeExpr rhs);
    static ModuleShapeExpr Mul(ModuleShapeExpr lhs, ModuleShapeExpr rhs);
    static ModuleShapeExpr FloorDiv(ModuleShapeExpr lhs, ModuleShapeExpr rhs);
    static ModuleShapeExpr Min(ModuleShapeExpr lhs, ModuleShapeExpr rhs);
    static ModuleShapeExpr Max(ModuleShapeExpr lhs, ModuleShapeExpr rhs);
    ModuleExtent Evaluate(const std::vector<std::vector<ModuleExtent>>& inputs) const;
    bool IsConstant() const noexcept;
    bool defined() const noexcept;
    void AppendCanonical(std::string& bytes) const;
private:
    struct Node;
    explicit ModuleShapeExpr(std::shared_ptr<const Node> node);
    static ModuleShapeExpr Binary(Kind kind, ModuleShapeExpr lhs, ModuleShapeExpr rhs);
    std::shared_ptr<const Node> node_;
};

struct ModuleAxisReference { std::size_t input_index{0}; std::size_t axis{0}; };
struct ModuleAxisGuard {
    std::size_t axis{0}; ModuleExtent lower{0}; ModuleExtent upper{~ModuleExtent{0}};
    ModuleExtent divisible_by{1}; std::optional<ModuleExtent> exact;
    std::optional<ModuleAxisReference> equal_to;
};
struct ModuleInputContract {
    DLDataType dtype{}; Device device; std::size_t rank{0};
    std::vector<ModuleAxisGuard> axis_guards;
};
struct ModuleTensorContract {
    DLDataType dtype{}; Device device; std::vector<ModuleShapeExpr> logical, physical, valid;
    std::size_t alignment{1}; std::string layout{"contiguous.row_major"};
    std::string scope{"global"}; std::size_t max_bytes{0};
};

class ModuleInvocationContract final {
public:
    static constexpr std::uint32_t kAbiVersion = 1;
    ModuleInvocationContract(std::vector<ModuleInputContract> inputs,
                             std::vector<ModuleTensorContract> outputs,
                             std::size_t run_byte_budget = 0,
                             std::uint32_t abi_version = kAbiVersion);
    const std::vector<ModuleInputContract>& inputs() const noexcept;
    const std::vector<ModuleTensorContract>& outputs() const noexcept;
    std::size_t run_byte_budget() const noexcept;
    std::uint32_t abi_version() const noexcept;
    bool IsConstantShape() const noexcept;
    std::string CanonicalBytes() const;
    void Validate(const codegen::KernelSignature& signature) const;
private:
    std::uint32_t abi_version_; std::vector<ModuleInputContract> inputs_;
    std::vector<ModuleTensorContract> outputs_; std::size_t run_byte_budget_;
};

struct ModuleInvocationOutput {
    runtime::NDArray storage; std::vector<ModuleExtent> logical, physical, valid;
};
struct ModuleInvocationResult { std::vector<ModuleInvocationOutput> outputs; AsyncOperation operation; };
enum class ModuleInvocationFailureKind { kInvalidContract, kDisabled, kGuard, kResource, kPreallocatedMismatch, kLaunch };
class ModuleInvocationError : public std::runtime_error {
public:
    ModuleInvocationError(ModuleInvocationFailureKind kind, const std::string& message)
        : std::runtime_error(message), kind_(kind) {}
    ModuleInvocationFailureKind kind() const noexcept { return kind_; }
private: ModuleInvocationFailureKind kind_;
};

}  // namespace kxc::api
