/*! \file src/runtime/internal/module_invocation_contract.h
 * \brief Source-private CompiledModule invocation-contract authoring seam.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "kxc/runtime/kernel_abi.h"
#include "kxc/runtime/module_invocation.h"

namespace kxc::api {

class ModuleShapeExpr final {
public:
    enum class Kind { kConst, kInputAxis, kAdd, kMul, kFloorDiv, kMin, kMax };
    static constexpr std::size_t kMaxDepth = 256;
    static constexpr std::size_t kMaxUniqueNodes = 1024;
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
    void Validate(const std::vector<std::size_t>& input_ranks) const;
    std::size_t unique_node_count() const;
    void AppendCanonical(std::string& bytes) const;
private:
    friend class ModuleInvocationContract;
    struct Node;
    std::pair<ModuleExtent, ModuleExtent> ValidateRanges(
        const std::vector<std::vector<ModuleExtent>>& input_lowers,
        const std::vector<std::vector<ModuleExtent>>& input_uppers) const;
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
struct ModuleInputContract { std::vector<ModuleAxisGuard> axis_guards; };
struct ModuleTensorContract {
    std::vector<ModuleShapeExpr> logical, physical, valid;
    std::size_t max_bytes{0};
};

/*! A one-element uint64 input buffer generated from its KernelArgSpec.
 *  Input-axis-sourced scalars are evaluated from caller input shapes at
 *  invocation time; state-sourced scalars are injected by the invoker from
 *  validated session state metadata (the M2 dynamic-stateful extent ABI). */
struct ModuleRuntimeExtentScalar {
    enum class Source : std::uint8_t { kInputAxis = 0, kStateExtent = 1 };
    Source source = Source::kInputAxis;
    /*! \brief Defined only for kInputAxis scalars. */
    std::optional<ModuleShapeExpr> expression;
    ModuleRuntimeExtentScalar() = default;
    ModuleRuntimeExtentScalar(ModuleShapeExpr input_expression)
        : source(Source::kInputAxis), expression(std::move(input_expression)) {}
    static ModuleRuntimeExtentScalar StateExtent() {
        ModuleRuntimeExtentScalar scalar;
        scalar.source = Source::kStateExtent;
        return scalar;
    }
};

class ModuleInvocationContract final {
public:
    static constexpr std::uint32_t kAbiVersion = 4;
    static constexpr std::size_t kMaxExpressions = 4096;
    ModuleInvocationContract(std::vector<ModuleInputContract> inputs,
                             std::vector<ModuleTensorContract> outputs,
                             std::vector<ModuleRuntimeExtentScalar> runtime_extent_scalars,
                             std::size_t run_byte_budget = 0,
                             std::uint32_t abi_version = kAbiVersion);
    const std::vector<ModuleInputContract>& inputs() const noexcept;
    const std::vector<ModuleTensorContract>& outputs() const noexcept;
    const std::vector<ModuleRuntimeExtentScalar>& runtime_extent_scalars() const noexcept;
    std::size_t run_byte_budget() const noexcept;
    std::uint32_t abi_version() const noexcept;
    bool IsConstantShape(const codegen::KernelSignature& signature) const noexcept;
    std::string CanonicalBytes() const;
    void Validate(const codegen::KernelSignature& signature) const;
private:
    std::uint32_t abi_version_; std::vector<ModuleInputContract> inputs_;
    std::vector<ModuleTensorContract> outputs_;
    std::vector<ModuleRuntimeExtentScalar> runtime_extent_scalars_;
    std::size_t run_byte_budget_;
};

}  // namespace kxc::api
