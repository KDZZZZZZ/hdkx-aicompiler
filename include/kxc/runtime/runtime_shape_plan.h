/*! \file include/kxc/runtime/runtime_shape_plan.h
 * \brief Default-off, CPU-only dynamic-shape launch contract.
 *
 * This is deliberately independent of Compiler, Relay, caches, and adaptive
 * runtime APIs.  Its bound launcher is restricted local evidence, not a
 * CompiledModule dynamic-output ABI.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace kxc::runtime {

using RuntimeShapeExtent = std::uint64_t;

/*! \brief Immutable checked extent expression. */
class RuntimeShapeExpr final {
public:
    enum class Kind { kConst, kInputAxis, kAdd, kMul, kFloorDiv, kMin, kMax };

    static RuntimeShapeExpr Const(RuntimeShapeExtent value);
    static RuntimeShapeExpr InputAxis(std::size_t input_index, std::size_t axis);
    static RuntimeShapeExpr Add(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs);
    static RuntimeShapeExpr Mul(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs);
    static RuntimeShapeExpr FloorDiv(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs);
    static RuntimeShapeExpr Min(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs);
    static RuntimeShapeExpr Max(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs);

    RuntimeShapeExtent Evaluate(
        const std::vector<std::vector<RuntimeShapeExtent>>& input_shapes) const;
    Kind kind() const;
    bool defined() const noexcept;

private:
    struct Node;
    explicit RuntimeShapeExpr(std::shared_ptr<const Node> node);
    static RuntimeShapeExpr Binary(Kind kind, RuntimeShapeExpr lhs, RuntimeShapeExpr rhs);
    static std::size_t Depth(const std::shared_ptr<const Node>& node);
    void AppendCanonical(std::string& bytes) const;
    std::shared_ptr<const Node> node_;
    friend class RuntimeShapePlan;
};

struct RuntimeShapeInputContract {
    std::string dtype;
    std::size_t rank{0};
    std::string device{"CPU:0"};
    std::uint32_t abi_version{1};
};

/*! \brief Shape and allocation contract for one dynamic output. */
struct RuntimeShapeTensorContract {
    std::string dtype;
    std::vector<RuntimeShapeExpr> logical;
    std::vector<RuntimeShapeExpr> physical;
    std::vector<RuntimeShapeExpr> valid;
    std::size_t alignment{1};
    std::string layout{"contiguous.row_major"};
    std::string scope{"global"};
    std::size_t max_bytes{0};
    std::string device{"CPU:0"};
    std::uint32_t abi_version{1};
};

struct RuntimeShapeInput {
    std::vector<RuntimeShapeExtent> shape;
    std::string dtype;
    std::string device{"CPU:0"};
    std::uint32_t abi_version{1};
};

struct RuntimeShapeOutput {
    RuntimeShapeTensorContract contract;
    std::vector<RuntimeShapeExtent> logical;
    std::vector<RuntimeShapeExtent> physical;
    std::vector<RuntimeShapeExtent> valid;
    void* data{nullptr};
    std::size_t bytes{0};

private:
    std::shared_ptr<void> owner_;
    friend class RuntimeShapeSession;
};

/*! \brief A deterministic test seam, explicitly fake and not CUDA evidence. */
class FakeRuntimeShapeCompletion final {
public:
    static std::shared_ptr<FakeRuntimeShapeCompletion> Pending();
    static std::shared_ptr<FakeRuntimeShapeCompletion> Completed();
    void Complete() noexcept;
    bool IsReady() const noexcept;

private:
    explicit FakeRuntimeShapeCompletion(bool ready) noexcept;
    struct State;
    std::shared_ptr<State> state_;
};

namespace detail {
/*! \brief Deterministically fails the next owner transfer; test seam only. */
void FailNextRuntimeShapeOwnerTransferForTest() noexcept;
}  // namespace detail

struct RuntimeShapeLaunchArgs {
    const std::vector<RuntimeShapeInput>& inputs;
    const std::vector<RuntimeShapeOutput>& outputs;
    /*! \brief Canonical full ABI bytes for this synchronous launch. */
    const std::string& exact_abi_fingerprint;
};

struct RuntimeShapeLaunchResult {
    bool accepted{true};
    std::string failure_reason;
    /*! \brief Optional deterministic fake completion for retention tests only. */
    std::shared_ptr<FakeRuntimeShapeCompletion> fake_completion;
};

/*! \brief Trusted local synchronous launcher contract.
 *
 * The launcher is called synchronously while the runtime owns all argument
 * storage. It must not retain argument references and must not submit
 * asynchronous work. It is a trusted local-evidence declaration, not a
 * CompiledModule ABI or an executor fence.
 */
using RuntimeShapeBoundLauncher =
    std::function<RuntimeShapeLaunchResult(const RuntimeShapeLaunchArgs&)>;

/*! \brief Preselected trusted synchronous entry; not a general CompiledModule ABI. */
struct RuntimeShapeReadyEntry {
    std::string module_label;
    std::string entry_symbol;
    std::uint32_t abi_version{1};
    bool ready{false};
    /*! \brief Full canonical ABI bytes, not a digest. */
    std::string exact_abi_fingerprint;
    RuntimeShapeBoundLauncher launcher;
    /*! \brief Opaque module-side lifetime owner retained by every result. */
    std::shared_ptr<void> module_lease;
};

struct RuntimeShapePlanSpec {
    static constexpr std::uint32_t kAbiVersion = 1;
    std::uint32_t abi_version{kAbiVersion};
    std::vector<RuntimeShapeInputContract> inputs;
    std::vector<RuntimeShapeTensorContract> outputs;
    RuntimeShapeReadyEntry entry;
    std::size_t run_byte_budget{0};
};

/*! \brief Frozen immutable shape plan with no mutable per-run state. */
class RuntimeShapePlan final {
public:
    static constexpr std::uint32_t kAbiVersion = RuntimeShapePlanSpec::kAbiVersion;
    RuntimeShapePlan() = default;
    explicit RuntimeShapePlan(RuntimeShapePlanSpec spec);

    /*! \brief Deterministic full canonical ABI bytes for a trusted entry binding. */
    static std::string ExactAbiFingerprint(
        const std::vector<RuntimeShapeInputContract>& inputs,
        const std::vector<RuntimeShapeTensorContract>& outputs);

    bool defined() const noexcept;
    void Validate() const;
    const RuntimeShapePlanSpec& spec() const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    std::mutex& launcher_mutex() const;
    friend class RuntimeShapeSession;
};

}  // namespace kxc::runtime
