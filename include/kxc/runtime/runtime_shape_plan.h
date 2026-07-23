/*! \file include/kxc/runtime/runtime_shape_plan.h
 * \brief Default-off restricted runtime shape launch contracts.
 *
 * This is deliberately independent of Compiler, Relay, caches, and adaptive
 * runtime APIs. CPU entries remain trusted local synchronous evidence; CUDA
 * entries use a separate, explicit completion ABI.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "kxc/runtime/device_stream.h"

namespace kxc::runtime {

using RuntimeShapeExtent = std::uint64_t;

enum class RuntimeShapeExecutionKind { kSynchronousCpu, kCudaAsync };

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

struct RuntimeShapeInputAxisReference {
    std::size_t input_index{0};
    std::size_t axis{0};
};

struct RuntimeShapeInputAxisGuard {
    std::size_t axis{0};
    RuntimeShapeExtent lower{0};
    RuntimeShapeExtent upper{0};
    RuntimeShapeExtent divisible_by{1};
    std::optional<RuntimeShapeExtent> exact;
    std::optional<RuntimeShapeInputAxisReference> equal_to;
};

struct RuntimeShapeInputContract {
    std::string dtype;
    std::size_t rank{0};
    std::string device{"CPU:0"};
    std::uint32_t abi_version{1};
    std::vector<RuntimeShapeInputAxisGuard> axis_guards;
    bool requires_data{false};
};

struct RuntimeShapeExtentScalar {
    std::uint32_t ordinal{0};
    std::string name;
    std::string symbol;
    std::size_t input_index{0};
    std::size_t axis{0};
    RuntimeShapeExtent lower{0};
    RuntimeShapeExtent upper{0};
    RuntimeShapeExtent divisible_by{1};
};

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
    const void* data{nullptr};
    std::size_t bytes{0};
    std::shared_ptr<void> owner;
    /*! \brief Required for CUDA inputs; backing storage must be on input.device. */
    ::kxc::Storage storage;
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
    ::kxc::Storage storage_;
    friend class RuntimeShapeSession;
};

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
void FailNextRuntimeShapeOwnerTransferForTest() noexcept;
/*! \brief Deterministically fails CUDA retention after launcher return; test seam only. */
void FailNextRuntimeShapeCudaRetentionForTest() noexcept;
}  // namespace detail

struct RuntimeShapeLaunchArgs {
    const std::vector<RuntimeShapeInput>& inputs;
    const std::vector<RuntimeShapeOutput>& outputs;
    const std::vector<RuntimeShapeExtent>& runtime_extent_values;
    const std::string& exact_abi_fingerprint;
};

struct RuntimeShapeLaunchResult {
    bool accepted{true};
    std::string failure_reason;
    std::shared_ptr<FakeRuntimeShapeCompletion> fake_completion;
};

using RuntimeShapeBoundLauncher =
    std::function<RuntimeShapeLaunchResult(const RuntimeShapeLaunchArgs&)>;

/*! \brief CUDA-only launcher: submission is accepted only with real completion. */
struct RuntimeShapeCudaLaunchArgs {
    const std::vector<RuntimeShapeInput>& inputs;
    const std::vector<RuntimeShapeOutput>& outputs;
    const std::vector<RuntimeShapeExtent>& runtime_extent_values;
    const std::string& exact_abi_fingerprint;
    ::kxc::DeviceStream stream;
};
using RuntimeShapeCudaBoundLauncher =
    std::function<::kxc::AsyncOperation(const RuntimeShapeCudaLaunchArgs&)>;

struct RuntimeShapeReadyEntry {
    std::string module_label;
    std::string entry_symbol;
    std::uint32_t abi_version{1};
    bool ready{false};
    std::string exact_abi_fingerprint;
    /*! \brief CPU-only synchronous callback; never used for CUDA async entries. */
    RuntimeShapeBoundLauncher launcher;
    /*! \brief CUDA async callback; must return a defined same-device completion. */
    RuntimeShapeCudaBoundLauncher cuda_launcher;
    RuntimeShapeExecutionKind execution_kind{RuntimeShapeExecutionKind::kSynchronousCpu};
    /*! \brief Required CUDA:N identity for kCudaAsync; empty for legacy CPU entries. */
    std::string device_contract;
    std::shared_ptr<void> module_lease;
    std::string artifact_identity;
    std::string tail_policy_identity;
};

struct RuntimeShapePlanSpec {
    static constexpr std::uint32_t kAbiVersion = 1;
    std::uint32_t abi_version{kAbiVersion};
    std::vector<RuntimeShapeInputContract> inputs;
    std::vector<RuntimeShapeTensorContract> outputs;
    std::vector<RuntimeShapeExtentScalar> runtime_extent_abi;
    RuntimeShapeReadyEntry entry;
    std::size_t run_byte_budget{0};
};

class RuntimeShapePlan final {
public:
    static constexpr std::uint32_t kAbiVersion = RuntimeShapePlanSpec::kAbiVersion;
    RuntimeShapePlan() = default;
    explicit RuntimeShapePlan(RuntimeShapePlanSpec spec);

    /*! \brief Full canonical ABI. Legacy CPU calls retain their v1 bytes. */
    static std::string ExactAbiFingerprint(
        const std::vector<RuntimeShapeInputContract>& inputs,
        const std::vector<RuntimeShapeTensorContract>& outputs,
        const std::vector<RuntimeShapeExtentScalar>& runtime_extent_abi = {},
        const std::string& artifact_identity = {},
        const std::string& tail_policy_identity = {},
        RuntimeShapeExecutionKind execution_kind = RuntimeShapeExecutionKind::kSynchronousCpu,
        const std::string& device_contract = {});

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
