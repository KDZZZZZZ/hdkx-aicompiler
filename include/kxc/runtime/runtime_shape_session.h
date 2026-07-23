/*! \file include/kxc/runtime/runtime_shape_session.h
 * \brief Restricted CPU synchronous and opt-in CUDA asynchronous shape session.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kxc/runtime/runtime_shape_plan.h"

namespace kxc::runtime {

/*! \brief Stable failure outcome for an upper control plane; no fallback is implicit. */
enum class RuntimeShapeFailureKind {
    kNone,
    kDisabled,
    kApplicabilityMiss,
    kShapeAbiRejected,
    kResourceExhausted,
    kLaunchRejected,
    kSubmissionFailed,
    kCompletionFailed,
};

enum class RuntimeShapeEventKind {
    kShapeEval,
    kAllocate,
    kKernel,
    kSubmission,
    kCompletion,
    /*! \brief Observed-completion retirement eligibility, never physical deallocation. */
    kRetire,
    kFailure,
};

struct RuntimeShapeEvent {
    RuntimeShapeEventKind kind{RuntimeShapeEventKind::kFailure};
    std::size_t output_index{static_cast<std::size_t>(-1)};
    std::size_t bytes{0};
    std::string detail;
};

class RuntimeShapeAsyncResult final {
public:
    bool ok() const noexcept;
    /*! \brief Base failure text is immutable; completion failure returns static text. */
    const std::string& failure_reason() const noexcept;
    RuntimeShapeFailureKind failure_kind() const noexcept;
    /*! \brief Immutable outputs published by RunAsync/Run. */
    const std::vector<RuntimeShapeOutput>& outputs() const noexcept;
    /*! \brief Returns a thread-safe value snapshot, including observed completion telemetry. */
    std::vector<RuntimeShapeEvent> events() const;
    /*! \brief Delegates CUDA completion polling; CPU fake completion remains test-only. */
    bool IsReady() const noexcept;
    /*! \brief Delegates CUDA completion waiting; CPU fake completion remains test-only. */
    void Wait() const noexcept;
    /*! \brief Exact CUDA output bytes held by this result, not process residency. */
    std::size_t retained_device_bytes() const noexcept;

private:
    struct State;
    explicit RuntimeShapeAsyncResult(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;
    friend class RuntimeShapeSession;
};

/*! \brief Frozen shape executor with separate CPU and CUDA callback contracts. */
class RuntimeShapeSession final {
public:
    explicit RuntimeShapeSession(RuntimeShapePlan plan);

    /*! \brief Preserves the existing CPU trusted synchronous call shape. */
    RuntimeShapeAsyncResult RunAsync(
        const std::vector<RuntimeShapeInput>& inputs,
        std::shared_ptr<void> caller_lease = {}) const;
    /*! \brief CUDA entries use this overload; stream may be undefined for CUDA default stream. */
    RuntimeShapeAsyncResult RunAsync(
        const std::vector<RuntimeShapeInput>& inputs, ::kxc::DeviceStream stream,
        std::shared_ptr<void> caller_lease = {}) const;
    RuntimeShapeAsyncResult Run(const std::vector<RuntimeShapeInput>& inputs,
                                std::shared_ptr<void> caller_lease = {}) const;
    const RuntimeShapePlan& plan() const noexcept;

private:
    RuntimeShapeAsyncResult RunImpl(const std::vector<RuntimeShapeInput>& inputs,
                                    ::kxc::DeviceStream stream,
                                    std::shared_ptr<void> caller_lease) const;
    RuntimeShapePlan plan_;
};

}  // namespace kxc::runtime
