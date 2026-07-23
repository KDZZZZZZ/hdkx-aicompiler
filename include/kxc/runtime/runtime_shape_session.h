/*! \file include/kxc/runtime/runtime_shape_session.h
 * \brief CPU:0/default-stream ShapeEval -> Allocate -> Kernel session.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kxc/runtime/runtime_shape_plan.h"

namespace kxc::runtime {

enum class RuntimeShapeEventKind { kShapeEval, kAllocate, kKernel, kFailure };

struct RuntimeShapeEvent {
    RuntimeShapeEventKind kind{RuntimeShapeEventKind::kFailure};
    std::size_t output_index{static_cast<std::size_t>(-1)};
    std::size_t bytes{0};
    std::string detail;
};

/*! \brief Result owns plan, module lease, outputs, run state, and caller lease. */
class RuntimeShapeAsyncResult final {
public:
    bool ok() const noexcept;
    const std::string& failure_reason() const noexcept;
    const std::vector<RuntimeShapeOutput>& outputs() const noexcept;
    const std::vector<RuntimeShapeEvent>& events() const noexcept;
    bool IsReady() const noexcept;
    /*! \brief Completes only the documented fake deterministic test seam. */
    void Wait() const noexcept;

private:
    struct State;
    explicit RuntimeShapeAsyncResult(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;
    friend class RuntimeShapeSession;
};

/*! \brief Separate default-off dynamic output session; no allocation reuse. */
class RuntimeShapeSession final {
public:
    explicit RuntimeShapeSession(RuntimeShapePlan plan);

    RuntimeShapeAsyncResult RunAsync(
        const std::vector<RuntimeShapeInput>& inputs,
        std::shared_ptr<void> caller_lease = {}) const;
    RuntimeShapeAsyncResult Run(const std::vector<RuntimeShapeInput>& inputs,
                                std::shared_ptr<void> caller_lease = {}) const;
    const RuntimeShapePlan& plan() const noexcept;

private:
    RuntimeShapePlan plan_;
};

}  // namespace kxc::runtime
