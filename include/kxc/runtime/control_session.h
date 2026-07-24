/*! \file include/kxc/runtime/control_session.h
 * \brief CPU default-stream executor for resolved control execution plans.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kxc/runtime/control_execution_plan.h"

namespace kxc::runtime {

struct ControlLoopIterationCount {
    ControlExecutionTaskId task_id{-1};
    std::int64_t iterations{0};
};

struct ControlRunResult {
    Array<NDArray> outputs;
    std::vector<std::string> events;
    std::vector<ControlLoopIterationCount> loop_iterations;
};

struct ControlRunAsyncResult : ControlRunResult {
    AsyncOperation completion;
};

/*! \brief A fixed-plan static executor; it has no compiler-side dependencies.
 *
 * Module snapshots are copied once while constructing the session, including
 * unselected branches, then shared by every run; no branch constant is copied
 * per run.  ControlRuntimeSession remains the authority that allocates outputs. */
class ControlRuntimeSession final {
public:
    explicit ControlRuntimeSession(ControlExecutionPlan plan);

    ControlRunResult Run(const Array<NDArray>& inputs) const;
    ControlRunAsyncResult RunAsync(const Array<NDArray>& inputs,
                                   const DeviceStream& stream) const;
    const ControlExecutionPlan& plan() const noexcept;

private:
    struct ConstantState;
    ControlExecutionPlan plan_;
    std::shared_ptr<const ConstantState> constants_;
};

}  // namespace kxc::runtime
