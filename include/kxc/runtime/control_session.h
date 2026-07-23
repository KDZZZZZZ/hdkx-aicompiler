/*! \file include/kxc/runtime/control_session.h
 * \brief CPU default-stream executor for resolved control execution plans.
 */
#pragma once

#include <cstdint>
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

/*! \brief A fixed-plan static executor; it has no compiler-side dependencies. */
class ControlRuntimeSession final {
public:
    explicit ControlRuntimeSession(ControlExecutionPlan plan);

    ControlRunResult Run(const Array<NDArray>& inputs) const;
    ControlRunAsyncResult RunAsync(const Array<NDArray>& inputs,
                                   const DeviceStream& stream) const;
    const ControlExecutionPlan& plan() const noexcept;

private:
    ControlExecutionPlan plan_;
};

}  // namespace kxc::runtime
