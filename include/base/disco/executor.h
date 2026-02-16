#pragma once

#include <memory>

#include "base/disco/ccl_backend.h"
#include "base/execution_plan.h"

namespace kxc {
namespace disco {

class ExecutionPlanExecutor {
public:
    explicit ExecutionPlanExecutor(DiscoSession session,
                                   std::shared_ptr<CCLBackend> ccl_backend = CreateCpuCCLBackend());

    Map<int, DRef> Execute(const ExecutionPlan& plan, const Map<int, DRef>& initial_values = {});
    DRef ExecuteForOutput(const ExecutionPlan& plan, const Map<int, DRef>& initial_values = {});

private:
    DRef EnsureValue(const ExecutionPlan& plan, int value_id, const DRef& prototype);
    void ExecuteKernel(const ExecutionPlan& plan, const KernelExecNode* kernel);
    void ExecuteComm(const ExecutionPlan& plan, const CommExecNode* comm);
    int ResolveWorkerForValue(const ExecutionPlan& plan, int value_id) const;
    int ResolveWorkerForVirtualDevice(const ExecutionPlan& plan, const VirtualDevice& vd) const;

    DiscoSession session_;
    std::shared_ptr<CCLBackend> ccl_backend_;
    Map<int, DRef> values_;
};

}  // namespace disco
}  // namespace kxc

