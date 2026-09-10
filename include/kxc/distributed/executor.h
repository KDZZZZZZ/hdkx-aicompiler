/*! \file include/kxc/distributed/executor.h
 * \brief 定义 Disco 分布式会话、DRef、执行器和通信后端接口。
 */

#pragma once

#include <memory>

#include "kxc/distributed/ccl_backend.h"
#include "kxc/distributed/execution_plan.h"
#include "kxc/runtime/compiled_module.h"

namespace kxc {
namespace disco {

class ExecutionPlanExecutor {
public:
    explicit ExecutionPlanExecutor(DiscoSession session,
                                   std::shared_ptr<CCLBackend> ccl_backend = CreateCpuCCLBackend());
    ExecutionPlanExecutor(DiscoSession session, api::CompiledModule module,
                          std::shared_ptr<CCLBackend> ccl_backend = CreateCpuCCLBackend());

    /*! Synchronous CPU execution. The caller must not mutate the plan, module
     * or input storage/registers until this call returns. Admission checks the
     * whole plan before communication or launch; each call owns fresh values. */
    Map<int, DRef> Execute(const ExecutionPlan& plan, const Map<int, DRef>& initial_values = {});
    DRef ExecuteForOutput(const ExecutionPlan& plan, const Map<int, DRef>& initial_values = {});
    Array<DRef> ExecuteForOutputs(const ExecutionPlan& plan, const Map<int, DRef>& initial_values = {});

private:
    void ExecuteKernel(const KernelExecNode* kernel, size_t node_index, Map<int, DRef>& values);
    void ExecuteComm(const ExecutionPlan& plan, const CommExecNode* comm, Map<int, DRef>& values);

    DiscoSession session_;
    std::shared_ptr<CCLBackend> ccl_backend_;
    api::CompiledModule module_{ObjectRef()};
};

}  // namespace disco
}  // namespace kxc
