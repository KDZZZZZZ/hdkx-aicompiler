/*! \file include/kxc/distributed/execution_plan.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <string>

#include "kxc/support/container.h"
#include "kxc/distributed/placement.h"
#include "kxc/pass/context.h"

namespace kxc::api { class CompiledModule; }
namespace kxc::runtime { class ExecutablePlan; }

namespace kxc {

inline constexpr int kDistributedExecutionContractVersion = 3;

enum class ExecNodeKind : int {
    kKernel = 0,
    kComm = 1,
    kBarrier = 2,
};

class ExecNodeBaseNode : public Object {
public:
    ExecNodeKind kind{ExecNodeKind::kKernel};
    Array<int> input_values;
    Array<int> output_values;
    Array<int> worker_set;

    KXC_OBJECT_DECLARE
};

class ExecNodeBase : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit ExecNodeBase(const ObjectRef& ref) : ObjectRef(ref) {}
    const ExecNodeBaseNode* operator->() const {
        return static_cast<const ExecNodeBaseNode*>(object_);
    }
};

class KernelExecNode : public ExecNodeBaseNode {
public:
    std::string op_name;
    std::string kernel_symbol;
    /*! Module-owned KernelSignature::CanonicalBytes(), checked before any work. */
    std::string kernel_abi;

    KXC_OBJECT_DECLARE
};

class KernelExec : public ExecNodeBase {
public:
    using ExecNodeBase::ExecNodeBase;
    explicit KernelExec(const ObjectRef& ref) : ExecNodeBase(ref) {}
    KernelExec(std::string op_name, Array<int> input_values,
               Array<int> output_values, Array<int> worker_set,
               std::string kernel_symbol = "", std::string kernel_abi = "");
    const KernelExecNode* operator->() const {
        return static_cast<const KernelExecNode*>(object_);
    }
};

struct CommExecAttrs {
    VirtualDevice src_virtual_device;
    VirtualDevice dst_virtual_device;
    std::string kind;
    std::string reduce_kind{"sum"};
    bool async{false};
    bool in_group{true};
    int group_id{0};
    int root_worker{0};
};

class CommExecNode : public ExecNodeBaseNode {
public:
    std::string op_name;
    CommExecAttrs attrs;

    KXC_OBJECT_DECLARE
};

class CommExec : public ExecNodeBase {
public:
    using ExecNodeBase::ExecNodeBase;
    explicit CommExec(const ObjectRef& ref) : ExecNodeBase(ref) {}
    CommExec(std::string op_name, CommExecAttrs attrs, Array<int> input_values,
             Array<int> output_values, Array<int> worker_set);
    const CommExecNode* operator->() const {
        return static_cast<const CommExecNode*>(object_);
    }
};

class BarrierExecNode : public ExecNodeBaseNode {
public:
    std::string tag;

    KXC_OBJECT_DECLARE
};

class BarrierExec : public ExecNodeBase {
public:
    using ExecNodeBase::ExecNodeBase;
    explicit BarrierExec(const ObjectRef& ref) : ExecNodeBase(ref) {}
    BarrierExec(std::string tag, Array<int> worker_set);
    const BarrierExecNode* operator->() const {
        return static_cast<const BarrierExecNode*>(object_);
    }
};

class ExecutionPlanNode : public Object {
public:
    Array<ObjectRef> nodes;
    Map<int, VirtualDevice> value_virtual_devices;
    Array<int> input_value_ids;
    Array<int> constant_value_ids;
    Map<int, Array<int64_t>> value_shapes;
    Map<int, std::string> value_dtypes;
    int num_values{0};
    PassContext pass_ctx;
    DiscoPlacement placement;
    /*! Ordered graph outputs; output_value is the checked first-output compatibility view. */
    Array<int> output_value_ids;
    int output_value{-1};

    KXC_OBJECT_DECLARE
};

class ExecutionPlan : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit ExecutionPlan(const ObjectRef& ref) : ObjectRef(ref) {}
    ExecutionPlan(Array<ObjectRef> nodes, Map<int, VirtualDevice> value_virtual_devices,
                  Array<int> input_value_ids, Array<int> constant_value_ids,
                  Map<int, Array<int64_t>> value_shapes, Map<int, std::string> value_dtypes,
                  int num_values, PassContext pass_ctx, DiscoPlacement placement,
                  int output_value);
    ExecutionPlan(Array<ObjectRef> nodes, Map<int, VirtualDevice> value_virtual_devices,
                  Array<int> input_value_ids, Array<int> constant_value_ids,
                  Map<int, Array<int64_t>> value_shapes, Map<int, std::string> value_dtypes,
                  int num_values, PassContext pass_ctx, DiscoPlacement placement,
                  Array<int> output_value_ids);
    /*! Bind an already compiled static, fresh-output CPU graph to explicit
     *  input/call workers and one output worker. Reuses the module's ABI and
     *  constants; inserts only declared point-to-point copies. No compilation,
     *  tensor allocation or communication occurs here. Input/output order is
     *  the source ExecutablePlan order; distributed value ids are local. */
    static ExecutionPlan FromExecutablePlan(const api::CompiledModule& module,
        const runtime::ExecutablePlan& source, const DiscoPlacement& placement,
        const Array<int>& input_workers, const Array<int>& call_workers,
        int output_worker);
    const ExecutionPlanNode* operator->() const {
        return static_cast<const ExecutionPlanNode*>(object_);
    }
    std::string ToString() const;
    /*! Validate placement, static value contracts and the complete use/def order. */
    void Validate() const;
};

bool IsCommunicationOpName(const std::string& op_name);

std::string SerializeExecutionPlanToJson(const ExecutionPlan& plan);
ExecutionPlan DeserializeExecutionPlanFromJson(const std::string& json_text);
ExecutionPlan LoadExecutionPlanFromJsonFile(const std::string& path);
void SaveExecutionPlanToJsonFile(const ExecutionPlan& plan, const std::string& path);

}  // namespace kxc
