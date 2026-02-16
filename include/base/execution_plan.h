#pragma once

#include <string>

#include "base/container.h"
#include "base/pass.h"
#include "relay/op.h"
#include "tir/stmt.h"

namespace kxc {

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
KXC_OBJECT_DEFINE(ExecNodeBaseNode)

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
    tir::PrimFunc primfunc;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(KernelExecNode)

class KernelExec : public ExecNodeBase {
public:
    using ExecNodeBase::ExecNodeBase;
    explicit KernelExec(const ObjectRef& ref) : ExecNodeBase(ref) {}
    KernelExec(std::string op_name, tir::PrimFunc primfunc, Array<int> input_values,
               Array<int> output_values, Array<int> worker_set);
    const KernelExecNode* operator->() const {
        return static_cast<const KernelExecNode*>(object_);
    }
};

class CommExecNode : public ExecNodeBaseNode {
public:
    std::string op_name;
    ObjectRef attrs;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(CommExecNode)

class CommExec : public ExecNodeBase {
public:
    using ExecNodeBase::ExecNodeBase;
    explicit CommExec(const ObjectRef& ref) : ExecNodeBase(ref) {}
    CommExec(std::string op_name, ObjectRef attrs, Array<int> input_values,
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
KXC_OBJECT_DEFINE(BarrierExecNode)

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
    int num_values{0};
    PassContext pass_ctx;
    int output_value{-1};

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ExecutionPlanNode)

class ExecutionPlan : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit ExecutionPlan(const ObjectRef& ref) : ObjectRef(ref) {}
    ExecutionPlan(Array<ObjectRef> nodes, Map<int, VirtualDevice> value_virtual_devices,
                  int num_values, PassContext pass_ctx, int output_value);
    const ExecutionPlanNode* operator->() const {
        return static_cast<const ExecutionPlanNode*>(object_);
    }
    std::string ToString() const;
};

bool IsCommunicationOpName(const std::string& op_name);

}  // namespace kxc

