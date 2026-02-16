#pragma once

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/container.h"
#include "base/object.h"
#include "tir/expr.h"

namespace kxc {
namespace te {

class Schedule;
class Stage;
class Operation;
class Tensor;

class OperationNode : public Object {
public:
    std::string name;
    std::string tag;
    Map<String, ObjectRef> attrs;

    virtual int num_outputs() const = 0;
    virtual tir::DataType output_dtype(int i) const = 0;
    virtual Array<tir::PrimExpr> output_shape(int i) const = 0;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(OperationNode)

class Operation : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    const OperationNode* operator->() const { return static_cast<const OperationNode*>(object_); }
    OperationNode* operator->() { return static_cast<OperationNode*>(const_cast<Object*>(object_)); }

    int num_outputs() const { return operator->()->num_outputs(); }
    tir::DataType output_dtype(int i) const { return operator->()->output_dtype(i); }
    Array<tir::PrimExpr> output_shape(int i) const { return operator->()->output_shape(i); }
};

}  // namespace te
}  // namespace kxc

namespace std {
template <>
struct hash<kxc::te::Operation> {
    size_t operator()(const kxc::te::Operation& k) const {
        return std::hash<const kxc::Object*>()(k.get());
    }
};
}

namespace kxc {
namespace te {

class TensorNode : public Object {
public:
    std::string name;
    Array<tir::PrimExpr> shape;
    tir::DataType dtype;
    Operation op;
    int value_index;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(TensorNode)

class Tensor : public ObjectRef {
public:
    using ObjectRef::ObjectRef;

    Tensor(Array<tir::PrimExpr> shape, tir::DataType dtype, Operation op, int value_index);

    const TensorNode* operator->() const { return static_cast<const TensorNode*>(object_); }

    tir::PrimExpr operator()(const Array<tir::PrimExpr>& indices) const;
    tir::PrimExpr operator()(const Array<tir::Var>& indices) const;

    template <typename... Args>
    tir::PrimExpr operator()(Args... args) const {
        return (*this)(Array<tir::PrimExpr>{tir::PrimExpr(args)...});
    }
};

class ProducerLoadNode : public tir::PrimExprNode {
public:
    Tensor tensor;
    Array<tir::PrimExpr> indices;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ProducerLoadNode)

class ProducerLoad : public tir::PrimExpr {
public:
    using PrimExpr::PrimExpr;
    ProducerLoad(Tensor tensor, Array<tir::PrimExpr> indices);
};

enum class IterVarType : int {
    kDataPar = 0,
    kThreadIndex = 1,
    kCommReduce = 2,
    kOrdered = 3,
    kOpaque = 4,
    kVectorized = 5,
    kParallel = 6,
    kUnrolled = 7,
};

class IterVarNode : public Object {
public:
    tir::Var var;
    tir::PrimExpr dom_min;
    tir::PrimExpr dom_extent;
    IterVarType iter_type;
    std::string thread_tag;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(IterVarNode)

class IterVar : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit IterVar(tir::PrimExpr min, tir::PrimExpr extent,
                     IterVarType type = IterVarType::kDataPar,
                     std::string thread_tag = "", std::string name = "rv");
    const IterVarNode* operator->() const { return static_cast<const IterVarNode*>(object_); }
    operator tir::PrimExpr() const { return operator->()->var; }
    operator tir::Var() const { return operator->()->var; }

    bool operator==(const IterVar& other) const { return object_ == other.object_; }
    bool operator!=(const IterVar& other) const { return !(*this == other); }
};

IterVar reduce_axis(tir::PrimExpr min, tir::PrimExpr extent, std::string name = "rv");

class StageNode : public Object {
public:
    Operation op;
    Array<IterVar> leaf_iter_vars;
    Array<IterVar> all_iter_vars;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(StageNode)

class Stage : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit Stage(Operation op);

    const StageNode* operator->() const { return static_cast<const StageNode*>(object_); }
    StageNode* operator->() { return static_cast<StageNode*>(const_cast<Object*>(object_)); }

    IterVar split(IterVar parent, tir::PrimExpr factor, IterVar* p_outer = nullptr,
                  IterVar* p_inner = nullptr);
    IterVar fuse(IterVar outer, IterVar inner);
    void reorder(const Array<IterVar>& order);
    void tile(IterVar x_parent, IterVar y_parent, tir::PrimExpr x_factor,
              tir::PrimExpr y_factor, IterVar* x_outer, IterVar* y_outer,
              IterVar* x_inner, IterVar* y_inner);
    void vectorize(IterVar var);
    void unroll(IterVar var);
    void parallel(IterVar var);
    void bind(IterVar var, IterVar thread_axis);
};

enum class ReduceType : int {
    kSum = 0,
    kMax = 1,
    kMin = 2,
};

class ReduceNode : public tir::PrimExprNode {
public:
    Array<IterVar> axis;
    Array<tir::PrimExpr> source;
    ReduceType reduce_type = ReduceType::kSum;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ReduceNode)

class Reduce : public tir::PrimExpr {
public:
    using PrimExpr::PrimExpr;
    Reduce(Array<IterVar> axis, Array<tir::PrimExpr> source,
           ReduceType type = ReduceType::kSum);
};

tir::PrimExpr sum(tir::PrimExpr expr, Array<IterVar> axis);
tir::PrimExpr max(tir::PrimExpr expr, Array<IterVar> axis);

class PlaceholderOpNode : public OperationNode {
public:
    Array<tir::PrimExpr> shape;
    tir::DataType dtype;

    int num_outputs() const override { return 1; }
    tir::DataType output_dtype(int i) const override {
        (void)i;
        return dtype;
    }
    Array<tir::PrimExpr> output_shape(int i) const override {
        (void)i;
        return shape;
    }

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(PlaceholderOpNode)

class PlaceholderOp : public Operation {
public:
    using Operation::Operation;
    PlaceholderOp(std::string name, Array<tir::PrimExpr> shape, tir::DataType dtype);
};

class ComputeOpNode : public OperationNode {
public:
    Array<tir::Var> axis;
    Array<IterVar> reduce_axis;
    Array<tir::PrimExpr> body;

    int num_outputs() const override { return static_cast<int>(body.size()); }
    tir::DataType output_dtype(int i) const override { return body[i].dtype(); }
    Array<tir::PrimExpr> shape;

    Array<tir::PrimExpr> output_shape(int i) const override {
        (void)i;
        return shape;
    }

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ComputeOpNode)

class ComputeOp : public Operation {
public:
    using Operation::Operation;
    ComputeOp(std::string name, std::string tag, Map<String, ObjectRef> attrs,
              Array<tir::Var> axis, Array<tir::PrimExpr> body,
              Array<tir::PrimExpr> shape);
};

Tensor placeholder(Array<tir::PrimExpr> shape,
                   tir::DataType dtype = tir::DataType::Float(32),
                   std::string name = "placeholder");

using FCompute = std::function<tir::PrimExpr(const Array<tir::Var>&)>;

Tensor compute(Array<tir::PrimExpr> shape, FCompute fcompute,
               std::string name = "compute", std::string tag = "",
               Map<String, ObjectRef> attrs = {});

IterVar thread_axis(tir::PrimExpr dom, std::string tag);

class ScheduleNode : public Object {
public:
    Array<Operation> outputs;
    Array<Stage> stages;
    Map<Operation, Stage> op_map;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ScheduleNode)

class Schedule : public ObjectRef {
public:
    using ObjectRef::ObjectRef;

    Stage operator[](const Operation& op) { return operator->()->op_map.at(op); }

    const ScheduleNode* operator->() const { return static_cast<const ScheduleNode*>(object_); }
    ScheduleNode* operator->() { return static_cast<ScheduleNode*>(const_cast<Object*>(object_)); }
};

Schedule create_schedule(const Array<Operation>& ops);

}  // namespace te
}  // namespace kxc