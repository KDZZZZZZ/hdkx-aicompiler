/*! \file include/te/te.h
 * \brief 定义 TE tensor、operation、reduce 和 schedule 表达层。
 */

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
class IterVar;

/*! \brief 将 TE/TIR 索引参数统一转换为 TIR PrimExpr。 */
tir::PrimExpr AsPrimExpr(const tir::PrimExpr& expr);
tir::PrimExpr AsPrimExpr(const tir::Var& var);
tir::PrimExpr AsPrimExpr(const IterVar& iter_var);

/*! \brief TE operation 的基类，描述 tensor 由哪个计算节点产生。 */
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

/*! \brief TE operation 的引用类型，提供输出个数、dtype 和 shape 查询。 */
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

/*! \brief TE tensor 节点，记录 shape、dtype、生产 operation 和输出索引。 */
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

/*! \brief TE tensor 的引用类型，可通过下标访问构造 ProducerLoad 表达式。 */
class Tensor : public ObjectRef {
public:
    using ObjectRef::ObjectRef;

    /*! \brief 创建一个由 op 的第 value_index 个输出产生的 tensor。 */
    Tensor(Array<tir::PrimExpr> shape, tir::DataType dtype, Operation op, int value_index);

    const TensorNode* operator->() const { return static_cast<const TensorNode*>(object_); }

    /*! \brief 以 PrimExpr 下标读取 tensor，返回 TE producer load。 */
    tir::PrimExpr operator()(const Array<tir::PrimExpr>& indices) const;
    /*! \brief 以 TIR Var 下标读取 tensor，便于 compute lambda 使用。 */
    tir::PrimExpr operator()(const Array<tir::Var>& indices) const;

    template <typename... Args>
    tir::PrimExpr operator()(Args... args) const {
        return (*this)(Array<tir::PrimExpr>{AsPrimExpr(args)...});
    }
};

/*! \brief 对 TE tensor 的符号读取，lowering 时会转换为 TIR Load/Buffer 访问。 */
class ProducerLoadNode : public tir::PrimExprNode {
public:
    Tensor tensor;
    Array<tir::PrimExpr> indices;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ProducerLoadNode)

/*! \brief ProducerLoad 的引用类型。 */
class ProducerLoad : public tir::PrimExpr {
public:
    using PrimExpr::PrimExpr;
    ProducerLoad(Tensor tensor, Array<tir::PrimExpr> indices);
};

/*! \brief TE 迭代变量的调度语义。 */
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

/*! \brief TE 迭代轴，描述数据并行、归约或线程绑定维度。 */
class IterVarNode : public Object {
public:
    tir::Var var;
    tir::PrimExpr dom_min;
    tir::PrimExpr dom_extent;
    IterVarType iter_type;
    std::string thread_tag;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE_WITH_KEY(IterVarNode, "kxc.te.IterVarNode")

/*! \brief TE 迭代轴引用类型，可隐式转换为底层 TIR Var/PrimExpr。 */
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

inline tir::PrimExpr AsPrimExpr(const tir::PrimExpr& expr) {
    return expr;
}

inline tir::PrimExpr AsPrimExpr(const tir::Var& var) {
    return var;
}

inline tir::PrimExpr AsPrimExpr(const IterVar& iter_var) {
    return iter_var->var;
}

/*! \brief 创建归约轴，供 sum/max 等归约表达式使用。 */
IterVar reduce_axis(tir::PrimExpr min, tir::PrimExpr extent, std::string name = "rv");

/*! \brief 单个 operation 的调度状态，保存 leaf/all iter vars。 */
class StageNode : public Object {
public:
    Operation op;
    Array<IterVar> leaf_iter_vars;
    Array<IterVar> all_iter_vars;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(StageNode)

/*! \brief 单个 operation 的调度接口，提供 split/fuse/reorder/bind 等 primitive。 */
class Stage : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit Stage(Operation op);

    const StageNode* operator->() const { return static_cast<const StageNode*>(object_); }
    StageNode* operator->() { return static_cast<StageNode*>(const_cast<Object*>(object_)); }

    /*! \brief 将 parent 轴按 factor 拆成 outer/inner 两个轴。 */
    IterVar split(IterVar parent, tir::PrimExpr factor, IterVar* p_outer = nullptr,
                  IterVar* p_inner = nullptr);
    /*! \brief 将相邻两个轴合并为一个轴。 */
    IterVar fuse(IterVar outer, IterVar inner);
    /*! \brief 重排当前 stage 的 leaf iter vars。 */
    void reorder(const Array<IterVar>& order);
    /*! \brief 对两个空间轴执行二维 tile，返回 outer/inner 组合。 */
    void tile(IterVar x_parent, IterVar y_parent, tir::PrimExpr x_factor,
              tir::PrimExpr y_factor, IterVar* x_outer, IterVar* y_outer,
              IterVar* x_inner, IterVar* y_inner);
    /*! \brief 标记轴为向量化执行。 */
    void vectorize(IterVar var);
    /*! \brief 标记轴为展开执行。 */
    void unroll(IterVar var);
    /*! \brief 标记轴为并行执行。 */
    void parallel(IterVar var);
    /*! \brief 将轴绑定到线程轴或 block/thread 语义。 */
    void bind(IterVar var, IterVar thread_axis);
};

/*! \brief TE 支持的归约类型。 */
enum class ReduceType : int {
    kSum = 0,
    kMax = 1,
    kMin = 2,
};

/*! \brief TE 归约表达式节点，保存归约轴、输入表达式和归约类型。 */
class ReduceNode : public tir::PrimExprNode {
public:
    Array<IterVar> axis;
    Array<tir::PrimExpr> source;
    ReduceType reduce_type = ReduceType::kSum;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ReduceNode)

/*! \brief TE 归约表达式引用类型。 */
class Reduce : public tir::PrimExpr {
public:
    using PrimExpr::PrimExpr;
    Reduce(Array<IterVar> axis, Array<tir::PrimExpr> source,
           ReduceType type = ReduceType::kSum);
};

/*! \brief 构造求和归约表达式。 */
tir::PrimExpr sum(tir::PrimExpr expr, Array<IterVar> axis);
/*! \brief 构造最大值归约表达式。 */
tir::PrimExpr max(tir::PrimExpr expr, Array<IterVar> axis);
/*! \brief 构造最小值归约表达式。 */
tir::PrimExpr min(tir::PrimExpr expr, Array<IterVar> axis);

/*! \brief placeholder operation，表示 TE 图的外部输入。 */
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

/*! \brief placeholder operation 引用类型。 */
class PlaceholderOp : public Operation {
public:
    using Operation::Operation;
    PlaceholderOp(std::string name, Array<tir::PrimExpr> shape, tir::DataType dtype);
};

/*! \brief compute operation，保存计算轴、归约轴和 body 表达式。 */
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

/*! \brief compute operation 引用类型。 */
class ComputeOp : public Operation {
public:
    using Operation::Operation;
    ComputeOp(std::string name, std::string tag, Map<String, ObjectRef> attrs,
              Array<tir::Var> axis, Array<tir::PrimExpr> body,
              Array<tir::PrimExpr> shape);
};

/*! \brief 创建一个输入 tensor。 */
Tensor placeholder(Array<tir::PrimExpr> shape,
                   tir::DataType dtype = tir::DataType::Float(32),
                   std::string name = "placeholder");

using FCompute = std::function<tir::PrimExpr(const Array<tir::Var>&)>;

/*! \brief 创建一个 compute tensor，fcompute 以符号轴构造输出表达式。 */
Tensor compute(Array<tir::PrimExpr> shape, FCompute fcompute,
               std::string name = "compute", std::string tag = "",
               Map<String, ObjectRef> attrs = {});

/*! \brief 创建线程轴，用于 bind 调度 primitive。 */
IterVar thread_axis(tir::PrimExpr dom, std::string tag);

/*! \brief TE schedule 节点，保存输出 operation 到 stage 的映射。 */
class ScheduleNode : public Object {
public:
    Array<Operation> outputs;
    Array<Stage> stages;
    Map<Operation, Stage> op_map;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ScheduleNode)

/*! \brief TE schedule 引用类型，可按 operation 取对应 stage。 */
class Schedule : public ObjectRef {
public:
    using ObjectRef::ObjectRef;

    Stage operator[](const Operation& op) { return operator->()->op_map.at(op); }

    const ScheduleNode* operator->() const { return static_cast<const ScheduleNode*>(object_); }
    ScheduleNode* operator->() { return static_cast<ScheduleNode*>(const_cast<Object*>(object_)); }
};

/*! \brief 从输出 operations 创建 schedule，并收集依赖 stage。 */
Schedule create_schedule(const Array<Operation>& ops);

}  // namespace te
}  // namespace kxc
