/*! \file include/kxc/te/te.h
 * \brief 定义 TE tensor、operation、reduce 和 schedule 表达层。
 */

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "kxc/support/container.h"
#include "kxc/support/object.h"
#include "kxc/tir/expr.h"

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

/*! \brief ProducerLoad 的引用类型。 */
class ProducerLoad : public tir::PrimExpr {
public:
    using PrimExpr::PrimExpr;
    ProducerLoad(Tensor tensor, Array<tir::PrimExpr> indices);
};

/*! \brief TE leaf 轴的来源或执行语义。 */
enum class IterVarType : int {
    kDataPar = 0,
    kCommReduce = 2,
    kVectorized = 5,
    kParallel = 6,
    kUnrolled = 7,
};

/*! \brief TE 迭代轴，保留静态域、归约来源和 lowering 可消费的执行语义。 */
class IterVarNode : public Object {
public:
    tir::Var var;
    tir::PrimExpr dom_min;
    tir::PrimExpr dom_extent;
    IterVarType iter_type{IterVarType::kDataPar};
    bool is_reduction{false};

    KXC_OBJECT_DECLARE
};

/*! \brief TE 迭代轴引用类型，可隐式转换为底层 TIR Var/PrimExpr。 */
class IterVar : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit IterVar(tir::PrimExpr min, tir::PrimExpr extent,
                     IterVarType type = IterVarType::kDataPar,
                     std::string name = "rv");
    const IterVarNode* operator->() const { return static_cast<const IterVarNode*>(object_); }
    operator tir::PrimExpr() const { return operator->()->var; }
    operator tir::Var() const { return operator->()->var; }

    bool operator==(const IterVar& other) const { return object_ == other.object_; }
    bool operator!=(const IterVar& other) const { return !(*this == other); }
};

/*! \brief 一次 exact-or-predicated split 的父子关系。 */
struct SplitRelation final {
    IterVar parent;
    IterVar outer;
    IterVar inner;
    tir::PrimExpr factor;
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

/*! \brief 单个 operation 的调度状态，保存 root/leaf 轴和 split 关系。 */
class StageNode : public Object {
public:
    Operation op;
    Array<IterVar> root_iter_vars;
    Array<IterVar> leaf_iter_vars;
    Array<IterVar> all_iter_vars;
    std::vector<SplitRelation> split_relations;

    KXC_OBJECT_DECLARE
};

/*! \brief 单个 operation 的最小可消费调度接口。 */
class Stage : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit Stage(Operation op);

    const StageNode* operator->() const { return static_cast<const StageNode*>(object_); }
    StageNode* operator->() { return static_cast<StageNode*>(const_cast<Object*>(object_)); }

    /*! \brief 将当前 leaf parent 按正静态 factor 拆成 outer/inner。 */
    IterVar split(IterVar parent, tir::PrimExpr factor, IterVar* p_outer = nullptr,
                  IterVar* p_inner = nullptr);
    /*! \brief 以当前 leaf 的无重复全排列重排循环。 */
    void reorder(const Array<IterVar>& order);
    /*! \brief 标记最内层数据并行 leaf 为向量化执行。 */
    void vectorize(IterVar var);
    /*! \brief 标记 data/reduction leaf 为展开执行。 */
    void unroll(IterVar var);
    /*! \brief 标记数据并行 leaf 为并行执行。 */
    void parallel(IterVar var);
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

/*! \brief TE schedule 节点，保存输出 operation 到完整 producer DAG stage 的映射。 */
class ScheduleNode : public Object {
public:
    Array<Operation> outputs;
    Array<Stage> stages;
    Map<Operation, Stage> op_map;
    std::string policy{"manual-v1"};

    KXC_OBJECT_DECLARE
};

/*! \brief TE schedule 引用类型，可按 operation 取对应 stage。 */
class Schedule : public ObjectRef {
public:
    using ObjectRef::ObjectRef;

    Stage operator[](const Operation& op) { return operator->()->op_map.at(op); }
    Stage operator[](const Operation& op) const { return operator->()->op_map.at(op); }

    const ScheduleNode* operator->() const { return static_cast<const ScheduleNode*>(object_); }
    ScheduleNode* operator->() { return static_cast<ScheduleNode*>(const_cast<Object*>(object_)); }
};

/*! \brief 从输出 operations 创建 schedule，并按 producer-first 顺序收集完整依赖 DAG。 */
Schedule create_schedule(const Array<Operation>& ops);

}  // namespace te
}  // namespace kxc
