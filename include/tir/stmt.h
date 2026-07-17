/*! \file include/tir/stmt.h
 * \brief 定义 TIR PrimExpr、Stmt、PrimFunc 和 pass 工具。
 */

#pragma once

#include <string>
#include <vector>

#include "base/container.h"
#include "base/expr.h"
#include "tir/expr.h"

namespace kxc {
namespace tir {

/*! \brief 所有 TIR statement 节点的基类。 */
class StmtNode : public Object {
public:
    KXC_OBJECT_DECLARE
    virtual void VisitAttrs(AttrVisitor& visitor) { (void)visitor; }
};
KXC_OBJECT_DEFINE(StmtNode)

/*! \brief TIR statement 的引用类型。 */
class Stmt : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    const StmtNode* operator->() const { return static_cast<const StmtNode*>(object_); }
};

/*! \brief 在语句作用域内绑定局部变量值的 let 语句节点。 */
class LetStmtNode : public StmtNode {
public:
    Var var;
    PrimExpr value;
    Stmt body;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(LetStmtNode)

/*! \brief LetStmt 引用类型。 */
class LetStmt : public Stmt {
public:
    using Stmt::Stmt;
    LetStmt(Var var, PrimExpr value, Stmt body);
};

/*! \brief 向 buffer 指针按下标写入的语句节点。 */
class StoreNode : public StmtNode {
public:
    Var buffer_var;
    PrimExpr value;
    PrimExpr index;
    PrimExpr predicate;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(StoreNode)

/*! \brief Store 语句引用类型。 */
class Store : public Stmt {
public:
    using Stmt::Stmt;
    Store(Var buffer_var, PrimExpr value, PrimExpr index, PrimExpr predicate = PrimExpr());
};

/*! \brief TIR for 循环的执行/调度类型。 */
enum class ForType {
    Serial = 0,
    Parallel = 1,
    Vectorized = 2,
    Unrolled = 3,
};

/*! \brief TIR for 循环节点。 */
class ForNode : public StmtNode {
public:
    Var loop_var;
    PrimExpr min;
    PrimExpr extent;
    ForType for_type;
    Stmt body;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ForNode)

/*! \brief For 语句引用类型。 */
class For : public Stmt {
public:
    using Stmt::Stmt;
    For(Var loop_var, PrimExpr min, PrimExpr extent, ForType for_type, Stmt body);
};

/*! \brief 条件分支语句节点。 */
class IfThenElseNode : public StmtNode {
public:
    PrimExpr condition;
    Stmt then_case;
    Stmt else_case;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(IfThenElseNode)

/*! \brief IfThenElse 语句引用类型。 */
class IfThenElse : public Stmt {
public:
    using Stmt::Stmt;
    IfThenElse(PrimExpr condition, Stmt then_case, Stmt else_case = Stmt());
};

/*! \brief 在 TIR 中声明临时 buffer 分配的语句节点。 */
class AllocateNode : public StmtNode {
public:
    Var buffer_var;
    DataType dtype;
    Array<PrimExpr> extents;
    PrimExpr condition;
    Stmt body;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(AllocateNode)

/*! \brief Allocate 语句引用类型。 */
class Allocate : public Stmt {
public:
    using Stmt::Stmt;
    Allocate(Var buffer_var, DataType dtype, Array<PrimExpr> extents, PrimExpr condition,
             Stmt body);
};

/*! \brief 给节点附加编译/调度属性的语句节点。 */
class AttrStmtNode : public StmtNode {
public:
    ObjectRef node;
    std::string attr_key;
    PrimExpr value;
    Stmt body;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(AttrStmtNode)

/*! \brief AttrStmt 语句引用类型。 */
class AttrStmt : public Stmt {
public:
    using Stmt::Stmt;
    AttrStmt(ObjectRef node, std::string attr_key, PrimExpr value, Stmt body);
};

class Range;
class IterVar;
class Buffer;
class BufferRegion;

/*! \brief 半开区间 [min, min + extent)，用于迭代域和 buffer region。 */
class RangeNode : public Object {
public:
    PrimExpr min;
    PrimExpr extent;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(RangeNode)

/*! \brief Range 引用类型。 */
class Range : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    Range(PrimExpr min, PrimExpr extent);
    const RangeNode* operator->() const { return static_cast<const RangeNode*>(object_); }
};

/*! \brief TIR block iter var 的迭代语义。 */
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

/*! \brief TIR block 迭代变量节点。 */
class IterVarNode : public Object {
public:
    Range dom;
    Var var;
    IterVarType iter_type;
    std::string thread_tag;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE_WITH_KEY(IterVarNode, "kxc.tir.IterVarNode")

/*! \brief TIR block 迭代变量引用类型。 */
class IterVar : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    IterVar(Range dom, Var var, IterVarType iter_type, std::string thread_tag = "");
    const IterVarNode* operator->() const { return static_cast<const IterVarNode*>(object_); }
};

/*! \brief TIR buffer 描述，包含数据指针、shape、stride 和对齐信息。 */
class BufferNode : public Object {
public:
    Var data;
    DataType dtype;
    Array<PrimExpr> shape;
    Array<PrimExpr> strides;
    PrimExpr elem_offset;
    std::string name;
    int data_alignment;
    int offset_factor;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(BufferNode)

/*! \brief Buffer 引用类型。 */
class Buffer : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    Buffer(Var data, DataType dtype, Array<PrimExpr> shape, Array<PrimExpr> strides,
           PrimExpr elem_offset, std::string name, int data_alignment, int offset_factor);
    const BufferNode* operator->() const { return static_cast<const BufferNode*>(object_); }
};

/*! \brief Buffer 的一组访问区间，用于 block reads/writes 分析。 */
class BufferRegionNode : public Object {
public:
    Buffer buffer;
    Array<Range> region;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(BufferRegionNode)

/*! \brief BufferRegion 引用类型。 */
class BufferRegion : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    BufferRegion(Buffer buffer, Array<Range> region);
    const BufferRegionNode* operator->() const { return static_cast<const BufferRegionNode*>(object_); }
};

/*! \brief TIR block 节点，描述局部计算块、读写区域和可选 init。 */
class BlockNode : public StmtNode {
public:
    Array<IterVar> iter_vars;
    Array<BufferRegion> reads;
    Array<BufferRegion> writes;
    std::string name_hint;
    Stmt body;
    Stmt init;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(BlockNode)

/*! \brief Block 语句引用类型。 */
class Block : public Stmt {
public:
    using Stmt::Stmt;
    Block(Array<IterVar> iter_vars, Array<BufferRegion> reads, Array<BufferRegion> writes,
          std::string name_hint, Stmt body, Stmt init = Stmt());
};

/*! \brief 顺序执行的语句列表节点。 */
class SeqStmtNode : public StmtNode {
public:
    Array<Stmt> seq;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(SeqStmtNode)

/*! \brief SeqStmt 语句引用类型。 */
class SeqStmt : public Stmt {
public:
    using Stmt::Stmt;
    explicit SeqStmt(Array<Stmt> seq);
};

/*! \brief 仅求值表达式的语句节点，常用于调用副作用函数。 */
class EvaluateNode : public StmtNode {
public:
    PrimExpr value;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(EvaluateNode)

/*! \brief Evaluate 语句引用类型。 */
class Evaluate : public Stmt {
public:
    using Stmt::Stmt;
    explicit Evaluate(PrimExpr value);
};

/*! \brief TIR primitive function 节点，是 codegen 的主要输入单位。 */
class PrimFuncNode : public Object {
public:
    Array<Var> params;
    Stmt body;
    Map<Var, Buffer> buffer_map;
    Map<String, ObjectRef> attrs;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(PrimFuncNode)

/*! \brief PrimFunc 引用类型，包含参数、函数体、buffer map 和 attrs。 */
class PrimFunc : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    PrimFunc(Array<Var> params, Stmt body, Map<Var, Buffer> buffer_map = {},
             Map<String, ObjectRef> attrs = {});
    const PrimFuncNode* operator->() const { return static_cast<const PrimFuncNode*>(object_); }
};

}  // namespace tir
}  // namespace kxc
