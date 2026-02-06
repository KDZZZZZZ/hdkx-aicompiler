#pragma once
#include "tir/expr.h"
#include "base/object.h"
#include "base/container.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <algorithm>

namespace kxc {
namespace te {

// Forward declarations
class Schedule;
class Stage;
class Operation;
class Tensor;

// --- Operation ---
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

} // namespace te
} // namespace kxc

namespace std {
    template<> struct hash<kxc::te::Operation> {
        size_t operator()(const kxc::te::Operation& k) const {
            return std::hash<const kxc::Object*>()(k.get());
        }
    };
}

namespace kxc {
namespace te {


// --- Tensor ---
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
    
    // Operator() for indexing - Defined later
    tir::PrimExpr operator()(const Array<tir::PrimExpr>& indices) const;
    
    // Overload for vector<Var>
    tir::PrimExpr operator()(const Array<tir::Var>& indices) const;

    template<typename... Args>
    tir::PrimExpr operator()(Args... args) const {
        return (*this)(Array<tir::PrimExpr>{tir::PrimExpr(args)...});
    }
};

// --- ProducerLoad (Expression for Tensor Access) ---
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
    ProducerLoad(Tensor tensor, Array<tir::PrimExpr> indices) {
        auto* node = new ProducerLoadNode();
        node->tensor = tensor;
        node->indices = indices;
        node->dtype = tensor->dtype;
        SetData(node);
    }
};

// --- Tensor Implementation ---
inline Tensor::Tensor(Array<tir::PrimExpr> shape, tir::DataType dtype, Operation op, int value_index) {
    auto* node = new TensorNode();
    node->shape = shape;
    node->dtype = dtype;
    node->op = op;
    node->value_index = value_index;
    if (op.defined()) {
        node->name = op->name;
        if (op->num_outputs() > 1) {
            node->name += ".v" + std::to_string(value_index);
        }
    }
    SetData(node);
}

inline tir::PrimExpr Tensor::operator()(const Array<tir::PrimExpr>& indices) const {
    return ProducerLoad(*this, indices);
}

inline tir::PrimExpr Tensor::operator()(const Array<tir::Var>& indices) const {
    Array<tir::PrimExpr> prim_indices;
    // prim_indices.reserve(indices.size()); // Array doesn't support reserve yet
    for(const auto& v : indices) {
        prim_indices.push_back(v);
    }
    return ProducerLoad(*this, prim_indices);
}

// --- IterVar (Iteration Variable for Reduction) ---

enum class IterVarType : int {
    kDataPar = 0,
    kThreadIndex = 1,
    kCommReduce = 2,
    kOrdered = 3,
    kOpaque = 4,
    kVectorized = 5,
    kParallel = 6,
    kUnrolled = 7
};

class IterVarNode : public Object {
public:
    tir::Var var;
    tir::PrimExpr dom_min;
    tir::PrimExpr dom_extent;
    IterVarType iter_type;
    std::string thread_tag; // For thread binding
    
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(IterVarNode)

class IterVar : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit IterVar(tir::PrimExpr min, tir::PrimExpr extent, IterVarType type = IterVarType::kDataPar, std::string thread_tag = "", std::string name = "rv") {
        auto* node = new IterVarNode();
        node->var = tir::Var(name);
        node->dom_min = min;
        node->dom_extent = extent;
        node->iter_type = type;
        node->thread_tag = thread_tag;
        SetData(node);
    }
    const IterVarNode* operator->() const { return static_cast<const IterVarNode*>(object_); }
    operator tir::PrimExpr() const { return operator->()->var; }
    operator tir::Var() const { return operator->()->var; }
    
    // operator== for std::find
    bool operator==(const IterVar& other) const {
        return object_ == other.object_;
    }
    
    // Define !=
    bool operator!=(const IterVar& other) const {
        return !(*this == other);
    }
};

inline IterVar reduce_axis(tir::PrimExpr min, tir::PrimExpr extent, std::string name = "rv") {
    return IterVar(min, extent, IterVarType::kCommReduce, "", name);
}

// --- Schedule Primitives ---

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
    
    // Schedule Primitives
    IterVar split(IterVar parent, tir::PrimExpr factor, IterVar* p_outer = nullptr, IterVar* p_inner = nullptr);
    IterVar fuse(IterVar outer, IterVar inner);
    void reorder(const Array<IterVar>& order);
    void tile(IterVar x_parent, IterVar y_parent, tir::PrimExpr x_factor, tir::PrimExpr y_factor, 
              IterVar* x_outer, IterVar* y_outer, IterVar* x_inner, IterVar* y_inner);
    void vectorize(IterVar var);
    void unroll(IterVar var);
    void parallel(IterVar var);
    void bind(IterVar var, IterVar thread_axis);
};

// --- Reduce (Expression) ---
enum class ReduceType : int {
    kSum = 0,
    kMax = 1,
    kMin = 2
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
    Reduce(Array<IterVar> axis, Array<tir::PrimExpr> source, ReduceType type = ReduceType::kSum) {
        auto* node = new ReduceNode();
        node->axis = axis;
        node->source = source;
        node->reduce_type = type;
        if(!source.empty()) node->dtype = source[0].dtype();
        SetData(node);
    }
};

inline tir::PrimExpr sum(tir::PrimExpr expr, Array<IterVar> axis) {
    return Reduce(axis, {expr}, ReduceType::kSum);
}

inline tir::PrimExpr max(tir::PrimExpr expr, Array<IterVar> axis) {
    return Reduce(axis, {expr}, ReduceType::kMax);
}

// --- Concrete Operations ---

// PlaceholderOp
class PlaceholderOpNode : public OperationNode {
public:
    Array<tir::PrimExpr> shape;
    tir::DataType dtype;
    
    int num_outputs() const override { return 1; }
    tir::DataType output_dtype(int i) const override { return dtype; }
    Array<tir::PrimExpr> output_shape(int i) const override { return shape; }
    
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(PlaceholderOpNode)

class PlaceholderOp : public Operation {
public:
    using Operation::Operation;
    PlaceholderOp(std::string name, Array<tir::PrimExpr> shape, tir::DataType dtype) {
        auto* node = new PlaceholderOpNode();
        node->name = name;
        node->shape = shape;
        node->dtype = dtype;
        SetData(node);
    }
};

// ComputeOp
class ComputeOpNode : public OperationNode {
public:
    Array<tir::Var> axis;
    Array<IterVar> reduce_axis; // Added reduce axis
    Array<tir::PrimExpr> body; 
    
    int num_outputs() const override { return body.size(); }
    tir::DataType output_dtype(int i) const override { return body[i].dtype(); }
    Array<tir::PrimExpr> shape;

    Array<tir::PrimExpr> output_shape(int i) const override { return shape; }
    
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ComputeOpNode)

class ComputeOp : public Operation {
public:
    using Operation::Operation;
    ComputeOp(std::string name, std::string tag, Map<String, ObjectRef> attrs, 
              Array<tir::Var> axis, Array<tir::PrimExpr> body, Array<tir::PrimExpr> shape) {
        auto* node = new ComputeOpNode();
        node->name = name;
        node->tag = tag;
        node->attrs = attrs;
        node->axis = axis;
        node->body = body;
        node->shape = shape;
        
        // Extract reduce_axis from body
        // Visitor to find Reduce nodes in body
        class ReduceAxisVisitor : public AttrVisitor { // Simplified, we don't have full IR visitor yet
        public:
            std::vector<IterVar> axes;
            void Visit(const ObjectRef& obj) {
                if (auto* reduce = obj.As<ReduceNode>()) {
                     for (const auto& ax : reduce->axis) {
                        axes.push_back(ax);
                     }
                }
                // Recursive visit would be needed for real IR
                // Here we assume simple structure or manual check
            }
        };

        // Manual recursive check for simple expressions (BinaryOp, Call, Reduce)
        std::function<void(const tir::PrimExpr&)> find_reduce = [&](const tir::PrimExpr& expr) {
             if (auto* reduce = expr.As<ReduceNode>()) {
                 for(auto& ax : reduce->axis) {
                     // Check duplicates
                     bool exists = false;
                     for(auto& exist_ax : node->reduce_axis) if(exist_ax == ax) exists = true;
                     if(!exists) node->reduce_axis.push_back(ax);
                 }
                 // Visit source
                 for(auto& src : reduce->source) find_reduce(src);
             } else if (auto* bin = expr.As<tir::BinaryOpNode>()) {
                 find_reduce(bin->a);
                 find_reduce(bin->b);
             } else if (auto* call = expr.As<tir::CallNode>()) {
                 for(auto& arg : call->args) find_reduce(arg);
             } else if (auto* sel = expr.As<tir::SelectNode>()) {
                 find_reduce(sel->condition);
                 find_reduce(sel->true_value);
                 find_reduce(sel->false_value);
             } else if (auto* not_node = expr.As<tir::NotNode>()) {
                 find_reduce(not_node->value);
             }
             /*
             } else if (auto* cast = expr.As<tir::CastNode>()) {
                 find_reduce(cast->value);
             } 
             */
             else if (auto* load = expr.As<ProducerLoadNode>()) {
                 for (auto& idx : load->indices) find_reduce(idx);
             }
        };

        for(auto& expr : body) {
            find_reduce(expr);
        }
        
        SetData(node);
    }
};

// --- Helper Functions ---

inline Tensor placeholder(Array<tir::PrimExpr> shape, tir::DataType dtype = tir::DataType::Float(32), std::string name = "placeholder") {
    PlaceholderOp op(name, shape, dtype);
    return Tensor(shape, dtype, op, 0);
}

// FCompute: std::function<PrimExpr(const std::vector<Var>&)>
// or FCompute: std::function<PrimExpr(Var, Var...)>
// We'll support varargs via a helper later, for now vector version
using FCompute = std::function<tir::PrimExpr(const Array<tir::Var>&)>;

inline Tensor compute(Array<tir::PrimExpr> shape, FCompute fcompute, std::string name = "compute", std::string tag = "", Map<String, ObjectRef> attrs = {}) {
    // 1. Create IterVars for the shape
    Array<tir::Var> axis;
    for (size_t i = 0; i < shape.size(); ++i) {
        axis.push_back(tir::Var("ax" + std::to_string(i)));
    }
    
    // 2. Compute body
    tir::PrimExpr body = fcompute(axis);
    
    // 3. Create Op
    ComputeOp op(name, tag, attrs, axis, {body}, shape);
    
    // 4. Return Tensor
    return Tensor(shape, body.dtype(), op, 0);
}

// --- Schedule (Implementation) ---

inline Stage::Stage(Operation op) {
    auto* node = new StageNode();
    node->op = op;
    
    // Initialize leaf_iter_vars from Operation
    // For ComputeOp, we have axis and reduce_axis.
    // We need to convert Var to IterVar or assume ComputeOp uses IterVars (which it doesn't currently)
    // Or we create IterVars from Vars.
    // For now, let's look at ComputeOpNode.
    
    if (auto* compute_op = op.As<ComputeOpNode>()) {
        for (const auto& var : compute_op->axis) {
            // Assume extent is in shape? Or we need proper domain inference.
            // Simplified: create IterVar with var name. Extent unknown or from shape.
            // We'll use 0 to -1 for now if unknown, or match shape.
            // Matching shape is hard without index.
            // Let's assume ComputeOp stores full IterVars in future.
            // For now, construct new IterVar wrapper around the Var.
            IterVar iv(0, 0, IterVarType::kDataPar, "", var->name_hint);
            node->leaf_iter_vars.push_back(iv);
            node->all_iter_vars.push_back(iv);
        }
        for (const auto& iv : compute_op->reduce_axis) {
            node->leaf_iter_vars.push_back(iv);
            node->all_iter_vars.push_back(iv);
        }
    }
    
    SetData(node);
}

inline IterVar Stage::split(IterVar parent, tir::PrimExpr factor, IterVar* p_outer, IterVar* p_inner) {
    // 1. Create new IterVars
    IterVar outer(0, 0, IterVarType::kDataPar, "", parent->var->name_hint + ".outer");
    IterVar inner(0, factor, IterVarType::kDataPar, "", parent->var->name_hint + ".inner");
    
    // 2. Update leaf_iter_vars
    auto* node = this->operator->();
    // std::find requires operator==, we implemented it for IterVar.
    // However, if leaf_iter_vars has been modified (e.g. by reorder), finding might fail if we don't have the exact same object reference?
    // We used ObjectRef equality (pointer equality), which is correct for same object.
    
    auto it = std::find(node->leaf_iter_vars.begin(), node->leaf_iter_vars.end(), parent);
    if (it != node->leaf_iter_vars.end()) {
        // Insert inner first, then replace current with outer
        // Wait, insert invalidates iterator?
        // Yes, insert before `it` invalidates `it` if reallocation happens.
        // Index based is safer.
        size_t idx = std::distance(node->leaf_iter_vars.begin(), it);
        node->leaf_iter_vars[idx] = outer;
        node->leaf_iter_vars.insert(node->leaf_iter_vars.begin() + idx + 1, inner);
    } else {
        // Error: parent not found in leaves
        // For debug, print or assert
        // std::cerr << "Warning: Split parent not found in leaves!" << std::endl;
    }
    
    // 3. Track
    node->all_iter_vars.push_back(outer);
    node->all_iter_vars.push_back(inner);
    
    if(p_outer) *p_outer = outer;
    if(p_inner) *p_inner = inner;
    
    return outer; // Return outer for convenience
}

inline IterVar Stage::fuse(IterVar outer, IterVar inner) {
    IterVar fused(0, 0, IterVarType::kDataPar, "", outer->var->name_hint + "." + inner->var->name_hint + ".fused");
    
    auto* node = this->operator->();
    auto it_outer = std::find(node->leaf_iter_vars.begin(), node->leaf_iter_vars.end(), outer);
    auto it_inner = std::find(node->leaf_iter_vars.begin(), node->leaf_iter_vars.end(), inner);
    
    if (it_outer != node->leaf_iter_vars.end() && it_inner != node->leaf_iter_vars.end()) {
        // Remove inner
        node->leaf_iter_vars.erase(it_inner);
        // Replace outer with fused
        // Re-find outer because erase might invalidate iterators if vector reallocates (though usually safe for erase later)
        it_outer = std::find(node->leaf_iter_vars.begin(), node->leaf_iter_vars.end(), outer);
        *it_outer = fused;
    }
    
    node->all_iter_vars.push_back(fused);
    return fused;
}

inline void Stage::reorder(const Array<IterVar>& order) {
    auto* node = this->operator->();
    // Validate all vars in order exist in leaves
    // Then replace leaves with order
    // Simplified: Just set leaves = order
    node->leaf_iter_vars = order;
}

inline void Stage::tile(IterVar x_parent, IterVar y_parent, tir::PrimExpr x_factor, tir::PrimExpr y_factor, 
          IterVar* x_outer, IterVar* y_outer, IterVar* x_inner, IterVar* y_inner) {
    split(x_parent, x_factor, x_outer, x_inner);
    split(y_parent, y_factor, y_outer, y_inner);
    
    auto* node = this->operator->();
    // Capture current leaves before reorder overwrites them
    Array<IterVar> current_leaves = node->leaf_iter_vars;
    
    reorder({*x_outer, *y_outer, *x_inner, *y_inner});
    
    // Note: This reorder implementation completely overwrites leaf_iter_vars.
    // If there were other axes (like reduction axis k), they are lost!
    // Correct reorder should be selective or user must provide full list.
    // For tile, we usually want to keep other axes.
    // Let's improve reorder or just append missing axes?
    // In TVM reorder takes variable args.
    // Here we take a vector.
    // If the vector is partial, we should probably keep others?
    // But usually reorder specifies the full loop nest order.
    // If we lose 'k', verify fails.
    
    // Let's try to recover missing axes for this simple implementation.
    Array<IterVar> new_leaves = node->leaf_iter_vars; // Should be just the reordered ones
    
    for(auto& iv : current_leaves) {
        // Skip parent axes that were split
        if (iv == x_parent || iv == y_parent) continue;

        // If iv is not in new_leaves, append it?
        // The current leaves contain x_outer, x_inner, y_outer, y_inner, and k.
        // We want [xo, yo, xi, yi, k] (or similar tiling order).
        // Let's find any leaf not in the new order and append it.
        bool found = false;
        for(auto& new_iv : new_leaves) {
            if(iv == new_iv) {
                found = true;
                break;
            }
        }
        if(!found) {
            new_leaves.push_back(iv);
        }
    }
    node->leaf_iter_vars = new_leaves;
}

inline void Stage::vectorize(IterVar var) {
    // In-place modification of IterVar type? 
    // Or create new IterVar? 
    // IterVar is ObjectRef, modifying the Node affects all references.
    // IterVarNode has iter_type.
    // We need mutable access to node.
    // const_cast hack or make accessors.
    const_cast<IterVarNode*>(var.operator->())->iter_type = IterVarType::kVectorized;
}

inline void Stage::unroll(IterVar var) {
    const_cast<IterVarNode*>(var.operator->())->iter_type = IterVarType::kUnrolled;
}

inline void Stage::parallel(IterVar var) {
    const_cast<IterVarNode*>(var.operator->())->iter_type = IterVarType::kParallel;
}

inline void Stage::bind(IterVar var, IterVar thread_axis) {
     const_cast<IterVarNode*>(var.operator->())->iter_type = IterVarType::kThreadIndex;
     const_cast<IterVarNode*>(var.operator->())->thread_tag = thread_axis->thread_tag;
}

inline IterVar thread_axis(tir::PrimExpr dom, std::string tag) {
    return IterVar(0, dom, IterVarType::kThreadIndex, tag, tag);
}
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
    
    Stage operator[](const Operation& op) {
        return operator->()->op_map.at(op);
    }
    
    const ScheduleNode* operator->() const { return static_cast<const ScheduleNode*>(object_); }
    ScheduleNode* operator->() { return static_cast<ScheduleNode*>(const_cast<Object*>(object_)); }
};

inline Schedule create_schedule(const Array<Operation>& ops) {
    auto* node = new ScheduleNode();
    node->outputs = ops;
    for(auto& op : ops) {
        Stage stage(op);
        node->stages.push_back(stage);
        node->op_map.Set(op, stage);
    }
    return Schedule(node);
}

} // namespace te
} // namespace kxc

