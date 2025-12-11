#pragma once
#include "tir/expr.h"
#include "base/object.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace kxc {
namespace te {

using namespace kxc::tir;

// Forward declarations
class Operation;
class Tensor;
class Schedule;
class Stage;

// --- Operation ---
class OperationNode : public Object {
public:
    std::string name;
    std::string tag;
    std::unordered_map<std::string, ObjectRef> attrs;
    
    virtual int num_outputs() const = 0;
    virtual DataType output_dtype(int i) const = 0;
    virtual std::vector<PrimExpr> output_shape(int i) const = 0;
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 50; }
};

class Operation : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    const OperationNode* operator->() const { return static_cast<const OperationNode*>(object_); }
    OperationNode* operator->() { return static_cast<OperationNode*>(const_cast<Object*>(object_)); }
    
    int num_outputs() const { return operator->()->num_outputs(); }
    DataType output_dtype(int i) const { return operator->()->output_dtype(i); }
    std::vector<PrimExpr> output_shape(int i) const { return operator->()->output_shape(i); }
};

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
    Var var;
    PrimExpr dom_min;
    PrimExpr dom_extent;
    IterVarType iter_type;
    std::string thread_tag; // For thread binding
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 57; }
};

class IterVar : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit IterVar(PrimExpr min, PrimExpr extent, IterVarType type = IterVarType::kDataPar, std::string thread_tag = "", std::string name = "rv") {
        auto* node = new IterVarNode();
        node->var = Var(name);
        node->dom_min = min;
        node->dom_extent = extent;
        node->iter_type = type;
        node->thread_tag = thread_tag;
        object_ = node;
        if(object_) object_->IncRef();
    }
    const IterVarNode* operator->() const { return static_cast<const IterVarNode*>(object_); }
    operator PrimExpr() const { return operator->()->var; }
    operator Var() const { return operator->()->var; }
    
    // operator== for std::find
    bool operator==(const IterVar& other) const {
        return object_ == other.object_;
    }
};

inline IterVar reduce_axis(PrimExpr min, PrimExpr extent, std::string name = "rv") {
    return IterVar(min, extent, IterVarType::kCommReduce, "", name);
}

// --- Schedule Primitives ---

class StageNode : public Object {
public:
    Operation op;
    std::vector<IterVar> leaf_iter_vars;
    std::vector<IterVar> all_iter_vars;
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 55; }
};

class Stage : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit Stage(Operation op);

    const StageNode* operator->() const { return static_cast<const StageNode*>(object_); }
    StageNode* operator->() { return static_cast<StageNode*>(const_cast<Object*>(object_)); }
    
    // Schedule Primitives
    IterVar split(IterVar parent, PrimExpr factor, IterVar* p_outer = nullptr, IterVar* p_inner = nullptr);
    IterVar fuse(IterVar outer, IterVar inner);
    void reorder(const std::vector<IterVar>& order);
    void tile(IterVar x_parent, IterVar y_parent, PrimExpr x_factor, PrimExpr y_factor, 
              IterVar* x_outer, IterVar* y_outer, IterVar* x_inner, IterVar* y_inner);
    void vectorize(IterVar var);
    void unroll(IterVar var);
    void parallel(IterVar var);
    void bind(IterVar var, IterVar thread_axis);
};

// --- Reduce (Expression) ---
class ReduceNode : public PrimExprNode {
public:
    std::vector<IterVar> axis;
    std::vector<PrimExpr> source;
    // Combiner combiner; // Simplified: Assume Sum
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 58; }
};

class Reduce : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    Reduce(std::vector<IterVar> axis, std::vector<PrimExpr> source) {
        auto* node = new ReduceNode();
        node->axis = axis;
        node->source = source;
        if(!source.empty()) node->dtype = source[0].dtype();
        object_ = node;
        if(object_) object_->IncRef();
    }
};

inline PrimExpr sum(PrimExpr expr, std::vector<IterVar> axis) {
    return Reduce(axis, {expr});
}

// --- Operation ---
// (Moved to top)

// --- Tensor ---

// --- Tensor ---
class TensorNode : public Object {
public:
    std::string name;
    std::vector<PrimExpr> shape;
    DataType dtype;
    Operation op;
    int value_index;
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 51; }
};

class Tensor : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    
    Tensor(std::vector<PrimExpr> shape, DataType dtype, Operation op, int value_index);
    
    const TensorNode* operator->() const { return static_cast<const TensorNode*>(object_); }
    
    // Operator() for indexing - Defined later
    PrimExpr operator()(const std::vector<PrimExpr>& indices) const;
    
    // Overload for vector<Var>
    PrimExpr operator()(const std::vector<Var>& indices) const;

    template<typename... Args>
    PrimExpr operator()(Args... args) const {
        return (*this)(std::vector<PrimExpr>{PrimExpr(args)...});
    }
};

// --- ProducerLoad (Expression for Tensor Access) ---
class ProducerLoadNode : public PrimExprNode {
public:
    Tensor tensor;
    std::vector<PrimExpr> indices;
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 52; }
};

class ProducerLoad : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    ProducerLoad(Tensor tensor, std::vector<PrimExpr> indices) {
        auto* node = new ProducerLoadNode();
        node->tensor = tensor;
        node->indices = indices;
        node->dtype = tensor->dtype;
        object_ = node;
        if (object_) object_->IncRef();
    }
};

// --- Tensor Implementation ---
inline Tensor::Tensor(std::vector<PrimExpr> shape, DataType dtype, Operation op, int value_index) {
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
    object_ = node;
    if (object_) object_->IncRef();
}

inline PrimExpr Tensor::operator()(const std::vector<PrimExpr>& indices) const {
    return ProducerLoad(*this, indices);
}

inline PrimExpr Tensor::operator()(const std::vector<Var>& indices) const {
    std::vector<PrimExpr> prim_indices;
    prim_indices.reserve(indices.size());
    for(const auto& v : indices) {
        prim_indices.push_back(v);
    }
    return ProducerLoad(*this, prim_indices);
}


// --- Concrete Operations ---

// PlaceholderOp
class PlaceholderOpNode : public OperationNode {
public:
    std::vector<PrimExpr> shape;
    DataType dtype;
    
    int num_outputs() const override { return 1; }
    DataType output_dtype(int i) const override { return dtype; }
    std::vector<PrimExpr> output_shape(int i) const override { return shape; }
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 53; }
};

class PlaceholderOp : public Operation {
public:
    using Operation::Operation;
    PlaceholderOp(std::string name, std::vector<PrimExpr> shape, DataType dtype) {
        auto* node = new PlaceholderOpNode();
        node->name = name;
        node->shape = shape;
        node->dtype = dtype;
        object_ = node;
        if (object_) object_->IncRef();
    }
};

// ComputeOp
class ComputeOpNode : public OperationNode {
public:
    std::vector<Var> axis;
    std::vector<IterVar> reduce_axis; // Added reduce axis
    std::vector<PrimExpr> body; 
    
    int num_outputs() const override { return body.size(); }
    DataType output_dtype(int i) const override { return body[i].dtype(); }
    // Shape is determined by iteration domain, usually assumed to be the bounding box of axes
    // For simplicity, we store shape explicitly or infer it? 
    // In TVM, ComputeOp stores the output shape logic. 
    // Here we'll simplify and store expected shape in the node for now.
    std::vector<PrimExpr> shape;

    std::vector<PrimExpr> output_shape(int i) const override { return shape; }
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 54; }
};

class ComputeOp : public Operation {
public:
    using Operation::Operation;
    ComputeOp(std::string name, std::string tag, std::unordered_map<std::string, ObjectRef> attrs, 
              std::vector<Var> axis, std::vector<PrimExpr> body, std::vector<PrimExpr> shape) {
        auto* node = new ComputeOpNode();
        node->name = name;
        node->tag = tag;
        node->attrs = attrs;
        node->axis = axis;
        node->body = body;
        node->shape = shape;
        
        // Extract reduce_axis from body if possible
        // This is a simplified analysis. In real TVM, this is done by traversing the body.
        // For our demo, we can try to find Reduce nodes in body.
        // Or we just assume the user provided body correctly.
        // We'll leave reduce_axis empty here or implement a visitor to find it.
        // Let's implement a simple visitor later if needed.
        // Quick hack: if body[0] is Reduce, extract axes.
        if (!body.empty()) {
            // Need to cast PrimExpr to Reduce. 
            // PrimExpr -> PrimExprNode. ReduceNode inherits PrimExprNode.
            // As<ReduceNode> won't work on PrimExpr directly unless PrimExpr::operator-> returns ReduceNode.
            // PrimExpr-> returns PrimExprNode.
            // But we can check type.
            const PrimExprNode* node_ptr = body[0].operator->();
            if (node_ptr->GetTypeId() == kKXC_OBJECT_TYPE + 58) { // ReduceNode
                  const ReduceNode* reduce = static_cast<const ReduceNode*>(node_ptr);
                  // this->reduce_axis = reduce->axis; // Error: ComputeOp wrapper has no reduce_axis member, ComputeOpNode does.
                  node->reduce_axis = reduce->axis;
             }
        }
        
        object_ = node;
        if (object_) object_->IncRef();
    }
};

// --- Helper Functions ---

inline Tensor placeholder(std::vector<PrimExpr> shape, DataType dtype = DataType::Float(32), std::string name = "placeholder") {
    PlaceholderOp op(name, shape, dtype);
    return Tensor(shape, dtype, op, 0);
}

// FCompute: std::function<PrimExpr(const std::vector<Var>&)>
// or FCompute: std::function<PrimExpr(Var, Var...)>
// We'll support varargs via a helper later, for now vector version
using FCompute = std::function<PrimExpr(const std::vector<Var>&)>;

inline Tensor compute(std::vector<PrimExpr> shape, FCompute fcompute, std::string name = "compute", std::string tag = "", std::unordered_map<std::string, ObjectRef> attrs = {}) {
    // 1. Create IterVars for the shape
    std::vector<Var> axis;
    for (size_t i = 0; i < shape.size(); ++i) {
        axis.push_back(Var("ax" + std::to_string(i)));
    }
    
    // 2. Compute body
    PrimExpr body = fcompute(axis);
    
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
    
    object_ = node;
    if(object_) object_->IncRef();
}

inline IterVar Stage::split(IterVar parent, PrimExpr factor, IterVar* p_outer, IterVar* p_inner) {
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

inline void Stage::reorder(const std::vector<IterVar>& order) {
    auto* node = this->operator->();
    // Validate all vars in order exist in leaves
    // Then replace leaves with order
    // Simplified: Just set leaves = order
    node->leaf_iter_vars = order;
}

inline void Stage::tile(IterVar x_parent, IterVar y_parent, PrimExpr x_factor, PrimExpr y_factor, 
          IterVar* x_outer, IterVar* y_outer, IterVar* x_inner, IterVar* y_inner) {
    split(x_parent, x_factor, x_outer, x_inner);
    split(y_parent, y_factor, y_outer, y_inner);
    
    auto* node = this->operator->();
    // Capture current leaves before reorder overwrites them
    std::vector<IterVar> current_leaves = node->leaf_iter_vars;
    
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
    std::vector<IterVar> new_leaves = node->leaf_iter_vars; // Should be just the reordered ones
    
    for(auto& iv : current_leaves) {
        // If iv is not in new_leaves, append it?
        // Wait, x_parent and y_parent are gone (replaced by split).
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

inline IterVar thread_axis(PrimExpr dom, std::string tag) {
    return IterVar(0, dom, IterVarType::kThreadIndex, tag, tag);
}
struct OpNodeHash {
    size_t operator()(const OperationNode* k) const {
        return std::hash<const OperationNode*>()(k);
    }
};

struct OpNodeEqual {
    bool operator()(const OperationNode* lhs, const OperationNode* rhs) const {
        return lhs == rhs;
    }
};

class ScheduleNode : public Object {
public:
    std::vector<Stage> stages;
    std::unordered_map<OperationNode*, Stage, OpNodeHash, OpNodeEqual> op_to_stage;
    std::vector<Operation> outputs;
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 56; }
    
    // Helper to get stage for op
    Stage operator[](Operation op) {
        if(op_to_stage.find(op.operator->()) != op_to_stage.end()) {
            return op_to_stage[op.operator->()];
        }
        // Fallback: Linear search in stages (if op_to_stage not fully populated)
        for(auto s : stages) {
            if(s->op.operator->() == op.operator->()) return s;
        }
        // Should not happen if constructed correctly
        return Stage(Operation());
    }
};

class Schedule : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit Schedule(std::vector<Operation> ops) {
        auto* node = new ScheduleNode();
        node->outputs = ops;
        
        // Simplified traversal: just add output ops as stages
        // Real impl does DFS traversal
        for(auto op : ops) {
            Stage stage(op);
            node->stages.push_back(stage);
            node->op_to_stage[op.operator->()] = stage;
        }
        
        object_ = node;
        if(object_) object_->IncRef();
    }
    const ScheduleNode* operator->() const { return static_cast<const ScheduleNode*>(object_); }
    ScheduleNode* operator->() { return static_cast<ScheduleNode*>(const_cast<Object*>(object_)); }
    
    Stage operator[](Operation op) {
        return operator->()->operator[](op);
    }
    
    Stage operator[](Tensor t) {
        return operator[](t->op);
    }
};

inline Schedule create_schedule(std::vector<Operation> ops) {
    return Schedule(ops);
}

} // namespace te
} // namespace kxc
