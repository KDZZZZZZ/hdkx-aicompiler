#include "relay/transforms/lower.h"
#include "relay/op_attr_types.h"
#include "relay/op.h"
#include "te/te.h"
#include "base/pass.h"
#include "tir/expr.h"
#include <unordered_map>
#include <iostream>

namespace kxc {
namespace relay {

using namespace kxc::relay;
// using namespace kxc::tir; // Removed to avoid ambiguity

class RelayToTEConverter : public RelayPassFunctor<std::vector<te::Tensor>> {
public:
    std::vector<te::Tensor> Convert(const Expr& expr) {
        return Visit(expr);
    }

protected:
    // Cache for visited nodes (Memoization)
    std::unordered_map<const Object*, std::vector<te::Tensor>> memo_;

    std::vector<te::Tensor> Visit(const Expr& expr) override {
        if (memo_.count(expr.get())) {
            return memo_.at(expr.get());
        }
        auto res = RelayPassFunctor::Visit(expr);
        memo_[expr.get()] = res;
        return res;
    }

    std::vector<te::Tensor> VisitVar(const VarNode* op, const Expr& ref) override {
        // Var -> Placeholder
        // We assume simple static shape [1, 1] if not provided, for testing purposes.
        std::vector<tir::PrimExpr> shape = {1, 1}; 
        tir::DataType dtype = tir::DataType::Float(32); 
        
        return {te::placeholder(shape, dtype, op->vid->name_hint)};
    }
    
    std::vector<te::Tensor> VisitConstant(const ConstantNode* op, const Expr& ref) override {
        std::vector<tir::PrimExpr> shape;
        for(auto d : op->data->shape) shape.push_back(d);
        tir::DataType dtype = tir::DataType::Float(32); 
        // Simplified mapping
        if (op->data->dtype == "float32") dtype = tir::DataType::Float(32);
        else if (op->data->dtype == "int32") dtype = tir::DataType::Int(32);
        
        return {te::placeholder(shape, dtype, "const")};
    }

    std::vector<te::Tensor> VisitCall(const CallNode* op, const Expr& ref) override {
        // 1. Visit Args
        std::vector<te::Tensor> inputs;
        for (const auto& arg : op->args) {
            auto res = Visit(arg);
            inputs.insert(inputs.end(), res.begin(), res.end());
        }

        // 2. Get Op and FTVMCompute
        if (auto* op_node = op->op.As<OpNode>()) {
             auto it = op_node->attrs.find("FTVMCompute");
             if (it != op_node->attrs.end()) {
                 if (auto* fcompute_ptr = std::any_cast<FTVMCompute>(&it->second)) {
                     kxc::Type out_type = ref.checked_type();
                     Attrs attrs(op->attrs); 
                     return {(*fcompute_ptr)(attrs, inputs, out_type)};
                 } else {
                     std::cerr << "Error: Failed to cast FTVMCompute (bad type) for " << op_node->name << std::endl;
                 }
             } else {
                 std::cerr << "Warning: No FTVMCompute found for " << op_node->name << std::endl;
             }
        }
        return {};
    }
    
    // Basic implementations for others
    std::vector<te::Tensor> VisitDefault(const Expr& expr) override {
        return {};
    }
};

// Simple TE -> TIR Generator (Simplified)
tir::Stmt BuildLoopNest(te::Tensor output) {
    if (!output.defined()) return tir::Stmt();
    auto op = output->op.As<te::ComputeOpNode>();
    
    // If it's a placeholder, no computation
    if (output->op.As<te::PlaceholderOpNode>()) {
        return tir::Stmt();
    }
    
    if (!op) {
        // Handle other ops?
        return tir::Stmt();
    }

    // 1. Create inner body: Store(output, value, index)
    // We assume 1D flat index for simplicity or just 0 for scalar test
    // Real implementation would flatten indices based on shape
    
    // For this demo, let's assume we just want to generate the structure:
    // for (ax0, 0, shape[0]) {
    //   for (ax1, 0, shape[1]) {
    //     output[ax0, ax1] = ...
    //   }
    // }
    
    // Inner Store
    tir::PrimExpr value = op->body[0];
    
    tir::PrimExpr index = 0;
    // Simple flatten logic
    if (!op->axis.empty()) {
        index = op->axis.back(); 
        // Iterate backwards to build stride?
        // Let's keep it simple: just use the last axis as index for visual verification
    }
    
    // Use Var(output->name) to represent the buffer variable in Store
    tir::Stmt body = tir::Store(tir::Var(output->name), value, index);
    
    // Wrap in Loops (Reverse order: inner to outer)
    for (int i = op->axis.size() - 1; i >= 0; --i) {
        tir::Var loop_var = op->axis[i];
        tir::PrimExpr extent = op->shape[i];
        body = tir::For(loop_var, 0, extent, tir::ForType::Serial, body);
    }
    
    return body;
}

tir::Stmt LowerToTIR(Function func) {
    RelayToTEConverter converter;
    auto outputs = converter.Convert(func->body);
    if (outputs.empty()) return tir::Stmt();
    
    te::Tensor out = outputs[0];
    
    // Recursive build?
    // A real Lower would topologically sort the graph and build Stmts for all Tensors.
    // Here we just build the output tensor's loop nest.
    
    return BuildLoopNest(out);
}

}
}
