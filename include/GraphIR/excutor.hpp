#pragma once
#include "base/capi.hpp"
#include "node.hpp"
#include "graph.hpp"
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <vector>
namespace kxcomp{

class Excutor:public object{//实现遍历、存放信息等基础功能
    SIMPLE_DECLARE_TYPE(Excutor,object)
public:
    GraphPtr graph;
    void traverse_graph(){
        for(auto node:graph->graph_nodes){
            if(this->identify(node)){
                this->optimize_op(node);
            }
        }
    }
    bool identify(NodePtr node);
    void optimize_op(NodePtr node);
    Excutor() = default;
    explicit Excutor(GraphPtr graph):graph(graph){
        if(graph.get()){
            this->graph = graph;
        }
    }
};
class FoldConstant:public Excutor{
    SIMPLE_DECLARE_TYPE(FoldConstant,Excutor)
public:
    bool identify(NodePtr node){
        // An operation node is foldable if all its inputs are constants,
        // it is not a constant itself, and it actually has inputs.
        if (node->op_type == "Constent") {
            return false;
        }
        if (node.prev_nodes.empty()) {
            return false;
        }

        for (const auto& prev : node.prev_nodes) {
            if (prev->op_type != "Constent") {
                return false;
            }
        }
        return true;
    };
    void optimize_op(NodePtr node){
        // Note: This implementation assumes the existence of a Tensor class
        // with basic arithmetic operations and a constructor to wrap results.
        // The actual tensor logic is commented out and needs to be implemented.

        std::vector<Tensor*> input_tensors;
        for (const auto& prev : node.prev_nodes) {
            // We know from identify() that all prev nodes are Constent.
            auto const_node = dynamic_cast<Constent*>(prev.get());
            if(const_node){
                input_tensors.push_back(const_node->attribute.value);
            }
        }

        Tensor* result_tensor = nullptr;

        // Perform operation based on node type
        if (node->op_type == "Add") {
            // result_tensor = tensor_add(input_tensors[0], input_tensors[1]);
        } else if (node->op_type == "Mul") {
            // result_tensor = tensor_mul(input_tensors[0], input_tensors[1]);
        } else if (node->op_type == "Sub") {
            // result_tensor = tensor_sub(input_tensors[0], input_tensors[1]);
        }
        // ... other foldable operations like Div, Pow, etc.

        if (result_tensor) {
            std::string new_const_name = "const_" + node->name;
            NodePtr new_const_node(make_object<Constent>(new_const_name, result_tensor).release());
            
            graph->replace_node(node, new_const_node);
        }
    }
};
class FuseOps:public Excutor{
    SIMPLE_DECLARE_TYPE(FuseOps,Excutor)
public:
    bool identify(NodePtr node);
    void optimize_op(NodePtr node);
};
class SimlifyInference:public Excutor{
    SIMPLE_DECLARE_TYPE(SimlifyInference,Excutor)
public:
    bool identify(NodePtr node);
    void optimize_op(NodePtr node);
};
class FastMath:public Excutor{
    SIMPLE_DECLARE_TYPE(FastMath,Excutor)
public:
    bool identify(NodePtr node);
    void optimize_op(NodePtr node);
};
class DynamicToStatic:public Excutor{
    SIMPLE_DECLARE_TYPE(DynamicToStatic,Excutor)
public:
    bool identify(NodePtr node);
    void optimize_op(NodePtr node);
};
class InferType:public Excutor{
    SIMPLE_DECLARE_TYPE(InferType,Excutor)
public:
    bool identify(NodePtr node);
    void optimize_op(NodePtr node);
};
class EliminateCommonSubexpression:public Excutor{
    SIMPLE_DECLARE_TYPE(EliminateCommonSubexpression,Excutor)
private:
    std::unordered_map<std::uint32_t, NodePtr> seen_expressions;

    uint32_t compute_cse_hash(NodePtr node) {
        if (node->cse_hash.has_value()) {
            return node->cse_hash.value();
        }

        std::string hash_str = node->op_type;
        std::hash<std::string> str_hasher;
        uint32_t hash_val = str_hasher(hash_str);

        if (auto trans_node = dynamic_cast<Transpose*>(node.get())) {
            for (auto p : trans_node->perm) {
                hash_val ^= (p + 27); // Add some salt
            }
        }
        
        for (auto& prev_node : node.prev_nodes) {
             hash_val ^= compute_cse_hash(prev_node);
        }

        node->cse_hash = hash_val;
        return hash_val;
    }

public:
    bool identify(NodePtr node){
        if (node->op_type == "Input" || node->op_type == "Output" || node->op_type == "Constent") {
            return false;
        }

        auto hash_val = compute_cse_hash(node);
        if (seen_expressions.count(hash_val)) {
            // A full implementation should check for hash collisions.
            return true;
        } else {
            seen_expressions[hash_val] = node;
            return false;
        }
    }
    void optimize_op(NodePtr node){
        auto hash_val = compute_cse_hash(node);
        NodePtr existing_node = seen_expressions[hash_val];

        auto consumers = node.next_nodes;
        for (auto& consumer : consumers) {
            for (auto& input_to_consumer : consumer.prev_nodes) {
                if (input_to_consumer.get() == node.get()) {
                    input_to_consumer = existing_node;
                }
            }
            
            auto it = std::find_if(existing_node.next_nodes.begin(), existing_node.next_nodes.end(), 
                                   [&](const NodePtr& n){ return n.get() == consumer.get(); });
            if(it == existing_node.next_nodes.end()){
                existing_node.next_nodes.push_back(consumer);
            }
        }
        
        graph->remove_node(node);
    }
};
class AlterOpLayout:public Excutor{
    SIMPLE_DECLARE_TYPE(AlterOpLayout,Excutor)
public:
    bool identify(NodePtr node);
    void optimize_op(NodePtr node);
};
class AnnotateUsedMemory:public Excutor{
    SIMPLE_DECLARE_TYPE(AnnotateUsedMemory,Excutor)
public:
    bool identify(NodePtr node);
    void optimize_op(NodePtr node);
};




SIMPLE_REGISTER_TYPE(Excutor)
SIMPLE_REGISTER_TYPE(FoldConstant)
SIMPLE_REGISTER_TYPE(FuseOps)
SIMPLE_REGISTER_TYPE(SimlifyInference)
SIMPLE_REGISTER_TYPE(FastMath)
SIMPLE_REGISTER_TYPE(DynamicToStatic)
SIMPLE_REGISTER_TYPE(InferType)
SIMPLE_REGISTER_TYPE(EliminateCommonSubexpression)
SIMPLE_REGISTER_TYPE(AlterOpLayout)
SIMPLE_REGISTER_TYPE(AnnotateUsedMemory)
}
