#include <vector>
#include "base/capi.hpp"
namespace kxcomp{

class Tensor;

SIMPLE_REGISTER_TYPE(Node)
SIMPLE_REGISTER_TYPE(Constent)
SIMPLE_REGISTER_TYPE(Gather)
SIMPLE_REGISTER_TYPE(Add)
SIMPLE_REGISTER_TYPE(Mul)
SIMPLE_REGISTER_TYPE(MatMul)
SIMPLE_REGISTER_TYPE(Div)
SIMPLE_REGISTER_TYPE(Pow)
SIMPLE_REGISTER_TYPE(Slice)
SIMPLE_REGISTER_TYPE(Reshape)
SIMPLE_REGISTER_TYPE(Transpose)
SIMPLE_REGISTER_TYPE(Concat)
SIMPLE_REGISTER_TYPE(Unsqueeze)
SIMPLE_REGISTER_TYPE(Sqrt)
SIMPLE_REGISTER_TYPE(Cast)
SIMPLE_REGISTER_TYPE(ReduceMean)
SIMPLE_REGISTER_TYPE(Softmax)
SIMPLE_REGISTER_TYPE(Split)
class NodePtr:public objectPtr<Node>{
public:
    std::vector<objectPtr<Node>> next_nodes;
    std::vector<objectPtr<Node>> prev_nodes;
    std::uint32_t node_hash_value;
    std::uint32_t node_hash_value_func(const std::vector<objectPtr<Node>>& next_nodes,const std::vector<objectPtr<Node>>& prev_nodes){
        std::uint32_t hash_value = 0;
        for(auto& node:next_nodes){
            hash_value += node->node_hash_value;
        }
        return hash_value;
    }
    //结构等价性验证、类型转换、节点查询
    NodePtr(std::vector<objectPtr<Node>> next_nodes,std::vector<objectPtr<Node>> prev_nodes = {}):next_nodes(next_nodes),prev_nodes(prev_nodes){
        this->node_hash_value = node_hash_value_func(next_nodes,prev_nodes);
    }
};
class Node:public object{
SIMPLE_DECLARE_TYPE(Node,object)
public:
    std::string name;
    std::string op_type;
    std::uint32_t node_hash_value;
    
    Node() = default;
    
    std::uint32_t node_hash_value_func(const std::string& name,const std::string& op_type){
        return std::hash<std::string>{}(name + op_type);
    }
};
//value类型待实现
class Constent:public Node{
    SIMPLE_DECLARE_TYPE(Constent,Node)
public:
    struct attribute{
        std::string name;
        Tensor* value;
    };
    std::string op_type = "Constent";
    attribute attribute;
    std::vector<Node> outputs;
    Constent() = default;
    Constent(std::string name,Tensor* value):Node() {
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->attribute.value = value;
    }
};
class Gather:public Node{
    SIMPLE_DECLARE_TYPE(Gather,Node)
public:
    std::string op_type = "Gather";
    std::string inputs_name;
    Tensor* indices;
    Gather() = default;
    Gather(std::string name,std::string inputs_name,Tensor* indices):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->indices = indices;
    }
};
class Add:public Node{
    SIMPLE_DECLARE_TYPE(Add,Node)
public:
    std::string op_type = "Add";
    std::string inputs_name;
    std::string inputs2_name;
    Add() = default;
    Add(std::string name,std::string inputs_name,std::string inputs2_name):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->inputs2_name = inputs2_name;
    }
};
class Mul:public Node{
    SIMPLE_DECLARE_TYPE(Mul,Node)
public:
    std::string op_type = "Mul";
    std::string inputs_name;
    std::string inputs2_name;
    Mul() = default;
    Mul(std::string name,std::string inputs_name,std::string inputs2_name):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->inputs2_name = inputs2_name;
    }
};
class MatMul:public Node{
    SIMPLE_DECLARE_TYPE(MatMul,Node)
public:
    std::string op_type = "MatMul";
    std::string inputs_name;
    std::string inputs2_name;
    MatMul() = default;
    MatMul(std::string name,std::string inputs_name,std::string inputs2_name):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->inputs2_name = inputs2_name;
    }
};
class Div:public Node{
    SIMPLE_DECLARE_TYPE(Div,Node)
public:
    std::string op_type = "Div";
    std::string inputs_name;
    std::string inputs2_name;
    Div() = default;
    Div(std::string name,std::string inputs_name,std::string inputs2_name):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->inputs2_name = inputs2_name;
    }
};
class Pow:public Node{
    SIMPLE_DECLARE_TYPE(Pow,Node)
public:
    std::string op_type = "Pow";
    std::string inputs_name;
    std::int32_t power;
    Pow() = default;
    Pow(std::string name,std::string inputs_name,std::int32_t power):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->power = power;
    }
};
class Slice:public Node{
    SIMPLE_DECLARE_TYPE(Slice,Node)
public:
    std::string op_type = "Slice";
    std::string inputs_name;
    std::vector<std::int32_t> starts;
    std::vector<std::int32_t> ends;
    std::vector<std::int32_t> axes;
    std::vector<std::int32_t> steps;
    Slice() = default;
    Slice(std::string name,std::string inputs_name,std::vector<std::int32_t> starts,std::vector<std::int32_t> ends,std::vector<std::int32_t> axes,std::vector<std::int32_t> steps):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->starts = starts;
        this->ends = ends;
        this->axes = axes;
        this->steps = steps;
    }
};
class Reshape:public Node{
    SIMPLE_DECLARE_TYPE(Reshape,Node)
public:
    std::string op_type = "Reshape";
    std::string inputs_name;
    bool allow_zero;
    std::vector<std::int32_t> shape;
    Reshape() = default;
    Reshape(std::string name,std::string inputs_name,std::vector<std::int32_t> shape,bool allow_zero):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->shape = shape;
        this->allow_zero = allow_zero;
    }
};
class Transpose:public Node{
    SIMPLE_DECLARE_TYPE(Transpose,Node)
public:
    std::string op_type = "Transpose";
    std::vector<std::int32_t> perm;
    std::string inputs_name;
    Transpose() = default;
    Transpose(std::string name,std::string inputs_name,std::vector<std::int32_t> perm):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->perm = perm;
    }
};
class Concat:public Node{
    SIMPLE_DECLARE_TYPE(Concat,Node)
public:
    std::string op_type = "Concat";
    std::string inputs_name;
    std::int32_t axis;
    Concat() = default;
    Concat(std::string name,std::string inputs_name,std::int32_t axis):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->axis = axis;
    }
};
class Unsqueeze:public Node{
    SIMPLE_DECLARE_TYPE(Unsqueeze,Node)
public:
    std::string op_type = "Unsqueeze";
    std::string inputs_name;
    std::vector<std::int32_t> axes;
    Unsqueeze() = default;
    Unsqueeze(std::string name,std::string inputs_name,std::vector<std::int32_t> axes):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->axes = axes;
    }
};
class Sqrt:public Node{
    SIMPLE_DECLARE_TYPE(Sqrt,Node)
public:
    std::string op_type = "Sqrt";
    std::string inputs_name;
    Sqrt() = default;
    Sqrt(std::string name,std::string inputs_name):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
    }
};
class Cast:public Node{
    SIMPLE_DECLARE_TYPE(Cast,Node)
public:
    std::string op_type = "Cast";
    std::string inputs_name;
    std::string to;
    Cast() = default;
    Cast(std::string name,std::string inputs_name,std::string to):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->to = to;
    }
};
class ReduceMean:public Node{
    SIMPLE_DECLARE_TYPE(ReduceMean,Node)
public:
    std::string op_type = "ReduceMean";
    std::string inputs_name;
    std::vector<std::int32_t> axes;
    bool keep_dims;
    ReduceMean() = default;
    ReduceMean(std::string name,std::string inputs_name,std::vector<std::int32_t> axes,bool keep_dims):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->axes = axes;
        this->keep_dims = keep_dims;
    }
};
class Softmax:public Node{
    SIMPLE_DECLARE_TYPE(Softmax,Node)
public:
    std::string op_type = "Softmax";
    std::string inputs_name;
    std::int32_t axis;
    Softmax() = default;
    Softmax(std::string name,std::string inputs_name,std::int32_t axis):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->axis = axis;
    }
};
class Split:public Node{
    SIMPLE_DECLARE_TYPE(Split,Node)
public:
    std::string op_type = "Split";
    std::string inputs_name;
    std::vector<std::int32_t> split;
    std::int32_t axis;
    Split() = default;
    Split(std::string name,std::string inputs_name,std::vector<std::int32_t> split_sizes):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->split = split;
        this->axis = axis;
    }
};
}
