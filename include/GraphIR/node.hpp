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
SIMPLE_REGISTER_TYPE(Input)
SIMPLE_REGISTER_TYPE(Output)
SIMPLE_REGISTER_TYPE(Shape)
SIMPLE_REGISTER_TYPE(Sub)
class NodePtr:public objectPtr<Node>{
public:
    std::vector<NodePtr> next_nodes;
    std::vector<NodePtr> prev_nodes;
    std::uint32_t hash_value;
    std::uint32_t structure_hash_value;

    NodePtr() = default;

    explicit NodePtr(Node* node_raw_ptr) : objectPtr<Node>(node_raw_ptr) {
        if (get()) {
            this->hash_value = get_node_hash_value(get()->name, get()->op_type);
            this->structure_hash_value = calcul_structure_hash_value();
        }
    }

    std::uint32_t calcul_structure_hash_value(){
        std::uint32_t hash_value = 0;
        for(auto& node:this->next_nodes){
            hash_value += node->node_hash_value;
        }
        for(auto& node:this->prev_nodes){
            hash_value += node->node_hash_value;
        }
        return hash_value;
    }
    std::uint32_t get_node_hash_value(const std::string& name,const std::string& op_type){
        return this->get()->node_hash_value_func(name,op_type);
    }
};
template<> NodePtr make_object<node>(const std::string& data) {
    node* ptr = new node(data);
    const int32_t type_index = node::RuntimeTypeIndex();
    ptr->SetTypeIndex(type_index);
    ptr->SetDeleter([](void* obj) { delete static_cast<node*>(obj); });
    return nodeptr(ptr);
}
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
    std::string indices_name;
    Gather() = default;
    Gather(std::string name,std::string inputs_name,std::string indices_name):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->indices_name = indices_name;
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
class Sub:public Node{
    SIMPLE_DECLARE_TYPE(Sub,Node)
public:
    std::string op_type = "Sub";
    std::string inputs_name;
    std::string inputs2_name;
    Sub() = default;
    Sub(std::string name,std::string inputs_name,std::string inputs2_name):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->inputs2_name = inputs2_name;
    }
};
class Shape:public Node{
    SIMPLE_DECLARE_TYPE(Shape,Node)
public:
    std::string op_type = "Shape";
    std::string inputs_name;
    Shape() = default;
    Shape(std::string name,std::string inputs_name):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
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
    std::vector<std::string> inputs_names;
    std::int32_t axis;
    Concat() = default;
    Concat(std::string name,std::vector<std::string> inputs_names,std::int32_t axis):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_names = inputs_names;
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
    Split(std::string name,std::string inputs_name,std::vector<std::int32_t> split, std::int32_t axis):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->inputs_name = inputs_name;
        this->split = split;
        this->axis = axis;
    }
};
class Input:public Node{
    SIMPLE_DECLARE_TYPE(Input,Node)
public:
    std::string op_type = "Input";
    Tensor* value;
    Input() = default;
    Input(std::string name,Tensor* value):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->value = value;
    }
};
class Output:public Node{
    SIMPLE_DECLARE_TYPE(Output,Node)
public:
    std::string op_type = "Output";
    Tensor* value;
    Output() = default;
    Output(std::string name,Tensor* value):Node(){
        this->node_hash_value = node_hash_value_func(name,op_type);
        this->name = name;
        this->value = value;
    }
};
}
