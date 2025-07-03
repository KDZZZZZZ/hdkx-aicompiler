#include "base/capi.hpp"
#include "node.hpp"
#include <vector>

namespace kxcomp{
typedef struct{
    std::string node_name;
    
}Graph_node;

SIMPLE_REGISTER_TYPE(Graph)
class Graph:public object{
    SIMPLE_DECLARE_TYPE(Graph,object)
public:
    std::string Graph_name;
    std::vector<objectPtr<Node>> graph;
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;

};
}