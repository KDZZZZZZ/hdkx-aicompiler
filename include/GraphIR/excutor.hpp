#pragma once
#include "base/capi.hpp"
#include "node.hpp"
#include "graph.hpp"
namespace kxcomp{
SIMPLE_REGISTER_TYPE(Excutor)
class Excutor:public object{//实现遍历、存放信息等基础功能
    SIMPLE_DECLARE_TYPE(Excutor,object)
public:
    Graph graph;
};
typedef struct{//TODO:支持非相关pass并行优化
    std::vector<Excutor> excutors;
}ExcutorStack;

}
