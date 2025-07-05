#pragma once
#include "capi.hpp"

namespace kxcomp{

class Tensor:public object{
    SIMPLE_DECLARE_TYPE(Tensor,object)
public:
};

SIMPLE_REGISTER_TYPE(Tensor)
}