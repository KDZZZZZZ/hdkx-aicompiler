#pragma once

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"

namespace kxc::test_support {

// The same graph is consumed by real CPU/CUDA request tests and offline CUDA
// lowering. Axis 2 / C=2 checks nontrivial row packing and multi-token append.
inline api::experimental::restricted_symbolic_shape::v1::BoundedCompileRequest
RequestBatchingGraph(const api::CompileConfig& config, int64_t axis = 1, int64_t append = 1) {
    const Array<int64_t> shape = axis == 1 ? Array<int64_t>{1,2,4} : Array<int64_t>{1,2,2,4};
    Array<int64_t> token_shape;
    for (int64_t dimension : shape) token_shape.push_back(dimension);
    token_shape[axis] = append;
    const Var past("past",TensorType(shape,"float32")), token("token",TensorType(token_shape,"float32"));
    const Var extra("extra",TensorType({1,2},"float32"));
    const auto present = Call(relay::Op::Get("concatenate"),{past,token},relay::ConcatenateAttrs::Create(axis));
    const auto body = Tuple({Call(relay::Op::Get("nn_relu"),{present}),present,
                             Call(relay::Op::Get("nn_relu"),{extra})});
    using Adapter = api::experimental::restricted_symbolic_shape::v1::RestrictedSymbolicShapeAdapter;
    return Adapter::MintBoundedCompileRequest(Adapter::Prepare(Function({past,token,extra},body),
        config,{{0,0,"B",1,3,1},{0,static_cast<size_t>(axis),"P",0,5,1},
                {1,0,"B",1,3,1},{2,0,"B",1,3,1},{2,1,"V",1,4,1}}));
}

}  // namespace kxc::test_support
