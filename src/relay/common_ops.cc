/*! \file src/relay/common_ops.cc
 * \brief 实现 Relay 节点、算子元数据、pass 工具和公共注册。
 */

#include "relay/op.h"
#include "relay/op_macros.h"
#include "relay/op_attr_types.h"
#include "relay/type_infer.h"
#include <iostream>

namespace kxc {
namespace relay {

// --- NN Ops ---
KXC_REGISTER_OP(nn_conv2d)
    .describe("2D convolution layer")
    .set_num_inputs(-1) // 数据、权重[, 偏置]
    .set_attr<FInferType>("FInferType", Conv2DInferType)
    .set_attr<std::string>("TAttrs", "Conv2DAttrs"); // Register the Attribute Type Key

KXC_REGISTER_OP(nn_relu)
    .describe("Rectified Linear Unit activation")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", UnarySameInferType)
    .set_attr<std::string>("TAttrs", "ReluAttrs");

KXC_REGISTER_OP(nn_dense)
    .describe("Dense (fully connected) layer")
    .set_num_inputs(2) // data, weight
    .set_attr<FInferType>("FInferType", DenseInferType)
    .set_attr<std::string>("TAttrs", "DenseAttrs");

// Gemm
KXC_REGISTER_OP(nn_gemm)
    .describe(R"doc(General matrix multiplication.
)doc")
    .set_num_inputs(3)
    .add_argument("A", "Tensor", "The first input tensor.")
    .add_argument("B", "Tensor", "The second input tensor.")
    .add_argument("C", "Tensor", "The third input tensor (bias).")
    .set_attr<FInferType>("FInferType", GemmInferType)
    .set_attr<std::string>("TAttrs", "GemmAttrs");

KXC_REGISTER_OP(nn_max_pool2d)
    .describe("2D max pooling")
    .set_num_inputs(1)
    .set_attr<FInferType>("FInferType", Pool2DInferType)
    .set_attr<std::string>("TAttrs", "MaxPool2DAttrs");

KXC_REGISTER_OP(nn_softmax)
    .describe("Softmax activation")
    .set_num_inputs(1)
    .set_attr<FInferType>("FInferType", SoftmaxInferType)
    .set_attr<std::string>("TAttrs", "SoftmaxAttrs");

// --- Tensor Ops ---
KXC_REGISTER_OP(add)
    .describe("Element-wise addition")
    .set_num_inputs(2)
    .set_attr<FInferType>("FInferType", AddInferType);

KXC_REGISTER_OP(subtract)
    .describe("Element-wise subtraction")
    .set_num_inputs(2)
    .set_attr<FInferType>("FInferType", SubtractInferType);

KXC_REGISTER_OP(multiply)
    .describe("Element-wise multiplication")
    .set_num_inputs(2)
    .set_attr<FInferType>("FInferType", MultiplyInferType);

// --- Control/Logic Ops ---
KXC_REGISTER_OP(greater)
    .describe("Element-wise greater than comparison")
    .set_num_inputs(2)
    .set_attr<FInferType>("FInferType", GreaterInferType);

// --- Device/Communication Ops ---
static OpRegEntry __make_OpEntry_device_copy__ =
    OpRegEntry(Op::Get("device.copy"))
        .describe("Copy tensor value across virtual devices")
        .set_num_inputs(1)
        .set_attr<FInferType>("FInferType", IdentityInferType)
        .set_attr<std::string>("TAttrs", "DeviceCopyAttrs");

static OpRegEntry __make_OpEntry_device_allreduce__ =
    OpRegEntry(Op::Get("device.allreduce"))
        .describe("Collective allreduce on distributed workers")
        .set_num_inputs(1)
        .set_attr<FInferType>("FInferType", IdentityInferType)
        .set_attr<std::string>("TAttrs", "CollectiveAttrs");

static OpRegEntry __make_OpEntry_device_broadcast_from_worker0__ =
    OpRegEntry(Op::Get("device.broadcast_from_worker0"))
        .describe("Collective broadcast from worker0")
        .set_num_inputs(1)
        .set_attr<FInferType>("FInferType", IdentityInferType)
        .set_attr<std::string>("TAttrs", "CollectiveAttrs");

static OpRegEntry __make_OpEntry_device_scatter_from_worker0__ =
    OpRegEntry(Op::Get("device.scatter_from_worker0"))
        .describe("Collective scatter from worker0")
        .set_num_inputs(1)
        .set_attr<FInferType>("FInferType", IdentityInferType)
        .set_attr<std::string>("TAttrs", "CollectiveAttrs");

static OpRegEntry __make_OpEntry_device_gather_to_worker0__ =
    OpRegEntry(Op::Get("device.gather_to_worker0"))
        .describe("Collective gather to worker0")
        .set_num_inputs(1)
        .set_attr<FInferType>("FInferType", IdentityInferType)
        .set_attr<std::string>("TAttrs", "CollectiveAttrs");

static OpRegEntry __make_OpEntry_device_send_to_worker__ =
    OpRegEntry(Op::Get("device.send_to_worker"))
        .describe("Point-to-point send to worker")
        .set_num_inputs(1)
        .set_attr<FInferType>("FInferType", IdentityInferType)
        .set_attr<std::string>("TAttrs", "CollectiveAttrs");

static OpRegEntry __make_OpEntry_device_recv_from_worker__ =
    OpRegEntry(Op::Get("device.recv_from_worker"))
        .describe("Point-to-point recv from worker")
        .set_num_inputs(1)
        .set_attr<FInferType>("FInferType", IdentityInferType)
        .set_attr<std::string>("TAttrs", "CollectiveAttrs");

} // namespace relay
} // namespace kxc
