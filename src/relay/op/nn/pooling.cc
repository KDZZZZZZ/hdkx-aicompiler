/*! \file src/relay/op/nn/pooling.cc
 * \brief 注册 Relay 算子及其 FRelayToTE compute。
 */

#include "relay/op_macros.h"
#include "relay/op.h"
#include "relay/op_attr_types.h"
#include "relay/type_infer.h"
#include "te/topi/nn.h"
#include <vector>
#include <string>
#include <stdexcept>

namespace kxc {
namespace relay {

namespace {
Array<int> Read2DPair(const std::vector<int64_t>& values, int default_value) {
    Array<int> out;
    if (values.size() >= 2) {
        out.push_back(static_cast<int>(values[0]));
        out.push_back(static_cast<int>(values[1]));
    } else {
        out.push_back(default_value);
        out.push_back(default_value);
    }
    return out;
}

Array<int> ReadPadding(const std::vector<int64_t>& values) {
    Array<int> out;
    if (values.size() >= 4) {
        out.push_back(static_cast<int>(values[0]));
        out.push_back(static_cast<int>(values[1]));
        out.push_back(static_cast<int>(values[2]));
        out.push_back(static_cast<int>(values[3]));
    } else if (values.size() >= 2) {
        out.push_back(static_cast<int>(values[0]));
        out.push_back(static_cast<int>(values[1]));
        out.push_back(static_cast<int>(values[0]));
        out.push_back(static_cast<int>(values[1]));
    } else {
        out.push_back(0);
        out.push_back(0);
    }
    return out;
}
}

te::Tensor MaxPool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    (void)out_type;
    if (inputs.size() != 1) {
        throw std::runtime_error("nn_max_pool2d expects exactly 1 input");
    }
    auto* p = attrs.As<MaxPool2DAttrsNode>();
    if (!p) {
        throw std::runtime_error("nn_max_pool2d expects MaxPool2DAttrs");
    }
    if ((p->layout != "NCHW" && !p->layout.empty()) || inputs[0]->shape.size() != 4) {
        throw std::runtime_error("nn_max_pool2d expects NCHW rank-4 input");
    }

    Array<int> pool_size = Read2DPair(p->pool_size, 1);
    Array<int> strides = Read2DPair(p->strides, 1);
    Array<int> padding = ReadPadding(p->padding);
    return te::topi::pool2d(inputs[0], pool_size, strides, padding, "max", p->ceil_mode, "T_max_pool2d");
}

te::Tensor AvgPool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    (void)out_type;
    if (inputs.size() != 1) {
        throw std::runtime_error("nn_avg_pool2d expects exactly 1 input");
    }
    auto* p = attrs.As<MaxPool2DAttrsNode>();
    if (!p) {
        throw std::runtime_error("nn_avg_pool2d expects MaxPool2DAttrs");
    }
    if ((p->layout != "NCHW" && !p->layout.empty()) || inputs[0]->shape.size() != 4) {
        throw std::runtime_error("nn_avg_pool2d expects NCHW rank-4 input");
    }

    Array<int> pool_size = Read2DPair(p->pool_size, 1);
    Array<int> strides = Read2DPair(p->strides, 1);
    Array<int> padding = ReadPadding(p->padding);
    return te::topi::pool2d(inputs[0], pool_size, strides, padding, "avg", p->ceil_mode, "T_avg_pool2d");
}

te::Tensor GlobalAvgPool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    (void)attrs;
    (void)out_type;
    if (inputs.size() != 1) {
        throw std::runtime_error("nn_global_avg_pool2d expects exactly 1 input");
    }
    if (inputs[0]->shape.size() != 4) {
        throw std::runtime_error("nn_global_avg_pool2d expects NCHW rank-4 input");
    }
    return te::topi::global_avg_pool2d(inputs[0], "T_global_avg_pool2d");
}

// ---------------------------------------------------------------------------
// 1. Operator Registration for Pooling
// ---------------------------------------------------------------------------

KXC_REGISTER_OP(nn_max_pool2d)
    .describe(R"doc(2D max pooling operation.

This operator performs max pooling on the input tensor.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "MaxPool2DAttrs")
    .set_attr<FInferType>("FInferType", Pool2DInferType)
    .set_attr<FRelayToTE>("FRelayToTE", MaxPool2DCompute);

KXC_REGISTER_OP(nn_avg_pool2d)
    .describe(R"doc(2D average pooling operation.

This operator performs average pooling on the input tensor.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    // Reusing MaxPool2DAttrs for AvgPool2D as they share structure (pool_size, strides, padding)
    // In real TVM, they might share a generic Pool2DAttrs.
    .set_attr<std::string>("TAttrs", "MaxPool2DAttrs")
    .set_attr<FInferType>("FInferType", Pool2DInferType)
    .set_attr<FRelayToTE>("FRelayToTE", AvgPool2DCompute); 

KXC_REGISTER_OP(nn_global_avg_pool2d)
    .describe(R"doc(Global average pooling operation.

Reduces the spatial dimensions (H, W) to 1x1 by averaging.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "GlobalAvgPool2DAttrs")
    .set_attr<FInferType>("FInferType", GlobalAvgPool2DInferType)
    .set_attr<FRelayToTE>("FRelayToTE", GlobalAvgPool2DCompute);

} // namespace relay
} // namespace kxc
