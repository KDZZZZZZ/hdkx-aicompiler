/*! \file src/relay/op/nn/pooling.cc
 * \brief 注册 Relay 算子及其 FRelayToTE compute。
 */

#include "kxc/relay/op_macros.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/topi/nn.h"
#include <string>
#include <stdexcept>

namespace kxc {
namespace relay {

namespace {
// 将零、一或二维池化属性规范化为高宽二元组。
Array<int> Read2DPair(const Array<int64_t>& values, int default_value) {
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

// 将紧凑 padding 属性展开为 TOPI 接受的二维或四维形式。
Array<int> ReadPadding(const Array<int64_t>& values) {
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

// 校验并生成最大池化的 TE 计算。
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

// 校验并生成平均池化的 TE 计算；属性布局与最大池化共享。
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

// 将 NCHW 输入的全部空间维归约为 1x1 TE 张量。
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
// 注册三类池化算子的属性、类型推导和 Relay-to-TE 入口。
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
    // 平均池化与最大池化共享 pool_size、strides、padding 等属性结构。
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

namespace kxc::builtin_anchor {
void RelayPoolingOps() {}
}  // namespace kxc::builtin_anchor
