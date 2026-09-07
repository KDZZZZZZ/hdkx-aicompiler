/*! \file src/relay/op/tensor/shape_ops.cc
 * \brief 注册受限形状值 Relay 算子及其 FRelayToTE compute。
 *
 * 本文件是 M3 S1/S2 形状值算子的唯一落点：
 * - shape_of 把输入张量的各维（静态常量或受限符号维）物化为 int64 行向量。
 * - 形状链的准入、越界与溢出拒绝由 src/compiler/shape/ 的受限准备路径负责；
 *   本层只提供类型规则与真实 TE/TIR 计算，不引入第二套形状解释器。
 */

#include "kxc/relay/op_macros.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/te.h"
#include "kxc/tir/expr.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace kxc {
namespace relay {

namespace {

void RequireInputCount(const char* op_name, const Array<te::Tensor>& inputs, size_t expected) {
    if (inputs.size() != expected) {
        throw std::runtime_error(std::string(op_name) + " expects exactly " +
                                 std::to_string(expected) + " input(s)");
    }
}

const TensorTypeNode* RequireTensorOutput(const char* op_name, const kxc::Type& out_type) {
    const auto* tensor_type = out_type.As<TensorTypeNode>();
    if (!tensor_type) {
        throw std::runtime_error(std::string(op_name) + " expects TensorType output");
    }
    return tensor_type;
}

te::Tensor RequireDefined(const char* op_name, const te::Tensor& tensor) {
    if (!tensor.defined()) {
        throw std::runtime_error(std::string(op_name) + " lowering returned undefined tensor");
    }
    return tensor;
}

Array<kxc::tir::PrimExpr> ShapeFromTensorType(const TensorTypeNode* type,
                                              const std::string& op_name) {
    Array<kxc::tir::PrimExpr> shape;
    for (int64_t dim : type->shape) {
        if (dim < 0) {
            throw std::runtime_error(op_name + " lowering requires static output shape");
        }
        shape.push_back(kxc::tir::IntImm(dim, kxc::tir::DataType::Int(64)));
    }
    return shape;
}

kxc::tir::PrimExpr LinearIndex(const Array<kxc::tir::PrimExpr>& indices,
                               const Array<kxc::tir::PrimExpr>& dims) {
    if (dims.empty()) {
        return kxc::tir::IntImm(0, kxc::tir::DataType::Int(64));
    }
    if (indices.size() != dims.size()) {
        throw std::runtime_error("restricted shape op index rank mismatch");
    }
    kxc::tir::PrimExpr linear = indices[0];
    for (size_t i = 1; i < indices.size(); ++i) {
        linear = linear * dims[i] + indices[i];
    }
    return linear;
}

Array<kxc::tir::PrimExpr> UnflattenIndex(kxc::tir::PrimExpr flat,
                                         const Array<kxc::tir::PrimExpr>& dims) {
    Array<kxc::tir::PrimExpr> out;
    for (size_t i = 0; i < dims.size(); ++i) {
        out.push_back(kxc::tir::IntImm(0, kxc::tir::DataType::Int(64)));
    }
    kxc::tir::PrimExpr cur = flat;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        out[static_cast<size_t>(i)] = cur % dims[static_cast<size_t>(i)];
        cur = cur / dims[static_cast<size_t>(i)];
    }
    return out;
}

// 通用行重排拷贝体：输出扁平索引映射回输入多维索引（行主序一致）。
kxc::tir::PrimExpr LinearRemapBody(const te::Tensor& input,
                                   const Array<kxc::tir::PrimExpr>& out_shape,
                                   const Array<kxc::tir::Var>& indices) {
    Array<kxc::tir::PrimExpr> out_indices;
    for (const auto& index : indices) {
        out_indices.push_back(index);
    }
    kxc::tir::PrimExpr flat = LinearIndex(out_indices, out_shape);
    Array<kxc::tir::PrimExpr> input_indices = UnflattenIndex(flat, input->shape);
    return input(input_indices);
}

// 读取并校验受限形状表达式三元组（与 InferType 同一约定）。
struct ShapeExprElement {
    int64_t kind;
    int64_t value;
    int64_t axis;
};

std::vector<ShapeExprElement> ReadShapeExpr(const std::string& op_name,
                                            const Array<int64_t>& kinds,
                                            const Array<int64_t>& values,
                                            const Array<int64_t>& axes) {
    if (kinds.empty() || values.size() != kinds.size() || axes.size() != kinds.size()) {
        throw std::runtime_error(op_name + " requires a resolved restricted shape "
                                 "expression (expr_kinds/expr_values/expr_axes)");
    }
    std::vector<ShapeExprElement> elements;
    elements.reserve(kinds.size());
    for (size_t i = 0; i < kinds.size(); ++i) {
        if (kinds[i] != kShapeExprKindConst && kinds[i] != kShapeExprKindInputAxis) {
            throw std::runtime_error(op_name + " shape expression kind must be 0 or 1");
        }
        elements.push_back(ShapeExprElement{kinds[i], values[i], axes[i]});
    }
    return elements;
}

// 由受限表达式 + 输入占位符形状构造输出 TE 形状：轴引用取占位符的
// 静态/extent-load 条目，常量取 IntImm；-1 输出类型在此拒绝。
Array<kxc::tir::PrimExpr> ShapeExprToTEShape(const std::string& op_name,
                                             const std::vector<ShapeExprElement>& elements,
                                             const te::Tensor& data) {
    Array<kxc::tir::PrimExpr> shape;
    for (const ShapeExprElement& element : elements) {
        if (element.kind == kShapeExprKindConst) {
            if (element.value < 0) {
                throw std::runtime_error(op_name +
                                         " shape expression constant must be >= 0");
            }
            shape.push_back(kxc::tir::IntImm(element.value, kxc::tir::DataType::Int(64)));
            continue;
        }
        if (element.value != 0) {
            throw std::runtime_error(
                op_name + " shape expression axis references must target input 0 (data)");
        }
        const int64_t data_rank = static_cast<int64_t>(data->shape.size());
        if (element.axis < 0 || element.axis >= data_rank) {
            throw std::runtime_error(op_name + " shape expression axis out of range");
        }
        shape.push_back(data->shape[static_cast<size_t>(element.axis)]);
    }
    if (shape.empty()) {
        throw std::runtime_error(op_name + " requires a rank >= 1 target");
    }
    return shape;
}

// 注意：kxc::tir::DataType 的 code 是 TIR 约定（0=Int, 1=UInt, 2=Float,
// Bool=UInt(1)），不是 DLDataType 的 kDL* 值。
kxc::tir::PrimExpr ConstantOfShapeValue(kxc::tir::DataType dtype, double value) {
    if (dtype == kxc::tir::DataType::Bool()) {
        return kxc::tir::IntImm(value != 0.0 ? 1 : 0,
                                kxc::tir::DataType::Bool());
    }
    if (dtype.code == 2) {  // TIR float
        return dtype.bits == 64 ? kxc::tir::FloatImm(value, dtype)
                                : kxc::tir::FloatImm(static_cast<float>(value), dtype);
    }
    if (dtype.code == 0) {  // TIR int
        return kxc::tir::IntImm(static_cast<int64_t>(value), dtype);
    }
    if (dtype.code == 1) {  // TIR uint
        return kxc::tir::IntImm(static_cast<int64_t>(value), dtype);
    }
    throw std::runtime_error("constant_of_shape has an unsupported fill dtype");
}

kxc::tir::DataType DTypeFromCastCodeForShape(int code) {
    switch (code) {
    case 0: return kxc::tir::DataType::Float(32);
    case 1: return kxc::tir::DataType::Int(32);
    case 2: return kxc::tir::DataType::Int(64);
    case 3: return kxc::tir::DataType::Float(64);
    case 4: return kxc::tir::DataType::Bool();
    case 5: return kxc::tir::DataType::Int(8);
    case 6: return kxc::tir::DataType::UInt(8);
    default:
        throw std::runtime_error("constant_of_shape has an unsupported dtype code");
    }
}

}  // namespace

// 把输入张量各维写入 int64[rank] 输出：静态维写入常量，受限符号维在
// bounded 降级中经既有 runtime extent ABI 读取真实运行期数值。
te::Tensor ShapeOfCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                          const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("shape_of", inputs, 1);
    const auto* tensor_type = RequireTensorOutput("shape_of", out_type);
    const size_t rank = inputs[0]->shape.size();
    if (rank == 0 || tensor_type->shape.size() != 1 ||
        tensor_type->shape[0] != static_cast<int64_t>(rank)) {
        throw std::runtime_error(
            "shape_of lowering output must be int64[rank(data)] with rank >= 1");
    }
    for (const kxc::tir::PrimExpr& extent : inputs[0]->shape) {
        if (!extent.defined()) {
            throw std::runtime_error("shape_of lowering received an undefined input extent");
        }
    }
    const kxc::tir::DataType index_dtype = kxc::tir::DataType::Int(32);
    const kxc::tir::DataType value_dtype = kxc::tir::DataType::Int(64);
    return RequireDefined("shape_of", te::compute(
        {kxc::tir::IntImm(static_cast<int64_t>(rank), kxc::tir::DataType::Int(64))},
        [input = inputs[0], rank, index_dtype, value_dtype](
            const Array<kxc::tir::Var>& indices) {
            // body(i) = i == 0 ? dim0 : i == 1 ? dim1 : ...；rank 在编译期固定，
            // 静态维为 IntImm，受限符号维为 runtime extent load。
            kxc::tir::PrimExpr selected = kxc::tir::IntImm(0, value_dtype);
            for (size_t axis = rank; axis > 0; --axis) {
                selected = kxc::tir::Select(
                    kxc::tir::EQ(indices[0],
                                 kxc::tir::IntImm(static_cast<int64_t>(axis - 1),
                                                  index_dtype)),
                    input->shape[axis - 1],
                    selected);
            }
            return selected;
        },
        "T_shape_of"));
}

// 受限形状表达式单元：把源张量的各维（静态/extent load）与常量元素
// 物化为 int64[len] 输出；这是 Shape→Gather→Concat 链折叠后的唯一形态。
te::Tensor ShapeExprCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                            const kxc::Type& out_type) {
    (void)out_type;
    RequireInputCount("shape_expr", inputs, 1);
    const auto* shape_expr_attrs = attrs.As<ShapeExprAttrsNode>();
    if (!shape_expr_attrs) {
        throw std::runtime_error("shape_expr expects ShapeExprAttrs");
    }
    const std::vector<ShapeExprElement> elements = ReadShapeExpr(
        "shape_expr", shape_expr_attrs->expr_kinds, shape_expr_attrs->expr_values,
        shape_expr_attrs->expr_axes);
    const te::Tensor source = inputs[0];
    const kxc::tir::DataType index_dtype = kxc::tir::DataType::Int(32);
    const kxc::tir::DataType value_dtype = kxc::tir::DataType::Int(64);
    const int64_t length = static_cast<int64_t>(elements.size());
    std::vector<kxc::tir::PrimExpr> element_values;
    element_values.reserve(elements.size());
    for (const ShapeExprElement& element : elements) {
        if (element.kind == kShapeExprKindConst) {
            element_values.push_back(
                kxc::tir::IntImm(element.value, kxc::tir::DataType::Int(64)));
        } else {
            const int64_t source_rank = static_cast<int64_t>(source->shape.size());
            if (element.value != 0 || element.axis < 0 ||
                element.axis >= source_rank) {
                throw std::runtime_error(
                    "shape_expr axis references must target input 0 within range");
            }
            element_values.push_back(source->shape[static_cast<size_t>(element.axis)]);
        }
    }
    return RequireDefined("shape_expr", te::compute(
        {kxc::tir::IntImm(length, kxc::tir::DataType::Int(64))},
        [element_values, length, index_dtype, value_dtype](
            const Array<kxc::tir::Var>& indices) {
            kxc::tir::PrimExpr selected = kxc::tir::IntImm(0, value_dtype);
            for (int64_t axis = length; axis > 0; --axis) {
                selected = kxc::tir::Select(
                    kxc::tir::EQ(indices[0],
                                 kxc::tir::IntImm(axis - 1, index_dtype)),
                    element_values[static_cast<size_t>(axis - 1)],
                    selected);
            }
            return selected;
        },
        "T_shape_expr"));
}

// 控制输入 reshape：目标来自已解析的受限形状表达式；控制张量只承载
// producer/consumer 对应关系，kernel 由 extent ABI 与输出形状完成拷贝。
te::Tensor ReshapeDynamicCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const kxc::Type& out_type) {
    (void)out_type;
    RequireInputCount("reshape_dynamic", inputs, 2);
    const auto* reshape_attrs = attrs.As<ReshapeDynamicAttrsNode>();
    if (!reshape_attrs) {
        throw std::runtime_error("reshape_dynamic expects ReshapeDynamicAttrs");
    }
    const std::vector<ShapeExprElement> elements = ReadShapeExpr(
        "reshape_dynamic", reshape_attrs->expr_kinds, reshape_attrs->expr_values,
        reshape_attrs->expr_axes);
    const Array<kxc::tir::PrimExpr> out_shape =
        ShapeExprToTEShape("reshape_dynamic", elements, inputs[0]);
    return RequireDefined("reshape_dynamic", te::compute(
        out_shape,
        [input = inputs[0], out_shape](const Array<kxc::tir::Var>& indices) {
            return LinearRemapBody(input, out_shape, indices);
        },
        "T_reshape_dynamic"));
}

// 受限目标 expand：逐轴广播拷贝，目标各维来自受限表达式。
te::Tensor ExpandCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                         const kxc::Type& out_type) {
    (void)out_type;
    RequireInputCount("expand", inputs, 2);
    const auto* expand_attrs = attrs.As<ExpandAttrsNode>();
    if (!expand_attrs) {
        throw std::runtime_error("expand expects ExpandAttrs");
    }
    const std::vector<ShapeExprElement> elements = ReadShapeExpr(
        "expand", expand_attrs->expr_kinds, expand_attrs->expr_values,
        expand_attrs->expr_axes);
    const Array<kxc::tir::PrimExpr> out_shape =
        ShapeExprToTEShape("expand", elements, inputs[0]);
    const te::Tensor data = inputs[0];
    const size_t rank = out_shape.size();
    if (rank != data->shape.size()) {
        throw std::runtime_error("expand lowering requires equal data and target rank");
    }
    return RequireDefined("expand", te::compute(
        out_shape,
        [data, rank](const Array<kxc::tir::Var>& indices) {
            Array<kxc::tir::PrimExpr> in_indices;
            for (size_t axis = 0; axis < rank; ++axis) {
                // in_dim == 1 → 广播；否则输出索引即输入索引（合同保证等值）。
                in_indices.push_back(kxc::tir::Select(
                    kxc::tir::EQ(data->shape[axis],
                                 kxc::tir::IntImm(1, kxc::tir::DataType::Int(64))),
                    kxc::tir::IntImm(0, kxc::tir::DataType::Int(64)),
                    indices[axis]));
            }
            return data(in_indices);
        },
        "T_expand"));
}

// 常量目标 constant_of_shape：输出形状来自显式常量目标，逐元素写填充值。
te::Tensor ConstantOfShapeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                  const kxc::Type& out_type) {
    (void)inputs;
    const auto* constant_attrs = attrs.As<ConstantOfShapeAttrsNode>();
    if (!constant_attrs) {
        throw std::runtime_error("constant_of_shape expects ConstantOfShapeAttrs");
    }
    const auto* tensor_type = RequireTensorOutput("constant_of_shape", out_type);
    Array<kxc::tir::PrimExpr> shape;
    for (int64_t dim : tensor_type->shape) {
        if (dim < 0) {
            throw std::runtime_error(
                "constant_of_shape lowering requires a static target shape");
        }
        shape.push_back(kxc::tir::IntImm(dim, kxc::tir::DataType::Int(64)));
    }
    const kxc::tir::DataType dtype =
        DTypeFromCastCodeForShape(constant_attrs->dtype_code);
    const kxc::tir::PrimExpr fill = ConstantOfShapeValue(dtype, constant_attrs->value);
    return RequireDefined("constant_of_shape", te::compute(
        shape,
        [fill](const Array<kxc::tir::Var>&) { return fill; },
        "T_constant_of_shape"));
}

// squeeze/unsqueeze：被移除/插入维恒为 1，线性索引一一对应。
te::Tensor SqueezeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                          const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("squeeze", inputs, 1);
    const auto* tensor_type = RequireTensorOutput("squeeze", out_type);
    const Array<kxc::tir::PrimExpr> out_shape =
        ShapeFromTensorType(tensor_type, "squeeze");
    return RequireDefined("squeeze", te::compute(
        out_shape,
        [input = inputs[0], out_shape](const Array<kxc::tir::Var>& indices) {
            return LinearRemapBody(input, out_shape, indices);
        },
        "T_squeeze"));
}

te::Tensor UnsqueezeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                            const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("unsqueeze", inputs, 1);
    const auto* tensor_type = RequireTensorOutput("unsqueeze", out_type);
    const Array<kxc::tir::PrimExpr> out_shape =
        ShapeFromTensorType(tensor_type, "unsqueeze");
    return RequireDefined("unsqueeze", te::compute(
        out_shape,
        [input = inputs[0], out_shape](const Array<kxc::tir::Var>& indices) {
            return LinearRemapBody(input, out_shape, indices);
        },
        "T_unsqueeze"));
}

KXC_REGISTER_OP(shape_expr)
    .describe(R"doc(Materialize a verified restricted shape expression from one source tensor.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The source tensor whose axes feed the expression.")
    .set_attr<std::string>("TAttrs", "ShapeExprAttrs")
    .set_attr<FInferType>("FInferType", ShapeExprInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ShapeExprCompute);

KXC_REGISTER_OP(reshape_dynamic)
    .describe(R"doc(Reshape data using a verified restricted shape expression target.)doc")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "Tensor", "int64 shape value from a restricted chain.")
    .set_attr<std::string>("TAttrs", "ReshapeDynamicAttrs")
    .set_attr<FInferType>("FInferType", ReshapeDynamicInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ReshapeDynamicCompute);

KXC_REGISTER_OP(expand)
    .describe(R"doc(Broadcast data to a verified restricted shape expression target.)doc")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "Tensor", "int64 shape value from a restricted chain.")
    .set_attr<std::string>("TAttrs", "ExpandAttrs")
    .set_attr<FInferType>("FInferType", ExpandInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ExpandCompute);

KXC_REGISTER_OP(constant_of_shape)
    .describe(R"doc(Fill a constant target shape with an explicit scalar value.)doc")
    .set_num_inputs(1)
    .add_argument("shape", "Tensor", "int64 control shape tensor.")
    .set_attr<std::string>("TAttrs", "ConstantOfShapeAttrs")
    .set_attr<FInferType>("FInferType", ConstantOfShapeInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ConstantOfShapeCompute);

KXC_REGISTER_OP(squeeze)
    .describe(R"doc(Remove provably unit dimensions along explicit axes.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "SqueezeAttrs")
    .set_attr<FInferType>("FInferType", SqueezeInferType)
    .set_attr<FRelayToTE>("FRelayToTE", SqueezeCompute);

KXC_REGISTER_OP(unsqueeze)
    .describe(R"doc(Insert unit dimensions at explicit axes.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "UnsqueezeAttrs")
    .set_attr<FInferType>("FInferType", UnsqueezeInferType)
    .set_attr<FRelayToTE>("FRelayToTE", UnsqueezeCompute);

}  // namespace relay
}  // namespace kxc
