/*! \file src/relay/type_infer.cc
 * \brief 实现 Relay 算子共享的类型推导规则。
 */

#include "kxc/relay/type_infer.h"

#include "kxc/te/topi/window.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kxc {
namespace relay {

namespace {

// 校验算子输入数量。
void RequireArity(const std::string& op_name, const Array<Type>& input_types, size_t expected) {
    if (input_types.size() != expected) {
        throw std::runtime_error(op_name + " expects " + std::to_string(expected) +
                                 " input(s), got " + std::to_string(input_types.size()));
    }
}

// 将临时 shape vector 转为对象系统 Array。
Array<int64_t> ToArray(const std::vector<int64_t>& shape) {
    Array<int64_t> out;
    for (int64_t dim : shape) {
        out.push_back(dim);
    }
    return out;
}

// 从 TensorType 复制出便于算法处理的 shape vector。
std::vector<int64_t> ShapeVector(const TensorTypeNode* type) {
    std::vector<int64_t> out;
    out.reserve(type->shape.size());
    for (size_t i = 0; i < type->shape.size(); ++i) {
        out.push_back(type->shape[i]);
    }
    return out;
}

// 从临时 shape 与 dtype 构造 TensorType。
Type MakeTensorType(const std::vector<int64_t>& shape, const std::string& dtype) {
    return TensorType(ToArray(shape), dtype);
}

// 要求输入为 TensorType 并返回节点指针。
const TensorTypeNode* RequireTensor(const std::string& op_name, const Type& type,
                                    const std::string& input_name) {
    const auto* tensor = type.As<TensorTypeNode>();
    if (!tensor) {
        throw std::runtime_error(op_name + " expects TensorType for " + input_name +
                                 ", got " + TypeToString(type));
    }
    return tensor;
}

// 要求两个张量 dtype 完全一致。
void RequireSameDType(const std::string& op_name, const TensorTypeNode* lhs,
                      const TensorTypeNode* rhs) {
    if (lhs->dtype != rhs->dtype) {
        throw std::runtime_error(op_name + " dtype mismatch: " + lhs->dtype + " vs " +
                                 rhs->dtype);
    }
}

// 判断维度是否为已知非负值。
bool IsKnown(int64_t dim) { return dim >= 0; }

// 比较两个维度，未知维度视为可兼容。
bool SameOrUnknown(int64_t lhs, int64_t rhs) {
    return !IsKnown(lhs) || !IsKnown(rhs) || lhs == rhs;
}

// 计算 shape 区间乘积；零维使结果为 0，否则含未知维度时返回 -1。
int64_t KnownProduct(const std::vector<int64_t>& shape, size_t begin, size_t end) {
    bool has_zero = false;
    bool has_unknown = false;
    for (size_t i = begin; i < end; ++i) {
        has_zero = has_zero || shape[i] == 0;
        has_unknown = has_unknown || !IsKnown(shape[i]);
    }
    if (has_zero) {
        return 0;
    }
    if (has_unknown) {
        return -1;
    }

    int64_t product = 1;
    for (size_t i = begin; i < end; ++i) {
        if (product > std::numeric_limits<int64_t>::max() / shape[i]) {
            throw std::overflow_error("shape product overflows int64");
        }
        product *= shape[i];
    }
    return product;
}

// 按 NumPy 规则推导两个 shape 的广播结果。
std::vector<int64_t> BroadcastShape(const std::string& op_name,
                                    const std::vector<int64_t>& lhs,
                                    const std::vector<int64_t>& rhs) {
    const size_t rank = std::max(lhs.size(), rhs.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const size_t lhs_pos = lhs.size() + i;
        const size_t rhs_pos = rhs.size() + i;
        const int64_t ldim = lhs_pos >= rank ? lhs[lhs_pos - rank] : 1;
        const int64_t rdim = rhs_pos >= rank ? rhs[rhs_pos - rank] : 1;
        if (ldim == rdim) {
            out[i] = ldim;
        } else if (ldim == 1) {
            out[i] = rdim;
        } else if (rdim == 1) {
            out[i] = ldim;
        } else if (!IsKnown(ldim) || !IsKnown(rdim)) {
            out[i] = -1;
        } else {
            throw std::runtime_error(op_name + " cannot broadcast shapes " +
                                     TypeToString(TensorType(ToArray(lhs), "shape")) + " and " +
                                     TypeToString(TensorType(ToArray(rhs), "shape")));
        }
    }
    return out;
}

// 将可能为负的轴规范化到合法非负范围。
int NormalizeAxis(const std::string& op_name, int64_t axis, int rank, bool allow_end = false) {
    const int upper = allow_end ? rank : rank - 1;
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis > upper) {
        throw std::runtime_error(op_name + " axis out of range: " + std::to_string(axis) +
                                 " for rank " + std::to_string(rank));
    }
    return static_cast<int>(axis);
}

// 规范化轴集合，并拒绝重复轴。
std::vector<int> NormalizeAxes(const std::string& op_name, const Array<int64_t>& axes,
                               int rank) {
    std::vector<int> out;
    if (axes.empty()) {
        for (int i = 0; i < rank; ++i) {
            out.push_back(i);
        }
        return out;
    }
    for (int64_t axis : axes) {
        int normalized = NormalizeAxis(op_name, axis, rank);
        if (std::find(out.begin(), out.end(), normalized) != out.end()) {
            throw std::runtime_error(op_name + " duplicate axis: " + std::to_string(axis));
        }
        out.push_back(normalized);
    }
    return out;
}

// 读取可选向量分量。
int64_t ReadVectorValue(const Array<int64_t>& values, size_t index, int64_t default_value) {
    return index < values.size() ? values[index] : default_value;
}

// 推导卷积或池化窗口对应的单个输出维度。
int64_t WindowOutputDim(const std::string& op_name, int64_t input, int64_t kernel,
                        int64_t pad_before, int64_t pad_after, int64_t stride,
                        int64_t dilation, bool ceil_mode) {
    if (stride <= 0 || dilation <= 0) {
        throw std::runtime_error(op_name + " stride and dilation must be positive");
    }
    if (!IsKnown(input) || !IsKnown(kernel)) {
        return -1;
    }
    const int64_t effective_kernel = dilation * (kernel - 1) + 1;
    if (input + pad_before + pad_after - effective_kernel < 0) {
        throw std::runtime_error(op_name + " window is larger than input");
    }
    return te::topi::WindowOutputExtent<int64_t>(input, kernel, pad_before, pad_after, stride,
                                                 dilation, ceil_mode);
}

// 推导二元逐元素算子的广播 shape 和结果 dtype。
Type BinaryBroadcastInferType(const std::string& op_name, const Array<Type>& input_types,
                              const std::string& out_dtype = "") {
    RequireArity(op_name, input_types, 2);
    const auto* lhs = RequireTensor(op_name, input_types[0], "lhs");
    const auto* rhs = RequireTensor(op_name, input_types[1], "rhs");
    RequireSameDType(op_name, lhs, rhs);
    return MakeTensorType(BroadcastShape(op_name, ShapeVector(lhs), ShapeVector(rhs)),
                          out_dtype.empty() ? lhs->dtype : out_dtype);
}

// 将前端 cast 整数编码映射到 Relay dtype 名称。
std::string CastDTypeFromCode(int code) {
    switch (code) {
    case 0:
        return "float32";
    case 1:
        return "int32";
    case 2:
        return "int64";
    case 3:
        return "float64";
    case 4:
        return "bool";
    case 5:
        return "int8";
    case 6:
        return "uint8";
    default:
        throw std::runtime_error("cast has unsupported dtype code: " + std::to_string(code));
    }
}

// 在属性未指定 out_dtype 时继承输入 dtype。
std::string AttrOutDTypeOrDefault(const std::string& out_dtype,
                                  const std::string& default_dtype) {
    return out_dtype.empty() ? default_dtype : out_dtype;
}

}  // namespace

// 推导 add 的广播结果类型。
Type AddInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("add", input_types);
}

// 推导 subtract 的广播结果类型。
Type SubtractInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("subtract", input_types);
}

// 推导 multiply 的广播结果类型。
Type MultiplyInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("mul", input_types);
}

// 推导 divide 的广播结果类型。
Type DivideInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("divide", input_types);
}

// equal 已验证的同类型输入 dtype 集合；输出固定为 bool。
bool IsEqualInputDType(const std::string& dtype) {
    return dtype == "int32" || dtype == "int64" || dtype == "float32";
}

// 推导 equal 的广播结果类型；数值相等输出 bool，不支持位相等语义。
Type EqualInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("equal", input_types, 2);
    const auto* lhs = RequireTensor("equal", input_types[0], "lhs");
    const auto* rhs = RequireTensor("equal", input_types[1], "rhs");
    RequireSameDType("equal", lhs, rhs);
    if (!IsEqualInputDType(lhs->dtype)) {
        throw std::runtime_error(
            "equal supports same-dtype int32, int64, or float32 inputs, got " + lhs->dtype);
    }
    return MakeTensorType(BroadcastShape("equal", ShapeVector(lhs), ShapeVector(rhs)), "bool");
}

// M4/M5 静态子集共享：Neg/Sigmoid 是 float32-only 的 fieldless 一元算子，
// shape 与 dtype 原样保持；其余 dtype 必须在类型推导处立即失败。
Type UnaryFloat32InferType(const std::string& op_name, const Array<Type>& input_types) {
    RequireArity(op_name, input_types, 1);
    const auto* data = RequireTensor(op_name, input_types[0], "data");
    if (data->dtype != "float32") {
        throw std::runtime_error(op_name +
                                 " requires float32 input in the M4/M5 static subset, got " +
                                 data->dtype);
    }
    return input_types[0];
}

// 推导 neg 的 shape 保持类型；仅开放 M4/M5 静态子集声明的 float32。
Type NegInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return UnaryFloat32InferType("neg", input_types);
}

// 推导 sigmoid 的 shape 保持类型；仅开放 M4/M5 静态子集声明的 float32。
Type SigmoidInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return UnaryFloat32InferType("sigmoid", input_types);
}

Type TanhInferType(const Attrs& attrs, const Array<Type>& input_types) {
    if (attrs.defined()) throw std::runtime_error("tanh does not accept attrs");
    return UnaryFloat32InferType("tanh", input_types);
}

Type ErfInferType(const Attrs& attrs, const Array<Type>& input_types) {
    if (attrs.defined()) throw std::runtime_error("erf does not accept attrs");
    return UnaryFloat32InferType("erf", input_types);
}

// 推导 pow 的二元广播类型；M5 S2 子集只声明同 dtype float32 输入。
Type PowInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("pow", input_types, 2);
    const auto* lhs = RequireTensor("pow", input_types[0], "lhs");
    const auto* rhs = RequireTensor("pow", input_types[1], "rhs");
    RequireSameDType("pow", lhs, rhs);
    if (lhs->dtype != "float32") {
        throw std::runtime_error("pow requires same-dtype float32 inputs in the M4/M5 "
                                 "static subset, got " +
                                 lhs->dtype);
    }
    return MakeTensorType(BroadcastShape("pow", ShapeVector(lhs), ShapeVector(rhs)), "float32");
}

// 推导 expand 的单输入形式：目标 shape 是导入期已解析进 ExpandAttrs 的常量
// 控制输入；每个对齐后的 data 维度必须是 1 或等于目标维度（numpy
// broadcast_to 规则），输出 rank 等于目标 rank，dtype 保持。
Type ExpandInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("expand", input_types, 1);
    const auto* data = RequireTensor("expand", input_types[0], "data");
    const auto* expand_attrs = attrs.As<ExpandAttrsNode>();
    if (!expand_attrs) {
        throw std::runtime_error("expand static type inference requires ExpandAttrs");
    }
    std::vector<int64_t> target;
    target.reserve(expand_attrs->target_shape.size());
    for (int64_t dim : expand_attrs->target_shape) {
        if (dim < 0) {
            throw std::runtime_error(
                "expand target_shape must carry non-negative static dimensions");
        }
        target.push_back(dim);
    }
    if (data->shape.size() > target.size()) {
        throw std::runtime_error("expand data rank must not exceed the target rank");
    }
    const size_t offset = target.size() - data->shape.size();
    for (size_t i = 0; i < data->shape.size(); ++i) {
        const int64_t dim = data->shape[i];
        if (!IsKnown(dim)) {
            // 未知维度留给 backend；导入边界要求静态 shape 后才进入这里。
            continue;
        }
        if (dim != target[offset + i] && dim != 1) {
            throw std::runtime_error("expand data dimension " + std::to_string(dim) +
                                     " at axis " + std::to_string(i) +
                                     " must be 1 or equal to the target dimension " +
                                     std::to_string(target[offset + i]));
        }
    }
    return MakeTensorType(target, data->dtype);
}

bool IsWhereBranchDType(const std::string& dtype) {
    return dtype == "float32" || dtype == "float64" || dtype == "int32" ||
           dtype == "int64" || dtype == "int8" || dtype == "uint8" || dtype == "bool";
}

bool IsConcatenateDType(const std::string& dtype) {
    return IsWhereBranchDType(dtype);
}

int64_t ClampPositiveStepEndpoint(int64_t endpoint, int64_t dim) {
    if (endpoint < 0) {
        return endpoint < -dim ? 0 : endpoint + dim;
    }
    return std::min(endpoint, dim);
}

// 推导 ONNX Where 的三元 trailing-axis 广播结果。
Type WhereInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("where", input_types, 3);
    const auto* condition = RequireTensor("where", input_types[0], "condition");
    const auto* x = RequireTensor("where", input_types[1], "x");
    const auto* y = RequireTensor("where", input_types[2], "y");
    if (condition->dtype != "bool") {
        throw std::runtime_error("where condition dtype must be bool");
    }
    if (!IsWhereBranchDType(x->dtype) || !IsWhereBranchDType(y->dtype)) {
        throw std::runtime_error(
            "where branch dtypes must be float32, float64, int32, int64, int8, uint8, or bool");
    }
    RequireSameDType("where", x, y);
    const std::vector<int64_t> condition_x =
        BroadcastShape("where", ShapeVector(condition), ShapeVector(x));
    return MakeTensorType(
        BroadcastShape("where", condition_x, ShapeVector(y)), x->dtype);
}

// 推导保持 shape 与 dtype 的一元算子类型。
Type UnarySameInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("unary", input_types, 1);
    RequireTensor("unary", input_types[0], "data");
    return input_types[0];
}

// 推导 cast 的 shape 保持和目标 dtype。
Type CastInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("cast", input_types, 1);
    const auto* data = RequireTensor("cast", input_types[0], "data");
    const auto* cast_attrs = attrs.As<CastAttrsNode>();
    const std::string dtype = cast_attrs ? CastDTypeFromCode(cast_attrs->to) : data->dtype;
    return MakeTensorType(ShapeVector(data), dtype);
}

// 推导批量矩阵乘的广播前缀与末两维。
Type MatMulInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("matmul", input_types, 2);
    const auto* lhs = RequireTensor("matmul", input_types[0], "a");
    const auto* rhs = RequireTensor("matmul", input_types[1], "b");
    RequireSameDType("matmul", lhs, rhs);

    const std::vector<int64_t> a = ShapeVector(lhs);
    const std::vector<int64_t> b = ShapeVector(rhs);
    if (a.size() < 2 || b.size() < 2) {
        throw std::runtime_error("matmul expects rank >= 2 inputs");
    }
    const int64_t k_a = a[a.size() - 1];
    const int64_t k_b = b[b.size() - 2];
    if (!SameOrUnknown(k_a, k_b)) {
        throw std::runtime_error("matmul reduction dimension mismatch");
    }

    std::vector<int64_t> a_batch(a.begin(), a.end() - 2);
    std::vector<int64_t> b_batch(b.begin(), b.end() - 2);
    std::vector<int64_t> out = BroadcastShape("matmul batch", a_batch, b_batch);
    out.push_back(a[a.size() - 2]);
    out.push_back(b[b.size() - 1]);
    return MakeTensorType(out, lhs->dtype);
}

// 推导 dense 的批维和 units 输出维度。
Type DenseInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("nn_dense", input_types, 2);
    const auto* data = RequireTensor("nn_dense", input_types[0], "data");
    const auto* weight = RequireTensor("nn_dense", input_types[1], "weight");
    RequireSameDType("nn_dense", data, weight);
    if (data->shape.size() != 2 || weight->shape.size() != 2) {
        throw std::runtime_error("nn_dense expects rank-2 data and weight");
    }
    if (!SameOrUnknown(data->shape[1], weight->shape[1])) {
        throw std::runtime_error("nn_dense input dimension mismatch");
    }
    const auto* dense_attrs = attrs.As<DenseAttrsNode>();
    const int64_t units = dense_attrs && dense_attrs->units > 0 ? dense_attrs->units : weight->shape[0];
    const std::string dtype =
        dense_attrs ? AttrOutDTypeOrDefault(dense_attrs->out_dtype, data->dtype) : data->dtype;
    return MakeTensorType({data->shape[0], units}, dtype);
}

// 推导 GEMM 转置、广播 bias 和输出 dtype。
Type GemmInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("nn_gemm", input_types, 3);
    const auto* a_type = RequireTensor("nn_gemm", input_types[0], "A");
    const auto* b_type = RequireTensor("nn_gemm", input_types[1], "B");
    const auto* c_type = RequireTensor("nn_gemm", input_types[2], "C");
    RequireSameDType("nn_gemm", a_type, b_type);
    RequireSameDType("nn_gemm", a_type, c_type);
    if (a_type->shape.size() != 2 || b_type->shape.size() != 2) {
        throw std::runtime_error("nn_gemm expects rank-2 A and B");
    }
    const auto* gemm_attrs = attrs.As<GemmAttrsNode>();
    const bool trans_a = gemm_attrs && gemm_attrs->transA != 0;
    const bool trans_b = gemm_attrs && gemm_attrs->transB != 0;

    const int64_t m = trans_a ? a_type->shape[1] : a_type->shape[0];
    const int64_t k_a = trans_a ? a_type->shape[0] : a_type->shape[1];
    const int64_t k_b = trans_b ? b_type->shape[1] : b_type->shape[0];
    const int64_t n = trans_b ? b_type->shape[0] : b_type->shape[1];
    if (!SameOrUnknown(k_a, k_b)) {
        throw std::runtime_error("nn_gemm reduction dimension mismatch");
    }

    const std::vector<int64_t> out_shape = {m, n};
    BroadcastShape("nn_gemm bias", ShapeVector(c_type), out_shape);
    return MakeTensorType(out_shape, a_type->dtype);
}

// 推导 NCHW/OIHW 卷积输出 shape 与 dtype。
Type Conv2DInferType(const Attrs& attrs, const Array<Type>& input_types) {
    if (input_types.size() != 2 && input_types.size() != 3) {
        throw std::runtime_error("nn_conv2d expects data, weight[, bias]");
    }
    const auto* data = RequireTensor("nn_conv2d", input_types[0], "data");
    const auto* weight = RequireTensor("nn_conv2d", input_types[1], "weight");
    RequireSameDType("nn_conv2d", data, weight);
    if (data->shape.size() != 4 || weight->shape.size() != 4) {
        throw std::runtime_error("nn_conv2d expects NCHW/OIHW rank-4 tensors");
    }
    const auto* conv_attrs = attrs.As<Conv2DAttrsNode>();
    if (conv_attrs && !conv_attrs->data_layout.empty() && conv_attrs->data_layout != "NCHW") {
        throw std::runtime_error("nn_conv2d only supports NCHW data layout in type inference");
    }
    if (conv_attrs && !conv_attrs->kernel_layout.empty() && conv_attrs->kernel_layout != "OIHW") {
        throw std::runtime_error("nn_conv2d only supports OIHW kernel layout in type inference");
    }

    const int64_t groups = conv_attrs ? std::max<int64_t>(1, conv_attrs->groups) : 1;
    if (IsKnown(data->shape[1]) && IsKnown(weight->shape[1]) &&
        data->shape[1] != weight->shape[1] * groups) {
        throw std::runtime_error("nn_conv2d input channel mismatch");
    }

    const te::topi::AxisPair2D strides =
        conv_attrs ? te::topi::ExpandPair2D(conv_attrs->strides, 1) : te::topi::AxisPair2D{};
    const te::topi::AxisPair2D dilation =
        conv_attrs ? te::topi::ExpandPair2D(conv_attrs->dilation, 1) : te::topi::AxisPair2D{};
    const te::topi::Padding2D padding =
        conv_attrs ? te::topi::ExpandPadding2D(conv_attrs->padding) : te::topi::Padding2D{};
    const int64_t kh = conv_attrs && !conv_attrs->kernel_size.empty()
                           ? ReadVectorValue(conv_attrs->kernel_size, 0, weight->shape[2])
                           : weight->shape[2];
    const int64_t kw = conv_attrs && !conv_attrs->kernel_size.empty()
                           ? ReadVectorValue(conv_attrs->kernel_size, 1, weight->shape[3])
                           : weight->shape[3];
    if (!SameOrUnknown(kh, weight->shape[2]) || !SameOrUnknown(kw, weight->shape[3])) {
        throw std::runtime_error("nn_conv2d kernel_size does not match weight shape");
    }

    const int64_t channels =
        conv_attrs && conv_attrs->channels > 0 ? conv_attrs->channels : weight->shape[0];
    if (input_types.size() == 3) {
        const auto* bias = RequireTensor("nn_conv2d", input_types[2], "bias");
        RequireSameDType("nn_conv2d", data, bias);
        if (bias->shape.size() != 1 || !SameOrUnknown(bias->shape[0], channels)) {
            throw std::runtime_error("nn_conv2d bias must be rank-1 with output channels");
        }
    }

    const int64_t oh = WindowOutputDim("nn_conv2d", data->shape[2], kh, padding.top, padding.bottom,
                                      strides.h, dilation.h, false);
    const int64_t ow = WindowOutputDim("nn_conv2d", data->shape[3], kw, padding.left, padding.right,
                                      strides.w, dilation.w, false);
    const std::string dtype =
        conv_attrs ? AttrOutDTypeOrDefault(conv_attrs->out_dtype, data->dtype) : data->dtype;
    return MakeTensorType({data->shape[0], channels, oh, ow}, dtype);
}

// 推导二维池化输出空间维度。
Type Pool2DInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("pool2d", input_types, 1);
    const auto* data = RequireTensor("pool2d", input_types[0], "data");
    if (data->shape.size() != 4) {
        throw std::runtime_error("pool2d expects NCHW rank-4 input");
    }
    const auto* pool_attrs = attrs.As<MaxPool2DAttrsNode>();
    if (pool_attrs && !pool_attrs->layout.empty() && pool_attrs->layout != "NCHW") {
        throw std::runtime_error("pool2d only supports NCHW layout in type inference");
    }
    const te::topi::AxisPair2D pool_size =
        pool_attrs ? te::topi::ExpandPair2D(pool_attrs->pool_size, 1) : te::topi::AxisPair2D{};
    const te::topi::AxisPair2D strides =
        pool_attrs ? te::topi::ExpandPair2D(pool_attrs->strides, 1) : te::topi::AxisPair2D{};
    const te::topi::AxisPair2D dilation =
        pool_attrs ? te::topi::ExpandPair2D(pool_attrs->dilation, 1) : te::topi::AxisPair2D{};
    const te::topi::Padding2D padding =
        pool_attrs ? te::topi::ExpandPadding2D(pool_attrs->padding) : te::topi::Padding2D{};
    const bool ceil_mode = pool_attrs && pool_attrs->ceil_mode;

    const int64_t oh = WindowOutputDim("pool2d", data->shape[2], pool_size.h, padding.top,
                                      padding.bottom, strides.h, dilation.h, ceil_mode);
    const int64_t ow = WindowOutputDim("pool2d", data->shape[3], pool_size.w, padding.left,
                                      padding.right, strides.w, dilation.w, ceil_mode);
    return MakeTensorType({data->shape[0], data->shape[1], oh, ow}, data->dtype);
}

// 推导全局平均池化的 NCHW 1x1 输出。
Type GlobalAvgPool2DInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("nn_global_avg_pool2d", input_types, 1);
    const auto* data = RequireTensor("nn_global_avg_pool2d", input_types[0], "data");
    if (data->shape.size() != 4) {
        throw std::runtime_error("nn_global_avg_pool2d expects NCHW rank-4 input");
    }
    return MakeTensorType({data->shape[0], data->shape[1], 1, 1}, data->dtype);
}

// 按 axis 合并 flatten 两侧维度。
Type FlattenInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("nn_flatten", input_types, 1);
    const auto* data = RequireTensor("nn_flatten", input_types[0], "data");
    if (data->shape.empty()) {
        throw std::runtime_error("nn_flatten expects rank >= 1");
    }
    const auto* flatten_attrs = attrs.As<FlattenAttrsNode>();
    const int axis = NormalizeAxis("nn_flatten", flatten_attrs ? flatten_attrs->axis : 1,
                                   static_cast<int>(data->shape.size()), true);
    const std::vector<int64_t> shape = ShapeVector(data);
    return MakeTensorType({KnownProduct(shape, 0, axis),
                           KnownProduct(shape, axis, shape.size())},
                          data->dtype);
}

// shape_of 的固定向量长度上限：Select 链与受限符号集都按此声明子集。
constexpr int kMaxShapeOfRank = 8;

// 推导 shape_of 的 int64[rank(data)] 输出；未知 rank 与超界 rank 拒绝。
Type ShapeOfInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("shape_of", input_types, 1);
    const auto* data = RequireTensor("shape_of", input_types[0], "data");
    const int64_t rank = static_cast<int64_t>(data->shape.size());
    if (rank < 1) {
        throw std::runtime_error("shape_of requires data rank >= 1");
    }
    if (rank > kMaxShapeOfRank) {
        throw std::runtime_error(
            "shape_of supports fixed rank up to " + std::to_string(kMaxShapeOfRank) +
            ", got rank " + std::to_string(rank));
    }
    return MakeTensorType({rank}, "int64");
}

namespace {

// 受限形状表达式元素：kind 0 常量 / kind 1 数据输入轴引用。
struct ShapeExprElement {
    int64_t kind;
    int64_t value;
    int64_t axis;
};

// 读取并校验受限形状表达式三元组（expr_kinds/expr_values/expr_axes）。
std::vector<ShapeExprElement> ReadShapeExpr(const std::string& op_name,
                                            const Array<int64_t>& kinds,
                                            const Array<int64_t>& values,
                                            const Array<int64_t>& axes) {
    if (kinds.empty() || values.size() != kinds.size() || axes.size() != kinds.size()) {
        throw std::runtime_error(
            op_name + " requires a resolved restricted shape expression "
            "(expr_kinds/expr_values/expr_axes); static admission has no proof");
    }
    std::vector<ShapeExprElement> elements;
    elements.reserve(kinds.size());
    for (size_t i = 0; i < kinds.size(); ++i) {
        if (kinds[i] != kShapeExprKindConst && kinds[i] != kShapeExprKindInputAxis && kinds[i] != kShapeExprKindInputAxisOffset) {
            throw std::runtime_error(op_name + " shape expression kind must be 0, 1 or 2");
        }
        elements.push_back(ShapeExprElement{kinds[i], values[i], axes[i]});
    }
    return elements;
}

// 求值受限形状表达式：kind 1 引用第 0 个输入（数据）轴；未知维透传 -1。
std::vector<int64_t> EvaluateShapeExpr(const std::string& op_name,
                                       const std::vector<ShapeExprElement>& elements,
                                       const TensorTypeNode* data) {
    const int64_t data_rank = static_cast<int64_t>(data->shape.size());
    std::vector<int64_t> dims;
    dims.reserve(elements.size());
    for (const ShapeExprElement& element : elements) {
        if (element.kind == kShapeExprKindConst) {
            if (element.value < 0) {
                throw std::runtime_error(
                    op_name + " shape expression constant must be >= 0; unresolved "
                    "-1 inference is not part of the restricted subset");
            }
            dims.push_back(element.value);
            continue;
        }
        if ((element.kind == kShapeExprKindInputAxis && element.value != 0) || element.value < 0) {
            throw std::runtime_error(op_name + " shape expression axis/offset metadata is invalid");
        }
        if (element.axis < 0 || element.axis >= data_rank) {
            throw std::runtime_error(op_name + " shape expression axis out of range");
        }
        const int64_t extent = data->shape[static_cast<size_t>(element.axis)];
        if (extent < -1 || (extent >= 0 && extent > std::numeric_limits<int64_t>::max() - element.value)) {
            throw std::runtime_error(op_name + " shape expression extent overflows int64");
        }
        dims.push_back(extent < 0 ? -1 : extent + element.value);
    }
    return dims;
}

// 校验元素总数（已知时），未知维（-1）跳过。
void RequireSameElementCount(const std::string& op_name,
                             const std::vector<int64_t>& data_dims,
                             const std::vector<int64_t>& target_dims) {
    const int64_t data_product = KnownProduct(data_dims, 0, data_dims.size());
    const int64_t target_product = KnownProduct(target_dims, 0, target_dims.size());
    if (data_product < 0 || target_product < 0) {
        return;  // 边界 -1 维：符号等值证明由受限准备路径负责。
    }
    if (data_product != target_product) {
        throw std::runtime_error(op_name + " element count mismatch");
    }
}

}  // namespace

// 推导 shape_expr 的 int64[len(expr)] 输出；表达式经受限准备验证。
Type ShapeExprInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("shape_expr", input_types, 1);
    const auto* source = RequireTensor("shape_expr", input_types[0], "data");
    const auto* shape_expr_attrs = attrs.As<ShapeExprAttrsNode>();
    if (!shape_expr_attrs) {
        throw std::runtime_error(
            "shape_expr requires ShapeExprAttrs; unrestricted use is rejected");
    }
    const std::vector<ShapeExprElement> elements = ReadShapeExpr(
        "shape_expr", shape_expr_attrs->expr_kinds, shape_expr_attrs->expr_values,
        shape_expr_attrs->expr_axes);
    (void)EvaluateShapeExpr("shape_expr",elements,source);
    return MakeTensorType({static_cast<int64_t>(elements.size())}, "int64");
}

// 推导 reshape_dynamic 的目标形状：目标表达式必须已被受限准备解析。
Type ReshapeDynamicInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("reshape_dynamic", input_types, 2);
    const auto* data = RequireTensor("reshape_dynamic", input_types[0], "data");
    const auto* control = RequireTensor("reshape_dynamic", input_types[1], "shape");
    if (control->dtype != "int64" || control->shape.size() != 1 ||
        !IsKnown(control->shape[0])) {
        throw std::runtime_error(
            "reshape_dynamic control must be a known-length int64 rank-1 tensor");
    }
    const auto* reshape_attrs = attrs.As<ReshapeDynamicAttrsNode>();
    if (!reshape_attrs) {
        throw std::runtime_error(
            "reshape_dynamic requires ReshapeDynamicAttrs; unrestricted use is rejected");
    }
    const std::vector<ShapeExprElement> elements = ReadShapeExpr(
        "reshape_dynamic", reshape_attrs->expr_kinds, reshape_attrs->expr_values,
        reshape_attrs->expr_axes);
    if (elements.size() != static_cast<size_t>(control->shape[0])) {
        throw std::runtime_error(
            "reshape_dynamic target expression length differs from its control tensor");
    }
    const std::vector<int64_t> target = EvaluateShapeExpr("reshape_dynamic", elements, data);
    if (target.empty()) {
        throw std::runtime_error("reshape_dynamic requires a rank >= 1 target");
    }
    RequireSameElementCount("reshape_dynamic", ShapeVector(data), target);
    return MakeTensorType(target, data->dtype);
}

// 推导 expand 的受限目标形状与广播合法性。
Type ExpandDynamicInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("expand_dynamic", input_types, 2);
    const auto* data = RequireTensor("expand_dynamic", input_types[0], "data");
    const auto* control = RequireTensor("expand_dynamic", input_types[1], "shape");
    if (control->dtype != "int64" || control->shape.size() != 1 ||
        !IsKnown(control->shape[0])) {
        throw std::runtime_error(
            "expand control must be a known-length int64 rank-1 tensor");
    }
    const auto* expand_attrs = attrs.As<ExpandDynamicAttrsNode>();
    if (!expand_attrs) {
        throw std::runtime_error(
            "expand requires ExpandDynamicAttrs; unrestricted use is rejected");
    }
    const std::vector<ShapeExprElement> elements = ReadShapeExpr(
        "expand_dynamic", expand_attrs->expr_kinds, expand_attrs->expr_values,
        expand_attrs->expr_axes);
    if (elements.size() != static_cast<size_t>(control->shape[0]) ||
        elements.size() != data->shape.size()) {
        throw std::runtime_error(
            "expand target length must equal both its control tensor and data rank");
    }
    const std::vector<int64_t> target = EvaluateShapeExpr("expand_dynamic", elements, data);
    for (size_t axis = 0; axis < target.size(); ++axis) {
        const int64_t data_dim = data->shape[axis];
        const int64_t target_dim = target[axis];
        if (!IsKnown(data_dim) || !IsKnown(target_dim)) {
            continue;  // 未知维兼容性由运行期合同校验。
        }
        if (data_dim != target_dim && data_dim != 1) {
            throw std::runtime_error(
                "expand cannot broadcast data axis " + std::to_string(axis) +
                " (extent " + std::to_string(data_dim) + ") to " +
                std::to_string(target_dim));
        }
    }
    return MakeTensorType(target, data->dtype);
}

// The two-input form is minted by restricted preparation: data anchors each
// runtime extent, while shape remains the original checked control tensor.
Type ConstantOfShapeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    if (input_types.size() != 1 && input_types.size() != 2) {
        throw std::runtime_error("constant_of_shape expects shape or data+shape");
    }
    const auto* control = RequireTensor("constant_of_shape", input_types[input_types.size() - 1], "shape");
    if (control->dtype != "int64" || control->shape.size() != 1 ||
        !IsKnown(control->shape[0])) {
        throw std::runtime_error(
            "constant_of_shape control must be a known-length int64 rank-1 tensor");
    }
    const auto* constant_attrs = attrs.As<ConstantOfShapeAttrsNode>();
    if (!constant_attrs) {
        throw std::runtime_error(
            "constant_of_shape requires ConstantOfShapeAttrs with an explicit "
            "constant target; data-dependent shapes are rejected");
    }
    std::vector<int64_t> target;
    if (input_types.size() == 2) {
        if (!constant_attrs->target.empty() || constant_attrs->dtype_code != 0) {
            throw std::runtime_error("constant_of_shape dynamic fill requires float32 and no static target");
        }
        const auto* data = RequireTensor("constant_of_shape", input_types[0], "data");
        target = EvaluateShapeExpr("constant_of_shape", ReadShapeExpr("constant_of_shape",
            constant_attrs->expr_kinds, constant_attrs->expr_values, constant_attrs->expr_axes), data);
    } else {
        if (!constant_attrs->expr_kinds.empty() || !constant_attrs->expr_values.empty() ||
            !constant_attrs->expr_axes.empty()) {
            throw std::runtime_error("constant_of_shape expression needs an explicit data anchor");
        }
        target.assign(constant_attrs->target.begin(), constant_attrs->target.end());
    }
    if (target.empty() || target.size() != static_cast<size_t>(control->shape[0])) {
        throw std::runtime_error(
            "constant_of_shape target must be nonempty and match its control length");
    }
    if (target.size() > static_cast<size_t>(kMaxShapeOfRank)) {
        throw std::runtime_error(
            "constant_of_shape supports rank up to " + std::to_string(kMaxShapeOfRank));
    }
    for (int64_t dim : target) {
        if (dim < 0 && !(input_types.size() == 2 && dim == -1)) {
            throw std::runtime_error("constant_of_shape target dimensions must be >= 0");
        }
    }
    const std::string dtype = CastDTypeFromCode(constant_attrs->dtype_code);
    const double value = constant_attrs->value;
    if (std::isnan(value) || (!std::isfinite(value) && dtype != "float32")) {
        throw std::runtime_error("constant_of_shape accepts infinities only for float32 and rejects NaN");
    }
    if ((dtype == "int32" || dtype == "int64" || dtype == "int8" || dtype == "uint8") &&
        std::trunc(value) != value) {
        throw std::runtime_error(
            "constant_of_shape integer fill value must be integral");
    }
    int64_t elements = 1;
    for (int64_t dim : target) {
        if (dim == -1) continue;  // Full bounds are proved by restricted preparation.
        if (dim != 0 &&
            elements > std::numeric_limits<int64_t>::max() / dim) {
            throw std::overflow_error("constant_of_shape target element count overflows int64");
        }
        elements *= dim == 0 ? 0 : dim;
    }
    const int64_t item_bytes = dtype == "bool" || dtype == "int8" || dtype == "uint8"
                                   ? 1
                                   : (dtype == "float32" || dtype == "int32" ? 4 : 8);
    if (elements > kMaxConstantOfShapeBytes / item_bytes) {
        throw std::runtime_error(
            "constant_of_shape output exceeds the declared byte cap");
    }
    return MakeTensorType(target, dtype);
}

Type TriluInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("trilu", input_types, 1);
    const auto* data = RequireTensor("trilu", input_types[0], "data");
    const auto* triangle = attrs.As<TriluAttrsNode>();
    if (!triangle || (triangle->upper != 0 && triangle->upper != 1) ||
        data->dtype != "float32" || data->shape.size() < 2) {
        throw std::runtime_error("trilu requires float32 rank >= 2 and upper 0 or 1");
    }
    for (int64_t dim : data->shape) {
        if (dim < -1) throw std::runtime_error("trilu has an invalid dimension");
    }
    return input_types[0];
}

// 推导 squeeze 的移除轴结果；被移除维必须静态已知为 1。
Type SqueezeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("squeeze", input_types, 1);
    const auto* data = RequireTensor("squeeze", input_types[0], "data");
    const auto* squeeze_attrs = attrs.As<SqueezeAttrsNode>();
    if (!squeeze_attrs || squeeze_attrs->axes.empty()) {
        throw std::runtime_error(
            "squeeze requires explicit axes; omit-axes cannot prove fixed rank here");
    }
    const int rank = static_cast<int>(data->shape.size());
    if (rank < 1) {
        throw std::runtime_error("squeeze requires data rank >= 1");
    }
    std::vector<bool> removed(static_cast<size_t>(rank), false);
    for (int64_t raw_axis : squeeze_attrs->axes) {
        const int axis = NormalizeAxis("squeeze", raw_axis, rank);
        if (removed[static_cast<size_t>(axis)]) {
            throw std::runtime_error("squeeze duplicate axis");
        }
        removed[static_cast<size_t>(axis)] = true;
        const int64_t dim = data->shape[static_cast<size_t>(axis)];
        if (dim != 1) {
            throw std::runtime_error(
                "squeeze requires each removed axis to be provably 1, got " +
                (dim < 0 ? std::string("unknown") : std::to_string(dim)));
        }
    }
    std::vector<int64_t> out;
    for (int axis = 0; axis < rank; ++axis) {
        if (!removed[static_cast<size_t>(axis)]) {
            out.push_back(data->shape[static_cast<size_t>(axis)]);
        }
    }
    if (out.empty()) {
        throw std::runtime_error("squeeze must keep at least one axis");
    }
    return MakeTensorType(out, data->dtype);
}

// 推导 unsqueeze 的插入轴结果；插入 1 维，结果 rank 在声明子集内。
Type UnsqueezeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("unsqueeze", input_types, 1);
    const auto* data = RequireTensor("unsqueeze", input_types[0], "data");
    const auto* unsqueeze_attrs = attrs.As<UnsqueezeAttrsNode>();
    if (!unsqueeze_attrs || unsqueeze_attrs->axes.empty()) {
        throw std::runtime_error("unsqueeze requires explicit axes");
    }
    const int64_t input_rank = static_cast<int64_t>(data->shape.size());
    const int64_t new_rank = input_rank + static_cast<int64_t>(unsqueeze_attrs->axes.size());
    if (new_rank > kMaxShapeOfRank) {
        throw std::runtime_error(
            "unsqueeze supports output rank up to " + std::to_string(kMaxShapeOfRank));
    }
    std::vector<int64_t> out(static_cast<size_t>(new_rank), -1);
    std::vector<bool> placed(static_cast<size_t>(new_rank), false);
    for (int64_t raw_axis : unsqueeze_attrs->axes) {
        int64_t axis = raw_axis < 0 ? raw_axis + new_rank : raw_axis;
        if (axis < 0 || axis >= new_rank) {
            throw std::runtime_error(
                "unsqueeze axis out of range for the provable output rank");
        }
        if (placed[static_cast<size_t>(axis)]) {
            throw std::runtime_error("unsqueeze duplicate axis");
        }
        placed[static_cast<size_t>(axis)] = true;
        out[static_cast<size_t>(axis)] = 1;
    }
    // 原始维按未放置位置依次填入，保持 ONNX unsqueeze 的顺序语义。
    size_t next = 0;
    for (size_t axis = 0; axis < out.size(); ++axis) {
        if (placed[axis]) {
            continue;
        }
        out[axis] = data->shape[next++];
    }
    return MakeTensorType(out, data->dtype);
}

// 解释 0、-1 与 allowzero 后推导 reshape 结果。
Type ReshapeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("reshape", input_types, 1);
    const auto* data = RequireTensor("reshape", input_types[0], "data");
    const auto* reshape_attrs = attrs.As<ReshapeAttrsNode>();
    if (!reshape_attrs) {
        throw std::runtime_error("reshape static type inference requires ReshapeAttrs");
    }
    const std::vector<int64_t> input_shape = ShapeVector(data);
    const int64_t input_product = KnownProduct(input_shape, 0, input_shape.size());
    std::vector<int64_t> out;
    std::vector<int64_t> known_shape;
    out.reserve(reshape_attrs->newshape.size());
    known_shape.reserve(reshape_attrs->newshape.size());

    int infer_index = -1;
    for (size_t i = 0; i < reshape_attrs->newshape.size(); ++i) {
        int64_t dim = reshape_attrs->newshape[i];
        if (dim > 0) {
            out.push_back(dim);
            known_shape.push_back(dim);
        } else if (dim == 0) {
            if (reshape_attrs->allowzero) {
                out.push_back(0);
                known_shape.push_back(0);
            } else {
                if (i >= input_shape.size()) {
                    throw std::runtime_error("reshape 0-dim copy index exceeds input rank");
                }
                out.push_back(input_shape[i]);
                known_shape.push_back(input_shape[i]);
            }
        } else if (dim == -1) {
            if (infer_index >= 0) {
                throw std::runtime_error("reshape allows at most one inferred -1 dimension");
            }
            infer_index = static_cast<int>(out.size());
            out.push_back(-1);
        } else {
            throw std::runtime_error("reshape only supports positive, 0, and -1 dimensions");
        }
    }
    const int64_t known_product = KnownProduct(known_shape, 0, known_shape.size());

    if (infer_index >= 0 && input_product >= 0 && known_product > 0) {
        if (input_product % known_product != 0) {
            throw std::runtime_error("reshape cannot infer -1 dimension from element count");
        }
        out[static_cast<size_t>(infer_index)] = input_product / known_product;
    } else if (infer_index < 0 && input_product >= 0 && known_product >= 0 &&
               input_product != known_product) {
        throw std::runtime_error("reshape element count mismatch");
    }
    return MakeTensorType(out, data->dtype);
}

// 按 perm 或逆序默认规则推导 transpose shape。
Type TransposeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("transpose", input_types, 1);
    const auto* data = RequireTensor("transpose", input_types[0], "data");
    const int rank = static_cast<int>(data->shape.size());
    std::vector<int64_t> perm;
    if (const auto* transpose_attrs = attrs.As<TransposeAttrsNode>()) {
        perm.assign(transpose_attrs->perm.begin(), transpose_attrs->perm.end());
    }
    if (perm.empty()) {
        for (int i = rank - 1; i >= 0; --i) {
            perm.push_back(i);
        }
    }
    if (static_cast<int>(perm.size()) != rank) {
        throw std::runtime_error("transpose perm rank mismatch");
    }
    std::vector<bool> seen(static_cast<size_t>(rank), false);
    std::vector<int64_t> out;
    out.reserve(perm.size());
    for (int64_t axis : perm) {
        const int normalized = NormalizeAxis("transpose", axis, rank);
        if (seen[static_cast<size_t>(normalized)]) {
            throw std::runtime_error("transpose duplicate axis");
        }
        seen[static_cast<size_t>(normalized)] = true;
        out.push_back(data->shape[static_cast<size_t>(normalized)]);
    }
    return MakeTensorType(out, data->dtype);
}

// 按 ONNX Gather 规则替换 data.axis，并验证静态索引类型边界。
Type GatherInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("gather", input_types, 2);
    const auto* data = RequireTensor("gather", input_types[0], "data");
    const auto* indices = RequireTensor("gather", input_types[1], "indices");
    if (data->shape.empty()) {
        throw std::runtime_error("gather requires data rank >= 1");
    }
    if (indices->dtype != "int32" && indices->dtype != "int64") {
        throw std::runtime_error("gather indices dtype must be int32 or int64");
    }
    const auto* gather_attrs = attrs.As<GatherAttrsNode>();
    if (!gather_attrs) {
        throw std::runtime_error("gather requires GatherAttrs");
    }
    const int axis = NormalizeAxis("gather", gather_attrs->axis,
                                   static_cast<int>(data->shape.size()));
    const int64_t extent = data->shape[static_cast<size_t>(axis)];
    if (indices->dtype == "int32" && IsKnown(extent) &&
        extent > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("gather int32 indices cannot address axis extent > INT32_MAX");
    }
    std::vector<int64_t> out;
    out.reserve(data->shape.size() + indices->shape.size() - 1);
    for (int i = 0; i < axis; ++i) out.push_back(data->shape[static_cast<size_t>(i)]);
    for (int64_t dim : indices->shape) out.push_back(dim);
    for (size_t i = static_cast<size_t>(axis) + 1; i < data->shape.size(); ++i) {
        out.push_back(data->shape[i]);
    }
    return MakeTensorType(out, data->dtype);
}

// 推导 exact-static binary concatenate 的输出 shape 与 dtype。
Type ConcatenateInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("concatenate", input_types, 2);
    const auto* lhs = RequireTensor("concatenate", input_types[0], "lhs");
    const auto* rhs = RequireTensor("concatenate", input_types[1], "rhs");
    const auto* concatenate_attrs = attrs.As<ConcatenateAttrsNode>();
    if (!concatenate_attrs) {
        throw std::runtime_error("concatenate requires ConcatenateAttrs");
    }
    if (!IsConcatenateDType(lhs->dtype) || !IsConcatenateDType(rhs->dtype)) {
        throw std::runtime_error(
            "concatenate dtype must be float32, float64, int32, int64, int8, uint8, or bool");
    }
    RequireSameDType("concatenate", lhs, rhs);
    if (lhs->shape.empty() || rhs->shape.empty()) {
        throw std::runtime_error("concatenate requires rank >= 1 inputs");
    }
    if (lhs->shape.size() != rhs->shape.size()) {
        throw std::runtime_error("concatenate input rank mismatch");
    }
    const int rank = static_cast<int>(lhs->shape.size());
    const int axis = NormalizeAxis("concatenate", concatenate_attrs->axis, rank);
    std::vector<int64_t> out = ShapeVector(lhs);
    for (int index = 0; index < rank; ++index) {
        const int64_t lhs_dim = lhs->shape[static_cast<size_t>(index)];
        const int64_t rhs_dim = rhs->shape[static_cast<size_t>(index)];
        if (lhs_dim < -1 || rhs_dim < -1 ||
            (index == axis && lhs_dim < 0 && rhs_dim < 0)) {
            throw std::runtime_error("concatenate requires at least one static concatenation axis and valid input dimensions");
        }
        if (index != axis && lhs_dim != rhs_dim) {
            throw std::runtime_error("concatenate non-axis dimensions must exactly match");
        }
    }
    const int64_t lhs_axis = lhs->shape[static_cast<size_t>(axis)];
    const int64_t rhs_axis = rhs->shape[static_cast<size_t>(axis)];
    if (lhs_axis < 0 || rhs_axis < 0) {
        out[static_cast<size_t>(axis)] = -1;
        return MakeTensorType(out, lhs->dtype);
    }
    if (lhs_axis > std::numeric_limits<int64_t>::max() - rhs_axis) {
        throw std::runtime_error("concatenate axis extent sum overflows int64");
    }
    out[static_cast<size_t>(axis)] = lhs_axis + rhs_axis;
    return MakeTensorType(out, lhs->dtype);
}

// 推导静态、常量分段的 Split；输出路数由 sections 明确给出，避免运行时猜测 ABI。
Type SplitInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("split", input_types, 1);
    const auto* data = RequireTensor("split", input_types[0], "data");
    const auto* split_attrs = attrs.As<SplitAttrsNode>();
    if (!split_attrs) {
        throw std::runtime_error("split requires SplitAttrs");
    }
    if (!IsConcatenateDType(data->dtype)) {
        throw std::runtime_error(
            "split dtype must be float32, float64, int32, int64, int8, uint8, or bool");
    }
    if (data->shape.empty()) {
        throw std::runtime_error("split requires data rank >= 1");
    }
    if (split_attrs->sections.size() < 2) {
        throw std::runtime_error("split requires at least two output sections");
    }
    const int axis = NormalizeAxis("split", split_attrs->axis,
                                   static_cast<int>(data->shape.size()));
    const int64_t extent = data->shape[static_cast<size_t>(axis)];
    if (extent < 0) {
        throw std::runtime_error("split requires a static split axis extent");
    }
    int64_t total = 0;
    for (size_t i = 0; i < split_attrs->sections.size(); ++i) {
        const int64_t section = split_attrs->sections[i];
        if (section < 0 || total > std::numeric_limits<int64_t>::max() - section) {
            throw std::runtime_error("split sections must be non-negative and fit int64");
        }
        total += section;
    }
    if (total != extent) {
        throw std::runtime_error("split sections must sum to the input axis extent");
    }
    Array<Type> outputs;
    for (int64_t section : split_attrs->sections) {
        std::vector<int64_t> shape = ShapeVector(data);
        shape[static_cast<size_t>(axis)] = section;
        outputs.push_back(MakeTensorType(shape, data->dtype));
    }
    return TupleType(std::move(outputs));
}

// 推导静态 +1 Slice 或已准备的 0:extent 前缀输出。
Type SliceInferType(const Attrs& attrs, const Array<Type>& input_types) {
    if (input_types.size() == 5) {
        throw std::runtime_error("slice source controls require restricted preparation before InferType");
    }
    if (input_types.size() != 1 && input_types.size() != 2 && input_types.size() != 3) {
        throw std::runtime_error("slice requires one static input or prepared prefix/window inputs");
    }
    const auto* data = RequireTensor("slice", input_types[0], "data");
    const auto* slice_attrs = attrs.As<SliceAttrsNode>();
    if (!slice_attrs) {
        throw std::runtime_error("slice requires SliceAttrs");
    }
    if (!IsConcatenateDType(data->dtype)) {
        throw std::runtime_error(
            "slice dtype must be float32, float64, int32, int64, int8, uint8, or bool");
    }
    const int rank = static_cast<int>(data->shape.size());
    if (rank < 1) {
        throw std::runtime_error("slice requires data rank >= 1");
    }
    if (input_types.size() == 2 || input_types.size() == 3) {
        const auto* anchor = RequireTensor("slice", input_types[1], "shape anchor");
        const auto* window_anchor = input_types.size() == 3
            ? RequireTensor("slice", input_types[2], "window shape anchor") : nullptr;
        if (!slice_attrs->starts.empty() || !slice_attrs->ends.empty() ||
            !slice_attrs->axes.empty() || !slice_attrs->steps.empty() ||
            slice_attrs->prefix_axis < 0 || slice_attrs->prefix_axis >= rank ||
            slice_attrs->extent_axis < 0 ||
            static_cast<size_t>(slice_attrs->extent_axis) >= anchor->shape.size() ||
            (input_types.size() == 2 && (slice_attrs->window_size < -1 ||
                                          slice_attrs->window_extent_axis != -1)) ||
            (input_types.size() == 3 && (slice_attrs->window_size != -2 ||
                                          slice_attrs->window_extent_axis < 0 ||
                                          static_cast<size_t>(slice_attrs->window_extent_axis) >=
                                              window_anchor->shape.size()))) {
            throw std::runtime_error("slice prepared prefix requires empty static controls and valid axes");
        }
        std::vector<int64_t> out = ShapeVector(data);
        for (int64_t dim : out) {
            if (dim < 0) throw std::runtime_error("slice prepared prefix requires a static table");
        }
        for (int64_t dim : anchor->shape) {
            if (dim < -1) throw std::runtime_error("slice shape anchor has an invalid dimension");
        }
        if (window_anchor) {
            for (int64_t dim : window_anchor->shape) {
                if (dim < -1) throw std::runtime_error("slice window shape anchor has an invalid dimension");
            }
        }
        const int64_t extent = anchor->shape[slice_attrs->extent_axis];
        const int64_t count = slice_attrs->window_size;
        const int64_t capacity = out[slice_attrs->prefix_axis];
        if (count >= -1 && (count > capacity || extent > capacity - (count < 0 ? 0 : count))) {
            throw std::runtime_error("slice prefix/window extent exceeds table capacity");
        }
        out[slice_attrs->prefix_axis] = count == -1 ? extent
            : count == -2 ? window_anchor->shape[slice_attrs->window_extent_axis] : count;
        return MakeTensorType(out, data->dtype);
    }
    if (slice_attrs->prefix_axis != -1 || slice_attrs->extent_axis != -1 || slice_attrs->window_size != -1) {
        throw std::runtime_error("slice prefix attrs require a prepared shape anchor");
    }
    const size_t count = slice_attrs->starts.size();
    if (count == 0 || slice_attrs->ends.empty() || slice_attrs->axes.empty() ||
        slice_attrs->steps.empty() || slice_attrs->ends.size() != count ||
        slice_attrs->axes.size() != count || slice_attrs->steps.size() != count) {
        throw std::runtime_error("slice starts, ends, axes, and steps must be nonempty and equal length");
    }
    std::vector<int64_t> out = ShapeVector(data);
    for (int64_t extent : out) {
        if (extent < -1) {
            throw std::runtime_error("slice input dimensions must be static or symbolic -1");
        }
    }
    std::vector<bool> seen(static_cast<size_t>(rank), false);
    for (size_t index = 0; index < count; ++index) {
        if (slice_attrs->steps[index] != 1) {
            throw std::runtime_error("slice requires every step to equal exactly +1");
        }
        const int axis = NormalizeAxis("slice", slice_attrs->axes[index], rank);
        if (seen[static_cast<size_t>(axis)]) {
            throw std::runtime_error("slice axes must be unique");
        }
        seen[static_cast<size_t>(axis)] = true;
        const int64_t dim = out[static_cast<size_t>(axis)];
        if (dim < 0) throw std::runtime_error("slice requires static sliced axes");
        const int64_t start = ClampPositiveStepEndpoint(slice_attrs->starts[index], dim);
        const int64_t end = ClampPositiveStepEndpoint(slice_attrs->ends[index], dim);
        out[static_cast<size_t>(axis)] = std::max(end - start, int64_t{0});
    }
    return MakeTensorType(out, data->dtype);
}

// 按 axes 与 keepdims 推导 reduce_mean 结果 shape。
Type ReduceMeanInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("reduce_mean", input_types, 1);
    const auto* data = RequireTensor("reduce_mean", input_types[0], "data");
    const auto* reduce_attrs = attrs.As<ReduceMeanAttrsNode>();
    const std::vector<int> axes =
        NormalizeAxes("reduce_mean", reduce_attrs ? reduce_attrs->axes : Array<int64_t>{},
                      static_cast<int>(data->shape.size()));
    const bool keepdims = !reduce_attrs || reduce_attrs->keepdims != 0;

    std::vector<bool> reduce_axis(data->shape.size(), false);
    for (int axis : axes) {
        reduce_axis[static_cast<size_t>(axis)] = true;
    }

    std::vector<int64_t> out;
    for (size_t i = 0; i < data->shape.size(); ++i) {
        if (reduce_axis[i]) {
            if (keepdims) {
                out.push_back(1);
            }
        } else {
            out.push_back(data->shape[i]);
        }
    }
    return MakeTensorType(out, data->dtype);
}

// 校验 softmax 轴并保持输入类型。
Type SoftmaxInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("softmax", input_types, 1);
    const auto* data = RequireTensor("softmax", input_types[0], "data");
    if (data->shape.empty()) {
        throw std::runtime_error("softmax requires rank at least 1");
    }
    if (data->dtype != "float32" && data->dtype != "float64") {
        throw std::runtime_error("softmax requires float32 or float64 input");
    }
    const auto* softmax_attrs = attrs.As<SoftmaxAttrsNode>();
    NormalizeAxis("softmax", softmax_attrs ? softmax_attrs->axis : -1,
                  static_cast<int>(data->shape.size()));
    return input_types[0];
}

Type MaskedSoftmaxInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("masked_softmax", input_types, 2);
    const auto* data = RequireTensor("masked_softmax", input_types[0], "data");
    const auto* mask = RequireTensor("masked_softmax", input_types[1], "mask");
    if (data->shape.empty() || (data->dtype != "float32" && data->dtype != "float64")) {
        throw std::runtime_error("masked_softmax requires rank >= 1 float32 or float64 data");
    }
    if (mask->dtype != "bool" || mask->shape.size() > data->shape.size()) {
        throw std::runtime_error("masked_softmax requires a bool mask broadcastable to data");
    }
    const auto* axis_attrs = attrs.As<SoftmaxAttrsNode>();
    if (attrs.defined() && !axis_attrs) {
        throw std::runtime_error("masked_softmax requires SoftmaxAttrs");
    }
    const int axis = NormalizeAxis("masked_softmax", axis_attrs ? axis_attrs->axis : -1,
                                   static_cast<int>(data->shape.size()));
    if (data->shape[axis] == 0) {
        throw std::runtime_error("masked_softmax requires a positive reduction extent");
    }
    const auto shape = ShapeVector(data);
    const auto broadcast = BroadcastShape("masked_softmax mask", shape, ShapeVector(mask));
    for (size_t i = 0; i < shape.size(); ++i) {
        if (IsKnown(shape[i]) && IsKnown(broadcast[i]) && shape[i] != broadcast[i]) {
            throw std::runtime_error("masked_softmax mask cannot expand the data shape");
        }
    }
    return input_types[0];
}

Type LayerNormInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("nn_layer_norm", input_types, 3);
    const auto* data = RequireTensor("nn_layer_norm", input_types[0], "data");
    const auto* scale = RequireTensor("nn_layer_norm", input_types[1], "scale");
    const auto* bias = RequireTensor("nn_layer_norm", input_types[2], "bias");
    const auto* layer_norm_attrs = attrs.As<LayerNormAttrsNode>();
    if (!layer_norm_attrs) {
        throw std::runtime_error("nn_layer_norm requires LayerNormAttrs");
    }
    if (data->dtype != "float32" || scale->dtype != "float32" || bias->dtype != "float32") {
        throw std::runtime_error("nn_layer_norm requires float32 data, scale, and bias");
    }
    if (data->shape.empty()) {
        throw std::runtime_error("nn_layer_norm requires data rank >= 1");
    }
    for (size_t index = 0; index < data->shape.size(); ++index) {
        if (data->shape[index] < 0) {
            throw std::runtime_error("nn_layer_norm requires non-negative static data dimensions");
        }
    }
    const int axis = NormalizeAxis("nn_layer_norm", layer_norm_attrs->axis,
                                   static_cast<int>(data->shape.size()));
    if (!std::isfinite(layer_norm_attrs->epsilon) || layer_norm_attrs->epsilon <= 0.0f) {
        throw std::runtime_error("nn_layer_norm epsilon must be finite and > 0");
    }
    if (layer_norm_attrs->accumulation_dtype != "float64") {
        throw std::runtime_error("nn_layer_norm accumulation_dtype must be float64");
    }
    const size_t suffix_rank = data->shape.size() - static_cast<size_t>(axis);
    if (scale->shape.size() != suffix_rank || bias->shape.size() != suffix_rank) {
        throw std::runtime_error("nn_layer_norm scale and bias shapes must exactly equal data.shape[axis:]");
    }
    for (size_t index = 0; index < suffix_rank; ++index) {
        const int64_t extent = data->shape[static_cast<size_t>(axis) + index];
        if (extent <= 0) {
            throw std::runtime_error("nn_layer_norm normalized suffix dimensions must be > 0");
        }
        if (scale->shape[index] != extent || bias->shape[index] != extent) {
            throw std::runtime_error("nn_layer_norm scale and bias shapes must exactly equal data.shape[axis:]");
        }
    }
    return input_types[0];
}

}  // namespace relay
}  // namespace kxc
