/*! \file src/relay/type_infer.cc
 * \brief 实现 Relay 算子共享的类型推导规则。
 */

#include "relay/type_infer.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kxc {
namespace relay {

namespace {

void RequireArity(const std::string& op_name, const Array<Type>& input_types, size_t expected) {
    if (input_types.size() != expected) {
        throw std::runtime_error(op_name + " expects " + std::to_string(expected) +
                                 " input(s), got " + std::to_string(input_types.size()));
    }
}

Array<int64_t> ToArray(const std::vector<int64_t>& shape) {
    Array<int64_t> out;
    for (int64_t dim : shape) {
        out.push_back(dim);
    }
    return out;
}

std::vector<int64_t> ShapeVector(const TensorTypeNode* type) {
    std::vector<int64_t> out;
    out.reserve(type->shape.size());
    for (size_t i = 0; i < type->shape.size(); ++i) {
        out.push_back(type->shape[i]);
    }
    return out;
}

Type MakeTensorType(const std::vector<int64_t>& shape, const std::string& dtype) {
    return TensorType(ToArray(shape), dtype);
}

const TensorTypeNode* RequireTensor(const std::string& op_name, const Type& type,
                                    const std::string& input_name) {
    const auto* tensor = type.As<TensorTypeNode>();
    if (!tensor) {
        throw std::runtime_error(op_name + " expects TensorType for " + input_name +
                                 ", got " + TypeToString(type));
    }
    return tensor;
}

void RequireSameDType(const std::string& op_name, const TensorTypeNode* lhs,
                      const TensorTypeNode* rhs) {
    if (lhs->dtype != rhs->dtype) {
        throw std::runtime_error(op_name + " dtype mismatch: " + lhs->dtype + " vs " +
                                 rhs->dtype);
    }
}

bool IsKnown(int64_t dim) { return dim >= 0; }

bool SameOrUnknown(int64_t lhs, int64_t rhs) {
    return !IsKnown(lhs) || !IsKnown(rhs) || lhs == rhs;
}

int64_t KnownProduct(const std::vector<int64_t>& shape, size_t begin, size_t end) {
    int64_t product = 1;
    for (size_t i = begin; i < end; ++i) {
        if (!IsKnown(shape[i])) {
            return -1;
        }
        product *= shape[i];
    }
    return product;
}

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

std::vector<int> NormalizeAxes(const std::string& op_name, const std::vector<int64_t>& axes,
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

int64_t ReadVectorValue(const std::vector<int64_t>& values, size_t index, int64_t default_value) {
    return index < values.size() ? values[index] : default_value;
}

std::vector<int64_t> ReadPair(const std::vector<int64_t>& values, int64_t default_value) {
    if (values.empty()) {
        return {default_value, default_value};
    }
    if (values.size() == 1) {
        return {values[0], values[0]};
    }
    return {values[0], values[1]};
}

std::vector<int64_t> ReadPadding(const std::vector<int64_t>& values) {
    if (values.empty()) {
        return {0, 0, 0, 0};
    }
    if (values.size() == 1) {
        return {values[0], values[0], values[0], values[0]};
    }
    if (values.size() == 2) {
        return {values[0], values[1], values[0], values[1]};
    }
    return {values[0], values[1], values[2], values[3]};
}

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
    const int64_t numerator = input + pad_before + pad_after - effective_kernel;
    if (numerator < 0) {
        throw std::runtime_error(op_name + " window is larger than input");
    }
    if (ceil_mode) {
        return (numerator + stride - 1) / stride + 1;
    }
    return numerator / stride + 1;
}

Type BinaryBroadcastInferType(const std::string& op_name, const Array<Type>& input_types,
                              const std::string& out_dtype = "") {
    RequireArity(op_name, input_types, 2);
    const auto* lhs = RequireTensor(op_name, input_types[0], "lhs");
    const auto* rhs = RequireTensor(op_name, input_types[1], "rhs");
    RequireSameDType(op_name, lhs, rhs);
    return MakeTensorType(BroadcastShape(op_name, ShapeVector(lhs), ShapeVector(rhs)),
                          out_dtype.empty() ? lhs->dtype : out_dtype);
}

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

std::string AttrOutDTypeOrDefault(const std::string& out_dtype,
                                  const std::string& default_dtype) {
    return out_dtype.empty() ? default_dtype : out_dtype;
}

}  // namespace

Type IdentityInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("identity", input_types, 1);
    return input_types[0];
}

Type AddInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("add", input_types);
}

Type SubtractInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("subtract", input_types);
}

Type MultiplyInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("mul", input_types);
}

Type DivideInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("divide", input_types);
}

Type PowInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("pow", input_types);
}

Type EqualInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("equal", input_types, "bool");
}

Type GreaterInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("greater", input_types, "bool");
}

Type UnarySameInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("unary", input_types, 1);
    RequireTensor("unary", input_types[0], "data");
    return input_types[0];
}

Type CastInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("cast", input_types, 1);
    const auto* data = RequireTensor("cast", input_types[0], "data");
    const auto* cast_attrs = attrs.As<CastAttrsNode>();
    const std::string dtype = cast_attrs ? CastDTypeFromCode(cast_attrs->to) : data->dtype;
    return MakeTensorType(ShapeVector(data), dtype);
}

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
    std::vector<int64_t> out = BroadcastShape("matmul", a_batch, b_batch);
    out.push_back(a[a.size() - 2]);
    out.push_back(b[b.size() - 1]);
    return MakeTensorType(out, lhs->dtype);
}

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

    const std::vector<int64_t> strides = conv_attrs ? ReadPair(conv_attrs->strides, 1) : std::vector<int64_t>{1, 1};
    const std::vector<int64_t> dilation = conv_attrs ? ReadPair(conv_attrs->dilation, 1) : std::vector<int64_t>{1, 1};
    const std::vector<int64_t> padding = conv_attrs ? ReadPadding(conv_attrs->padding) : std::vector<int64_t>{0, 0, 0, 0};
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

    const int64_t oh = WindowOutputDim("nn_conv2d", data->shape[2], kh, padding[0], padding[2],
                                      strides[0], dilation[0], false);
    const int64_t ow = WindowOutputDim("nn_conv2d", data->shape[3], kw, padding[1], padding[3],
                                      strides[1], dilation[1], false);
    const std::string dtype =
        conv_attrs ? AttrOutDTypeOrDefault(conv_attrs->out_dtype, data->dtype) : data->dtype;
    return MakeTensorType({data->shape[0], channels, oh, ow}, dtype);
}

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
    const std::vector<int64_t> pool_size =
        pool_attrs ? ReadPair(pool_attrs->pool_size, 1) : std::vector<int64_t>{1, 1};
    const std::vector<int64_t> strides =
        pool_attrs ? ReadPair(pool_attrs->strides, 1) : std::vector<int64_t>{1, 1};
    const std::vector<int64_t> dilation =
        pool_attrs ? ReadPair(pool_attrs->dilation, 1) : std::vector<int64_t>{1, 1};
    const std::vector<int64_t> padding =
        pool_attrs ? ReadPadding(pool_attrs->padding) : std::vector<int64_t>{0, 0, 0, 0};
    const bool ceil_mode = pool_attrs && pool_attrs->ceil_mode;

    const int64_t oh = WindowOutputDim("pool2d", data->shape[2], pool_size[0], padding[0],
                                      padding[2], strides[0], dilation[0], ceil_mode);
    const int64_t ow = WindowOutputDim("pool2d", data->shape[3], pool_size[1], padding[1],
                                      padding[3], strides[1], dilation[1], ceil_mode);
    return MakeTensorType({data->shape[0], data->shape[1], oh, ow}, data->dtype);
}

Type GlobalAvgPool2DInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("nn_global_avg_pool2d", input_types, 1);
    const auto* data = RequireTensor("nn_global_avg_pool2d", input_types[0], "data");
    if (data->shape.size() != 4) {
        throw std::runtime_error("nn_global_avg_pool2d expects NCHW rank-4 input");
    }
    return MakeTensorType({data->shape[0], data->shape[1], 1, 1}, data->dtype);
}

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

Type ReshapeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    if (input_types.size() != 1 && input_types.size() != 2) {
        throw std::runtime_error("reshape expects data and optional legacy shape input");
    }
    const auto* data = RequireTensor("reshape", input_types[0], "data");
    const auto* reshape_attrs = attrs.As<ReshapeAttrsNode>();
    if (!reshape_attrs) {
        throw std::runtime_error("reshape static type inference requires ReshapeAttrs");
    }
    const std::vector<int64_t> input_shape = ShapeVector(data);
    const int64_t input_product = KnownProduct(input_shape, 0, input_shape.size());
    std::vector<int64_t> out;
    out.reserve(reshape_attrs->newshape.size());

    int infer_index = -1;
    int64_t known_product = 1;
    for (size_t i = 0; i < reshape_attrs->newshape.size(); ++i) {
        int64_t dim = reshape_attrs->newshape[i];
        if (dim > 0) {
            out.push_back(dim);
            known_product *= dim;
        } else if (dim == 0) {
            if (reshape_attrs->allowzero) {
                out.push_back(0);
                known_product = 0;
            } else {
                if (i >= input_shape.size()) {
                    throw std::runtime_error("reshape 0-dim copy index exceeds input rank");
                }
                out.push_back(input_shape[i]);
                if (IsKnown(input_shape[i])) {
                    known_product *= input_shape[i];
                } else {
                    known_product = -1;
                }
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

Type ShapeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("shape", input_types, 1);
    const auto* data = RequireTensor("shape", input_types[0], "data");
    return MakeTensorType({static_cast<int64_t>(data->shape.size())}, "int64");
}

Type TransposeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("transpose", input_types, 1);
    const auto* data = RequireTensor("transpose", input_types[0], "data");
    const int rank = static_cast<int>(data->shape.size());
    std::vector<int64_t> perm;
    if (const auto* transpose_attrs = attrs.As<TransposeAttrsNode>()) {
        perm = transpose_attrs->perm;
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

Type ReduceMeanInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("reduce_mean", input_types, 1);
    const auto* data = RequireTensor("reduce_mean", input_types[0], "data");
    const auto* reduce_attrs = attrs.As<ReduceMeanAttrsNode>();
    const std::vector<int> axes =
        NormalizeAxes("reduce_mean", reduce_attrs ? reduce_attrs->axes : std::vector<int64_t>{},
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

Type SoftmaxInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("softmax", input_types, 1);
    const auto* data = RequireTensor("softmax", input_types[0], "data");
    const auto* softmax_attrs = attrs.As<SoftmaxAttrsNode>();
    if (!data->shape.empty()) {
        NormalizeAxis("softmax", softmax_attrs ? softmax_attrs->axis : -1,
                      static_cast<int>(data->shape.size()));
    }
    return input_types[0];
}

Type ConcatenateInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("concatenate", input_types, 1);
    const auto* tuple = input_types[0].As<TupleTypeNode>();
    if (!tuple || tuple->fields.empty()) {
        throw std::runtime_error("concatenate expects a non-empty tuple of tensors");
    }
    const auto* first = RequireTensor("concatenate", tuple->fields[0], "field[0]");
    const int axis = NormalizeAxis("concatenate",
                                   attrs.As<ConcatAttrsNode>() ? attrs.As<ConcatAttrsNode>()->axis : 0,
                                   static_cast<int>(first->shape.size()));
    std::vector<int64_t> out = ShapeVector(first);
    for (size_t i = 1; i < tuple->fields.size(); ++i) {
        const auto* current = RequireTensor("concatenate", tuple->fields[i],
                                            "field[" + std::to_string(i) + "]");
        RequireSameDType("concatenate", first, current);
        if (current->shape.size() != first->shape.size()) {
            throw std::runtime_error("concatenate rank mismatch");
        }
        for (size_t dim = 0; dim < out.size(); ++dim) {
            if (static_cast<int>(dim) == axis) {
                if (IsKnown(out[dim]) && IsKnown(current->shape[dim])) {
                    out[dim] += current->shape[dim];
                } else {
                    out[dim] = -1;
                }
            } else if (!SameOrUnknown(out[dim], current->shape[dim])) {
                throw std::runtime_error("concatenate non-axis dimension mismatch");
            }
        }
    }
    return MakeTensorType(out, first->dtype);
}

Type SplitInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("split", input_types, 1);
    const auto* data = RequireTensor("split", input_types[0], "data");
    const auto* split_attrs = attrs.As<SplitAttrsNode>();
    if (!split_attrs || split_attrs->split.empty()) {
        throw std::runtime_error("split requires SplitAttrs with split points or sections");
    }
    const int axis =
        NormalizeAxis("split", split_attrs->axis, static_cast<int>(data->shape.size()));
    const int64_t axis_dim = data->shape[static_cast<size_t>(axis)];

    Array<Type> fields;
    if (split_attrs->split.size() == 1) {
        const int64_t sections = split_attrs->split[0];
        if (sections <= 0) {
            throw std::runtime_error("split sections must be positive");
        }
        int64_t segment = -1;
        if (IsKnown(axis_dim)) {
            if (axis_dim % sections != 0) {
                throw std::runtime_error("split sections do not divide axis dimension");
            }
            segment = axis_dim / sections;
        }
        for (int64_t i = 0; i < sections; ++i) {
            std::vector<int64_t> shape = ShapeVector(data);
            shape[static_cast<size_t>(axis)] = segment;
            fields.push_back(MakeTensorType(shape, data->dtype));
        }
    } else {
        int64_t previous = 0;
        for (int64_t point : split_attrs->split) {
            if (point < previous || (IsKnown(axis_dim) && point > axis_dim)) {
                throw std::runtime_error("split points must be sorted within axis dimension");
            }
            std::vector<int64_t> shape = ShapeVector(data);
            shape[static_cast<size_t>(axis)] = point - previous;
            fields.push_back(MakeTensorType(shape, data->dtype));
            previous = point;
        }
        std::vector<int64_t> shape = ShapeVector(data);
        shape[static_cast<size_t>(axis)] = IsKnown(axis_dim) ? axis_dim - previous : -1;
        fields.push_back(MakeTensorType(shape, data->dtype));
    }
    return TupleType(fields);
}

Type WhereInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    RequireArity("where", input_types, 3);
    const auto* cond = RequireTensor("where", input_types[0], "condition");
    const auto* lhs = RequireTensor("where", input_types[1], "x");
    const auto* rhs = RequireTensor("where", input_types[2], "y");
    if (cond->dtype != "bool") {
        throw std::runtime_error("where condition must have bool dtype");
    }
    RequireSameDType("where", lhs, rhs);
    std::vector<int64_t> value_shape =
        BroadcastShape("where", ShapeVector(lhs), ShapeVector(rhs));
    return MakeTensorType(BroadcastShape("where", ShapeVector(cond), value_shape), lhs->dtype);
}

Type GatherInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("gather", input_types, 2);
    const auto* data = RequireTensor("gather", input_types[0], "data");
    const auto* indices = RequireTensor("gather", input_types[1], "indices");
    if (indices->dtype != "int32" && indices->dtype != "int64") {
        throw std::runtime_error("gather indices must be int32 or int64");
    }
    const auto* gather_attrs = attrs.As<GatherAttrsNode>();
    const int axis = NormalizeAxis("gather", gather_attrs ? gather_attrs->axis : 0,
                                   static_cast<int>(data->shape.size()));
    std::vector<int64_t> out;
    for (int i = 0; i < axis; ++i) {
        out.push_back(data->shape[static_cast<size_t>(i)]);
    }
    const std::vector<int64_t> index_shape = ShapeVector(indices);
    out.insert(out.end(), index_shape.begin(), index_shape.end());
    for (size_t i = static_cast<size_t>(axis + 1); i < data->shape.size(); ++i) {
        out.push_back(data->shape[i]);
    }
    return MakeTensorType(out, data->dtype);
}

}  // namespace relay
}  // namespace kxc
