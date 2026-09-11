/*! \file include/kxc/relay/type_infer.h
 * \brief 声明 Relay 算子共享的类型推导规则。
 */

#pragma once

#include "kxc/relay/op.h"

namespace kxc {
namespace relay {

Type AddInferType(const Attrs& attrs, const Array<Type>& input_types);
Type SubtractInferType(const Attrs& attrs, const Array<Type>& input_types);
Type MultiplyInferType(const Attrs& attrs, const Array<Type>& input_types);
Type DivideInferType(const Attrs& attrs, const Array<Type>& input_types);
Type EqualInferType(const Attrs& attrs, const Array<Type>& input_types);
bool IsEqualInputDType(const std::string& dtype);
Type LessInferType(const Attrs& attrs, const Array<Type>& input_types);
Type SigmoidInferType(const Attrs& attrs, const Array<Type>& input_types);
Type TanhInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ErfInferType(const Attrs& attrs, const Array<Type>& input_types);
Type PowInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ExpandInferType(const Attrs& attrs, const Array<Type>& input_types);
Type WhereInferType(const Attrs& attrs, const Array<Type>& input_types);
Type UnarySameInferType(const Attrs& attrs, const Array<Type>& input_types);
Type CastInferType(const Attrs& attrs, const Array<Type>& input_types);
Type MatMulInferType(const Attrs& attrs, const Array<Type>& input_types);
Type DenseInferType(const Attrs& attrs, const Array<Type>& input_types);
Type GemmInferType(const Attrs& attrs, const Array<Type>& input_types);
Type Conv2DInferType(const Attrs& attrs, const Array<Type>& input_types);
Type Pool2DInferType(const Attrs& attrs, const Array<Type>& input_types);
Type GlobalAvgPool2DInferType(const Attrs& attrs, const Array<Type>& input_types);
Type FlattenInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ReshapeInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ReshapeDynamicInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ShapeOfInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ShapeExprInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ExpandDynamicInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ConstantOfShapeInferType(const Attrs& attrs, const Array<Type>& input_types);
Type TriluInferType(const Attrs& attrs, const Array<Type>& input_types);
Type SqueezeInferType(const Attrs& attrs, const Array<Type>& input_types);
Type UnsqueezeInferType(const Attrs& attrs, const Array<Type>& input_types);
Type TransposeInferType(const Attrs& attrs, const Array<Type>& input_types);
Type GatherInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ConcatenateInferType(const Attrs& attrs, const Array<Type>& input_types);
Type SplitInferType(const Attrs& attrs, const Array<Type>& input_types);
Type SliceInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ReduceMeanInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ReduceMaxInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ReduceMinInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ArgMaxInferType(const Attrs& attrs, const Array<Type>& input_types);
Type SoftmaxInferType(const Attrs& attrs, const Array<Type>& input_types);
Type MaskedSoftmaxInferType(const Attrs& attrs, const Array<Type>& input_types);
Type LayerNormInferType(const Attrs& attrs, const Array<Type>& input_types);
}  // namespace relay
}  // namespace kxc
