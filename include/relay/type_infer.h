/*! \file include/relay/type_infer.h
 * \brief 声明 Relay 算子共享的类型推导规则。
 */

#pragma once

#include "relay/op.h"

namespace kxc {
namespace relay {

Type IdentityInferType(const Attrs& attrs, const Array<Type>& input_types);
Type AddInferType(const Attrs& attrs, const Array<Type>& input_types);
Type SubtractInferType(const Attrs& attrs, const Array<Type>& input_types);
Type MultiplyInferType(const Attrs& attrs, const Array<Type>& input_types);
Type DivideInferType(const Attrs& attrs, const Array<Type>& input_types);
Type PowInferType(const Attrs& attrs, const Array<Type>& input_types);
Type EqualInferType(const Attrs& attrs, const Array<Type>& input_types);
Type GreaterInferType(const Attrs& attrs, const Array<Type>& input_types);
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
Type ShapeInferType(const Attrs& attrs, const Array<Type>& input_types);
Type TransposeInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ReduceMeanInferType(const Attrs& attrs, const Array<Type>& input_types);
Type SoftmaxInferType(const Attrs& attrs, const Array<Type>& input_types);
Type ConcatenateInferType(const Attrs& attrs, const Array<Type>& input_types);
Type SplitInferType(const Attrs& attrs, const Array<Type>& input_types);
Type WhereInferType(const Attrs& attrs, const Array<Type>& input_types);
Type GatherInferType(const Attrs& attrs, const Array<Type>& input_types);

}  // namespace relay
}  // namespace kxc
