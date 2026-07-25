/*! \file src/compiler/internal/te_to_tir.h
 * \brief Private TE-DAG to PrimFunc assembly for primitive units.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lowered_function.h"
#include "kxc/te/te.h"

namespace kxc::relay::internal {

struct ConstantTensor {
    te::Tensor tensor;
    String key;
    runtime::NDArray value;
};

struct PrimFuncIdentity {
    String symbol;
    int64_t unit_id{-1};
    String operator_name;
    int64_t operator_schema_version{-1};
    String structural_hash;
};

// Shared fail-closed contract for the current int32 iteration domain and
// int64/size_t row-major addressing/allocation domain.
bool EvaluateStaticLoweringInt64(const tir::PrimExpr& expression,
                                 int64_t* result);
void ValidateStaticLoweringTensor(const Array<tir::PrimExpr>& shape,
                                  tir::DataType dtype,
                                  const std::string& context);

LoweredFunction LowerTensorGraphToTIR(
    const Array<te::Tensor>& inputs,
    const std::vector<ConstantTensor>& constants,
    const Array<te::Tensor>& outputs,
    const PrimFuncIdentity& identity);

}  // namespace kxc::relay::internal
