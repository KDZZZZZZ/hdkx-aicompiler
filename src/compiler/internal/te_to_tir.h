/*! \file src/compiler/internal/te_to_tir.h
 * \brief Private TE-DAG to PrimFunc assembly for primitive units.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lowered_function.h"
#include "kxc/target/target.h"
#include "kxc/te/te.h"

namespace kxc::relay::internal {

inline constexpr const char* kDefaultTESchedulePolicy =
    "target-default-v1";

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

// Compiler-owned target policy: CUDA stays serial for BindCudaThreads.
te::Schedule BuildDefaultTESchedule(const Array<te::Tensor>& outputs,
                                    const Target& target);
std::string CanonicalTEScheduleContract(const te::Schedule& schedule,
                                        const Target& target);
std::string GetTEScheduleContract(const tir::PrimFunc& function);

LoweredFunction LowerTensorGraphToTIR(
    const Array<te::Tensor>& inputs,
    const std::vector<ConstantTensor>& constants,
    const Array<te::Tensor>& outputs,
    const te::Schedule& schedule,
    const Target& target,
    const PrimFuncIdentity& identity);

}  // namespace kxc::relay::internal
