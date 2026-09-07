/*! \file src/compiler/internal/te_to_tir.h
 * \brief Private TE-DAG to PrimFunc assembly for primitive units.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "lowered_function.h"
#include "kxc/target/target.h"
#include "kxc/te/te.h"

namespace kxc::relay::internal {

inline constexpr const char* kDefaultTESchedulePolicy =
    "target-default-v1";
inline constexpr const char* kBoundedDynamicTESchedulePolicy =
    "bounded-dynamic-serial-v1";
// Dynamic stateful KV kernels keep static shapes; their state extent buffers
// may appear only inside stage bodies (masks and index arithmetic).
inline constexpr const char* kStatefulKvTESchedulePolicy =
    "stateful-kv-serial-v1";

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
// Bounded dynamic kernels are deliberately serial and LLVM/CPU-only.
te::Schedule BuildBoundedDynamicTESchedule(
    const Array<te::Tensor>& outputs, const Target& target);
// Shape-value units materialize static-shaped outputs whose kernel bodies
// consume runtime extents as stored values; the buffer list authorizes them.
te::Schedule BuildBoundedDynamicTESchedule(
    const Array<te::Tensor>& outputs, const Target& target,
    const Array<tir::Var>& runtime_extent_buffers);
// Creates and recognizes the only accepted TE shape expression for a
// generated uint64[1] runtime-extent buffer.
tir::PrimExpr LoadRuntimeExtent(const tir::Var& buffer);
// Serial unsplit schedule with reductions for static-shape dynamic stateful
// kernels (LLVM/CPU only).
te::Schedule BuildStatefulKvTESchedule(const Array<te::Tensor>& outputs,
                                       const Target& target);
bool MatchRuntimeExtentLoad(const tir::PrimExpr& expression,
                            const Array<tir::Var>& buffers,
                            size_t* buffer_index);
std::string CanonicalTEScheduleContract(
    const te::Schedule& schedule, const Target& target,
    const Array<tir::Var>& runtime_extent_buffers = {},
    // stateful 路径：由调用方声明、只在 body 中出现的 extent 下标。
    const std::vector<size_t>& body_only_runtime_extents = {},
    // bounded 路径：由 lowering 扫描 compute body 得出、作为存储值被消费
    // 而非驱动循环轴的 extent buffer（形状值单元）。两条路径互斥。
    const std::unordered_set<const Object*>* body_consumed_extents = nullptr);
std::string GetTEScheduleContract(const tir::PrimFunc& function);

// runtime_extent_buffers is the canonical physical scalar order; the bounded
// producer must reuse that same order when authoring ModuleInvocationContract.
LoweredFunction LowerTensorGraphToTIR(
    const Array<te::Tensor>& inputs,
    const std::vector<ConstantTensor>& constants,
    const Array<te::Tensor>& outputs,
    const te::Schedule& schedule,
    const Target& target,
    const PrimFuncIdentity& identity,
    const Array<tir::Var>& runtime_extent_buffers = {},
    const std::vector<size_t>& body_only_runtime_extents = {});

}  // namespace kxc::relay::internal
