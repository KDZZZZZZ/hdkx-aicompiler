/*! \file src/compiler/internal/primitive_compiler.h
 * \brief Shared primitive lowering, optimization, cache, and backend compiler.
 */

#pragma once

#include <memory>
#include <vector>

#include "execution_contract.h"
#include "primitive_unit.h"
#include "primitive_cache.h"
#include "kxc/compiler/compile_config.h"
#include "kxc/runtime/compiled_module.h"

namespace kxc::profiling {
class ProfileContext;
}

namespace kxc::api::internal {

struct CompiledPrimitive final {
    PrimitiveUnitId unit_id{-1};
    tir::PrimFunc diagnostic_tir;
    PrimitiveArtifactPin pin;
    bool cache_hit{false};
};

struct CompiledPrimitiveBatch final {
    std::vector<CompiledPrimitive> primitives;
    Map<String, runtime::NDArray> constants;
};

CompiledPrimitiveBatch CompilePrimitiveUnits(
    const std::vector<PrimitiveUnit>& units,
    const std::vector<LogicalValueContract>& values,
    const CompileConfig& config,
    const CompilerExecutionContract& contract);

CompiledPrimitiveBatch CompilePrimitiveUnits(
    const std::vector<PrimitiveUnit>& units,
    const std::vector<LogicalValueContract>& values,
    const CompileConfig& config,
    const CompilerExecutionContract& contract,
    const std::vector<PrimitiveUnitId>& requested_unit_ids);

CompiledModule AssemblePrimitiveModule(
    const CompiledPrimitiveBatch& batch,
    const std::vector<PrimitiveUnit>& units, const Target& target,
    std::shared_ptr<profiling::ProfileContext> profile_context = nullptr);

}  // namespace kxc::api::internal
