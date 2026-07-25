/*! \file src/compiler/internal/primitive_compiler.h
 * \brief Shared primitive lowering, optimization, cache, and backend compiler.
 */

#pragma once

#include <memory>
#include <vector>

#include "../../codegen/internal/compiled_kernel.h"
#include "execution_contract.h"
#include "primitive_unit.h"
#include "kxc/compiler/artifact.h"
#include "kxc/compiler/compile_config.h"
#include "kxc/runtime/compiled_module.h"

namespace kxc::profiling {
class ProfileContext;
}

namespace kxc::api::internal {

struct CompiledPrimitive final {
    PrimitiveUnitId unit_id{-1};
    String symbol;
    UnitSemanticKey semantic_key;
    PrimitiveArtifactKey artifact_key;
    tir::PrimFunc tir;
    codegen::KernelSignature signature;
    codegen::KernelLaunchMetadata launch_metadata;
    codegen::CompiledKernel kernel;
    ArtifactPin pin;
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

CompiledModule AssemblePrimitiveModule(
    const CompiledPrimitiveBatch& batch, const Target& target,
    std::shared_ptr<profiling::ProfileContext> profile_context = nullptr);

}  // namespace kxc::api::internal
