/*! \file src/compiler/internal/compile_state.h
 * \brief Defines the immutable multi-primitive compiler state machine.
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "../../codegen/internal/compiled_kernel.h"
#include "kxc/compiler/artifact.h"
#include "kxc/relay/relay.h"
#include "kxc/runtime/executable_plan.h"
#include "kxc/runtime/kernel_abi.h"
#include "kxc/runtime/ndarray.h"
#include "kxc/target/target.h"
#include "kxc/tir/stmt.h"

namespace kxc::api {

enum class CompileStage : int {
    kValidated = 0,
    kRelayOptimized = 1,
    kLowered = 2,
    kTIROptimized = 3,
    kSignatureBuilt = 4,
    kBackendCompiled = 5,
};

/*! \brief One operator compilation record carried atomically across stages. */
struct PrimitiveCompileState {
    int64_t unit_id{-1};
    String symbol;
    String operator_identity;
    UnitSemanticKey semantic_key;
    tir::PrimFunc tir;
    std::optional<codegen::KernelSignature> signature;
    std::optional<codegen::KernelLaunchMetadata> launch_metadata;
    std::optional<codegen::CompiledKernel> kernel;
    bool cache_hit{false};
};

class CompileResultNode final : public Object {
public:
    KXC_OBJECT_DECLARE

private:
    friend class CompileResult;

    CompileStage stage_{CompileStage::kValidated};
    Target target_;
    std::optional<Function> relay_;
    std::vector<PrimitiveCompileState> primitives_;
    std::vector<ArtifactPin> artifact_pins_;
    std::optional<runtime::ExecutablePlan> plan_;
    Map<String, runtime::NDArray> constants_;
};

/*! \brief Immutable snapshot of all per-unit compiler facts at one stage. */
class CompileResult : public ObjectRef {
public:
    static CompileResult Validate(Target target, Function relay);
    explicit CompileResult(const ObjectRef& ref);

    CompileResult AfterRelayOptimization(Function optimized_relay) const;
    CompileResult AfterLowering(
        std::vector<PrimitiveCompileState> lowered_primitives,
        runtime::ExecutablePlan plan,
        const Map<String, runtime::NDArray>& constants) const;
    CompileResult AfterTIROptimization(
        std::vector<tir::PrimFunc> optimized_tir) const;
    CompileResult AfterSignatures(
        std::vector<codegen::KernelSignature> signatures) const;
    CompileResult AfterBackends(
        std::vector<codegen::KernelLaunchMetadata> launch_metadata,
        std::vector<codegen::CompiledKernel> kernels,
        std::vector<bool> cache_hits = {},
        std::vector<ArtifactPin> artifact_pins = {}) const;

    void ValidateState() const;
    CompileStage stage() const;
    std::string stage_name() const;
    Target target() const;
    Function validated_relay() const;
    Function optimized_relay() const;
    std::vector<PrimitiveCompileState> primitives() const;
    std::vector<ArtifactPin> artifact_pins() const;
    runtime::ExecutablePlan plan() const;
    Map<String, runtime::NDArray> constants() const;
    const CompileResultNode* operator->() const;

private:
    explicit CompileResult(CompileResultNode* node);
};

}  // namespace kxc::api
