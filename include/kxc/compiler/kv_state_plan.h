/*! \file include/kxc/compiler/kv_state_plan.h
 * \brief Production compilation entry for the M2 dynamic stateful KV contract.
 *
 * Declares a small session-owned KV cache computation (append + read) and
 * compiles it through the production TE -> TIR -> LLVM chain into a
 * CompiledModule plus a dynamic stateful ExecutablePlan. The declaration is
 * the only authoring surface: kernels, invocation contracts, and the plan are
 * generated, never hand-assembled.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <dlpack/dlpack.h>

#include "kxc/compiler/compile_config.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"

namespace kxc::api {

/*! \brief KV tensor layout [batch, seq, kv_heads, head_dim]; the sequence
 *  axis carries the dynamic valid extent. Same family as the M9 E2 decode
 *  signature at controlled scale. */
struct KvStateLayout final {
    int64_t batch{1};
    int64_t kv_heads{1};
    int64_t head_dim{1};
    /*! \brief Physical capacity in tokens along the sequence axis. */
    int64_t capacity{1};
};

/*! \brief Which read computation the plan exposes after each append. */
enum class KvStateReadKind : int {
    /*! \brief Masked copy of the valid region; invalid region reads as zero. */
    kValidRegion = 0,
    /*! \brief Tiny fixed-weight causal attention over the valid region;
     *  requires the K/V state pair. */
    kCausalAttention = 1,
};

/*! \brief Complete declaration of one dynamic stateful KV plan. */
struct KvStatePlanDeclaration final {
    Device device{Device::CPU()};
    /*! \brief float32 in the v1 contract. */
    DLDataType dtype{DLDataType{kDLFloat, 32, 1}};
    KvStateLayout layout;
    /*! \brief Physical tokens-input extent along the sequence axis; every run
     *  declares its append count explicitly and it must fit this bound. */
    int64_t max_append_tokens{1};
    /*! \brief 1 = one cache state (valid-region read), 2 = K/V pair
     *  (causal-attention read). */
    int64_t state_count{1};
    /*! \brief Sentinel fill for the invalid capacity region at session state
     *  construction; must be finite. */
    double invalid_fill{0.0};
    KvStateReadKind read_kind{KvStateReadKind::kValidRegion};
};

/*! \brief Immutable production compile result for one stateful declaration. */
struct KvStatePlan final {
    CompiledModule module;
    runtime::ExecutablePlan plan;
    /*! \brief One ordered artifact identity per plan call, in call order;
     *  feeds BuildPlanAbiFingerprint for the plan/module identity. */
    std::vector<OrderedArtifactIdentity> artifacts;
};

/*! \brief Compiles the declaration through the production lowering chain.
 *
 *  CPU/LLVM only in v1; requires KXC_USE_LLVM plus the dynamic compiled
 *  module ABI gate for the produced plan to be executable.
 */
KvStatePlan CompileKvStatePlan(const KvStatePlanDeclaration& declaration,
                               CompileConfig config);

}  // namespace kxc::api
