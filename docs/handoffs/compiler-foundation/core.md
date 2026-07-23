# Compiler Foundation Core handoff

> **Branch:** `feature/compiler-foundation-core`  
> **Base:** `e295a73` (foundation roadmap on top of `3b95aca`)  
> **Status:** Core-track implementation complete; blocked only on independent
> 02–06 branches recording actual CoreContract v1 consumption.  
> **Runtime boundary:** unchanged — `RuntimeSession` remains a static
> `CompiledModule + ExecutablePlan` data-plane executor.

## Delivered review stages

| Stage | Commit | Delivered contract |
|---|---|---|
| Capability truth | `cf6fcc0` | `CapabilityVerifier` fails closed at compiler entry, post-graph-pass and pre-partition; stable locator/missing-capability diagnostics; static-exact only. |
| Identity split | `96223cf` | `GraphValueLocator`, `UnitSemanticKey`, `ArtifactKey`, `DispatchKey`, `PlanVariantKey`; unit-local canonical mapping excludes graph ids/symbol/storage; full canonical equality despite digest collision. |
| Artifact lifecycle | `2b0a2e3` | Immutable `PrimitiveArtifactPin`, eviction-safe ownership, production same-key singleflight, bounded entries/bytes/in-flight/failures, retry records and explicit backpressure; symbol/signature alias validation. |
| Metadata/pipeline authority | `22b6c3d` | JSON -> generated C++ -> strict checker chain for operator/pass metadata; implementation-only binding tables; `PipelineResolver` is production order/invariant/fingerprint source. |
| Frozen cross-track contract | `0acdb8d` | CoreContract v1 artifact/request/ticket/outcome/observer/selected-plan DTOs and deterministic capability/resolver/store/coordinator/assembler fakes. |
| Legacy/docs convergence | `8e3c45a` | Whole-graph `LowerToTIR` is compatibility/testing only; stale guides are archived; production claims point to per-unit `Compiler::Compile`. |

## Frozen v1 integration surface

Other tracks may consume only these public/value contracts and deterministic
fakes until their production integration review:

- `include/kxc/compiler/capability.h`
- `include/kxc/compiler/identity.h`
- `include/kxc/compiler/pipeline.h`
- `include/kxc/compiler/foundation_contract.h`
- `test/support/compiler_foundation_fakes.h`

They must not include `src/compiler/internal/*` or `src/runtime/internal/*`, and
must not use object addresses, graph-local value/unit/storage ids, link symbols,
request heat or `kDynamicDimension == -1` as semantic/artifact/dispatch identity.
Generation `0` plus explicit `static-exact` applicability is the fallback oracle.
There is no fuzzy fallback.

## Verification run in this worktree

Configured without downloads or external access:

```text
cmake --preset dev-ninja-cpu
CUDA: disabled
LLVM: requested but not found in the environment
ONNX Python fixture: skipped because onnx/numpy are not installed
```

Passing focused executables:

- `compiler_capability_test`
- `compiler_identity_test`
- `compiler_foundation_contract_test`
- `pipeline_resolver_test`
- `primitive_cache_test` (12-thread same-key merge, pin/eviction, collision,
  failure/retry/bounds/backpressure)
- `graph_partition_test`
- `compiler_contract_test`
- `compiler_extension_contract_test`
- `pass_pipeline_test`
- `operator_compilation_test` (non-LLVM contract cases in this environment)
- `executable_plan_test`
- `kernel_signature_test`
- `compiled_module_test`
- `runtime_session_test`

Passing generated/static checks:

- `check_relay_op_contract`: 19/19
- `check_pass_contract`: 19/19
- `check_include_layers`
- `check_public_headers`
- `git diff --check`

The LLVM-only numerical cache tests now include a graph-renumbered, different
symbol reuse case. They are compiled/run when LLVM is available; this worktree
could not execute that conditional block because no local LLVM package exists.
No test was weakened or disabled to hide that environment limitation.

## Remaining hard blockers and required follow-up

1. **02 Shape:** consume CoreContract v1 with `DispatchKey`; exact shape first;
   prove no `-1` enters semantic/artifact/dispatch/profile DTOs.
2. **03 Adaptive:** consume `ArtifactPin`, request/ticket/outcome and observer;
   preserve the ready-cache singleflight semantics or layer service scheduling
   above it; never move coordination into `RuntimeSession`.
3. **04 Control flow:** use explicit capability mode/fake rejection until its
   executable task contract is integrated; do not bypass the three gates.
4. **05 Region/runtime plan:** prove graph renumbering does not alter region
   semantic keys and retain per-call fallback; consume selected immutable
   artifacts only.
5. **06 NLP/GPU validation:** report rejection/cache miss/compile failure
   separately; do not infer capability from importer or legacy `LowerToTIR`.
6. **LLVM integration environment:** on an LLVM-enabled builder, run
   `operator_compilation_test`, `op_numeric_llvm_test`, and the broader compiler
   regression to execute the new cached-symbol relocation path numerically.

Once items 1–5 have branch-local mock/fake evidence, M1 can move from “blocked”
to “completed” without further Core-track architecture changes. Any DTO version
change must state compatibility, key/ABI/fingerprint impact and fallback.
