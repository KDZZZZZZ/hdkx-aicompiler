# Compiler Foundation Core handoff — Track01 final-review closure

> **Branch:** `feature/compiler-foundation-core`
> **Status:** repository-local Track01 boundary is implemented and CPU-verified; this is
> **not** a claim of LLVM green or Track03 adaptive-coordinator completion. The LLVM
> workflow is configured, but this machine/current session has no LLVM-enabled green
> record. LLVM execution and final review remain pending.
> **Runtime boundary:** unchanged — `RuntimeSession` remains a static
> `CompiledModule + ExecutablePlan` data-plane executor.

## Closed in this round

| Requirement | Production implementation | Evidence |
|---|---|---|
| Executable capability truth | `CapabilityResult` distinguishes `unsupported`, `eligible_but_not_executable`, and `executable`. `supported` is true only after `CapabilityVerifier` executes the real normalized per-unit compile path through lowering, target schedule, signature and backend. This executable probe intentionally participates in production cache/singleflight; it is not a metadata-only query. Backend-disabled builds and incomplete CUDA snapshots remain eligible but not executable. | `test/compiler_capability_test.cpp`: control flow/dynamic/tuple rejection, `nn_gemm transA`, custom lowering shape mismatch, backend unavailable, CUDA compute capability, CUDA reduction schedule, and `supported => Compiler::Compile` matrix. |
| One production execution plan | `Compiler::Compile` resolves one Relay/TIR execution contract per invocation. Compiler Relay plans explicitly contain pre/post `infer_type`; `PipelineExecutor` revalidates step identity, binding, phase, invariant transitions, target requirements and canonical bytes. Relay `checked_type` has an executable validator run after every normalized step once proven. Unknown invariants cannot become production preconditions; unsupported declarations must be `declarative_only` and are not treated as proof. | `test/pipeline_resolver_test.cpp`; production integration in `src/compiler/compiler.cc`; compatibility-only inference remains in `LowerToTIR`/`LowerOperatorCallsToTIR`. |
| Artifact identity from execution | Artifact keys consume canonical bytes from the exact resolved Relay plan, per-unit lowering version, exact TIR plan/schedule, ABI and backend version. Digest remains indexing/diagnostic only; full canonical bytes decide equality. | `test/compiler_identity_test.cpp`, `test/primitive_cache_test.cpp`, LLVM block in `test/operator_compilation_test.cpp`. |
| Public production transaction adapter | `ProductionArtifactCacheAdapter::Acquire/Wait/Publish/Fail` maps public `CompileRequest`, transaction/ticket, `CompileOutcome` and `ArtifactPin` onto the real primitive-cache lease/flight. The executable candidate and primitive lease remain opaque; `Compiler::Compile` itself uses this adapter. Exact same-key callers share the internal flight/ticket, terminal calls on copied transactions linearize, abandoned owners fail/release waiters, and publish returns a production-backed eviction-safe pin. Read-only `Lookup` remains non-owning on miss. | `primitive_cache_test::public_production_merge_cancel`, `public_production_failure_retry_backpressure`, `public_production_concurrent_singleflight`, `public_transaction_terminal_race_abandonment`, `production_adapter_eviction_pin`; production integration in `src/compiler/compiler.cc`. |
| Core-owned cache policy | Core owns static-exact full-`ArtifactKey` singleflight, failure TTL, snapshot cancellation before acquire/wait, global in-flight backpressure, byte/entry eviction and pin lifetime. Dispatch-aware requests, non-normal priority and non-global budget requests fail closed instead of silently dropping policy. | `ProductionCompileOwnership`; public-to-production cache tests in `primitive_cache_test`. |
| CPU CI closure | CTest registers all 23 CPU/core executables plus Relay/pass contracts, include-layer and compiled public-header checks. CPU workflow runs the complete `cpu` label. | Local CPU-only run: **27/27 passed**. |
| LLVM CI closure | LLVM workflow explicitly builds and runs operator compilation (including symbol relocation/cache reuse), LLVM codegen, per-op numeric, and ONNX compile tests. | Workflow configured only. This machine/current session has no LLVM package and no green LLVM record; execution remains pending on an LLVM-enabled builder. |
| Legacy documentation | `Compiler::Compile -> per-unit LowerGraph -> CompiledModule + ExecutablePlan` is the production authority. Whole-graph `relay::LowerToTIR` is compatibility/testing-only and cannot establish support. | `MODULE_GUIDE.md`, `OP_SUPPORT_MATRIX.md`, Track01 plan/roadmap. |

## Frozen public/value surface

This closure bumps `kCompilerFoundationContractVersion` from 1 to 2. V2 adds the
opaque production transaction/candidate API and executable/declarative invariant
classification; normalized pipeline/pass-contract canonical identity therefore
changes. Artifact/semantic key field definitions and `RuntimeSession` ABI do not
change. Cross-track consumers must explicitly update their version assertion and
regenerate/re-resolve pipeline fingerprints; there is no silent v1 fallback.

- `include/kxc/compiler/capability.h`
- `include/kxc/compiler/identity.h`
- `include/kxc/compiler/pipeline.h`
- `include/kxc/compiler/foundation_contract.h`
- `include/kxc/compiler/compiler.h` (`CompiledGraph::artifact_pins`)
- `test/support/compiler_foundation_fakes.h`

Consumers must not include `src/compiler/internal/*` or `src/runtime/internal/*`.
Graph-local value/unit/storage ids, object addresses, link symbols, request heat and
`kDynamicDimension == -1` do not enter semantic/artifact/dispatch identity. There is
no fuzzy fallback. The fake store/coordinator remain deterministic cross-track test
utilities. Core's production adapter is deliberately static-exact. Only the production
compiler backend SPI can create an opaque `ProductionArtifactCandidate`; external
control-plane consumers cannot reconstruct executables from `ArtifactRecord` and must
not acquire a primitive owner and recursively call `Compiler::Compile` for the same
key. Track03 coordinates whole compile/plan requests above this primitive transaction
and owns live asynchronous cancellation, queue/priority/per-model/per-target budget
policy, dispatch-aware coordination, generation publication, canary/rollback and hot
swap. None of those policies may be copied into `RuntimeSession`.

## Verification performed locally

Configured and rerun in this session without downloads or external access:

```text
cmake --preset dev-ninja-cpu
CUDA: disabled
LLVM: requested but not found
ONNX C++ fixture: skipped because onnx/numpy are not installed
```

Commands and results:

```bash
cmake --build out/build/dev-ninja-cpu --parallel 2
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure \
  --label-regex '(^|;)cpu(;|$)'
# 27/27 passed

python python/tools/check_relay_op_contract.py --root .
python python/tools/check_pass_contract.py --root .
python tools/architecture/check_include_layers.py --root .
python tools/architecture/check_public_headers.py --root . --compile
# included in the 27/27 CTest result
```

Before handoff, `git diff --check` is required. No network, push, or merge is part of
this track.

## Remaining review/integration items

1. Run the LLVM CI selection on an LLVM-enabled builder and retain the job record:
   `operator_compilation_test`, `codegen_llvm_test`, `op_numeric_llvm_test`, and
   `onnx_importer_test`.
2. Supervisor reviews the stronger capability semantics, executable invariant
   validation, and public production transaction/pin adapter. This handoff does not
   self-approve.
3. Tracks 02–06 record their branch-local CoreContract consumption separately.
   Their consumption is required for M1 integration, but is no longer described as
   the only blocker or as evidence that Track01 was already complete.
4. Any DTO/key/ABI/pipeline change must state compatibility, canonical identity
   impact, and exact fallback. `RuntimeSession` must remain compiler/cache-free.
