# Compiler Foundation Core handoff — supervisor fix round 2

> **Branch:** `feature/compiler-foundation-core`
> **Status:** implementation ready for supervisor re-review; **not** “Core complete” and
> not “blocked only by 02–06”. LLVM-enabled CI execution and final review remain open.
> **Runtime boundary:** unchanged — `RuntimeSession` remains a static
> `CompiledModule + ExecutablePlan` data-plane executor.

## Closed in this round

| Requirement | Production implementation | Evidence |
|---|---|---|
| Executable capability truth | `CapabilityResult` distinguishes `unsupported`, `eligible_but_not_executable`, and `executable`. `supported` is true only after `CapabilityVerifier` executes the real normalized per-unit compile path through lowering, target schedule, signature and backend. This executable probe intentionally participates in production cache/singleflight; it is not a metadata-only query. Backend-disabled builds and incomplete CUDA snapshots remain eligible but not executable. | `test/compiler_capability_test.cpp`: control flow/dynamic/tuple rejection, `nn_gemm transA`, custom lowering shape mismatch, backend unavailable, CUDA compute capability, CUDA reduction schedule, and `supported => Compiler::Compile` matrix. |
| One production execution plan | `Compiler::Compile` resolves one Relay/TIR execution contract per invocation. Compiler Relay plans explicitly contain pre/post `infer_type`; `PipelineExecutor` executes the `NormalizedPipeline` and revalidates step identity, binding, phase, invariant transitions, target requirements, canonical bytes, checked type and CUDA launch metadata. `LowerGraph` no longer runs an unrecorded InferType pass. | `test/pipeline_resolver_test.cpp`; production integration in `src/compiler/compiler.cc`; compatibility-only inference remains in `LowerToTIR`/`LowerOperatorCallsToTIR`. |
| Artifact identity from execution | Artifact keys consume canonical bytes from the exact resolved Relay plan, per-unit lowering version, exact TIR plan/schedule, ABI and backend version. Digest remains indexing/diagnostic only; full canonical bytes decide equality. | `test/compiler_identity_test.cpp`, `test/primitive_cache_test.cpp`, LLVM block in `test/operator_compilation_test.cpp`. |
| Public production adapter | `CompiledGraph::artifact_pins` exposes CoreContract `ArtifactPin`s backed by opaque strong owners of real `PrimitiveArtifactPin`s. `ProductionArtifactCacheAdapter` provides read-only full-key lookup/stats; a miss never creates a singleflight owner. | `primitive_cache_test::production_adapter_read_only_pin`; LLVM cache test checks compiler-returned pins resolve in the production cache. |
| Production pin/cache/singleflight | Existing full-canonical cache, same-key singleflight, failure TTL, bounds/backpressure, ABI relocation and eviction-safe pin semantics are preserved. | `primitive_cache_test`, `compiler_foundation_contract_test`, LLVM relocation case in `operator_compilation_test`. |
| CPU CI closure | CTest registers all 23 CPU/core executables plus Relay/pass contracts, include-layer and compiled public-header checks. CPU workflow runs the complete `cpu` label. | Local CPU-only run: **27/27 passed**. |
| LLVM CI closure | LLVM workflow explicitly builds and runs operator compilation (including symbol relocation/cache reuse), LLVM codegen, per-op numeric, and ONNX compile tests. | Workflow configured; this worktree has no LLVM package, so execution remains an external CI requirement. |
| Legacy documentation | `Compiler::Compile -> per-unit LowerGraph -> CompiledModule + ExecutablePlan` is the production authority. Whole-graph `relay::LowerToTIR` is compatibility/testing-only and cannot establish support. | `MODULE_GUIDE.md`, `OP_SUPPORT_MATRIX.md`, Track01 plan/roadmap. |

## Frozen public/value surface

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
utilities; production cache retention is supplied by the adapter above.

## Verification performed locally

Configured without downloads or external access:

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
2. Supervisor reviews the stronger capability semantics, normalized execution
   identity, and public production-pin adapter. This handoff does not self-approve.
3. Tracks 02–06 record their branch-local CoreContract consumption separately.
   Their consumption is required for M1 integration, but is no longer described as
   the only blocker or as evidence that Track01 was already complete.
4. Any DTO/key/ABI/pipeline change must state compatibility, canonical identity
   impact, and exact fallback. `RuntimeSession` must remain compiler/cache-free.
