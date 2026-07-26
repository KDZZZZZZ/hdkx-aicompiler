# Architecture status

> **Status:** Current implementation guide. Source and repeatable tests are the
> final authority for capability claims.

## Compiler and runtime

The compiler publishes two immutable runtime products from one prepared Relay
program model.

```text
Static:
Function + immutable CompileConfig
  -> PrepareRelayProgram
  -> BuildValueGraph -> PartitionValueGraph
  -> PrimitiveUnit[] -> CompilePrimitiveUnits
  -> AssembleCompiledGraph -> CompiledGraph -> RuntimeSession

Residual native control:
Function + immutable CompileConfig
  -> PrepareRelayProgram
  -> LowerPreparedRelayToControlPlanWithSidecar
  -> PrimitiveUnit[] -> CompilePrimitiveUnits
  -> BindControlPlanForRuntime
  -> CompiledControlFlowGraph -> ControlRuntimeSession
```

`Compiler::Compile` is the static entry and rejects residual control topology.
`Compiler::CompileControlFlowExact` is gated by
`KXC_ENABLE_CONTROL_RUNTIME`; it accepts only the native static-exact control
subset and requires LLVM CPU:0/default-stream artifacts for executable
production output.

Preparation produces typed ANF and a residual control profile. The profile is
the sole topology choice. Static and control builders each resolve ordinary
Relay calls during their own deterministic traversal and share logical values,
primitive units, primitive compilation, artifact keys, cache behavior, ABI,
and artifact-pin ownership.

## Publication and ownership

`CompilePrimitiveUnits` is the sole lowering/TIR/signature/cache/backend path.
It returns a complete primitive batch. `AssembleCompiledGraph` validates
ordered pins and publishes the static module and plan without repeating prior
work. Control binding validates each ready artifact against the control task
before publishing its sealed runtime plan.

The primitive cache keeps its singleflight, failure, backpressure, eviction,
and owner-cleanup mechanisms internal. Public artifact pins are read-only
views of real retained artifacts. Runtime receives immutable modules and plans;
it does not inspect Relay, lower primitives, compile artifacts, or mutate the
cache.

Exact shape and adaptive replacement reuse the static preparation/primitive
compilation/assembly chain. Restricted symbolic shape only makes exact
concrete decisions; it does not create a generic symbolic compilation or
execution path. A restricted decision can be materialized into a compilable
concrete Function and routed at the request boundary by a caller-owned map
keyed on oracle-space dispatch keys; compile intent stays with the caller's
explicit `Compiler::Compile`, and a route miss never compiles implicitly.

### Data contracts

The one-way rule constrains *control and execution* dependencies, not the
naming of shared data types. `NDArray`, `DeviceStream`, `DeviceInfo` /
`DeviceAttributes` in `include/kxc/runtime/device_info.h`, and the types in
`include/kxc/runtime/kernel_abi.h` are neutral data contracts: compiler-tier
code may name them, because the artifacts it produces must conform to them.
`relay::Constant` holding a `runtime::NDArray` is therefore correct and not a
layering violation, and `target/target.h` including `runtime/device_info.h` is
a legal downward edge under the declared module order (runtime sits below
target).

What the rule forbids is the reverse edge. Runtime must not include
`kxc/compiler/`, `kxc/relay/`, `kxc/te/`, or `kxc/tir/` headers, and must not
inspect Relay, lower primitives, or select variants. That direction is what
keeps `RuntimeSession` a static plan executor.

One deliberate carve-out exists: the compiled-module implementation under
`src/runtime/internal/` names `Target` because published modules record the
target they were built for. `tools/architecture/check_include_layers.py`
models this as the `runtime_executable` seam; it is intentional, enforced, and
not a module cycle.

## Deliberate limits

- Dynamic, ragged, data-dependent, bucketed, and polymorphic execution are not
  supported.
- Compiler publication is single-target. Distributed runtime, CPU CCL,
  workers, sessions, and execution-plan data structures remain independent
  until they can launch real compiled modules with numerical coverage.
  Disposition (2026-07-26): the module stays in the tree and in the default
  build as roadmap work; issue #12 (real `CompiledKernel` invocation) is the
  tracked next step, and reaching it is the revisit condition for this bullet.
- CUDA and LLVM numerical claims require their enabled local toolchain and
  hardware; a CPU-only LLVM-disabled build does not establish those claims.

## Source layout

`src/compiler/` keeps only the orchestration surface at top level —
`compiler.cc`, `compile_config.cc`, `pipeline_resolver.cc`, matching the
public entry headers. Every other implementation lives in a domain
subdirectory (`abi/`, `adaptive/`, `analysis/`, `cache/`, `control_flow/`,
`graph/`, `identity/`, `lowering/`, `primitive/`, `shape/`); private headers
stay in `internal/`.

Cross-module includes between `src/` modules are src-rooted (the `src/`
directory is a declared include root for module objects and test
executables), e.g. `#include "runtime/internal/compiled_module_node.h"` —
never `../` filesystem traversal. Same-module includes of `internal/` headers
remain includer-relative.

Contract wellformedness has one implementation: `shape::ContractDefect` /
`shape::IsExactContract` in `include/kxc/shape/shape.h`. The specialization
verifier and both shape adapters delegate to it; do not write another copy.
Relay AST snapshot cloning lives in `src/compiler/analysis/relay_snapshot.cc`
behind `src/compiler/internal/relay_snapshot.h`, not in the shape layer.

## Repository policy

Large binary fixtures are generated or fetched, never committed.
`resnet18.onnx` (45 MB) predates this rule and is permanently in history; it
stays tracked, but it is the last of its kind. New fixtures follow the
workbench precedent: commit the generator or fetch script, not the artifact.

## Verification

Use the configured CPU matrix to build compiler, pass, shape, adaptive,
control, runtime, and header/layer targets, then run full CTest with
`--no-tests=error`. Run the Relay and pass contract scripts alongside
`git diff --check`. Treat unavailable CUDA-host validation as blocked, not as a
successful substitute.
