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
execution path.

## Deliberate limits

- Dynamic, ragged, data-dependent, bucketed, and polymorphic execution are not
  supported.
- Compiler publication is single-target. Distributed runtime, CPU CCL,
  workers, sessions, and execution-plan data structures remain independent
  until they can launch real compiled modules with numerical coverage.
- CUDA and LLVM numerical claims require their enabled local toolchain and
  hardware; a CPU-only LLVM-disabled build does not establish those claims.

## Verification

Use the configured CPU matrix to build compiler, pass, shape, adaptive,
control, runtime, and header/layer targets, then run full CTest with
`--no-tests=error`. Run the Relay and pass contract scripts alongside
`git diff --check`. Treat unavailable CUDA-host validation as blocked, not as a
successful substitute.
