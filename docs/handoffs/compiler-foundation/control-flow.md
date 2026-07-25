# Compiler Foundation / Control-flow handoff

## Current authority

`Compiler::CompileControlFlowExact` is the only control-flow publication entry.
It is available only when `KXC_ENABLE_CONTROL_RUNTIME` is enabled and follows
the shared preparation/primitive chain:

```text
Relay Function + immutable CompileConfig
  -> PrepareRelayProgram
  -> LowerPreparedRelayToControlPlanWithSidecar
  -> PrimitiveUnit[]
  -> CompilePrimitiveUnits
  -> BindControlPlanForRuntime
  -> sealed immutable runtime::ControlExecutionPlan
  -> CompiledControlFlowGraph -> runtime::ControlRuntimeSession
```

`PrepareRelayProgram` produces typed ANF and the residual control profile.
This entry requires residual native control; a program without it must use
`Compiler::Compile` and the static fast path. The control builder owns regions,
branches, Phi routing, bounded loops, live values, and task dependencies. Each
ordinary Relay call is resolved once in that topology traversal, and each task
references the shared `PrimitiveUnit` boundary.

`CompilePrimitiveUnits` is shared with static compilation. Binding accepts only
ready artifacts, verifies task-to-unit correspondence, module entry, signature,
and ordered non-output values, and retains the selected public pins through a
single opaque shared owner. The result stays executable after its compiler
wrapper is copied or released without exposing cache or generation authority.

The preparation and binding seams are source-private:

- `src/compiler/control_flow/control_plan.h`
- `src/compiler/control_flow/internal_lowering.h`
- `src/runtime/internal/control_execution_plan_spec.h`
- `src/runtime/internal/control_execution_plan_access.h`

Installed callers can inspect or copy a compiler-minted
`ControlExecutionPlan`; they cannot author a runnable plan or bind artifacts.

## Supported subset and fail-closed boundary

Production control requires available LLVM CPU:0/default-stream artifacts and
supports structured Relay `If`, exact Phi forwarding, and
condition-before-body `While` with a positive static maximum trip count. Values
have static dtype/rank/shape, kernels are pure, inputs are read-only, and
outputs are freshly allocated.

The path rejects recursion, general ONNX Loop, unbounded loops, dynamic carried
shape, dynamic graph memory planning, non-CPU placement, non-default streams,
host callbacks, synchronization effects, unsupported alias contracts, malformed
Phi/backedge mappings, dynamic invocation contracts, and invalid backend
completion. Runtime execution contains no Relay, compiler, primitive-cache,
shape policy, adaptive controller, or background compilation dependency.

## Verification

Run `control_plan_test`, `relay_control_plan_test`, and
`control_runtime_integration_test` with the control gate on, together with full
CTest and the public-header/include-layer checks. LLVM-off runs prove
source-level behavior and fail-closed gating only; numerical production control
requires an LLVM-enabled CPU build.
