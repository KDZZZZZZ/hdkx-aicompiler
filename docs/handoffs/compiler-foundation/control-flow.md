# Compiler Foundation / Control-flow handoff

## Current authority

The only installed production minting path is:

```text
Relay Function
  -> Compiler::CompileControlFlowExact
  -> compiler-private ControlPlan preparation
  -> compiler-private real-artifact binding
  -> sealed immutable runtime::ControlExecutionPlan
  -> runtime::ControlRuntimeSession
```

`Compiler::Compile` remains the static-dataflow API and rejects Relay control
flow. The complete compiler/runtime path is default OFF behind the single
`KXC_ENABLE_CONTROL_RUNTIME` gate.

The preparation DTO and authoring seams are source-private:

- `src/runtime/internal/control_plan.h`
- `src/compiler/control_flow/internal_lowering.h`
- `src/runtime/internal/control_execution_plan_spec.h`
- `src/runtime/internal/control_execution_plan_access.h`

The former installed `kxc/compiler/control_flow.h` and
`kxc/runtime/control_plan.h` headers were deleted. There is no installed
`LowerRelayToControlPlan`, `BindControlPlanForRuntime`, `ControlKernelBinding`,
`ControlFlowArtifactLease`, fixture revision, generation, or mutable execution
plan spec.

Installed callers can only copy and inspect a compiler-minted
`ControlExecutionPlan`; its defined constructor and `BoundControlKernel`
constructor are private. Read-only task/region DTOs are not accepted by any
installed executable-plan factory.

## Supported subset

The production path supports only static-exact CPU control:

- Relay `If` with structured then/else regions and exact Phi forwarding;
- condition-before-body `While` with a positive static `max_trip_count`;
- static CPU:0 values, default stream, fixed dtype/rank/shape;
- pure kernels with read-only inputs and freshly allocated outputs;
- real one-call `Compiler::Compile` artifacts for every kernel task.

The lowering freezes the Relay call payload in a compiler-private sidecar.
Diagnostic `kernel_ref` and source locators never select an executable.
Binding uses task id, a ready module entry, exact ordered non-output values, and
signature output order. Constants are snapshotted by each bound module entry.

A single opaque shared owner retains the selected production `ArtifactPin`s.
Every bound kernel holds that same owner. A copied execution plan therefore
remains executable after the `CompiledControlFlowGraph` wrapper is destroyed;
no lease or generation authority is exposed.

## Fail-closed boundary

The path rejects recursion, general ONNX Loop, unbounded loops, dynamic carried
shape, dynamic graph memory planning, non-CPU placement, non-default streams,
host callbacks, synchronization effects, unsupported alias contracts, malformed
Phi/backedge mappings, dynamic invocation contracts, and invalid backend
completion.

Runtime execution contains no Relay, Compiler, primitive cache, adaptive
controller, shape policy, or background compilation dependency.

## Verification

CPU-only gate-OFF and gate-ON builds must run:

- `control_plan_test`
- `control_plan_reference_executor_test`
- `relay_control_plan_test`
- `control_runtime_integration_test`
- full CTest, public-header manifest, and include-layer checks

Real numerical production `If`/`While` cases remain conditional on an LLVM
build. A local LLVM-OFF pass proves source behavior and fail-closed gating, not
LLVM execution evidence.
