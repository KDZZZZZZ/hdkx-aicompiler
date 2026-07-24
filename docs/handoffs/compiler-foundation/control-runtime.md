# Compiler Foundation / Control runtime handoff

## Runtime contract

`runtime::ControlRuntimeSession` consumes only a sealed
`runtime::ControlExecutionPlan`. It cannot construct, resolve, compile, cache,
or replace a kernel.

The installed execution plan exposes read-only metadata and immutable regions,
tasks, values, branches, loops, and bound kernels. Its authoring spec and
constructors are private source contracts. Public code cannot turn copied DTOs
or a caller-selected module entry into an executable plan.

Each `BoundControlKernel` snapshots one ready module entry, constants,
`KernelSignature`, launch metadata, and executable. It retains an opaque owner
minted by the compiler. Launch delegates to the canonical compiled-module
invoker and backend `AsyncOperation`; no callback completion protocol exists.

## Execution semantics

- one CPU:0 default stream;
- static exact value contracts;
- branch executes exactly one selected region and applies validated Phi values;
- bounded condition-before-body loop records exact iterations and fails when a
  still-true condition exceeds `max_trip_count`;
- graph inputs/constants are read-only and may alias;
- each kernel output is freshly allocated;
- constants must match the private module snapshot;
- async results retain the plan, bound modules, storage, and opaque pin owner
  until backend completion.

`KXC_ENABLE_CONTROL_RUNTIME` is default OFF. Gate-OFF construction/launch fails
before backend dispatch.

## Unsupported

There is no recursion, arbitrary loop topology, dynamic carried shape, general
ONNX Loop, dynamic allocation task, cross-device copy, borrowed stream,
workspace protocol, runtime compiler callback, adaptive lookup, or public
fixture binding API.

## Tests

Source-private fixture builders are used only by focused tests. They pass an
explicit test-owned retention object through the same private binding path; no
test contract is compiled into production sources.

The focused suite covers branch non-selection, Phi and loop differential
semantics, zero/one/multiple/exhausted trips, duplicate operands, read-only
input aliasing, exact argument/output order, aligned constants, malformed
plans, invalid completion, copied-plan retention, and gate-OFF rejection.
LLVM production numerical evidence remains conditional on an LLVM-enabled
build.
