# Repository implementation overview

## Read the compiler by authority

The active compiler path is deliberately small:

```text
Function + immutable CompileConfig
  -> PrepareRelayProgram
  -> static: BuildValueGraph -> PartitionValueGraph
  -> control: LowerPreparedRelayToControlPlanWithSidecar
  -> PrimitiveUnit[]
  -> CompilePrimitiveUnits
  -> static: AssembleCompiledGraph
  -> control: BindControlPlanForRuntime
  -> immutable runtime plan
```

Start at `include/kxc/compiler/compiler.h` and
`src/compiler/compiler.cc`. `Compiler::Compile` publishes static
`CompiledGraph` values. `Compiler::CompileControlFlowExact`, behind
`KXC_ENABLE_CONTROL_RUNTIME`, publishes a `CompiledControlFlowGraph` only for
the supported residual-control subset.

`PrepareRelayProgram` lives in `src/compiler/analysis/relay_program.cc`. It
runs Relay preparation and records the residual capability profile. That
profile selects topology once: the static builder is
`src/compiler/graph/value_graph.cc` plus `src/compiler/graph/partition.cc`;
the control builder is `src/compiler/control_flow/relay_control_plan.cc`.
Each builder validates the nodes it owns and resolves an ordinary call once.

`PrimitiveUnit` is the shared boundary. The implementation in
`src/compiler/primitive_compiler.cc` lowers each unit, runs the TIR pipeline,
builds the kernel ABI and artifact identity, uses the primitive cache, and
returns ready artifact pins. `AssembleCompiledGraph` in
`src/compiler/compiler.cc` creates the static module and plan from ordered
pins. `src/compiler/control_flow/production_control_flow.cc` binds the same
ready artifacts to the control topology.

## Ownership and execution

`CompiledGraph` and `CompiledControlFlowGraph` are immutable publication
products. Their runtime plans receive ready module entries, value contracts,
and retained artifact ownership. `RuntimeSession` and `ControlRuntimeSession`
validate and execute those plans; they do not depend on Relay, the compiler, or
the primitive cache.

The cache implementation is private to `src/compiler/cache`. Its singleflight,
failure propagation, backpressure, eviction, and pin lifetime rules are
consumed through `CompilePrimitiveUnits`. `ArtifactPin` is a read-only public
view, not cache mutation authority.

## Adjacent features

Exact shape preparation and adaptive replacement reuse the static chain and
static assembly. Restricted symbolic shape is an exact-decision surface only;
it does not lower or execute symbolic Relay. Distributed runtime and CPU CCL
remain independent runtime facilities: compiler publication is single-target
until an execution plan can launch real compiled modules with numerical proof.

## Useful verification entry points

- `test/compiler_contract_test.cpp`, `test/compiler_identity_test.cpp`, and
  `test/operator_compilation_test.cpp` cover static compiler contracts.
- `test/primitive_cache_test.cpp` covers cache ownership and failure behavior.
- `test/control_plan_test.cpp` and
  `test/control_runtime_integration_test.cpp` cover control topology and
  binding.
- `test/shape_production_exact_test.cpp`,
  `test/restricted_symbolic_shape_test.cpp`, and
  `test/adaptive_preparation_test.cpp` cover the adjacent feature boundaries.
- `check_include_layers`, `check_public_headers`,
  `python/tools/check_relay_op_contract.py`, and
  `python/tools/check_pass_contract.py` protect source contracts.

LLVM and CUDA execution are environment-dependent. An LLVM/CUDA-disabled CPU
run validates configured source behavior but is not numerical backend evidence.
