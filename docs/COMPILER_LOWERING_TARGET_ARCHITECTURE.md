# Compiler authority chain

> **Status:** Implemented
> **Scope:** Relay preparation, static and control topology construction,
> primitive compilation, assembly, and runtime binding.

## Authority

`CompileConfig::Create` snapshots and validates the target, optimization level,
and profiling options. The resulting handle is read-only. A compiler entry
validates that immutable input once at its trust boundary; internal stages pass
the same configuration and resolved execution contract forward.

Production has one Relay preparation authority:

```text
Function + CompileConfig
  -> PrepareRelayProgram
  -> residual control-capability decision
```

Preparation runs the permitted Relay pipeline, produces typed ANF, and records
the residual profile. The profile is the only topology decision: no residual
control selects static dataflow; residual native control selects structured
control. Runtime never repeats this decision.

## Static compiled graph

`Compiler::Compile` accepts only the static result of preparation:

```text
PrepareRelayProgram
  -> BuildValueGraph
  -> PartitionValueGraph
  -> PrimitiveUnit[]
  -> CompilePrimitiveUnits
  -> AssembleCompiledGraph
  -> immutable CompiledGraph
  -> RuntimeSession
```

`BuildValueGraph` is the static topology authority. Its deterministic traversal
validates Relay values and resolves each ordinary call once. It produces the
logical value contracts consumed by `PartitionValueGraph`; partitioning creates
ordered `PrimitiveUnit` values and the static executable topology.

`CompilePrimitiveUnits` is the only primitive backend path. For each requested
unit it performs per-unit lowering, TIR pipeline execution, ABI construction,
artifact-key construction, cache acquisition, and backend compilation. It
returns a complete `CompiledPrimitiveBatch`; its internal phases are expressed
by the call stack and error context, not a separately published lifecycle.

`AssembleCompiledGraph` only validates ordered artifact pins against the
prepared units and target, then builds the module, static executable plan, and
immutable graph. It does not re-run Relay, lowering, cache lookup, or backend
compilation.

## Structured control graph

`Compiler::CompileControlFlowExact` is enabled only by
`KXC_ENABLE_CONTROL_RUNTIME` and accepts a prepared program whose residual
profile requires native control topology:

```text
PrepareRelayProgram
  -> LowerPreparedRelayToControlPlanWithSidecar
  -> PrimitiveUnit[]
  -> CompilePrimitiveUnits
  -> BindControlPlanForRuntime
  -> immutable CompiledControlFlowGraph
  -> ControlRuntimeSession
```

The control topology owns regions, branches, Phi routing, bounded
condition-before-body loops, and live values. It uses the same resolved-call,
logical-value, primitive-unit, primitive-compiler, and artifact-pin contracts
as the static path. Binding verifies each task against a ready module entry,
signature, ordered non-output values, and one shared retention owner before it
mints the runtime control plan.

The production subset requires LLVM CPU:0/default-stream artifacts. In an
LLVM-disabled build, the control entry remains fail-closed rather than
pretending to have executable CPU artifacts.

## Ownership and runtime boundary

`PrimitiveArtifactPin` is the internal retention authority. Public
`ArtifactPin` is a read-only view minted only from a real internal pin.
`CompilePrimitiveUnits` owns primitive-cache acquisition and completion;
`AssembleCompiledGraph` and control binding retain the pins required by their
immutable result.

The primitive cache keeps singleflight ownership, waiter/failure propagation,
backpressure, eviction, and owner cleanup internal. Cache hits and completed
backend work both yield the same ready pin contract.

`CompiledGraph` contains a `CompiledModule`, static `ExecutablePlan`, and
public pins. `CompiledControlFlowGraph` contains a sealed
`ControlExecutionPlan`. Runtime plans contain ready artifacts and runtime
value contracts, never Relay, operator-registry state, compiler callbacks, or
cache mutation authority.

## Exact shape and adaptive reuse

The exact-shape adapter prepares the same static graph, calls
`CompilePrimitiveUnits`, and publishes through `AssembleCompiledGraph`.
Adaptive replacement compiles only its requested primitive ids, substitutes
those pins into its baseline ordered pin vector, and also calls
`AssembleCompiledGraph`. Both paths therefore preserve the static module/plan
and pin ownership rules instead of maintaining a parallel assembly path.

Restricted symbolic shape is decision-only. It may freeze a concrete exact
representative and mint exact requests, but it does not compile, cache,
allocate, or execute a symbolic Relay graph.

## Verification boundary

The CPU-enabled matrix builds the compiler, topology, cache, shape, adaptive,
header, and include-layer checks; full CTest and the Relay/pass contract
scripts verify the configured source tree. LLVM and CUDA numerical execution
require their respective enabled toolchains and devices.
