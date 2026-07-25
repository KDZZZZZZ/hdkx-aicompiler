# Compiler Foundation Core handoff

## Current authority

`CompileConfig::Create` makes an immutable target/configuration snapshot.
`Compiler::Compile` is the static publication authority:

```text
PrepareRelayProgram
  -> BuildValueGraph -> PartitionValueGraph
  -> PrimitiveUnit[]
  -> CompilePrimitiveUnits
  -> AssembleCompiledGraph
  -> immutable CompiledGraph
```

`CompilePrimitiveUnits` owns per-unit lowering, TIR processing, ABI and
artifact-key construction, primitive-cache acquisition, backend compilation,
and completion. It returns one complete batch of ready primitive pins.
`AssembleCompiledGraph` consumes only the prepared static graph, ordered pins,
and constants to validate correspondence and publish the module and executable
plan. It never repeats preparation, lowering, cache lookup, or backend work.

`include/kxc/compiler/artifact.h` exposes immutable `ArtifactRecord` and
read-only opaque `ArtifactPin` views. A defined public pin comes only from a
real internal primitive pin. Clients cannot create, publish, fail, or mutate a
cache entry.

The internal primitive cache owns static-exact full-key singleflight, failure
propagation, bounded in-flight backpressure, byte/entry eviction, and pin
retention. `CompilePrimitiveUnits` acquires all leases before waiting and
publishes work owned by the current compilation before resolving waiters, so
opposite key acquisition orders do not create cross-flight wait cycles.

Exact shape and adaptive replacement reuse this chain. Adaptive compilation
may request only selected units, but substitutes their ready pins into the
baseline ordered vector and publishes through `AssembleCompiledGraph`.

`RuntimeSession` remains compiler/cache-free. Consumers must not include
`src/compiler/internal/*` or `src/runtime/internal/*`.

## Verification

Use primitive-cache, compiler/operator, exact-shape, and adaptive regressions
for this boundary, plus the public-header, include-layer, Relay-contract, and
pass-contract checks. LLVM/CUDA execution remains environment-dependent.
