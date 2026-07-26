# Compiler Foundation / Shape handoff

## Current authority

The repository-only experimental Shape headers are source-tree checked but are
not installed or exported and carry no source or binary compatibility promise.

`shape_specialization.h` defines graph templates, exact profiles, exact
requests, and changed-unit inputs. `shape_exact.h`, gated by
`KXC_ENABLE_SHAPE_PRODUCTION_EXACT`, is the concrete Relay adapter. Its
preparation and publication use the static compiler authority:

```text
PrepareRelayProgram
  -> BuildValueGraph -> PartitionValueGraph
  -> PrimitiveUnit[]
  -> CompilePrimitiveUnits
  -> AssembleCompiledGraph
  -> immutable exact CompiledGraph variant
```

The adapter accepts one concrete profile and static `RuntimeSession` execution.
It does not maintain a separate lowering, cache, module, or plan assembly path.
`ExactPlanVariant` carries a `DispatchKey` minted from
`BuildStaticExactDispatchKey(template semantic key, oracle profile key)` and
reconstructible from public identity builders.

The oracle-derived and plan-derived `ShapeProfileKey` spaces intentionally
diverge today: the oracle key encodes the full template canonical plus
bindings under policy `"exact"`, while the adaptive plan-derived key encodes
only the plan input boundary under policy `"static-exact-plan-v2"`. The
divergence is locked by an explicit inequality assertion in
`test/shape_production_exact_test.cpp`; aligning the two spaces (including
adaptive route-identity migration) is a separate future plan (issue #46).

`restricted_symbolic_shape.h`, gated by
`KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE`, is exact-decision-only. `Prepare`
freezes a concrete representative through the exact adapter, overlays explicit
nonempty symbol bindings, and accepts only fixed-rank `relu`/`sqrt` and
equal-shape `add`/`mul`. `MintExact` produces immutable request snapshots. It
does not compile, cache, allocate, execute, or lower a symbolic Relay graph.

The restricted adapter additionally closes the dispatch loop inside the oracle
key space. `MaterializeExactFunction` replays the frozen unit dataflow into a
compilable concrete Function for a minted decision; compile intent stays with
the caller invoking `Compiler::Compile` explicitly. `ExactDispatchKey` mints
the route identity for a decision (one route family per prepared template),
`VerifyCompiledExactVariant` binds a compiled graph to its authorizing
decision by boundary shape/dtype and unit count, and `BindingsFromInputShapes`
converts request-boundary input shapes into canonical bindings, failing closed
on conflicting shared symbols, rank mismatch, or static-axis mismatch. The
control plane is caller-owned (a plain map in
`test/shape_exact_dispatch_test.cpp`); there is no production variant table,
router, or dispatcher class, request-boundary lookups never compile, and a
route miss is an explicit failure with primitive cache statistics unchanged.
Publishing exact variants into the adaptive controller remains a non-goal
until a real hot-swap need for a single shape variant appears.

Source-private module invocation contracts validate compiler/runtime-authored
invocations. They do not consume restricted symbolic decisions or establish a
generic symbolic compilation path.

## Unsupported

Dynamic, ragged, data-dependent, bucketed, and polymorphic shapes; generic
symbolic Relay-to-module lowering; dynamic graph memory planning; dynamic
output allocation; workspace schemas; and tail transforms are unsupported.
`-1` remains a legacy input-ABI sentinel, not symbolic Shape support.

## Verification

Use the CPU-only LLVM/CUDA-disabled configured build with exact and restricted
symbolic Shape gates enabled. Build the shape tests plus
`check_include_layers` and `check_public_headers`, then run full CTest with
`--no-tests=error`. Backend execution claims still require an enabled backend
and its environment.
