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

`restricted_symbolic_shape.h`, gated by
`KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE`, is exact-decision-only. `Prepare`
freezes a concrete representative through the exact adapter, overlays explicit
nonempty symbol bindings, and accepts only fixed-rank `relu`/`sqrt` and
equal-shape `add`/`mul`. `MintExact` produces immutable request snapshots. It
does not compile, cache, allocate, execute, or lower a symbolic Relay graph.

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
