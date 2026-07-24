# Core contracts, identity, and primitive cache

## Current normative boundary

The public compiler artifact surface is `include/kxc/compiler/artifact.h`:
immutable `ArtifactRecord` plus opaque read-only `ArtifactPin`. A defined pin can only be minted by the compiler's real primitive
cache. `CompiledGraph::artifact_pins()` exposes these retained real pins; no
public cache mutation, lookup, transaction, candidate, request, ticket, or
policy DTO exists.

`Compiler::Compile` and `src/compiler/internal/primitive_cache.h` are the sole
primitive-cache mutation authority. The internal cache uses complete typed
`PrimitiveArtifactKey::canonical_bytes()` as authority, never a digest alone. It
implements static-exact same-key singleflight, bounded in-flight backpressure,
negative failure TTL, bounded ready-entry/byte eviction, and eviction-safe pins.
Compiler-owned failure cleanup releases unfinished owners.

Adaptive v2, not this cache, owns scheduling, routing, dispatch and generation
policy, priorities, budgets, and asynchronous cancellation. `RuntimeSession`
never performs cache lookup or compilation.

## Required evidence

Internal primitive-cache tests cover concurrent same-key flights, digest-collision
separation, waiter success/failure results, failure TTL/bounds, in-flight
backpressure, post-eviction and oversized pins, and near-`UINT64_MAX` byte
accounting. Compiler/operator/shape tests cover the opaque public pins carried by
real compiled graphs.

## Historical note

Earlier drafts in this plan discussed fake stores, generic compile requests and
tickets, selected/frozen plan DTOs, and a public production cache adapter. Those
were planning-only designs and are not current API or ownership claims.
