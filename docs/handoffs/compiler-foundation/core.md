# Compiler Foundation Core handoff

## Current boundary

`include/kxc/compiler/artifact.h` exposes only immutable `ArtifactRecord` and
read-only opaque `ArtifactPin` views. A defined public pin is
minted only from a real internal primitive-cache pin; clients cannot create,
publish, fail, or otherwise mutate cache entries.

`Compiler::Compile` and `src/compiler/internal/primitive_cache.h` are the sole
mutation authority. The internal primitive cache owns static-exact full-key
singleflight, negative TTL, bounded in-flight backpressure, byte/entry eviction,
and pin retention. `Compiler::BuildBackends` acquires every primitive lease
before waiting: it publishes all leases it owns first, then resolves waiters.
This prevents opposite key acquisition orders from forming cross-flight wait
cycles. A defined `CompiledGraph` is minted only by the compiler-private
access seam from compiler-derived module, plan, pins, and graph key; public
callers can only consume an existing graph and its opaque real pins.

Adaptive v2 owns scheduling, routing, dispatch/generation, priority/budget, and
asynchronous coordination. Those policies are not represented by the public
artifact view or by a generic foundation cache API.

`RuntimeSession` remains compiler/cache-free. Consumers must not include
`src/compiler/internal/*` or `src/runtime/internal/*`.

## Verification

Use the internal primitive-cache regressions plus compiler/operator/shape tests
for cache behavior, and the public-header/include/contract checks for the public
surface. LLVM/CUDA execution remains environment-dependent.
