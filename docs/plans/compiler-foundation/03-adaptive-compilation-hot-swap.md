# 03 — Adaptive compilation / hot-swap

## Status

The production-path experimental gate is a preparation seam, not a standalone
adaptive lifecycle. `KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2` is default OFF and is
the sole whole-plan lifecycle authority when enabled.

## Preparation contracts

`adaptive_production_experimental.h` exposes only:

- `ProductionCompileRequest`
- `ProductionExecutionRequest`
- `ProductionPathCompilerAdapter`
- `PreparedCandidate`
- `PrepareCandidate` and validation helpers

Requests snapshot their compiler config and exact baseline. Preparation creates
a pinned immutable candidate and static `RuntimeSession`, rejecting malformed
graphs and incompatible exact ABI/call mappings. It does not route, publish,
mint generations, lease, execute, observe, quarantine, or roll back.

## v2 authority

`AdaptiveHotSwapController` v2 owns queueing, singleflight, cancellation and
deadline isolation, retry/negative-cache policy, generation issuance,
transactional publication, exact routing, health/quarantine/rollback,
discoverability eviction, byte accounting, and completion retention.

A `GenerationLease` directly owns `shared_ptr<const PreparedCandidate>` and
binds it to a monotonic generation, route, selected plan key, Plan ABI,
validation receipt, and producer bytes. It exposes read-only candidate, graph,
and session accessors. There is no frozen-plan lifecycle wrapper. Eviction only
removes controller discoverability: retained leases and completion dependencies
continue to own the candidate pins/session.

## Build/test gates

Preparation-only gate:

```bash
cmake -S . -B out/adaptive-preparation-on -G Ninja \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION=ON
cmake --build out/adaptive-preparation-on \
  --target run_adaptive_preparation_experimental_tests
```

The target is `adaptive_preparation_experimental_test`, label
`adaptive;adaptive-preparation-experimental;cpu`. It covers request snapshots,
candidate validation, malformed graph rejection, exact identity, and
pin/session lifetime. It is created only when preparation is ON and v2 is OFF;
the v2 binary includes the same preparation cases when v2 is ON.

v2 gate:

```bash
cmake -S . -B out/adaptive-v2-on -G Ninja \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=ON
cmake --build out/adaptive-v2-on --target run_adaptive_hot_swap_v2_tests
```

The v2 suite additionally verifies that an externally retained lease keeps its
candidate pins and session alive after route eviction and controller
destruction. TSan, CUDA pending completion, native/device-resident bytes,
authentication/attestation, and numeric-health claims require separate passing
environments and are not implied here.
