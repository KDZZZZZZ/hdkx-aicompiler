# 03 — Adaptive compilation / hot-swap

## Status

`KXC_ENABLE_ADAPTIVE_HOT_SWAP` is default OFF and is the only adaptive
lifecycle gate. Preparation is an internal hot-swap stage, not a separately
enabled feature.

## Hot-swap authority

`AdaptiveHotSwapController` owns queueing, singleflight, cancellation and
deadline isolation, retry/negative-cache policy, generation issuance,
transactional publication, exact routing, health/quarantine/rollback,
discoverability eviction, byte accounting, and completion retention.

Its internal preparation validates the exact compiler request and candidate,
then creates a pinned immutable candidate and static `RuntimeSession`. It does
not expose an independently enabled preparation lifecycle.

A `GenerationLease` directly owns `shared_ptr<const PreparedCandidate>` and
binds it to a monotonic generation, route, selected plan key, Plan ABI,
validation receipt, and producer bytes. Eviction only removes controller
discoverability: retained leases and completion dependencies keep candidate
pins/session alive.

## Build/test gate

```bash
cmake -S . -B out/adaptive-on -G Ninja \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_ADAPTIVE_HOT_SWAP=ON
cmake --build out/adaptive-on --target run_adaptive_hot_swap_tests
```

The hot-swap suite verifies external lease retention after route eviction and
controller destruction. TSan, CUDA pending completion, native/device-resident
bytes, authentication/attestation, and numeric-health claims require separate
passing environments and are not implied here.
