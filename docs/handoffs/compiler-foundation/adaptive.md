# Compiler Foundation Adaptive / Hot-Swap handoff

> State: default-off experimental; not production-ready.

## v2 lifecycle

`KXC_ENABLE_ADAPTIVE_HOT_SWAP` is the only adaptive gate and the sole
lifecycle authority. Its internal preparation stage snapshots the compiler
configuration and exact baseline, validates candidates, and creates a pinned
static `RuntimeSession`; it is not independently buildable or testable.

`AdaptiveHotSwapController` owns queueing, singleflight, cancellation and
deadline isolation, retry/negative-cache policy, generation issuance,
transactional publication, exact routing, health/quarantine/rollback,
discoverability eviction, byte accounting, and completion retention.
`GenerationLease` directly owns its prepared candidate, pins, and session.

```bash
cmake -S . -B out/adaptive-on -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_ADAPTIVE_HOT_SWAP=ON
cmake --build out/adaptive-on --target run_adaptive_hot_swap_tests \
  check_include_layers check_public_headers
ctest --test-dir out/adaptive-on --output-on-failure \
  -L adaptive-hot-swap -L cpu
```

The adaptive headers remain source-tree experiments and are not installed or
exported SDK API.

No claim is made for backend hard cancellation, authentication/attestation,
numeric health truth, native/device-resident byte accounting, CUDA pending
completion, or TSan unless the corresponding environment actually passes it.
