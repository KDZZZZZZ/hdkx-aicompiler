# Compiler Foundation Adaptive / Hot-Swap handoff

> State: default-off experimental; not production-ready.

## Preparation seam

`kxc::api::adaptive::experimental::production_path` is preparation only. Its
repository contracts are `ProductionCompileRequest`, `ProductionExecutionRequest`,
`ProductionPathCompilerAdapter`, `PreparedCandidate`, and `PrepareCandidate`.
The adaptive headers are source-tree experiments in `KXC_EXPERIMENTAL_HEADERS`;
they are not installed/exported SDK API and carry no compatibility promise.
The request snapshots `CompileConfig`/`Target`, derives exact graph, dispatch,
and Plan ABI identities, and retains the verified baseline pins. Preparation
rejects malformed Relay graphs, dynamic shapes, wrong call mapping, ABI changes,
and an empty injected receipt. A prepared candidate owns its `CompiledGraph`,
pins, immutable `RuntimeSession`, selected-artifact identity, and receipt.

It does not publish, route, mint generations, lease artifacts, execute,
quarantine, roll back, observe, queue, retry, or make health decisions. Those
are not compatibility aliases and are deliberately absent. Production adapters
remain the injection seam for v2 or another injected compiler.

Focused preparation gate:

```bash
cmake -S . -B out/adaptive-preparation-on -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION=ON
cmake --build out/adaptive-preparation-on --target \
  adaptive_preparation_experimental_test check_include_layers check_public_headers
ctest --test-dir out/adaptive-preparation-on --output-on-failure \
  -L adaptive-preparation-experimental -L cpu
```

The preparation test covers request snapshot/exact identity, candidate
validation (including factory attack rejection), malformed graph rejection, and
candidate pin/session lifetime. It is built only when preparation is ON and v2
is OFF; the v2 binary includes those cases when v2 is ON.

## v2 sole lifecycle authority

`kxc::api::adaptive::hot_swap::v2` is the sole adaptive routing, generation,
lease, health, quarantine, rollback, and asynchronous execution authority.
Its `GenerationLease` directly owns `shared_ptr<const PreparedCandidate>` plus
the monotonic generation, exact route, selected `PlanVariantKey`, Plan ABI,
validation receipt, and producer-reported bytes. Read-only `candidate()`,
`compiled_graph()`, and `session()` accessors expose the prepared payload;
there is no frozen-plan wrapper.

`GenerationAuthority::MakeLease` validates and binds that candidate directly.
Publication remains transactional under the route lock, rejects non-monotonic
authority output, preserves cancellation/deadline isolation, and keeps
negative-cache, quarantine, rollback, byte accounting, completion retention,
and static RuntimeSession data-plane isolation unchanged. Route eviction only
removes controller discoverability: external leases and completion retention
continue to own candidate pins and sessions.

```bash
cmake -S . -B out/adaptive-v2-on -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=ON
cmake --build out/adaptive-v2-on --target run_adaptive_hot_swap_v2_tests \
  check_include_layers check_public_headers
ctest --test-dir out/adaptive-v2-on --output-on-failure \
  -L adaptive-hot-swap-v2 -L cpu
```

The v2 suite includes an external-lease regression: after route eviction and
controller destruction, the retained lease still launches its candidate session
and retains pins; both release when the final lease drops.

No claim is made for backend hard cancellation, authentication/attestation,
numeric health truth, native/device-resident byte accounting, CUDA pending
completion, or TSan unless the corresponding environment actually passes it.
GitHub Actions enablement state is unchanged.
