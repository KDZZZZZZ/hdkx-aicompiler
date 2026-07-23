# Compiler Foundation Integration Handoff

> **Local checkpoint:** W3 restricted adapters integrated and locally reviewed
>
> **Base:** `7157ca6` (`ci(compiler): close W2 integration checkpoint`)
>
> **Current feature merge head:** `00f3f3c` (`merge: integrate adaptive generation authority`)
>
> **Remote state:** W2 refs are published; this W3 checkpoint is local until an explicit publish decision
>
> **Scope:** restricted/default-OFF W3 capability slices, not completion of the W0–W4 roadmap

This handoff is the integration authority for the Compiler Foundation work. The
track-specific handoffs remain useful implementation detail, but capability and
evidence claims must not exceed this document.

## 1. Non-negotiable boundaries

The W3 integrations do not weaken the static data-plane architecture:

- `RuntimeSession` remains a static, strongly typed executor. It does not own
  Relay, Compiler, primitive-cache, Shape-specialization, adaptive-policy, or
  background-compilation dependencies.
- Restricted dynamic Shape executes through the separate
  `RuntimeShapePlan` / `RuntimeShapeSession` path.
- Adaptive compilation and generation selection stay in an upper control
  plane. A selected same-ABI generation may be replaced, but a physical
  Shape/layout/workspace ABI change still requires a new `PlanVariant`.
- Default `Compiler::Compile()` remains fail closed for Relay `If` and `While`.
  The real control-artifact path is the separate gated
  `CompileControlFlowExact()` API.
- Exact reuse is still the oracle. Bucket or polymorphic reuse requires an
  immutable decision, an explicit guard, tail-safety proof, and exact artifact
  binding; dimension inequality alone is never reuse authority.
- Logical Shape, physical capacity/profile, and valid extent remain separate
  contracts.
- Graph-local locators such as `value_id` are not semantic artifact identity.
- Manifest, generation, validation-receipt, and lease values are trusted
  process-local declarations. They are not provenance, authentication, or
  remote security credentials.
- All new W3 feature gates are OFF by default.

## 2. Integrated W3 changes

First-parent integration history after the W2 checkpoint:

```text
7721d9f merge: integrate restricted symbolic shape authority
55f9846 merge: integrate runtime shape task slice
eb40c6e merge: integrate adaptive hot swap authority v2
c49a9a5 merge: integrate gated Relay control artifact path
3b323a6 merge: bound adaptive v2 metadata authority
cb4b15c merge: integrate restricted shape runtime bridge
ef866a9 merge: register restricted Shape LLVM E2E
bb919da merge: integrate bounded Relay While path
b2ec7cd merge: integrate CUDA-safe runtime shape completion
00f3f3c merge: integrate adaptive generation authority
```

### 2.1 Restricted symbolic Shape and Shape-to-runtime bridge

Implemented:

- Fixed-rank symbolic overlays for the approved `relu`, `sqrt`, `add`, and
  `mul` subset.
- Immutable exact, bucket, and polymorphic decisions with explicit guards,
  valid extents, tail proof, semantic bindings, artifact identity, and
  unit-granular invalidation.
- A frozen runtime-only bridge with complete ABI fingerprinting, input-axis
  guards, polymorphic extent-scalar ABI, and local CPU numerical callbacks.
- Distinct physical plans for exact and bucket variants.
- Conditional LLVM test registration that compiles real static ReLU variants,
  checks exact and padded/cropped bucket results, and rejects a guard miss
  before launch.

Evidence limit:

- The bridge is trusted local runtime evidence, not a generic dynamic-output
  `CompiledModule` ABI.
- The LLVM E2E is implemented and CI-registered but not locally executed in
  this environment because LLVM is unavailable.
- This is not a general symbolic constraint solver, ragged Shape system,
  arbitrary broadcast proof, or arbitrary runtime Shape function framework.

### 2.2 Dynamic allocation and asynchronous CUDA completion

Implemented:

- `RuntimeShapePlan` / `RuntimeShapeSession` evaluate checked `ShapeExpr`
  programs before allocation and launch.
- Runtime byte budgets, checked byte arithmetic, OOM/fallback events, zero
  extents, exact physical allocation accounting, and concurrent
  `PlanVariant` isolation.
- CPU remains a trusted synchronous callback path.
- A separately gated CUDA path requires one exact CUDA device and a
  caller-supplied matching stream.
- CUDA callbacks may only enqueue work synchronously. After callback return,
  the runtime creates and records the completion event on that stream. This
  makes completion provenance runtime-owned rather than callback-forgeable.
- Result/completion ownership retains input and output storage, plan, caller
  state, and module leases until the runtime-owned event proves completion.
- Submission/record/wait failures use typed categories; ambiguous post-submit
  failures synchronize or conservatively retain/quarantine state rather than
  publishing unsafe reclamation.
- `retained_device_bytes()` reports exact storage capacity owned by this
  result. It is not global GPU-memory or allocator-cache accounting.

Local CUDA evidence:

- `runtime_shape_cuda_async_test`: **3/3 cases passed** on
  **NVIDIA GeForce GTX 1650**, driver **580.159.03**.
- The test covers deterministic pending-event retention, post-callback failure,
  launch submission failure, event-record failure, recovery, and final lease
  release.

Evidence limit:

- This proves the restricted RuntimeShape completion/lifetime protocol on one
  local GPU. It does not prove CUDA compiler/codegen numerical correctness,
  broad-rank execution, multi-GPU behavior, stream capture, performance, or
  production GPU CI.
- `compute-sanitizer` is unavailable locally, so memcheck/racecheck evidence is
  not claimed.
- The restricted runtime still does not integrate generic dynamic outputs into
  the ordinary `CompiledModule`/static-memory-plan ABI.

### 2.3 Adaptive generation authority and same-ABI replacement

Implemented behind `KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2`:

- Callable/runtime Plan ABI v4 is separated from the selected artifact
  identity.
- W2 candidate preparation is prepare-only; v2 owns the single transactional
  publish point.
- Authority issues opaque generation leases bound to route, selected artifact,
  callable ABI, validation receipt, and producer-reported bytes.
- Different launchers/artifacts/provenance records can replace one another
  under the same callable ABI; physical Plan ABI changes remain disallowed.
- Generation numbers are monotonic and nonwrapping. Old generation leases keep
  old artifacts alive for in-flight executions.
- Bounded worker pool, bounded singleflight, per-waiter cancellation/deadline,
  negative cache/retry state, admission-before-publication, producer-byte
  budget eviction, quarantine/rollback, observer isolation, callback
  re-entry rejection, and bounded route/tombstone/history metadata.
- Commit/cancel linearization and serialized health-consumption prevent stale
  compilation or health reports from reviving superseded selections.
- Repeated replacement, cancellation, retry, failure, quarantine, and eviction
  paths have focused concurrency stress coverage.

Evidence limit:

- Cancellation controls waiters and unscheduled/scheduler work. It does not
  promise hard interruption of an arbitrary backend compiler.
- Retry is bounded policy state but remains caller/request driven; no
  autonomous distributed retry service is claimed.
- Health, authority, byte accounting, and generation leases are process-local
  trusted contracts, not remote consensus or authenticated provenance.
- A local TSan binary was built, but the host TSan runtime aborted before the
  test with `FATAL: ThreadSanitizer: unexpected memory mapping`. Therefore no
  local TSan race-clean claim is made. A dedicated GitHub Actions TSan job is
  registered as the external gate.

### 2.4 Real restricted Relay `If` and bounded `While`

Implemented behind `KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION`:

- Relay now has an explicit, deterministically printed `While` node with
  mandatory nonnegative `max_trip_count` and exact static carried-state type.
- Registration, visitors/mutators, type inference, ANF normalization, manual
  Relay walkers, capability checks, semantic identity, and control-plan
  lowering understand the node.
- `While` lowers to exact `LoopSpec` carried-value contracts: initial value,
  body argument, backedge, and result must have one exact static value
  contract. Tuple-carried state is flattened deterministically.
- Conditions are CPU scalar booleans; execution is CPU:0/default-stream only.
- `If` and `While` kernel tasks are resolved through real per-kernel
  `Compiler::Compile()` in the gated production route, with typed process-local
  artifact leases.
- Zero-, one-, and multi-trip behavior, tuple-carried state, nested control,
  max-trip failure, artifact retention, and generation-overflow rejection are
  covered.
- Default `Compiler::Compile(If/While)` remains fail closed.

Evidence limit:

- Local LLVM-OFF tests validate Relay construction, typing, ANF, exact
  control-plan contracts, capability gates, reference/runtime semantics, and
  fail-closed production behavior. Real LLVM condition/body numerical
  execution is CI-registered but not locally validated.
- The node is a restricted bounded `While`, not arbitrary recursion, break or
  continue, ONNX Loop import, mutable loop state, dynamic-Shape loop-carried
  values, multi-device control, or general loop optimization.
- Runtime retention is conservative; loop-aware allocation reuse/liveness
  optimization is not claimed.

## 3. Feature gates

All remain OFF unless explicitly enabled:

```text
KXC_ENABLE_SHAPE_PRODUCTION_EXACT
KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
KXC_ENABLE_RUNTIME_SHAPE_TASKS
KXC_ENABLE_RUNTIME_SHAPE_CUDA
KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE
KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
KXC_ENABLE_CONTROL_RUNTIME
KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION
KXC_ENABLE_REGION_TASK_DAG
```

Important dependencies:

- restricted Shape runtime bridge requires production-exact Shape, restricted
  symbolic Shape, and runtime Shape tasks;
- RuntimeShape CUDA requires runtime Shape tasks and a compiled CUDA backend;
- enabling adaptive v2 causes CMake to explicitly enable the experimental
  adaptive production adapter on which it depends;
- Relay control-artifact compilation is independently gated; constructing and
  executing the resulting plan through `ControlRuntimeSession` additionally
  requires the control-runtime gate;
- the default-OFF configuration must continue to compile and reject unavailable
  capability paths deterministically.

## 4. Local verification evidence

The following results were reproduced on the combined W3 integration tree:

| Configuration | Result | Evidence tier |
|---|---:|---|
| LLVM OFF, CUDA OFF, W3 gates OFF, CPU label | **43/43 passed** | validated locally |
| LLVM OFF, CUDA OFF, W3 gates ON, CPU label | **47/47 passed** | validated locally |
| GCC 13.3 ASan+UBSan, 11 focused W3 tests | **11/11 passed** | validated locally |
| RuntimeShape CUDA local hardware suite | **3/3 passed** | validated locally on one GPU |
| Relay operator contract | **24/24** | validated locally |
| Pass contract | **20/20** | validated locally |
| Adaptive v2 repeated stress | **50/50 invocations passed** | validated locally |
| Adaptive v2 TSan | host runtime aborted before test | not validated; CI registered |
| Restricted Shape LLVM JIT E2E | target/source registered | implemented/source-and-CI-registration |
| Relay control LLVM E2E | target/source registered | implemented/source-and-CI-registration |
| ONNX importer tests | dependencies unavailable | not locally validated |
| Compute Sanitizer | tool unavailable | not validated |

The ASan+UBSan focused set is:

```text
shape_production_exact_test
restricted_symbolic_shape_test
restricted_shape_runtime_bridge_test
runtime_shape_session_test
task_plan_test
task_executor_test
runtime_session_test
relay_control_plan_test
control_runtime_integration_test
adaptive_production_experimental_test
adaptive_hot_swap_v2_test
```

Local environment constraints observed during configuration:

```text
GCC/G++ 13.3.0
Python3 with onnx and numpy was not found; skipping onnx_importer_test target.
LLVM unavailable locally.
compute-sanitizer unavailable locally.
```

## 5. CI registration

`.github/workflows/ci.yml` now registers:

- CPU matrix states that name and configure W2/W3 adapters explicitly;
- default-OFF and W3-ON CPU paths;
- an exact 11-target ASan+UBSan W3 suite;
- a dedicated adaptive-v2 TSan job;
- LLVM build/run registration for restricted Shape JIT and production Relay
  control artifacts, including bounded `While`;
- the existing ONNX job, which provisions its Python dependencies before
  configuring and running the importer test.

CI registration is not validation. At the time of this W3 checkpoint, the
externally observed repository-owned GitHub workflow state was
`disabled_manually`; this work does not re-enable it. LLVM, ONNX, TSan, and
remote CPU status may be upgraded only after authorized green runs.
No GPU-hosted GitHub runner is registered by this checkpoint.

## 6. Capability status

| Capability | Status at this checkpoint |
|---|---|
| Static exact ValueGraph/CompiledModule/RuntimeSession path | validated baseline |
| Restricted symbolic overlay and exact guard authority | implemented/local-evidence |
| Restricted Shape-to-Runtime CPU bridge | implemented/local-evidence |
| Runtime Shape allocation on CPU | implemented/local-evidence |
| RuntimeShape CUDA completion and retention | implemented/local single-GPU evidence |
| Generic dynamic-output `CompiledModule` ABI | unsupported |
| General symbolic/ragged/broadcast solver | unsupported |
| Same-callable-ABI adaptive artifact replacement | implemented/local-evidence, experimental |
| Distributed/authenticated adaptive authority | unsupported |
| Backend-independent hard compiler cancellation | unsupported |
| Relay exact `If` and bounded static `While` control plans | implemented/local-evidence |
| LLVM numerical execution of W3 Shape/control paths | implemented/source-and-CI-registration |
| General loops/recursion/dynamic carried Shape | unsupported |
| Full Transformer/KV-cache/decode/sampling | unsupported |
| Broad CUDA numerical/race/performance production support | unsupported |

## 7. Remaining W4 work

Do not promote this restricted W3 checkpoint as the final dynamic compiler
architecture. Remaining work includes:

1. A general dynamic-output `CompiledModule` calling convention and integration
   with production memory planning.
2. Broader symbolic constraints, Shape functions, broadcast semantics, and
   specialization-policy evidence.
3. Backend-specific hard cancellation where safely supported and, if needed,
   durable/distributed generation authority.
4. General control-flow lowering, frontend loop import, dynamic loop-carried
   values, and loop-aware memory reuse.
5. Full Transformer/KV-cache/decode/sampling workloads.
6. CUDA codegen/numerical coverage, Compute Sanitizer, race/lifetime tests,
   broad-rank correctness, multi-stream/multi-device behavior, and performance
   gates on supported hardware.
7. Authorized remote CI runs and an explicit integration/publish decision.

## 8. Branch and publication policy

- `origin/dev` remains unchanged at `3b95aca`.
- Published W2 refs remain at `7157ca6`:
  - `origin/integration/compiler-foundation`
  - `origin/baseline/compiler-foundation-w2`
- Do not force-update an immutable baseline.
- Publishing a W3 integration head and creating an immutable W3 baseline
  require a separate explicit decision.
- Merging the integration line into `dev` remains a repository-review decision;
  this handoff does not authorize it.
