# Historical W3 Runtime Shape handoff (superseded by W4-1)

> Historical migration text only. The described RuntimeShape APIs were removed in W4-1; see `INTEGRATION.md`.

`KXC_ENABLE_RUNTIME_SHAPE_TASKS=OFF` and `KXC_ENABLE_RUNTIME_SHAPE_CUDA=OFF`
remain the defaults. The established CPU:0 path remains `ShapeEval -> Allocate
-> RuntimeShapeBoundLauncher` and its launcher is trusted synchronous-only.

## Explicit execution contracts

- `RuntimeShapeExecutionKind::kSynchronousCpu` uses only
  `RuntimeShapeBoundLauncher`; it cannot silently become asynchronous.
- `RuntimeShapeExecutionKind::kCudaAsync` uses the sibling
  `RuntimeShapeCudaBoundLauncher`, which returns only
  `RuntimeShapeCudaLaunchResult {accepted, failure_reason}`. The callback may
  synchronously enqueue work only on the supplied stream; it must not use other
  streams/threads, retain its arguments, or provide an event/`AsyncOperation`.
  `accepted=false` is the trusted explicit proof that no work was submitted; it
  reports `kLaunchRejected` and releases the unsubmitted outputs. The runtime
  creates an event before the callback and records it on that supplied stream
  immediately after an accepted return or a throw, so accepted work has a
  runtime-owned pending completion fence. It is available only when all three
  conditions hold:
  runtime-shape tasks are enabled, `KXC_ENABLE_RUNTIME_SHAPE_CUDA=ON`, and a
  CUDA backend was actually compiled (`KXC_USE_CUDA=1`). CMake rejects an
  enabled CUDA runtime-shape gate without those prerequisites.
- A CUDA entry has one canonical `CUDA:N` device contract. Every input and
  output contract, supplied input `Storage`, and supplied stream must match it;
  an omitted stream is the CUDA default stream for that device. Layout remains
  `contiguous.row_major`, scope remains `global`, and checked output sizes plus
  the per-plan byte budget are validated before any allocation or submission.
- CUDA input must supply same-device `Storage`, its exact shape-derived byte
  count, and its matching data pointer. CUDA output is allocated with
  `Storage::Alloc`; there is no host allocation or CPU fallback.

`ExactAbiFingerprint` preserves legacy CPU v1 bytes. CUDA entries append the
explicit execution-kind and device contract binding, so a CPU/synchronous ABI
cannot be reused as a CUDA/async ABI.

## Lifetime, completion, and outcomes

The runtime-owned CUDA completion handle retains output `Storage` and a
separate retention object containing the frozen plan/module lease, caller
lease, and input owners/storage. The result holds the completion but the
completion does not hold the result, so this has no ownership cycle. `IsReady`
and `Wait` delegate to that runtime-created `AsyncOperation`; callback code has
no API for fake or wrong-stream completion. Outputs, base events, and base
failure text are immutable after `Run` returns. Polling only updates atomic
completion/failure flags; `events()` returns a value snapshot that synthesizes
completion/retire/failure telemetry, so concurrent polling/getters are safe.
The CPU-only `FakeRuntimeShapeCompletion` remains only a deterministic legacy
retention seam.

`retained_device_bytes()` is exact per-result CUDA output storage retained by
that result/completion; it is not a process-wide resident-memory claim. It can
remain nonzero after completion because `AsyncOperation` intentionally retains
its storage until final completion/result ownership drops. `kRetire` is emitted
only with an observed CUDA completion as logical retirement eligibility; it
never claims a physical free and is never synthesized by result destruction.

After a callback could have submitted work, runtime event-recording,
completion-wrapper, retention, and callback-throw failures synchronize the
supplied stream before outputs are cleared. If synchronization cannot prove
completion, the complete run state (outputs, input/module/caller leases,
stream, runtime event, and completion handle) is permanently quarantined and
the failed result reports retained bytes instead of assuming no work was
submitted. This is intentionally process-lifetime retention: safety takes
precedence over reclamation and matches `CudaModuleLauncher`'s post-launch
error protocol. Explicit rejection has the stronger no-submission declaration,
so its unrecorded runtime event and outputs can be released immediately.
Failure-event telemetry allocation is contained where a result already exists;
the initial result-state `make_shared` can still throw before any result can be
represented.

`RuntimeShapeFailureKind` lets an upper control plane distinguish disabled,
applicability/guard miss, shape/ABI rejection, resource exhaustion/OOM, launch
rejection, submission failure, and completion failure. `RuntimeShapeSession`
does not select variants, retry, or fallback on capacity/guard failure. Any
fallback is an explicit upper-plane action to another prevalidated frozen
`PlanVariant`.

## Coverage and evidence

CPU focused coverage exercises gate ON/OFF, exact ABI mismatch, scalar ABI,
required CPU data, zero-byte output, layout/scope restrictions, checked sizes,
budget and callback failure, RAII transfer, fake CPU retention, leases, and
serialized trusted callbacks plus a concurrent poll/getter stress test suitable
for TSan. The conditional CUDA test uses real CUDA events and host callbacks to
check that the runtime-owned event stays pending across session/owner
destruction, post-submit event-record and retention failures synchronize before
clearing output, throw-after-submit is fenced, and explicit rejection clears
unsubmitted outputs. It also statically verifies that callback outcomes cannot
contain a forged `AsyncOperation`. CTest uses skip code 77 when no CUDA hardware
exists.

Local CUDA validation ran the conditional `runtime_shape_cuda_async_test` with
`KXC_ENABLE_RUNTIME_SHAPE_CUDA=ON` against an available CUDA device. This is
restricted runtime stream/event/retention evidence only; it does not establish
CUDA kernel code generation, JIT, or general dynamic-shape support.
