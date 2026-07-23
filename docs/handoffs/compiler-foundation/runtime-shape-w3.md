# Runtime Shape W3 handoff

`KXC_ENABLE_RUNTIME_SHAPE_TASKS=OFF` and `KXC_ENABLE_RUNTIME_SHAPE_CUDA=OFF`
remain the defaults. The established CPU:0 path remains `ShapeEval -> Allocate
-> RuntimeShapeBoundLauncher` and its launcher is trusted synchronous-only.

## Explicit execution contracts

- `RuntimeShapeExecutionKind::kSynchronousCpu` uses only
  `RuntimeShapeBoundLauncher`; it cannot silently become asynchronous.
- `RuntimeShapeExecutionKind::kCudaAsync` uses the sibling
  `RuntimeShapeCudaBoundLauncher`, which returns a defined same-device
  `kxc::AsyncOperation`. It is available only when all three conditions hold:
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

The CUDA completion handle retains output `Storage` and a separate retention
object containing the frozen plan/module lease, caller lease, and input
owners/storage. The result holds the completion but the completion does not
hold the result, so this has no ownership cycle. `IsReady` and `Wait` delegate
to the returned `AsyncOperation`; async mode accepts neither fake completion
nor a missing/wrong-device completion. The CPU-only
`FakeRuntimeShapeCompletion` remains only a deterministic legacy retention
seam.

`retained_device_bytes()` is exact per-result CUDA output storage retained by
that result/completion; it is not a process-wide resident-memory claim. It can
remain nonzero after completion because `AsyncOperation` intentionally retains
its storage until final completion/result ownership drops. Events distinguish
shape evaluation, allocation, CPU kernel acceptance or CUDA submission,
observed completion, failure, and ownership retirement. Retire explicitly does
not claim physical deallocation.

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
serialized trusted callbacks. The conditional CUDA test uses a real CUDA event
and stream blocker, checks pending retention across session/owner destruction,
wait/accounting, budget and guard failure before launch, rejected/wrong-device
completion, and CTest skip code 77 when no CUDA hardware exists.

Local CUDA validation ran the conditional `runtime_shape_cuda_async_test` with
`KXC_ENABLE_RUNTIME_SHAPE_CUDA=ON` against an available CUDA device. This is
restricted runtime stream/event/retention evidence only; it does not establish
CUDA kernel code generation, JIT, or general dynamic-shape support.
