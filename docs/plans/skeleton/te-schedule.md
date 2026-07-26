# TE Schedule → TIR target-aware skeleton

> Status: implemented and verified for `skeleton-te-schedule`.
> Scope: the production per-`PrimitiveUnit` TE → TIR path only; no autotuner,
> cache hierarchy, tensorization, or second CUDA launch-policy authority.

## 1. Current gap

The production chain is:

```text
Resolved Relay Call
  -> TOPI / te::compute DAG
  -> LowerTensorGraphToTIR
  -> target-selected TIR pipeline
  -> LLVM or CUDA codegen
```

Before this task, `te::Schedule` / `te::Stage` were disconnected from that
chain. `LowerTensorGraphToTIR` rebuilt loops directly from `ComputeOp::axis`,
so `split`, `reorder`, `vectorize`, `unroll`, and `parallel` only mutated dead
metadata. `create_schedule` also claimed to collect dependencies but created
stages only for the listed outputs. Artifact identity used the constant string
`per-unit-schedule-v2`, not the schedule that produced the kernel.

The existing CUDA authority is intentionally separate:
`tir::BindCudaThreads(PrimFunc, Target)` proves one-dimensional independent
writes, creates `ThreadBinding`, and owns launch metadata. TE scheduling must
not duplicate that proof or mint CUDA launch dimensions.

## 2. Reference model and deliberate subset

### TVM TE / TensorIR

TVM TE records stage/iteration transformations, then schedule lowering infers
leaf domains and materializes those relations as TIR loops. TensorIR makes the
same separation more explicit: schedule primitives transform loop/block IR and
a trace identifies the transformation contract. We adopt only the small common
core: a stage owns root axes, ordered leaf axes, split relations, and leaf
execution kinds; TE → TIR is the sole consumer that materializes them.

We do **not** copy TVM's complete `InferBound`, attach paths, cache read/write,
compute-at, tensorize, or auto-scheduler. Static exact shapes and the current
flat-buffer TIR need none of those yet.

### IREE / XLA

IREE keeps target configuration and loop/materialization transforms in compiler
pipelines while GPU workgroup mapping remains a later backend transform. XLA
likewise keeps HLO computation semantics separate from backend-specific launch
configuration. The useful rule here is authority separation: an explicit
`Target` selects a deterministic TE default policy, while existing TIR passes
remain authoritative for CUDA thread binding and launch metadata.

## 3. Target model

```text
TE compute DAG
  + explicit te::Schedule
  + explicit immutable Target snapshot
  + PrimFunc identity
      |
      v
validated scheduled TE → TIR
  - root-axis substitution from split relations
  - leaf loop order from reorder
  - TIR ForType from vectorize/unroll/parallel
  - tail predicate for non-divisible split
  - kxc.te.schedule_contract PrimFunc attr
      |
      +-> CPU: LLVM (ForType is semantic/diagnostic; backend remains correct)
      |
      +-> CUDA: unchanged serial TE loop -> existing BindCudaThreads pass
```

Authority rules:

1. `Target` is passed explicitly to default-schedule construction and lowering;
   ambient `PassContext` may be merged only to preserve placement attrs, never
   to choose the schedule.
2. `Schedule` is the only TE loop-transformation record. `ComputeOp` remains
   pure computation.
3. TE → TIR is the only consumer/materializer of TE schedule relations.
4. `BindCudaThreads` remains the only CUDA thread/launch authority. TE has no
   `bind` or `thread_axis` API in this subset.
5. The exact canonical schedule contract is attached to the `PrimFunc` and is
   used in `PrimitiveArtifactKey`; a policy-version placeholder is insufficient.
   It uses structural stage/axis ordinals, never graph-local tensor names or IDs.

## 4. Safe semantics

- `split(parent, factor)` requires a current leaf and a positive static factor.
  It creates `outer = ceil_div(extent, factor)` and `inner = factor`, substitutes
  `parent = min + outer * factor + inner`, and emits a bound predicate when the
  split is not exact.
- `reorder(order)` requires an exact, duplicate-free permutation of all current
  leaves. Data leaves must remain before reduction leaves, because the current
  reduction lowering materializes init before its reduction loops.
- `vectorize` accepts only the innermost data-parallel leaf; `parallel`
  accepts data-parallel leaves only.
- `unroll` accepts a data or reduction leaf.
- Every annotation requires a current leaf and may be applied once.
- CPU defaults conservatively split/vectorize an exact innermost elementwise
  axis, parallelize an independent outer data axis, and unroll a small leaf.
- CUDA defaults leave TE loops serial. Unsupported graphs continue to fail in
  `BindCudaThreads`; there is no CPU fallback and no synthetic launch metadata.

## 5. Minimal change plan

1. Make `create_schedule` collect the full producer DAG and make `Stage` retain
   root axes plus split relations. Delete the unconsumed `fuse`, `tile`, `bind`,
   and `thread_axis` surface instead of pretending it is supported.
2. Add strict validation to the five retained schedule primitives.
3. Add a compiler-owned target default policy and pass both `Schedule` and
   `Target` explicitly into `LowerTensorGraphToTIR`.
4. Materialize leaf order, split substitution/tails, and `ForType` in TIR; make
   the TIR printer expose non-serial loop kinds.
5. Canonicalize the actual stage/axis/primitive trace, attach it as
   `kxc.te.schedule_contract`, and feed that value into primitive artifact keys
   without graph-local names defeating cross-graph cache reuse.
6. Add focused schedule consumer/negative/identity tests. Reuse existing LLVM
   compiler numerical tests and CUDA `BindCudaThreads`/compiler tests.
7. Run build, full CTest, public-header/include-layer checks, and `rg` audits.

## 6. Ponytail QA record

### Round 1 — YAGNI and reuse

**Q:** Do we need a new schedule IR, generic transform dialect, or autotuner?

**A:** No. Reuse the existing `Schedule`/`Stage`, `tir::For`, `ForType`, static
extent validation, `Target`, and artifact key. Add only the split relation that
lowering needs. Skip fuse/compute-at/cache/tensorize/autotuning until a real
consumer and numerical test require them.

**Decision:** one small relation model, one lowering consumer, no dependency.

### Round 2 — architecture alignment

**Q:** Does target-aware TE scheduling create a second target, pass, or CUDA
launch authority?

**A:** No. `CompileConfig::target` is passed down explicitly. Compiler lowering
selects the TE default policy; TE itself does not depend on the Target module.
The normalized TIR pipeline still selects `bind_cuda_threads`, and that existing
pass alone proves independence and owns `ThreadBinding` plus launch metadata.

**Decision:** computation (TE), loop materialization (TE → TIR), target pass
selection (compiler), and CUDA launch mapping (TIR pass) remain one-way.

### Round 3 — consumer, identity, and dead-code audit

**Q:** Can every retained schedule API be shown to affect a consumer and can
cache identity distinguish the result?

**A:** The lowering directly consumes `split` relations, leaf order, and each of
`vectorize`/`unroll`/`parallel`; tests inspect changed TIR and execute LLVM
numerically. `create_schedule`, stages, roots, leaves, and split records all
feed lowering/canonicalization. The old `fuse`, `tile`, `bind`, and
`thread_axis` have no safe production consumer and are deleted. The canonical
schedule attr survives TIR passes and becomes the artifact key's schedule
field, so different real schedules cannot alias.

**Decision:** no retained no-op Schedule primitive and no constant-only
schedule identity.

## 7. Acceptance evidence

Required checks:

```bash
cmake --build <build-dir> -j2
ctest --test-dir <build-dir> --output-on-failure --no-tests=error
python3 tools/architecture/check_include_layers.py --root .
python3 tools/architecture/check_public_headers.py --root . --compile
rg -n 'LowerTensorGraphToTIR|LowerPrimitiveUnit|create_schedule' include src test
rg -n '\b(fuse|tile|bind|thread_axis)\b' include/kxc/te src/te src/compiler/lowering test/te_schedule_test.cpp
rg -n 'kxc.te.schedule_contract|schedule_version|BuildPrimitiveArtifactKey' src test
```

CPU default-schedule TIR and LLVM numerical execution are mandatory. CUDA
hardware execution is run when the configured toolkit/device is available;
otherwise the hardware test is reported as skipped, while synthetic target
contract tests must still prove serial TE output, exclusive
`BindCudaThreads` authority, and fail-closed unsupported scheduling.

### Recorded result

- `dev-ninja-cpu`: full build succeeded; CTest **40/40** passed, including
  `te_schedule_test`, `codegen_llvm_test`, and `op_numeric_llvm_test`.
- `dev-ninja` with CUDA toolkit: full build succeeded; non-hardware CTest
  **40/40** passed. `te_schedule_test` and `cuda_schedule_test` passed against
  complete synthetic CUDA targets.
- CUDA source emission/negative tests passed, but real launch was skipped:
  the local device probe reported `CUDA device is unavailable` and `nvidia-smi`
  reported an NVML driver/library mismatch. No CPU fallback was accepted.
- Include-layer audit passed for 265 files; all 87 installed and 7 experimental
  public headers compiled; dead Schedule API and artifact-consumer `rg` audits
  passed; `git diff --check` passed.
