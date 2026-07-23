# Runtime Shape W3 handoff

`KXC_ENABLE_RUNTIME_SHAPE_TASKS=OFF` remains the default. With the gate on,
`RuntimeShapeSession` evaluates a frozen CPU:0 plan in the fixed synchronous
order `ShapeEval -> Allocate -> trusted launcher`.

## Contract boundary

- `RuntimeShapeBoundLauncher` is a trusted local **synchronous-only** callback.
  It receives runtime-owned argument storage, must not retain arguments, and
  must not submit async device work. It is not a `CompiledModule` ABI or an
  executor/fence contract.
- `RuntimeShapePlan::ExactAbiFingerprint` emits canonical full ABI bytes for
  input/output contracts (including expressions, dtype, rank, layout, scope,
  input-data requirements, bounds, device, ABI versions), ordered polymorphic
  scalar ABI mappings, selected artifact identity, and bucket tail-policy
  identity. Scalar ordinals have no gaps or duplicates and their names,
  symbols, input axes, ranges, and divisibility domains are validated. Plan
  construction requires an exact byte match with the selected trusted entry;
  evaluated scalar values and the same bytes are passed to the launcher.
- `FakeRuntimeShapeCompletion::Pending()` is only a deterministic result
  lifetime-retention simulation. `RunAsync` has a compatibility name but has
  already invoked the launcher when it returns; `Wait()` only completes that
  fake token and never waits for device work.

## Safety constraints

- v1 accepts only `layout == "contiguous.row_major"`, `scope == "global"`, and
  `CPU:0`. `RuntimeShapeInputContract::requires_data` defaults to false for
  existing runtime-only plans; when true, the caller must provide the exact
  shape-derived byte count and a non-null pointer for a nonempty CPU input
  before ShapeEval. Zero extents are valid and produce zero-byte, null-data
  outputs.
- Expression depth is bounded (64) and expression node count is bounded
  (4096). Arithmetic and byte calculations are checked.
- Output allocation first enters a `unique_ptr` RAII owner, then transfers to
  the result `shared_ptr`; a throwing transfer cannot leak raw storage.
  `FailNextRuntimeShapeOwnerTransferForTest` is a deterministic proxy for that
  transfer path, not a deterministic injection into a `shared_ptr` control
  block allocation. The latter is implementation-dependent and has no
  deterministic seam; focused ASan/LSan covers the no-leak property.
- The shared trusted launcher is serialized by a plan-owned mutex. Callback
  exceptions, including non-`std::exception` values, become failed results.
  A kernel event is emitted only after callback acceptance.
- Each result retains the frozen plan/module lease, caller lease, output
  owners, and optional fake token. Output storage is fresh for every run; no
  allocation reuse occurs.

Focused coverage exercises gate ON/OFF, exact ABI mismatch, scalar ABI mapping,
required CPU input data, zero-byte output, layout/scope restrictions,
expression depth, callback failures, RAII transfer proxy, synchronous fake
retention, module/caller/output lease lifetime, and serialized concurrent
callbacks. Bridge relu/add numeric checks are trusted synchronous local CPU
evidence only, including predicated bucket-tail/crop and guard-miss behavior;
they are not `CompiledModule`, JIT, LLVM-lowering, or asynchronous-device proof.
