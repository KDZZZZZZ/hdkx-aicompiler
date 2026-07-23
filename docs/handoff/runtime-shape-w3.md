# Runtime Shape W3 handoff

`KXC_ENABLE_RUNTIME_SHAPE_TASKS=OFF` is the default.  With the gate enabled,
`RuntimeShapeSession` executes a frozen `RuntimeShapePlan` on CPU:0/default
in the fixed order `ShapeEval -> Allocate -> Kernel`.

Public API:

- `include/kxc/runtime/runtime_shape_plan.h`: immutable expression DTO,
  input/output contracts, frozen plan, and the runtime-owned bound launcher.
- `include/kxc/runtime/runtime_shape_session.h`: result/events and per-run
  session API.

The bound launcher deliberately is **restricted local evidence**, not a real
`CompiledModule` dynamic-output signature. `FakeRuntimeShapeCompletion` is a
deterministic retention-test seam and explicitly is **not CUDA evidence**.

Each call evaluates shapes and allocates fresh CPU memory privately. The async
result retains its plan (and thus entry module lease), output owners, private
run state, and supplied opaque caller lease. There is no allocation reuse or
mutable shared shape state.

Validation rejects zero extents, checked arithmetic/byte overflow, input
rank/dtype/device/ABI mismatches, invalid `valid <= logical <= physical`, bad
alignment, output max-byte limits, and aggregate byte-budget overflow before
launch. Allocation and launch errors are returned with a failure event/reason.
