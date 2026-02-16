# Multi-Device Communication IR Plan

## 1. Current Status

- Relay nodes now carry `virtual_device_` metadata.
- `PassContext` can be inferred from Relay placement and shared across Relay/TIR passes.
- `LowerToTIR` writes pass-context related attrs into `tir::PrimFunc`.

This is enough to make device-aware decisions in both IR layers.

## 2. Core Idea for Multi-Device Communication

When producer and consumer are placed on different `VirtualDevice`, communication should be
made explicit in IR, not left as an implicit runtime side effect.

Recommended split:

1. Relay level: insert communication operators at placement boundaries.
2. TIR level: lower these operators into runtime communication calls/events.

## 3. Relay-Level Representation

Add dedicated Relay ops (recommended):

- `device.copy`
- `device.allreduce`
- `device.broadcast`
- `device.scatter`
- `device.gather`

For each op, attrs should include:

- `src_virtual_device` (ObjectRef)
- `dst_virtual_device` (ObjectRef) (if applicable)
- `group_id` / `axis` / `reduce_kind` (for collectives)
- `async` (bool)

Pass to implement:

- `InsertDeviceCommunicationPass`
  - Traverse dataflow.
  - If edge crosses device boundary (`producer_vd != consumer_vd`), wrap producer output with
    `device.copy`.
  - If collective pattern detected, emit collective op instead of N copies.

## 4. TIR-Level Representation

Lower communication ops into explicit TIR side effects:

- Use `tir::Evaluate(tir::Call(DataType::Void(), "...", args))`.
- Suggested builtin names:
  - `runtime.device_copy`
  - `runtime.device_copy_async`
  - `runtime.allreduce`
  - `runtime.broadcast`
  - `runtime.scatter`
  - `runtime.gather`
  - `runtime.event_record`
  - `runtime.event_wait`

Minimal synchronous copy example:

```cpp
tir::Evaluate(
  tir::Call(
    tir::DataType::Void(),
    "runtime.device_copy",
    {src_ptr, dst_ptr, nbytes,
     tir::IntImm(src_dev_type), tir::IntImm(src_dev_id),
     tir::IntImm(dst_dev_type), tir::IntImm(dst_dev_id)}
  )
);
```

## 5. Ordering and Async Semantics

For async communication, add explicit dependency primitives:

1. `event_record(stream_src, event)`
2. `event_wait(stream_dst, event)`
3. Async copy/collective call

This avoids hidden stream-order assumptions and allows later scheduling passes to reorder safely.

## 6. Where to Use PassContext

- Relay pass:
  - Read `PassContext::Current().virtual_devices()` for cluster-level decisions.
  - Read per-node `virtual_device_` for edge-level insertion.
- TIR pass:
  - Read `PassContext::Current()` to select backend-specific runtime symbol or ABI.
  - Read `PrimFunc` attrs (already attached) when pass runs standalone.

## 7. Incremental Implementation Path

1. Add only `device.copy` in Relay + lowering to `runtime.device_copy`.
2. Add async copy with event calls.
3. Add collective ops (`allreduce/broadcast`) and map to NCCL/RCCL/Gloo or CPU fallback.
4. Add optimization pass:
   - fuse adjacent copies
   - remove redundant round-trips
   - overlap compute/communication via async streams.

## 8. Current Code Landing (Phase-1)

The following components are now implemented in-tree:

- `DiscoPlacement` and `WorkerPlacement`:
  - `include/base/disco_placement.h`
  - `src/base/disco_placement.cc`
- `PassContext` extended with Disco topology:
  - `disco_placement()` / `has_disco_placement()` / `WithDiscoPlacement(...)`
  - attrs now include `kxc.pass_ctx.disco_placement`
- Relay communication attrs:
  - `DeviceCopyAttrs`
  - `CollectiveAttrs`
- Relay communication ops registered:
  - `device.copy`
  - `device.allreduce`
  - `device.broadcast_from_worker0`
  - `device.scatter_from_worker0`
  - `device.gather_to_worker0`
  - `device.send_to_worker`
  - `device.recv_from_worker`
- New Relay transforms:
  - `BuildDiscoPlacementPass`
  - `InsertDeviceCommunicationPass`
  - `LowerRelayToExecPlanPass`
- `LowerToTIR` guard:
  - raises error if `device.*` ops are present
- Execution plan IR:
  - `include/base/execution_plan.h`
  - `src/base/execution_plan.cc`
  - `ExecNodeKind`, `KernelExec`, `CommExec`, `BarrierExec`, `ExecutionPlan`
- Self Disco runtime skeleton:
  - session + dref + cpu ccl + executor
  - `include/base/disco/*`
  - `src/base/disco/*`
- Runtime registry entry points:
  - `kxc.disco.session.threaded`
  - `kxc.disco.empty`
  - `kxc.disco.copy`
  - `kxc.disco.allreduce`
  - `kxc.disco.broadcast_from_worker0`
  - `kxc.disco.scatter_from_worker0`
  - `kxc.disco.gather_to_worker0`
  - `kxc.disco.send_to_worker`
  - `kxc.disco.recv_from_worker`
  - `kxc.disco.execute_plan`
  - `kxc.disco.execute_plan_output`
