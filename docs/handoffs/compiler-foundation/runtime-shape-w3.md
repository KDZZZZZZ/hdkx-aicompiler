# Historical W3 RuntimeShape handoff (superseded)

This file is a migration tombstone, not a current capability document.

W4-1 deleted the former `RuntimeShapePlan`, `RuntimeShapeSession`, fake
completion, trusted CPU/CUDA launcher callbacks, restricted Shape runtime
bridge, tests, and feature gates.  Supplying one of the removed CMake cache
variables now fails configuration rather than selecting a compatibility path.
There is no hidden or deprecated callback executor.

The W3 CPU/CUDA results remain historical evidence for the immutable W3
checkpoint only.  They are not evidence for the replacement module ABI.
Current architecture, gates, evidence, and unsupported scope are recorded in
[`INTEGRATION.md`](INTEGRATION.md).
