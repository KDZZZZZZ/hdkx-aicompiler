# Restricted Shape Runtime Bridge

`KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE=OFF` is the default. Enabling it requires all of:

- `KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON`
- `KXC_ENABLE_RUNTIME_SHAPE_TASKS=ON`
- `KXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON`

The bridge accepts a frozen restricted decision and a trusted synchronous local launcher descriptor, then returns an immutable `RuntimeShapePlan`. Runtime input-axis guards, input-data requirements, ordered polymorphic runtime scalars (ordinal/name/symbol/input-axis/domain), selected artifact identity, and bucket tail-policy identity are canonical entry-ABI data and are checked before ShapeEval, allocation, or launcher invocation. A descriptor must bind the exact final-unit artifact identity; bucket descriptors must also bind the exact `BucketPolicy::CanonicalString()` identity.

Only the final output of the current tree-only relu/sqrt/equal-shape add/mul subset is bridged. Bridge inputs require caller-owned CPU data with the exact shape-derived byte count. Exact decisions bind exact input guards and constants. Bucket decisions use dynamic logical/valid extents with a fixed validated physical boundary and a predicated/cropped tail contract. Polymorphic decisions use dynamic logical/physical/valid extents; their scalar values are evaluated from checked input axes and passed to the launcher in ABI ordinal order.

The relu/add checks are trusted synchronous local CPU numeric evidence: they verify logical values and bucket tail/crop behavior, including guard misses. This is not `CompiledModule` evidence: it does not create a `CompiledModule`, establish a dynamic-output `CompiledModule` ABI, prove JIT/LLVM lowering, or establish device asynchronous safety.
