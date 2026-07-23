# Restricted Shape Runtime Bridge

`KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE=OFF` is the default. Enabling it requires all of:

- `KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON`
- `KXC_ENABLE_RUNTIME_SHAPE_TASKS=ON`
- `KXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON`

The bridge accepts a frozen restricted decision and a trusted synchronous local launcher descriptor, then returns an immutable `RuntimeShapePlan`. Runtime input-axis guards are canonical ABI data and are checked before ShapeEval, allocation, or launcher invocation.

Only the final output of the current tree-only relu/sqrt/equal-shape add/mul subset is bridged. Exact decisions bind exact input guards and constants. Bucket decisions use dynamic logical/valid extents with a fixed validated physical boundary and tail contract. Polymorphic decisions use dynamic logical/physical/valid extents after validating the runtime-extent ABI.

This is trusted synchronous local evidence only. It does not create a `CompiledModule`, establish a dynamic-output `CompiledModule` ABI, prove JIT/LLVM lowering, or establish device asynchronous safety.
