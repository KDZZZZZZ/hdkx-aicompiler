# W4-1 migration record: dynamic CompiledModule ABI

The restricted Shape runtime bridge was deleted in W4-1.  It no longer provides
an execution path, launcher descriptor, completion protocol, or CMake gate.

`CompiledModule` is the sole invocation owner.  An immutable, versioned
`ModuleInvocationContract` resolves guards and all logical/physical/valid
extents before allocation, injects immutable constants, and launches the real
`CompiledKernel`.  `KernelSignature` is the sole physical ABI authority:
contracts contain only guards, extent expressions, limits, and generated
scalar expressions.  Generated extent scalars occupy explicit ordered
`kRuntimeExtent` signature slots and are one-element `uint64` buffers;
callers never provide them.  This role is backend registration, not proof that
arbitrary machine code consumes the scalar semantically; that remains backend
E2E evidence.

`RuntimeSession` and control execution fail closed for non-static contracts
until graph dynamic-memory planning exists. `ModuleShapeExpr` is CompiledModule
invocation typestate, not a generic Relay symbolic-to-module lowering path;
restricted symbolic Shape exact decisions do not feed it. Generic Relay
symbolic lowering, buckets/polymorphic execution, ragged/data-dependent
shapes, workspace schemas, and tail transforms remain unsupported. A backend
must explicitly register and consume a generated scalar descriptor; no bridge
path exists.
