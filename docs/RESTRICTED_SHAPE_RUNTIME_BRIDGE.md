# W4-1 migration record: dynamic CompiledModule ABI

The restricted Shape runtime bridge was deleted in W4-1.  It no longer provides
an execution path, launcher descriptor, completion protocol, or CMake gate.

`CompiledModule` is the sole invocation owner.  An immutable, versioned
`ModuleInvocationContract` resolves guards and all logical/physical/valid
extents before allocation, injects immutable constants, and launches the real
`CompiledKernel`.  Generated extent scalars are ordered, one-element `uint64`
input buffers immediately after caller data-input slots in `KernelSignature`;
their descriptors and ordered ASTs are canonical ABI bytes.  Callers never
provide these buffers.

`RuntimeSession` and control execution fail closed for non-static contracts
until graph dynamic-memory planning exists.  Generic Relay symbolic lowering,
buckets/polymorphic execution, ragged/data-dependent shapes, workspace
schemas, and tail transforms remain unsupported.  A backend must explicitly
register and consume a generated scalar descriptor; no bridge path exists.
