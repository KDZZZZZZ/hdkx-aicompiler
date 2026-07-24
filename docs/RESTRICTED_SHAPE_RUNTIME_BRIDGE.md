# W4-1 migration record: dynamic CompiledModule ABI

The restricted Shape runtime bridge was deleted in W4-1.  It no longer provides
an execution path, launcher descriptor, completion protocol, or CMake gate.

`CompiledModule` is now the sole owner of entry invocation.  Every entry binds
its physical `KernelSignature` to an immutable versioned
`ModuleInvocationContract`; `CompiledModule::Invoke` validates input guards,
resolves checked input-axis expressions, allocates physical output storage,
injects module constants, and launches the real `CompiledKernel`.
`RuntimeSession` and resolved control execution use the same internal
preallocated-output hook.

Restricted symbolic Shape remains an upper-plane decision facility only.
Bucket/polymorphic execution is unsupported until the compiler emits a real
module invocation contract and matching code generation.  W4-1 supports only
input-shape-derived tensor extents (including zero extents); it does not claim
data-dependent or ragged shapes, scalar extent codegen, workspace schemas,
generic symbolic Relay lowering, tail transforms, or dynamic graph planning.
