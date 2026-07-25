# W4-1 migration record: dynamic CompiledModule ABI

The restricted Shape runtime bridge was deleted in W4-1.  It no longer provides
an execution path, launcher descriptor, completion protocol, or CMake gate.

`CompiledModule` is the sole invocation owner.  A source-private immutable,
versioned `ModuleInvocationContract` resolves guards and all logical/physical/valid
extents before allocation, injects immutable constants, and launches the real
`CompiledKernel`.  `KernelSignature` is the sole physical ABI authority:
contracts contain only guards, extent expressions, limits, and generated
scalar expressions.  Generated extent scalars occupy explicit ordered
`kRuntimeExtent` signature slots and are one-element `uint64` buffers;
callers never provide them.  This role is backend registration, not proof that
arbitrary machine code consumes the scalar semantically; that remains backend
E2E evidence.

`RuntimeSession` and control execution fail closed for non-static contracts
until graph dynamic-memory planning exists. Source-private `ModuleShapeExpr` is
CompiledModule invocation typestate, not a generic Relay symbolic-to-module lowering path;
restricted symbolic Shape exact decisions do not feed it. Generic Relay
symbolic lowering, buckets/polymorphic execution, ragged/data-dependent
shapes, workspace schemas, and tail transforms remain unsupported. A backend
must explicitly register and consume a generated scalar descriptor; no bridge
path exists.

## Static artifact reassembly and publication

For static plans, backend recompilation is per `PrimitiveUnit`. A complete ordered
`PrimitiveArtifactPin` vector is the artifact selection: its index is the unit id,
and its pins remain the sole ABI, executable, and lifetime authority.
`AssembleCompiledGraph()` is the shared compiler-internal free assembler for Normal,
Shape exact, and Adaptive paths. It consumes prepared static topology, ordered pins,
and immutable constants to create a new immutable `CompiledGraph`; that graph is the
Adaptive publication generation, while leases retain old generations for execution.

Adaptive replaces only explicitly requested units, then reassembles the complete
static graph. Equal ordered artifact keys against the current route are a no-op.
Control flow remains separate: its resolved-plan assembler produces
`ControlExecutionPlan`, not a static `CompiledGraph`.

Only callers of `CompileAndPublish()` or `Submit()` create compile intent.
`Acquire()` and `RunAsync()` only select and execute an already published generation.
Thus compile-before-run, use-current-generation, and run-old-while-submitting-a-
replacement are API compositions, not a mode enum or policy hierarchy.
