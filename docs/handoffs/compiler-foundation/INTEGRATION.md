# Compiler foundation integration handoff

## W4-1: canonical dynamic CompiledModule invocation ABI

W4-1 replaces the former runtime-shape/bridge execution slice.  The old
RuntimeShape plan/session, fake completion, trusted launcher descriptors and
all related CMake gates are deleted.  Historical W3 evidence is intentionally
not an API claim.

Every `CompiledModule` entry owns one immutable version-1
`ModuleInvocationContract`.  The contract's canonical bytes bind logical input
guards/output extents to the physical `KernelSignature`; exact runtime ABI
fingerprints include those bytes and no longer include graph-local value or
storage identifiers.  `BuildCompiledModule` automatically makes a constant
contract for static signatures and rejects dynamic physical outputs unless a
matching contract is supplied.

Public `CompiledModule::Invoke(symbol, data_inputs, stream[, budget])` accepts
only data inputs.  It validates shape/dtype/device/layout, resolves checked
input-axis expressions, enforces `valid <= logical <= physical`, checks bytes,
alignment/layout/scope and budgets, allocates outputs, injects immutable module
constants, and obtains completion only from `CompiledKernel::Launch`.
`RuntimeSession` preserves preallocated memory-plan reuse through the same
internal resolver.  Task-DAG and resolved control launches delegate there too.

The default is OFF (`KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI`).  Static contracts
remain executable with the gate OFF; nonconstant contracts fail before
allocation or launch.  W4-1 supports input-shape-derived extents and zero
extents only.  It does not support data-dependent/ragged shapes, scalar or
workspace codegen, arbitrary tail transforms, generic symbolic Relay lowering,
or dynamic graph memory planning.

Public C++ API additions require recompilation; no ABI compatibility is
claimed.
