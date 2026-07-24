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
internal resolver, but rejects nonstatic/scalar invocation contracts at session
construction because it has no dynamic graph memory plan.  Task-DAG and
resolved control launches delegate there too.  A `BoundControlKernel` snapshots
the selected entry into an immutable one-entry module before canonical
invocation, so later mutation of a caller-owned module table cannot relocate or
discard its bound symbol, signature, contract, or constants.

The default is OFF (`KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI`).  Static contracts
remain executable with the gate OFF; nonconstant contracts fail before
allocation or launch.  W4-1 supports input-shape-derived extents and zero
extents only.  It does not support data-dependent/ragged shapes, scalar or
workspace codegen, arbitrary tail transforms, generic symbolic Relay lowering,
or dynamic graph memory planning.  Rank-zero static tensors remain valid
static ABI values; this is not runtime-scalar code generation.

### Backend evidence

`compiled_module_dynamic_llvm_test` is conditionally registered only when both
`KXC_USE_LLVM` and `KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI` are enabled.  It
constructs real TIR with physical parameters `(input, generated uint64[1]
extent, output)`, compiles it through `CodeGenLLVM` and `LLVMJITEngine`, builds
a `CompiledModule`, and invokes the public ABI for `S=0,2,3`.  The JIT body
loads the generated scalar and uses that load as its loop bound while writing
`2*S` deterministic numeric outputs; the test also verifies a guard miss.
This checkout has no local LLVM toolchain, so this is **source plus explicit CI
registration only**, not local LLVM execution evidence.

CUDA has no additional W4 public-Invoke evidence in this change: local CUDA
tooling/hardware is unavailable, and existing CUDA tests are not claimed as
dynamic-module coverage.  No callback completion is used by the module ABI.

Generic Relay emission, dynamic graph memory planning, bucket/polymorphic
execution, ragged/data-dependent outputs, and data-dependent output allocation
remain unsupported.

Public C++ API additions require recompilation; no ABI compatibility is
claimed.
