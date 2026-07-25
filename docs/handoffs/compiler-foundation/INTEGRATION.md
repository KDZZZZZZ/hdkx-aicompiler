# Compiler foundation integration handoff

## W4-1: canonical dynamic CompiledModule invocation ABI

W4-1 replaces the former runtime-shape/bridge execution slice.  The old
RuntimeShape plan/session, fake completion, trusted launcher descriptors and
all related CMake gates are deleted.  Historical W3 evidence is intentionally
not an API claim.

Every `CompiledModule` entry owns one immutable version-2 source-private
`ModuleInvocationContract`. The contract's canonical bytes bind logical input
guards/output extents to the physical `KernelSignature`, which is the sole
source for dtype, device, rank, alignment, layout, and scope; exact runtime ABI
fingerprints include those bytes and no longer include graph-local value or
storage identifiers.  `BuildCompiledModule` automatically makes a constant
contract for static signatures and rejects dynamic physical outputs unless a
matching contract is supplied.

Public `CompiledModule::Invoke(symbol, data_inputs, stream[, budget])` accepts
only data inputs.  It validates each matching physical signature argument,
resolves checked input-axis expressions, enforces `valid <= logical <= physical`,
checks bytes and budgets (the contract budget is an immutable upper bound and a caller budget can only tighten it), allocates outputs from matching signature specs,
injects immutable module constants, and obtains completion only from
`CompiledKernel::Launch`.
`RuntimeSession` preserves preallocated memory-plan reuse through the same
internal resolver, but rejects nonstatic/scalar invocation contracts at session
construction because it has no dynamic graph memory plan.  Resolved control
launches delegate there too.  A `BoundControlKernel` snapshots
the selected entry into an immutable one-entry module before canonical
invocation, so later mutation of a caller-owned module table cannot relocate or
discard its bound symbol, signature, contract, or constants.

The default is OFF (`KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI`).  Static contracts
remain executable with the gate OFF; nonconstant contracts fail before
allocation or launch.  W4-1 supports input-shape-derived extents, zero extents,
and an explicit module-generated `kRuntimeExtent` `uint64[1]` buffer ABI.  It does
not support data-dependent/ragged shapes, workspace codegen, arbitrary tail
transforms, generic symbolic Relay lowering, compiler-emitted scalar contracts,
or dynamic graph memory planning.  Rank-zero static tensors remain valid
static ABI values and are distinct from generated extent-scalar buffers.

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

CUDA has no additional W4 public-Invoke evidence in this change.  The local W3
CUDA machine evidence applied to the deleted callback runtime and is not
reused as evidence for this ABI; existing CUDA tests are not claimed as W4
dynamic-module coverage.  No callback completion is used by the module ABI.

### Local verification

The post-cleanup branch was rebuilt after the final contract and gate isolation fixes:

| Configuration | Result | Evidence tier |
|---|---:|---|
| LLVM OFF, CUDA OFF, all foundation gates OFF, CPU label | **37/37 passed** | validated locally |
| LLVM OFF, CUDA OFF, all foundation gates ON, CPU label | **38/38 passed** | validated locally |
| LLVM OFF, CUDA OFF, adaptive ON with Shape gates OFF, CPU label | **38/38 passed** | validated locally |
| ASan+UBSan: module/session/control focused set | **3/3 passed** | validated locally |
| Relay operator contract | **24/24 passed** | validated locally |
| Pass contract | **20/20 passed** | validated locally |
| NLP reference/capability checker | **PASS** | validated locally |
| Include-layer/public-header/YAML/diff checks | **PASS** | validated locally |
| Real LLVM dynamic scalar E2E | source and CI registered | not executed locally |
| W4 CUDA public-Invoke E2E | absent | unsupported evidence |

Legacy RuntimeShape/bridge cache variables now fail CMake configuration with a
migration diagnostic.  No live legacy API symbol remains under `include/`,
`src/`, or `test/`.

Generic Relay emission, dynamic graph memory planning, bucket/polymorphic
execution, ragged/data-dependent outputs, and data-dependent output allocation
remain unsupported. The compiler Shape bucket/polymorphic contract universe was
deleted; no guarded policy/profile/request or compatibility alias remains.

`ModuleShapeExpr` and `ModuleInvocationContract` are source-private
CompiledModule invocation typestate for compiler/runtime-authored contracts;
public clients cannot author or inspect them. They do not consume restricted
symbolic Shape decisions and do not provide generic Relay symbolic-to-module
lowering. An uncertain post-launch completion failure is
retained for process lifetime rather than risking early release; this is a
safety quarantine, not bounded failure recovery or normal memory accounting.

Public C++ API additions require recompilation; no ABI compatibility is
claimed. The default-off Shape and Adaptive control-plane headers are tracked
separately in `KXC_EXPERIMENTAL_HEADERS` for repository tests and are not part
of installed/exported package API.
