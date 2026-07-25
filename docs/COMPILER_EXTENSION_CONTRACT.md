# Compiler Extension Contract

This document defines the mandatory contract for adding Relay operators and
compiler passes. It is normative: an extension is incomplete until its runtime
metadata, implementation binding, validation, tests, and documentation agree.

## 1. Stable identities

- Operator identity is its canonical registered name plus schema version.
- Pass identity is `(IR dialect, canonical name)` plus schema version.
- Compilation-unit semantic identity is derived from normalized operator/attrs,
  unit-local logical-to-boundary mapping, boundary tensor contracts, effect and
  alias semantics. Graph-local value/unit/storage ids, object addresses and link
  symbols are excluded.
- Graph value ids remain plan-routing locators. For one frozen static plan call,
  PrimFunc `global_symbol`, KernelSignature symbol, CompiledModule entry and
  ExecutablePlan call symbol agree. A ready cached artifact may be rebound through
  a validated signature/link alias; that symbol never decides cache equivalence.

Registries must reject conflicting duplicate identities. Registry enumeration
and serialization must be deterministic.

## 2. New operator requirements

An operator must be added in this order:

1. Declare its machine-readable contract in `contracts/relay_op_contract.json`.
2. Define or reuse an attrs schema with typed fields, defaults, legal ranges,
   and stable serialization order. Every non-fieldless `BaseAttrsNode` must
   implement `SerializeCanonical(CanonicalAttrWriter&)` and emit every field
   exactly once in schema order; compilation-unit identity includes this
   type-and-value serialization.
3. Regenerate `src/relay/generated/relay_op_contract.inc`; the generated
   `OperatorSpec` is authoritative for critical metadata, while the registration
   block binds type/lowering implementations and argument documentation.
4. Implement type validation and output-type inference.
5. Implement lowering using only the current Call's explicit inputs, attrs,
   checked type, and target capabilities.
6. Add schema, type, lowering, ABI, target-capability, and numeric tests.
7. Update the operator integration documentation and support matrix.

A normal compute operator is eligible for a compilation unit only when its
specification is complete and its lowering binding matches the declared output
arity. Lowering must not recurse into producer Calls. Unsupported targets must
fail explicitly; empty tensors, empty PrimFuncs, no-op kernels, and silent
backend fallback are not valid implementations.

Purity, effects, determinism, aliasing, and in-place behavior must be declared.
An omitted declaration is interpreted conservatively. Runtime code must never
branch on an operator name.

## 3. New pass requirements

A pass must be added in this order:

1. Declare its machine-readable contract and named/default membership in
   `contracts/pass_contract.json`.
2. Regenerate `src/pass/generated/pass_contract.inc`; C++ binding tables map only
   the generated implementation key to a function.
3. Validate dialect, scope, phase, required invariants, analyses, and target
   capabilities before execution.
4. Transform only the declared IR and scope.
5. Preserve or invalidate metadata and analyses exactly as declared.
6. Validate produced invariants after execution.
7. Add hit, no-hit, invalid-input, metadata, and determinism tests. Declared
   idempotence requires a second-run test.
8. Update `docs/PASS_CONTRACT.md`.

Adding a pass does not add it to a default pipeline. Default pipeline changes
require an explicit contract-order update and a separate review.

Framework passes that need operator semantics must query generic
`OperatorSpec` fields. Hard-coded operator-name allowlists are forbidden.

## 4. Pass scopes

| Scope | Input | May change | Must not do |
|---|---|---|---|
| graph | Complete Relay Function | Relay topology before unit freezing | Retain stale checked types |
| compilation unit | One unit and explicit boundary values | Unit-local Relay IR | Read or merge producer/consumer units |
| PrimFunc | One PrimFunc | Unit-local TIR | Change another unit or symbol identity |
| module | A set of compiled entries | Declared module metadata | Silently change a frozen Kernel ABI |

Graph topology changes finish before final value and unit identities are
assigned. After unit freezing, changing unit inputs or outputs requires a return
to the graph phase and a fresh partition.

## 5. Layer interaction

```text
Frontend / FFI
  -> validated Relay Call and Attrs
Operator registry + Relay type/graph pipeline
  -> typed Relay Function
Value graph + per-operator partition
  -> one CompilationUnit per normal compute Call
Unit lowering
  -> one PrimFunc per CompilationUnit
PrimFunc pass pipeline
  -> one validated PrimFunc with stable symbol
Kernel ABI + Codegen
  -> immutable signature and executable entry per link symbol
CompiledModule + ExecutablePlan
  -> frozen link symbols plus graph-local value routing
RuntimeSession
  -> NDArray allocation, launch, and completion only
```

Backend batching does not weaken unit identity: multiple PrimFuncs for one
target may share one LLVM JIT or CUDA module, but each unit keeps its own
symbol, signature, launch metadata, module entry, and `KernelCall`.

Primitive artifact keys contain the full `UnitSemanticKey`, compile-relevant
Target capability identity, normalized `PipelineResolver` fingerprint, kernel
ABI version, schedule version and backend implementation version. Digests are
indexes only; complete canonical bytes decide equality. Graph value/unit/storage
ids, object addresses, request heat and link symbols are never artifact identity.

Executable-plan storage reuse is a physical allocation decision, not a value
identity or aliasing mechanism. Only equal-contract intermediates with strictly
non-overlapping call intervals may share a storage id. Inputs, constants, graph
outputs, declared aliases, and async-live values always retain dedicated
storage. Runtime execution remains ordered on one stream and retains replaced
storage until the final completion object is safe to release.

The allowed dependencies are one-way:

- Frontend may depend on Relay contracts.
- Compiler may depend on Relay, Pass, TE, TIR, Codegen contracts, and Runtime
  executable-plan contracts.
- Codegen may depend on TIR and Runtime kernel ABI contracts.
- Runtime must not depend on Relay, TE, TIR, operator registry, pass registry,
  Compiler internals, or backend-private types.

## 6. Rewrite rules across the contract boundary

When a Relay pass creates or replaces a Call:

- the target operator must already be registered;
- attrs and arity must validate against `OperatorSpec`;
- changed type or shape information must be invalidated and rebuilt;
- effect and alias contracts must permit deletion, duplication, or reordering.

When a PrimFunc pass changes a function:

- the unit semantic identity remains stable; any link-symbol alias remains an
  independently validated module/plan contract;
- declared parameter roles and constant keys remain consistent;
- any ABI-changing transformation runs before ABI freeze and declares that
  phase explicitly.

No pass may mutate the operator or pass registry while a pipeline is running.

## 7. Required checks

Compiler preparation and the selected topology builder enforce the fail-closed
Relay contract at their actual boundaries. `PipelineResolver` is the production
source of pass order, invariant transitions and artifact fingerprint; direct
named pipelines remain compatibility/testing entry points.

The minimum local verification for extension-contract changes is:

```powershell
cmake --build out/build/dev-mingw-cpu --target `
  registry_test pass_pipeline_test compiler_extension_contract_test `
  check_relay_op_contract check_pass_contract `
  check_include_layers check_public_headers -j 4
```

Run the produced test executables from `test/`. Backend or numeric changes also
require the corresponding LLVM/CUDA tests; unsupported backends must have an
explicit diagnostic test.
