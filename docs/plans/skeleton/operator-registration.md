# Relay declaration-driven operator registration

## Goal

Make `contracts/relay_op_contract.json` the single declaration authority for Relay
operator metadata and, incrementally, for the registration binding itself.  This
follows the useful parts of TVM's Op+attribute registry, MLIR ODS/TableGen,
ONNX Runtime schemas, and XLA HLO registration: one declarative schema owns
stable names and structural metadata, while handwritten implementations retain
only executable callbacks.

## Phase 1/2 shape

The existing JSON remains the only schema input; do not add YAML, a second
registry, or a runtime parser.  The generator continues to emit
`src/relay/generated/relay_op_contract.inc` for `OperatorSpec` and additionally
emits the checked-in, compiled
`src/relay/generated/relay_op_registration.cc` for entries with a `registration`
binding.  The first migrated entry is `add`:

- JSON owns the canonical name, schema metadata, registration description,
  arguments, and callback symbol names.
- The generated TU declares and binds the handwritten `AddInferType` and
  `AddCompute` callbacks with the existing `FInferType` and `FRelayToTE`
  signatures.
- `math.cc` keeps the compute implementation, but no longer registers `add`.
  Thus each operator has exactly one `KXC_REGISTER_OP` authority.

The generated TU is listed in `KXC_RELAY_IR_SOURCES` and its
`RelayGeneratedOpBindings` anchor is called by `RegisterBuiltins`.  This matters
for static archives: the direct reference prevents linker dead-stripping of the
TU and consequently retains its static registration initializer.

## Contract and symbol rules

A generated binding has `description`, ordered `arguments`, `type_infer_symbol`,
and `relay_to_te_symbol`.  Symbol names must be C++ identifiers.  The generator
only emits a single-TE binding for this phase.  Both callbacks are externally
visible, non-static functions with the existing exact callback signatures;
implementation TUs own definitions and generated TUs own only declarations and
registration calls.  This avoids ABI-sensitive wrapper functions and lets normal
C++ compilation catch declaration/definition disagreement.

Canonical Relay names, `schema_version`, `FInferType`/`FRelayToTE` keys,
`kxc.relay.op._make.<name>` FFI names, and kernel identity inputs remain
unchanged.  Callback symbol spellings are build-local implementation details and
are not serialized into `OperatorSpec`, FFI names, or kernel identities.  An
alias is not a migration mechanism: aliases remain forbidden by the contract.

## Incremental migration

1. Add a complete `registration` object to one simple, fieldless elementwise
   op, regenerate both files, move only its hand-written macro block, then test
   lookup, FFI, inference, and lowering.
2. Migrate further fixed-arity single-TE ops by adding equivalent JSON bindings
   and deleting their manual macro blocks in the same change.
3. Add generated support for attrs, variable arity, multi-output, and complex
   schemas only when an operator needs each feature, with one end-to-end test
   each.

During migration JSON is authoritative for schema metadata everywhere; a manual
registration is temporarily permitted only for entries without a `registration`
object.  A generated entry may never have a manual registration.

## Checker and tests

The checker validates binding field shape, generated-file freshness, exactly one
registration, generated source placement, callback definition presence and exact
signatures, CMake compilation membership, and builtin-anchor reachability.  It
also preserves existing FFI, type, lowering, backend-reference, and test checks.

Regression coverage must prove the generated `add` binding is reachable from a
static-link consumer, has the expected canonical FFI helper, infers broadcast
shape, and lowers through `FRelayToTE` to TIR.  CTest runs the contract checker,
registry, inference, and applicable CPU/LLVM suites.

## Non-goals

This phase does not create attrs classes, generate shape/compute code, alter
kernel naming or compilation identity, add distributed operators, parse JSON at
runtime, or migrate complex NLP/training ops.  Those need their own concrete
schema extensions and tests.

## Ponytail QA (full)

1. **A — reuse:** the initial sketch considered a generated binding table plus
   a second registry API. Rejected it: JSON plus the existing generator already
   express the needed data. Revision: add one optional `registration` object and
   emit ordinary existing registration macros.
2. **B — link/visibility/identity:** the initial sketch did not prove a static
   archive consumer pulls the new object. Revision: emit a named generated
   anchor, call it from `RegisterBuiltins`, require CMake source membership, and
   require non-static exact callback signatures. Keep callback symbols out of
   `OperatorSpec`; canonical FFI and kernel identity stay unchanged.
3. **C — real consumption:** a generated `.inc` would still leave the manual
   macro path authoritative. Rejected it. Revision: generate a compiled `.cc`,
   delete the migrated manual macro block, and have the checker reject stale
   output, missing anchor/source-list wiring, missing callbacks, or a duplicate
   registration. Registry/inference/lowering tests provide the end-to-end proof.
