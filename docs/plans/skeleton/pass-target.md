# Pass/Target execution skeleton

## Scope

This is the smallest executable extension of the existing `PassSpec` /
`PassContext` / normalized-pipeline contract. It does not add scheduling
algorithms, CUDA reductions, a second context type, an analysis cache, or a
new pass registry.

## Model

`PassContext` remains the sole per-invocation context. A pipeline entry point
has an overload taking it explicitly; the legacy overload uses the current
scoped context for compatibility. `PipelineExecutor` constructs one context
from its target and passes it through the explicit overload. This makes a
new target-aware binding receive both context and target without relying on a
bare global. Scopes are thread-local and nest/restore, so independent threads
and reentrant pipeline calls do not share mutable execution state. Pass
functions remain responsible for their own thread-safety claim.

`PassSpec::target_requirements` is an ordered, validated list of target
capability predicates. The initial grammar deliberately contains only facts
needed now: `kind=<kind>`, `attr.exists>0`,
`attr.max_threads_per_block>0`, and
`attr.max_shared_memory_per_block>=0`. A target-aware pass has a nonempty list;
a target-independent pass has an empty one. The generic matcher is used by
direct bindings, resolver, and executor, so adding a new target pass adds
predicates to the contract rather than adding a `cuda` branch to execution
machinery.

A normalized plan records the canonical capability snapshot actually passed to
execution plus the predicates of all executed steps. Its canonical bytes
include both these requirements and every transition, so fingerprints describe
execution rather than intention.

## Analysis state and invariant proof

The resolver and executor replay the same transition function. For an
IR-changing pass, only analyses explicitly listed in `preserved_analyses`
survive; for a non-changing pass, only `invalidated_analyses` are removed.
A name cannot be both preserved and invalidated. Thus analysis metadata
changes executable state instead of remaining a declaration.

`prim_func_defined` is the first executable TIR invariant: a `PrimFunc` and
its body must be defined and have the expected node types. CUDA thread binding
produces it, and the executor validates it after that step. This is not a
schedule proof; CUDA launch metadata continues to be recovered and validated
from TIR attrs by the existing typed accessor.

Canonical pass identity remains `(dialect, name, occurrence)`. Binding keys,
phases, target predicates, and transitions are rechecked before execution.

## Non-goals

- No specific schedule or CUDA reduction implementation.
- No general analysis cache or invalidation framework.
- No arbitrary predicate language or runtime target discovery.
- No implicit pass insertion or alternate fingerprint view.

## Plan and tests

1. Add target predicates to generated `PassSpec` metadata and generic matching.
2. Bind Relay/TIR passes through explicit `PassContext` overloads while keeping
   direct legacy calls source-compatible.
3. Replay analysis transitions in resolver/executor and add the TIR proof.
4. Declare CUDA binding requirements/proof in the machine contract and
   regenerate its C++ form.
5. Test CPU-only synthetic targets for predicate mismatch, direct/executor
   consistency, invalid analysis preservation, TIR proof, and CUDA attr
   recovery. Run generator/checker, CPU and LLVM CTest, plus binding/include
   audits.

## Ponytail QA

### A — reuse context

Rejected a target-context or pass-manager object. `PassContext` already owns
Target, is scoped, and restores nesting; explicit overloads are enough. The
executor constructs one merged context and passes it to every binding instead
of nesting a legacy current-context call.

### B — metadata must execute

Deleted the redundant `target_dependent` field and rejected passive analysis
metadata. Predicates are matched in direct, resolver, and executor paths;
analysis state is replayed and transition tampering fails validation.

### C — one path, no hidden recovery

Checked direct APIs, executor, TIR positive/negative proof, target mismatch,
and analysis preserve/invalidate replay. Direct and executor paths use the same
binding overload and predicate matcher; exact target snapshots and ordered
steps identify execution; existing typed CUDA attr recovery remains the single
source of launch metadata. No schedule or reduction implementation was added.
