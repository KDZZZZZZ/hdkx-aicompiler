# Shape / profile / dispatch control-plane skeleton

## Goal

Provide one fail-closed, caller-owned control-plane path for a finite set of
**exact** input profiles:

```text
concrete named input shapes -> BindingSet -> ExactOracle -> DispatchKey
                                             -> published variant lookup
```

`ShapeProgram` remains the sole shape evaluator and `experimental_identity.h`
remains the sole authority for `ShapeProfileKey`, `DispatchKey`,
`PlanVariantKey`, and `PlanAbiFingerprint`. Compilation stays explicit:
`ExactProfileRouteTable` never calls `Compiler::Compile`, cache APIs, or a
runtime session.

This takes the useful common part of several established designs, without
copying their runtime scope:

* TensorRT optimization profiles make permitted input ranges explicit and pick
  a prebuilt engine/profile rather than compiling on a request.
* XLA dynamic dimensions distinguish symbolic dimensions from bounded,
  executable specializations; this skeleton accepts only a finite exact point
  after constraint verification.
* IREE HAL dispatch separates dispatch selection from executable publication;
  this table selects only already-published variants.
* ONNX Runtime symbolic shape inference and profile providers keep symbolic
  binding separate from the provider-specific executable; here `ShapeProgram`
  owns the former and compiled variants own the latter.

## Model and boundaries

### Symbol binding

`BindExactInputShapes(GraphTemplate, vector<ConcreteInputShape>)` accepts
named input logical shapes in template order-independent form. It supports
only two input-axis forms:

* `DimExpr::Const(v)`, which must equal `v`;
* `DimExpr::Symbol(name)`, which binds that named symbol consistently.

Every concrete extent is non-negative. Repeated symbols must agree. It then
uses `ShapeProgram::Evaluate` to require all declared symbols, bounds,
divisibility, and exact logical/physical/valid contracts. Arithmetic dim
expressions are deliberately not inverted. This is named-dimension binding,
not a claim to support general `Shape`, `ConstantOfShape`, data-dependent
outputs, or arbitrary dynamic lowering.

### Concrete profile and route

An `ExactOracle` remains the concrete profile proof. The route is always
`BuildStaticExactDispatchKey(template.key(), oracle.profile().key())`; no
second route key exists. Profiles are exact: two profiles whose evaluated graph
inputs are identical overlap and are rejected at publish time, even if unused
bindings make their profile keys differ.

### Variant ownership and compatibility

`ExactProfileRouteTable` freezes one `GraphTemplate` and owns copies of
caller-published `CompiledGraph`s, so artifact pins remain alive; lookup
returns an owning value that survives table growth or destruction. It is
created for exactly one primitive target capability fingerprint. Publishing
requires every retained primitive artifact to match that fingerprint and
requires caller-supplied expected ABI metadata to equal the result of the
existing `BuildPlanAbiFingerprint`. It constructs the existing `PlanVariantKey` and
rejects duplicate/overlapping profiles. The table uses those existing identity
builders; it does not mint equivalents.

The caller must first bind a compiled graph to an oracle using its real
producer-specific validator (for example,
`RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant`). The frozen
template then validates oracle ownership; the table validates target and
expected ABI compatibility, profile overlap, and its own route/variant
consistency. It does not become a second lowering verifier.

### Compile / publish / lookup

* **Compile:** caller-only and explicit (`Compiler::Compile`).
* **Publish:** `ExactProfileRouteTable::Publish`; accepts an already compiled,
  producer-validated variant plus expected `PlanAbiFingerprint`, reconstructs
  and checks that ABI, then stores the variant.
* **Lookup:** `Lookup`; derives the canonical route from the oracle and either
  returns an owning immutable value or throws. It has no compile/cache side
  effect and cannot return a reference invalidated by later publication.

## Fail-closed rules

Reject malformed/unknown/duplicate input names, negative extents, non-direct
symbol axes, inconsistent shared symbols, constraints or exact-contract
failures, a foreign oracle/template, graph/profile/key disagreement, an empty
or target-incompatible variant, undefined or mismatched expected ABI,
duplicate or overlapping inputs, and a route miss. No default profile,
fallback bucket, widening, cache fill, or implicit compile exists.

## Non-goals

No distributed publication, hot swap, eviction, asynchronous compilation,
profile generation, bucket/range matching, dynamic allocation, dynamic Relay
lowering, CUDA requirement, adaptive-controller integration, or reopening
closed issue #46. The existing exact/oracle and `DispatchKey` key-space split
remains unchanged.

## Implementation and tests

* Add source-tree experimental `shape_control.h/.cc`, using existing Shape,
  identity, and compiler contracts only.
* Add `shape_control_test`: direct named binding tests; duplicate/overlap and
  foreign/key failures; CPU LLVM explicit compilation, publish, hit, miss,
  target and ABI mismatch, identity reconstruction, and execution.
  LLVM-disabled configurations keep the pure control-plane cases and skip only
  compilation-dependent assertions.
* Register the source only with existing shape-production/restricted object
  sets and the CPU shape test target.
* Delete the always-false `PreparedGraphTemplate::multi_profile_supported()`
  API. The production exact adapter documents its actual boundary directly:
  one concrete Relay/empty-binding profile per preparation. The independent
  finite table routes only already explicit compiled profiles.

## Ponytail QA record (full, three rounds)

### Round 1 — delete abstractions and dead surface

Rejected a router hierarchy, profile cache, compile callback, target wrapper,
and custom hash/key types. Replaced the provisional publication pimpl/default
state with one value type, return-by-value lookup, and a finite owning vector.
Deleted the constant `multi_profile_supported()` API and the test-local route
map superseded by this control plane.

### Round 2 — one evaluator and one identity authority

Kept `ShapeProgram` as the only evaluator, `ExactOracle` as applicability
proof, and `BuildStaticExactDispatchKey`/`BuildPlanVariantKey`/
`BuildPlanAbiFingerprint` as the only identity builders. Removed the unused
`GraphCanonical`/partition/constant canonicalization chain from the exact
adapter so no dormant second graph-identity encoding remains. Also removed the
provisional duplicate compiled-boundary verifier; the real restricted exact
adapter remains responsible for binding materialized code to its decision. No
change to the closed #46 oracle-vs-plan key-space split exists.

### Round 3 — real consumer and fail-closed paths

The named binder now feeds the restricted adapter's real request path, and
published variants feed CPU LLVM `RuntimeSession` execution. Publication
checks target and reconstructed expected ABI; lookup checks the frozen template
and canonical route. Malformed binding, foreign oracle, duplicate/overlap,
target/ABI mismatch, and miss paths throw without compiler/cache calls.
