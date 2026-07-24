# Compiler Foundation / Shape handoff

## Current status

The repository-only experimental Shape surface has two independent, narrow
pieces. These headers are listed in `KXC_EXPERIMENTAL_HEADERS` for source-tree
checks but are deliberately absent from install/export package targets; they
carry no source or binary compatibility promise.

- `shape_specialization.h` supplies `GraphTemplate`, exact profiles, exact
  requests, and `ChangedUnitIndices` inputs. It does not compile or execute.
- `shape_exact.h` is the default-OFF production exact Relay/compiler adapter.
  Its production bridge remains concrete Relay, one empty profile, and static
  `RuntimeSession` only.

`restricted_symbolic_shape.h` is default-OFF and is an **exact-decision-only
control plane**. `Prepare` freezes a concrete representative through the exact
adapter, overlays explicitly bound nonempty symbols, and accepts only fixed-rank
`relu`/`sqrt` and equal-shape `add`/`mul`. `MintExact` creates immutable exact
request snapshots. It does not compile, cache, allocate, execute, or lower a
symbolic Relay graph.

## Deleted and unsupported

The guarded bucket/polymorphic contract universe was deleted: its contract header,
implementation, deterministic fake resolver/plan types, focused test, and CMake
and CTest registration no longer exist. There is no bucket or polymorphic
policy, profile, request, dispatch kind, alias, compatibility shim, or generic
dynamic lowering path.

Bucket execution, polymorphic execution, generic Relay symbolic-to-module
lowering, dynamic graph memory planning, dynamic output allocation, ragged or
data-dependent shapes, workspace schemas, and tail transforms are unsupported.
`-1` remains a legacy input ABI sentinel, not symbolic Shape support.

`ModuleShapeExpr` and `ModuleInvocationContract` are separate CompiledModule
invocation typestate. They validate manually supplied/module-lowered invocation
contracts; they neither consume restricted symbolic decisions nor establish a
generic Relay symbolic-to-module lowering path.

## Verification

Use CPU-only builds with LLVM and CUDA disabled:

```bash
cmake -S . -B out/shape-restricted-off -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=OFF \
  -DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=OFF \
  -DKXC_BUILD_PASS_TESTS=ON -DKXC_BUILD_CODEGEN_TESTS=OFF
cmake --build out/shape-restricted-off --target run_restricted_symbolic_shape_test --parallel 2
ctest --test-dir out/shape-restricted-off --output-on-failure \
  -R '^restricted_symbolic_shape_test$'

cmake -S . -B out/shape-restricted-on -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON \
  -DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON \
  -DKXC_BUILD_PASS_TESTS=ON -DKXC_BUILD_CODEGEN_TESTS=OFF
cmake --build out/shape-restricted-on --target \
  run_shape_system_test run_shape_specialization_test \
  run_shape_production_exact_test run_restricted_symbolic_shape_test \
  check_include_layers check_public_headers --parallel 2
ctest --test-dir out/shape-restricted-on --output-on-failure --no-tests=error
```
