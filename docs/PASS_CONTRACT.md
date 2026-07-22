# Pass Contract

`PassSpec` is IR-independent metadata for deciding whether a pass can be safely
scheduled. It does not define or store rewrite algorithms.

## Required Fields

- `name`: canonical pass name within its dialect.
- `schema_version`: positive metadata schema version.
- `dialect`: `relay` or `tir`.
- `scope`: current supported pipeline scopes are Relay `graph` and TIR
  `prim_func`.
- `phase`: scheduling phase, such as `relay_optimize`, `tir_optimize`, or
  `tir_schedule`.
- `opt_level`: minimum opt level associated with the pass metadata.
- `implementation_key`: stable FFI-style implementation binding key.
- `deterministic`, `idempotent`, `thread_safe`, `target_dependent`: behavioral
  claims that tests and contract checks can validate.

The implementation key must point to an existing transform entry such as
`kxc.relay.transform.fold_constant` or `kxc.tir.transform.simplify_expr`.

## Registry Rules

- `(dialect, name)` is the unique pass identity.
- Empty `name`, `phase`, or `implementation_key` fails validation.
- Duplicate identities fail validation.
- Relay passes cannot use `prim_func` scope.
- TIR passes cannot use `graph` scope.
- Pipeline entry points validate dialect, scope, phase, and implementation
  binding before executing a pass.
- `PassSpec` never owns function objects. Pipeline files keep local function
  bindings and map them through `implementation_key`.

## Default Pipelines

Relay `optimize_default`:

1. `fold_tuple_get_item`
2. `fold_constant`
3. `simplify_expr`
4. `canonicalize_cast`
5. `remove_standalone_reshapes`
6. `eliminate_dead_let`
7. `annotate_memory_scope`
8. `capture_post_dfs_index_in_spans`
9. `infer_type`

`eliminate_common_subexpr` remains registered and callable, but is intentionally
not in the default Relay pipeline.

TIR `optimize_default`:

1. `fold_constant`
2. `simplify_expr`
3. `force_narrow_index_to_i32`
4. `convert_for_loops_serial`
5. `loop_partition`
6. `unroll_loop`
7. `vectorize_loop`
8. `remove_no_op`

`bind_cuda_threads` remains registered and callable as a target-dependent
`tir_schedule` pass, but the compiler still invokes CUDA binding explicitly at
the existing scheduling boundary.

## Machine Contract

`test/pass_contract.json` is the machine-readable contract for existing pass
metadata and default order. `python/tools/check_pass_contract.py` cross-checks:

- contract pass entries against C++ pipeline bindings;
- implementation keys against FFI registrations;
- default pipeline order against `GetDefaultPassOrder()`;
- default membership flags against the JSON pipeline lists.

Common conservative declarations live in `pass_defaults`; each pass entry is
merged over those defaults before required-field validation. A missing field in
both places is a contract error. This keeps empty invariant/analysis sets and
the default `thread_safe=false` policy explicit without duplicating them in
every entry.

Adding a pass requires adding or updating contract metadata before adding it to a
default pipeline. Adding a pass implementation does not automatically make it a
default pass.
