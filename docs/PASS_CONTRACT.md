# Pass 契约

`PassSpec` 是与 IR 解耦的元数据，用于判断 Pass 能否安全调度。
它不定义也不存储改写算法。该角色在端到端编译器中的位置见
[架构总览](ARCHITECTURE.md)。

## 必填字段

- `name`：该 IR 方言下的规范 Pass 名称。
- `schema_version`：正整数元数据 Schema 版本。
- `dialect`：`relay` 或 `tir`。
- `scope`：当前支持的流水线作用域为 Relay 的 `graph` 与 TIR 的 `prim_func`。
- `phase`：调度阶段，例如 `relay_optimize`、`tir_optimize`、`tir_schedule`。
- `opt_level`：与该 Pass 元数据绑定的最低优化级别。
- `implementation_key`：稳定的 FFI 风格实现绑定键。
- `required_invariants`、`produced_invariants`：生产不变量账本。
- `declarative_only_invariants`：只有声明、没有可执行证明的元数据；此类名称不能满足生产前置条件。
- `target_requirements`：有序目标谓词。目前支持以下语法：`kind=<kind>`、`attr.exists>0`、
  `attr.max_threads_per_block>0`、`attr.max_shared_memory_per_block>=0`。
- `may_change_ir`、`deterministic`、`idempotent`、`thread_safe`：可由测试和契约检查验证的行为声明。

实现键必须指向已存在的变换条目，例如
`kxc.relay.transform.fold_constant` 或 `kxc.tir.transform.simplify_expr`。

## 注册表规则

- `(dialect, name)` 是 Pass 的唯一身份。
- `name`、`phase`、`implementation_key` 任一为空都不通过验证。
- 重复身份不通过验证。
- Relay Pass 不允许使用 `prim_func` 作用域。
- TIR Pass 不允许使用 `graph` 作用域。
- 流水线入口会先校验 IR 方言、作用域、阶段、实现绑定与不变量分类，再执行 Pass。
- 不变量和分析列表中的名称必须唯一且非空。声明为“仅声明”的名称也必须出现在该 Pass 的必需或产出声明中。
- 每个生产前置条件，以及每个标记为“已证明”的不变量，都必须有对应 IR 方言的可执行校验器。
  不支持的产出元数据必须标记为“仅声明”，且不能加入“已证明”集合。
- 同一目标谓词匹配器会控制直接绑定、resolver 输出和执行器回放。
  未知谓词不通过契约校验。
- 对会修改 IR 的 Pass，只能保留显式声明的分析；对于不修改 IR 的 Pass，显式声明失效的分析会被移除。
  同一项分析不能同时声明为保留和失效。
- 执行器在 `NormalizedPipeline` 的每一步之后，都要验证初始“已证明”集合与完整“已证明”集合。
  可执行注册表会校验 Relay 的 `checked_type`/`anf` 与 TIR 的 `prim_func_defined`。
- `PassSpec` 不拥有函数对象。流水线文件保留本地函数绑定，通过 `implementation_key`
  映射到函数，并接收调用时明确的 `PassContext`。

## 默认流水线

Relay `optimize_default`：

1. `fold_tuple_get_item`
2. `fold_constant`
3. `simplify_expr`
4. `canonicalize_cast`
5. `remove_standalone_reshapes`
6. `eliminate_dead_let`
7. `annotate_memory_scope`
8. `capture_post_dfs_index_in_spans`
9. `infer_type`

`eliminate_common_subexpr` 保持注册且可调用，但有意不放入默认 Relay 流水线。

TIR `optimize_default`：

1. `fold_constant`
2. `simplify_expr`
3. `force_narrow_index_to_i32`
4. `convert_for_loops_serial`
5. `loop_partition`
6. `unroll_loop`
7. `vectorize_loop`
8. `remove_no_op`

`bind_cuda_threads` 注册为 `tir_schedule` Pass，需要可用的 CUDA 目标以及有效的线程与共享内存能力。
规范化 TIR 编译流水线会显式选中并执行它。它的精确步骤、依赖与标准目标能力快照
都会进入流水线身份；执行时使用的能力变化会影响指纹。

## 机器契约

`contracts/pass_contract.json` 是现有 Pass 元数据与默认顺序的机器可读契约。
`python/tools/check_pass_contract.py` 会交叉校验：

- 契约中的 Pass 条目与 C++ 流水线绑定；
- 实现键与 FFI 注册；
- 默认流水线顺序与 `GetDefaultPassOrder()`；
- 默认成员标记与 JSON 流水线列表；
- 目标谓词的语法与唯一性。

保守默认声明集中在 `pass_defaults`。校验必填字段前，每个 Pass 条目都会先与默认值合并。
若两处都缺失某字段，则视为契约错误。合并机制让不变量与分析的空集合、
`declarative_only_invariants` 集合及默认的 `thread_safe=false` 策略显式生效，
无需在每个条目重复声明。

新增 Pass 必须先补充或更新契约元数据，之后才能加入默认流水线。
只新增 Pass 实现不会让它自动成为默认 Pass。
