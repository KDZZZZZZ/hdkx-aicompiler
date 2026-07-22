# Relay/TIR Pass 实现流程与当前实现清单

更新时间：2026-02-20

## 1. 目标与范围

本文档总结两件事：

1. 在当前仓库中新增一个 Relay/TIR pass 的标准实现流程。  
2. 当前已经落地的 pass、pipeline、工具层与测试覆盖。

---

## 2. 实现 Pass 的标准流程

### Step 1: 明确规则和 IR 作用域

- 先确定是改写 `Relay` 还是 `TIR`。  
- 规则尽量保持“局部、可组合、幂等”，例如：
  - `x + 0 -> x`
  - `TupleGetItem(Tuple(fields), i) -> fields[i]`
  - 删除 `Evaluate(0)`。

### Step 2: 写 pass 本体（Mutator）

- Relay: 继承 `RelayPass`，在 `Visit*` 中做 rewrite。  
  文件位置示例：`src/relay/transforms/*.cc`  
- TIR: 继承 `TIRPass`，重写 `Visit*`。  
  文件位置示例：`src/tir/transforms/*.cc`

### Step 3: 复用/补齐 pass_utils

- Relay 公共工具：`include/relay/pass_utils.h`、`src/relay/pass_utils.cc`  
- TIR 公共工具：`include/tir/pass_utils.h`、`src/tir/pass_utils.cc`

避免把分析逻辑重复散落在每个 pass 里（如常量 0/1 判定、no-op 判定、SeqStmt 扁平化等）。

### Step 4: 暴露头文件 API

- Relay pass 头文件放在 `include/relay/transforms/`。  
- TIR pass 头文件放在 `include/tir/transforms/`。  
- 统一导出形如：
  - `Function XxxPass(const Function& func);`
  - `tir::PrimFunc XxxPass(const tir::PrimFunc& func);`

### Step 5: 接入 pipeline 与全局注册

- Relay pipeline：`src/relay/transforms/pipeline.cc`  
- TIR pipeline：`src/tir/transforms/pipeline.cc`

需要做三件事：

1. 在 pass 名字到函数映射表中注册字符串名。  
2. 如需要，加入默认别名 `optimize_default` 的顺序。  
3. 暴露全局函数入口（`kxc.*.transform.*`）。

### Step 6: 接入构建

在 `CMakeLists.txt` 里加入：

- 新增的 `.cc` 源文件；
- 测试可执行（如果是新的测试文件）。

### Step 7: 写测试

当前 pass 测试入口：`test/pass_pipeline_test.cpp`。  
建议至少覆盖：

1. 单个 rule 命中与不命中。  
2. pipeline 顺序一致性（与手工串联结果相同）。  
3. 未知 pass 名抛错。  
4. 默认 pipeline 幂等。  
5. 关键元信息保持（例如 Relay `virtual_device_`）。

### Step 8: 文档同步

至少同步：

- runtime API 文档（全局函数名）；  
- 本文档（规则、默认顺序、限制）。

---

## 3. 当前已实现的基础设施

### 3.1 Pipeline API

#### Relay

- C++ API:  
  - `kxc::relay::RunRelayPassPipeline(const Function&, const Array<String>&)`
- Registry:
  - `kxc.relay.transform.run_pipeline`
- 默认别名：
  - `optimize_default -> fold_tuple_get_item -> fold_constant -> simplify_expr -> canonicalize_cast -> remove_standalone_reshapes -> eliminate_dead_let -> annotate_memory_scope -> capture_post_dfs_index_in_spans -> infer_type`

#### TIR

- C++ API:
  - `kxc::tir::RunTIRPassPipeline(const tir::PrimFunc&, const Array<String>&)`
- Registry:
  - `kxc.tir.transform.run_pipeline`
- 默认别名：
  - `optimize_default -> simplify_expr -> remove_no_op -> convert_for_loops_serial`

说明：未知 pass 名会抛 `runtime_error`。

### 3.2 通用工具层

- Relay: 常量判定、副作用保守分析、use-count、`virtual_device_` 复制、Call op 名提取。  
- TIR: 常量判定、no-op 判定、`SeqStmt` 扁平化、空/单语句归一化构造。

### 3.3 调试打印

- Relay 打印：
  - `include/relay/pass/print_ir.h`
  - `src/relay/pass/print_ir.cc`
- TIR 打印：
  - `include/tir/pass/print_ir.h`
  - `src/tir/pass/print_ir.cc`

---

## 4. 当前已实现 Pass 清单

## 4.1 Relay（3 个）

1. `fold_tuple_get_item`  
文件：`src/relay/transforms/fold_tuple_get_item.cc`  
规则：`TupleGetItem(Tuple(fields), i) -> fields[i]`（索引合法时）。

2. `simplify_expr`  
文件：`src/relay/transforms/simplify_expr.cc`  
规则：
- `x + 0 -> x`
- `0 + x -> x`
- `x * 1 -> x`
- `1 * x -> x`
- `x - 0 -> x`
- `x / 1 -> x`

3. `eliminate_dead_let`  
文件：`src/relay/transforms/eliminate_dead_let.cc`  
规则：删除“无副作用且未被使用”的 `Let` 绑定。

对应 registry 单入口：

- `kxc.relay.transform.fold_tuple_get_item`
- `kxc.relay.transform.simplify_expr`
- `kxc.relay.transform.eliminate_dead_let`

## 4.2 TIR（3 个）

1. `simplify_expr`  
文件：`src/tir/transforms/simplify_expr.cc`  
规则：
- `x + 0 -> x`、`0 + x -> x`
- `x * 1 -> x`、`1 * x -> x`
- `x - 0 -> x`
- `x / 1 -> x`

2. `remove_no_op`  
文件：`src/tir/transforms/remove_no_op.cc`  
规则：
- 删除 `Evaluate(0)`
- 删除 no-op 循环体
- 扁平化嵌套 `SeqStmt` 并清理空语句

3. `convert_for_loops_serial`  
文件：`src/tir/transforms/convert_for_loops_serial.cc`  
规则：把 `ForType::{Parallel, Vectorized, Unrolled}` 规范为 `Serial`。

对应 registry 单入口：

- `kxc.tir.transform.simplify_expr`
- `kxc.tir.transform.remove_no_op`
- `kxc.tir.transform.convert_for_loops_serial`

---

## 5. 测试与验证现状

测试文件：`test/pass_pipeline_test.cpp`

当前覆盖：

1. Relay 三个 pass 的正反例。  
2. TIR 三个 pass 的正例。  
3. Relay/TIR pipeline 顺序一致性。  
4. 未知 pass 名抛错。  
5. `optimize_default` 幂等。  
6. Relay `virtual_device_` 保持检查。

构建与运行（Windows）：

```powershell
cmake -S . -B build -DKXC_BUILD_PASS_TESTS=ON
cmake --build build --target pass_pipeline_test
cmake --build build --target run_pass_pipeline_test
```

---

## 6. 后续新增 Pass 的最小 Checklist

1. 在 `include/*/transforms/` 增加头文件 API。  
2. 在 `src/*/transforms/` 实现 pass。  
3. 复用或补充 `pass_utils`。  
4. 在对应 `pipeline.cc` 注册字符串名与 registry 入口。  
5. 视需要加入 `optimize_default` 顺序。  
6. 更新 `CMakeLists.txt`。  
7. 在 `test/pass_pipeline_test.cpp` 增加 rule 与 pipeline 用例。  
8. 更新 docs（本文档 + runtime API 文档）。

## Batch-2 Pass Update (2026-02-20)

This repository now includes a second batch of passes (6 Relay + 4 TIR).

### New Relay Pass Entries

- kxc.relay.transform.fold_constant
- kxc.relay.transform.canonicalize_cast
- kxc.relay.transform.remove_standalone_reshapes
- kxc.relay.transform.eliminate_common_subexpr
- kxc.relay.transform.annotate_memory_scope
- kxc.relay.transform.capture_post_dfs_index_in_spans

Relay optimize_default order:

- fold_tuple_get_item -> fold_constant -> simplify_expr -> canonicalize_cast -> remove_standalone_reshapes -> eliminate_dead_let -> annotate_memory_scope -> capture_post_dfs_index_in_spans -> infer_type

`eliminate_common_subexpr` 保留显式入口，但结构键完善前不进入默认链。

### New TIR Pass Entries

- kxc.tir.transform.fold_constant
- kxc.tir.transform.force_narrow_index_to_i32
- kxc.tir.transform.loop_partition
- kxc.tir.transform.unroll_loop
- kxc.tir.transform.vectorize_loop

TIR optimize_default order:

- fold_constant -> simplify_expr -> force_narrow_index_to_i32 -> convert_for_loops_serial -> loop_partition -> unroll_loop -> vectorize_loop -> remove_no_op

### New C++ APIs

Relay:

- kxc::relay::FoldConstantPass(const Function&)
- kxc::relay::CanonicalizeCastPass(const Function&)
- kxc::relay::RemoveStandaloneReshapesPass(const Function&)
- kxc::relay::EliminateCommonSubexprPass(const Function&)
- kxc::relay::AnnotateMemoryScopePass(const Function&)
- kxc::relay::CapturePostDfsIndexInSpansPass(const Function&)

TIR:

- kxc::tir::FoldConstantPass(const tir::PrimFunc&)
- kxc::tir::ForceNarrowIndexToI32Pass(const tir::PrimFunc&)
- kxc::tir::LoopPartitionPass(const tir::PrimFunc&)
- kxc::tir::UnrollLoopPass(const tir::PrimFunc&)
- kxc::tir::VectorizeLoopPass(const tir::PrimFunc&)

## PassSpec Contract Update (2026-07-22)

Pass scheduling now has an IR-independent metadata layer:

- Public metadata: `include/kxc/pass/pass.h`
- Registry implementation: `src/pass/pass.cc`
- Machine contract: `test/pass_contract.json`
- Checker: `python/tools/check_pass_contract.py`
- Contract document: `docs/PASS_CONTRACT.md`

`PassSpec` records canonical pass identity, dialect, scope, phase, opt level,
behavioral claims, and a stable `implementation_key`. It does not store function
objects or concrete rewrite logic.

Relay and TIR pipeline files keep their existing pass algorithms and default
orders, but now bind each implementation through a `PassSpec` and validate
dialect, scope, phase, and implementation key before dispatch.

The default pipeline orders remain:

- Relay: `fold_tuple_get_item -> fold_constant -> simplify_expr -> canonicalize_cast -> remove_standalone_reshapes -> eliminate_dead_let -> annotate_memory_scope -> capture_post_dfs_index_in_spans -> infer_type`
- TIR: `fold_constant -> simplify_expr -> force_narrow_index_to_i32 -> convert_for_loops_serial -> loop_partition -> unroll_loop -> vectorize_loop -> remove_no_op`

Explicit-only passes remain registered but outside default order:

- Relay: `eliminate_common_subexpr`
- TIR: `bind_cuda_threads`
