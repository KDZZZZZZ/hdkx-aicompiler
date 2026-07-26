---
name: kxc-pass-target-mounting
description: Add KXC Relay/TIR passes through the generated pass contract and explicit Target/PassContext execution model. Use for pass registration, pipeline ordering, target predicates, analysis preservation/invalidation, executable invariants, CUDA scheduling passes, or pipeline identity changes.
compatibility: KXC repository with pass contract v4-style resolver/executor enforcement.
---

# KXC Pass / Target 挂载

先完整读取 [总则](../kxc-capability-mounting/SKILL.md)。具体 TE schedule/backend 语义继续读取 `kxc-te-backend-mounting`。

## 唯一 owner

```text
contracts/pass_contract.json
python/tools/generate_pass_contract.py
python/tools/check_pass_contract.py
src/pass/generated/pass_contract.inc
include/kxc/pass/pass.h, src/pass/pass.cc
src/compiler/pipeline_resolver.cc              resolve/validate/execute/identity
src/relay/transforms/pipeline.cc               Relay implementation bindings
src/tir/transforms/pipeline.cc                 TIR implementation bindings
include/kxc/pass/context.h, src/pass/context.cc sole invocation context
```

Pass identity 是 `(dialect, name, occurrence)`；Target snapshot、ordered steps、requirements 和 transitions 都进入 normalized pipeline canonical bytes。

## 新 Pass 步骤

1. **实现纯 pass 函数**
   - Relay 或 TIR 类型明确，不在函数内部注册另一份 metadata。
   - 先定义是否改 IR、deterministic/idempotent/thread-safe 语义。
2. **更新 `pass_contract.json`**
   - name/schema/dialect/scope/phase/opt level。
   - required/produced/declarative-only invariants。
   - preserved/invalidated analyses。
   - target requirements 与 implementation key。
3. **加入 pipeline（仅需要时）**
   - 明确 opt level 和 CPU/CUDA policy。
   - 不依赖 resolver 隐式插入。
4. **绑定 implementation key**
   - 在 Relay/TIR 唯一 binding table 加一项。
   - Target-aware binding 接收显式 `PassContext`；不要从裸全局恢复 Target。
5. **生成并检查**

```bash
python3 python/tools/generate_pass_contract.py \
  --matrix contracts/pass_contract.json \
  --output src/pass/generated/pass_contract.inc
python3 python/tools/check_pass_contract.py --root .
```

## Target predicate

当前 grammar 只包含实际需要的简单事实，如 `kind=cuda` 和已定义的 capability attrs。

- 新 pass 能用现有谓词就复用。
- 确需新硬件事实时，一次性扩展 predicate validation、matcher、generator/checker 和正负例。
- Predicate 必须由 direct binding、resolver、executor 同一 matcher 消费。
- 不要重新增加被删除的被动 `target_dependent` bool。
- Synthetic complete Target 足以验证 policy；真实硬件只负责运行证据。

## Analysis transition

- `may_change_ir=true`：只有显式 preserved 且此前存在的 analysis 保留。
- `may_change_ir=false`：显式 invalidated 被移除。
- 同一 analysis 不得同时 preserve/invalidate。
- 不声明不存在的 analysis cache；只有真实 consumer 出现时才扩展 production analysis。

## Invariant

- Production required invariant 必须有 executable validator，不能只是字符串。
- Produced invariant 必须恰好归类为 executable 或 declarative-only。
- 新 executable invariant 同时实现 Relay/TIR validator、resolver replay、executor post-pass check 和 malformed IR 负例。
- `prim_func_defined` 只是结构证明，不冒充 schedule independence proof。

## Context 与嵌套

- `PipelineExecutor` 从 CompileConfig Target merge 出一个 context，并显式传给每个 binding。
- Legacy overload 只为兼容，使用 scoped current context。
- Scope 必须 thread-local、嵌套恢复；新 pass 不保存 context 引用到调用结束之后。
- Placement 与 compile Target 冲突必须拒绝，不能静默覆盖 device identity。

## 测试

- Contract freshness 与单 implementation binding。
- Resolver：phase/order/occurrence、target mismatch、canonical tamper。
- Direct vs executor：同一 Target predicate 和 context 行为。
- Analysis：preserve/invalidate replay及 tamper 负例。
- Invariant：有效 IR、malformed IR、produced proof。
- CUDA pass：synthetic Target 正例与缺 attr/kind mismatch 负例。

```bash
ctest --test-dir <build> --output-on-failure \
  -R 'pass_contract|pass_pipeline_test|pipeline_resolver_test|compiler_contract_test|cuda_schedule_test'
```

## 禁止

- 第二个 pass registry/context/Target wrapper。
- 在 executor 中按 pass 名硬编码 CUDA 分支。
- 未列入 normalized pipeline 的隐藏 pass。
- 只有 metadata 没 transition/validator consumer。
- 将具体优化算法塞进通用 Pass Manager 骨架。

完成标准：machine contract、binding、resolver、explicit context、transition/invariant、canonical identity 和正负例全部一致。
