---
name: kxc-capability-mounting
description: KXC compiler capability attachment workflow. Use when adding or reviewing Relay operators/importers, TE scheduling primitives or policies, TIR passes, LLVM/CUDA backend support, RuntimeSession stateful features, exact shape profiles, KV-cache/training/NLP capabilities, or any change that must plug into the existing compiler skeleton without creating a second authority or dead API.
compatibility: KXC repository after the foundation-skeleton-first integration.
---

# KXC 能力挂载总则

“挂载”是把能力接到现有契约及真实 consumer 上，不是再造一套框架，也不是动态插件机制。

先按任务读取对应领域 skill：

- Relay schema、InferType、TE callback、ONNX importer：
  [kxc-relay-operator-mounting](../kxc-relay-operator-mounting/SKILL.md)
- TE primitive、schedule、TIR lowering、LLVM/CUDA codegen：
  [kxc-te-backend-mounting](../kxc-te-backend-mounting/SKILL.md)
- KV cache、参数/optimizer state、alias、异步生命周期：
  [kxc-runtime-state-mounting](../kxc-runtime-state-mounting/SKILL.md)
- symbolic binding、exact profile、有限路由：
  [kxc-shape-profile-mounting](../kxc-shape-profile-mounting/SKILL.md)
- Pass、Target predicate、analysis/invariant：
  [kxc-pass-target-mounting](../kxc-pass-target-mounting/SKILL.md)

同时遵循 `ponytail`：只扩展当前能力真正需要的最小契约。

## 唯一权威表

| 语义 | 唯一 owner | 禁止的第二套实现 |
| --- | --- | --- |
| Relay operator schema/注册声明 | `contracts/relay_op_contract.json` | 另一份 YAML、runtime JSON parser、generated op 的手写注册 |
| Pass metadata/顺序/Target 要求 | `contracts/pass_contract.json` + `PipelineResolver/Executor` | 第二个 pass registry、隐藏插入、裸全局 Target |
| Shape 求值 | `ShapeProgram` | 路由器内再写 shape evaluator |
| Profile/dispatch/plan identity | `experimental_identity.h` 中既有 builders | 新 route key、字符串拼装 identity |
| Runtime value/state/alias | `ExecutablePlan` + `RuntimeSession` | 模型专用 state registry、第二张 value table |
| TE loop transform | `te::Schedule/Stage`，由 TE→TIR lowering 消费 | 只改 metadata 的 no-op primitive |
| CUDA thread/launch mapping | `tir::BindCudaThreads` | TE `bind/thread_axis`、另一份 launch policy |
| Kernel callable ABI | `KernelSignature`/`CompiledModule` | 从 TIR 或 op 名临时猜 ABI |

若新能力与表中 owner 冲突，扩展 owner；不要绕开它。

## 完整挂载链

任何非文档能力都必须闭合：

```text
声明 -> 校验 -> 真实 consumer -> identity/version -> 正例 -> 负例
```

缺一项即未完成：

- 只有字段没有 consumer：死 API。
- 只有 consumer 没声明：隐藏策略。
- 行为改变但 identity 不变：错误 cache reuse。
- 只有正例：不支持路径可能静默 fallback。
- 只有单元结构测试：不能证明生产链真正接入。

## 不轻易阻塞：降级验证阶梯

遇到环境缺失时继续完成可证明部分，绝不伪造成功：

1. **现有 build 失效**：重新配置 CPU preset，不把 stale build 当代码阻塞。
2. **LLVM 路径**：本仓库 CPU/LLVM 是强制门禁；查找 `llvm-config` 和 CMake 发现结果，修复本分支引入或暴露的最小 gate 问题。
3. **没有 CUDA toolkit**：完成 target-neutral 实现、synthetic CUDA `Target`、policy/IR/source/负例；记录真实编译未执行。
4. **有 toolkit、无 GPU/driver**：完成 CUDA source compilation 或 emission；仅硬件 launch 标记 skip。
5. **硬件能力不支持**：明确 fail closed；不得回退 CPU 后声称 CUDA 通过。
6. **现有 owner 缺少一个表达能力**：在 owner 上加最小字段/关系和版本，不新建旁路。
7. 只有涉及数据丢失、ABI 兼容选择、安全边界或互斥产品决策时才停下来问用户。

报告 blocker 时必须给出执行命令、错误和已通过的替代证据，而不是只说“环境不支持”。

## 标准工作流

1. **确认基线**
   - `git status --short --branch`
   - 找到声明、consumer、identity、测试四个位置。
   - `rg` 全部 caller；不要只修 issue 点名路径。
2. **参考成熟实现**
   - 针对新语义查看 TVM/XLA/MLIR/IREE/ONNX Runtime/TensorRT 中最接近的一到两个契约。
   - 记录要采用的不变量；不要复制完整框架。
3. **写最小目标模型**
   - owner 是谁、输入是什么、谁消费、哪里 fail closed、identity 哪一层变化。
4. **三轮 Ponytail QA**（契约或架构变化必须记录）
   - A：能否复用既有类型/registry/evaluator？
   - B：是否产生第二 authority，identity 是否覆盖行为？
   - C：是否有真实 consumer、正负例和死代码审计？
5. **实现一条纵向切片**
   - 一个代表能力先贯通声明到执行；不要先堆一批未接线 API。
6. **验证**
   - 先领域测试，再 CPU/LLVM 全量；CUDA 按上面的阶梯。
7. **集成审计**
   - 生成物 freshness、单注册/单定义、旧路径、未消费字段、跨层 include、分布式越界。

## 常见组合

| 能力 | 应加载的领域 skill |
| --- | --- |
| 新 NLP op/importer | Relay operator + TE/backend |
| CUDA reduction/indirect load | TE/backend；若新增 pass 再加 Pass/Target |
| KV cache decode | Runtime state + Shape/profile；需要新 op 时再加 Relay |
| 参数原地更新/optimizer state | Runtime state + Relay/TE；不得创建 training-only runtime |
| Reverse AD | Relay/Pass + TE/backend；state 只用于真正持久值 |
| 新 exact profile | Shape/profile；编译仍由 producer 显式执行 |
| distributed execution | 默认不在挂载范围；只有用户明确重启 #12 时单独设计 |

## 完成定义

- 生产路径实际调用新能力，不靠测试专用入口。
- 不支持的 target/shape/alias/schedule 在 launch 或 cache publication 前拒绝。
- ABI、schedule、memory-plan 或 pipeline 语义变化已更新 canonical identity/version。
- CPU/LLVM 数值正例和至少一个 preflight 负例通过。
- CUDA 缺失时 synthetic 契约仍通过，且没有 CPU fallback。
- 生成器/checker、public headers、include layers、`git diff --check` 通过。
- `rg` 无旧 owner、重复注册、未消费字段或保留的 no-op API。
- 不使用 `git add .`；只暂存本任务文件，尤其不要误提交机器本地 skill。

## 通用验证命令

优先复用已配置 build；没有时：

```bash
cmake --preset dev-ninja-cpu
cmake --build --preset dev-ninja-cpu -j2
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error
```

契约与架构：

```bash
python3 python/tools/generate_relay_op_contract.py \
  --matrix contracts/relay_op_contract.json \
  --output src/relay/generated/relay_op_contract.inc \
  --registration-output src/relay/generated/relay_op_registration.cc --check
python3 python/tools/check_relay_op_contract.py --root .
python3 python/tools/generate_pass_contract.py \
  --matrix contracts/pass_contract.json \
  --output src/pass/generated/pass_contract.inc --check
python3 python/tools/check_pass_contract.py --root .
python3 tools/architecture/check_include_layers.py --root .
python3 tools/architecture/check_public_headers.py --root . --compile
git diff --check
```

静态库重复 strong symbol 审计：

```bash
nm --no-demangle --defined-only --extern-only <build>/libkxc_runtime.a \
  | awk '$2 ~ /^[TDBR]$/ {print $2, $3}' | sort | uniq -d
```

结果必须为空。构造函数的 demangled C1/C2 别名不是重复定义，因此这里故意使用 mangled name。
