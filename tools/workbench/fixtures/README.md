# 样例 Bundle

## 合成样例（随仓库提供）

```bash
node tools/workbench/scripts/make-fixture.mjs            # 生成
node tools/workbench/scripts/make-fixture.mjs --scale 40 # 放大事件量做虚拟化压测
```

| 目录 | 内容 |
| --- | --- |
| `bundles/baseline` | 完整编译 + 运行时，222 个事件 |
| `bundles/candidate` | 同上，但 `fold_tuple_get_item` 被人为放慢 3.4x，缓存命中率下降 |
| `bundles/raw-unanalyzed` | 与基线相同，但 `diagnosis.json` 是 C++ 写出的占位版本 |

生成器输出格式逐字段对照 `src/base/profiling.cc` 编写，并用真实产物校正：Pass 名使用 snake_case，`relay_pipeline`/`tir_pipeline` 是独立组件，产物路径形如 `artifacts/<run_id>/{relay,tir,lower}/...`。

## 真实产物（不入库，需自行生成）

`bundles/real-compile` 和 `bundles/real-analyzed` 是编译器真实生成的 Bundle，**不随仓库提供**。它们是校验合成样例是否失真的唯一依据，也是若干测试的数据源；缺失时相关用例会明确跳过，不会静默通过。

重新生成：

```bash
# 1. 构建并运行 profile_bundle_test
cmake -S . -B out/build/wb-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=clang++ -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF
cmake --build out/build/wb-cpu --target profile_bundle_test
KXC_PROFILE_ENABLE=1 ./out/build/wb-cpu/profile_bundle_test

# 2. 未分析版本
cp -r profile_bundle_test_output tools/workbench/fixtures/bundles/real-compile

# 3. 跑过 Python 诊断的版本
cp -r profile_bundle_test_output tools/workbench/fixtures/bundles/real-analyzed
PYTHONPATH=python python -m kxc_agent.cli analyze_bundle \
  --bundle tools/workbench/fixtures/bundles/real-analyzed

# 4. 刷新索引
node tools/workbench/scripts/make-fixture.mjs
```

生成器会自动把目录名以 `real-` 开头的 Bundle 纳入 `bundles/index.json`。

> 仓库中提交的 `index.json` 只包含三个合成样例。本地生成真实产物后，索引会多出两条记录，`git status` 会显示该文件被修改。这是预期行为，**不要提交**，否则新克隆的仓库会在下拉框中出现两个加载即失败的条目。

### 这两份产物验证过什么

- 合成样例的字段命名与真实产物一致，曾发现并修正过三处失真。
- Python 生成的 `diagnosis.json` 比 C++ 版本多出 `evidence`、`next_steps`，以及顶层的 `bundle_path`（绝对路径，需脱敏）和 `schema_version`。
- 同一份代码连续运行两次，毫秒级 Pass 耗时可能相差 1.4 到 1.9 倍。因此，`compare` 的单次运行结论会降级为 `suspected_*`。

### 已知缺口

`profile_bundle_test` 只执行编译，不执行推理，因此真实产物中**没有 `runtime`、`cache`、`execution_plan` 或 `device_api` 事件**。依赖这些事件的图表目前只用合成样例验证过。补齐验证需要能执行推理的目标，例如依赖 LLVM 开发库的 `op_numeric_llvm_test`。
