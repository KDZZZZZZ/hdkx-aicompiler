# 样例 bundle

## 合成样例（随仓库提供）

```bash
node tools/workbench/scripts/make-fixture.mjs           # 生成
node tools/workbench/scripts/make-fixture.mjs --scale 40 # 放大事件量做虚拟化压测
```

| 目录 | 内容 |
| --- | --- |
| `bundles/baseline` | 完整编译 + 运行时，222 个事件 |
| `bundles/candidate` | 同上，但 `fold_tuple_get_item` 被人为放慢 3.4×、缓存命中率下降 |
| `bundles/raw-unanalyzed` | 同 baseline，但 `diagnosis.json` 是 C++ 写的占位版本 |

生成器的输出格式逐字段对照 `src/base/profiling.cc` 写成，并用真实产物校正过
（pass 名 snake_case、`relay_pipeline`/`tir_pipeline` 是独立 component、
artifact 路径为 `artifacts/<run_id>/{relay,tir,lower}/...` 等）。

## 真实产物（不入库，需自行生成）

`bundles/real-compile` 与 `bundles/real-analyzed` 是编译器真实跑出来的 bundle，
**不随仓库提供**。它们是验证合成样例有没有失真的唯一依据，也是几组测试的数据源；
缺失时相关用例会显式跳过（不会静默通过）。

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

生成器会自动把目录名以 `real-` 开头的 bundle 纳入 `bundles/index.json`。

### 这两份产物验证过什么

- 合成样例的字段命名与真实产物一致（发现过三处失真并已修正）
- Python 的 `diagnosis.json` 比 C++ 版多 `evidence`、`next_steps`，
  以及顶层的 `bundle_path`（绝对路径，需脱敏）与 `schema_version`
- 同一份代码连跑两次，毫秒级 pass 耗时可差 1.4~1.9 倍——
  据此把 compare 的单次运行结论降级为 `suspected_*`

### 已知缺口

`profile_bundle_test` 只跑编译，不执行推理，因此真实产物里
**没有任何 runtime / 缓存 / execution_plan / device_api 事件**。
依赖这些事件的图表目前只用合成样例验证过。要补齐需要能执行推理的目标
（`op_numeric_llvm_test`），而它依赖 LLVM 开发库。
