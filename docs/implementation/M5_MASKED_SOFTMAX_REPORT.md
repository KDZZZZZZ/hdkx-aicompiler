# M5/M3 全 mask 注意力技术报告

日期：2026-09-09。范围：显式 `masked_softmax` 的 Relay/FFI、生产 TE lowering、静态与有界 CPU/LLVM 执行。

## 达到的效果

新增 `masked_softmax(logits, mask, axis=-1)`：布尔 mask 中 True 的位置参与归一化，False 的位置输出精确正零；一整行均为 False 时，该行也输出精确正零。被屏蔽位置中的 NaN/Inf 不污染结果。普通 `softmax` 保留原有语义。

实际编译的 `QK → scale → masked_softmax → V` 注意力图已与独立标量参考对齐：概率的绝对误差不超过 `2e-6`，最终 context 不超过 `3e-6`，全 mask 查询行的概率和 context 均为精确零。一份有界产物运行四组 B/S/T，每次提交四个 LLVM kernel，运行期间不访问或修改 primitive cache。

这是可执行的显式算子能力。前端入口是 C++ Relay 与 FFI，尚未增加 ONNX Attention 导入，也没有把实际 MiniMind 图中的普通 Softmax 自动改写为新算子。

## 方法与合同

[ONNX Attention 的全 mask 测试](https://onnx.ai/onnx/operators/onnx__Attention.html)采用全 mask 查询行输出零的约定；[PyTorch Transformer 教程](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)也说明 SDPA 对这类行的零输出策略。本次在独立的显式算子中采用这一约定，不更改标准 ONNX Softmax。

合同接受 rank≥1 的 float32/float64 logits，归约轴须合法且长度为正；mask 为 bool，可以按尾轴广播到 logits，但不能扩大 logits 的形状。非归约轴可以为空，mask 可以是标量。数值保证针对**参与归一化的有限 logits**；运行时不扫描数据并拒绝可见 NaN/Inf，也不承诺它们的归一化结果。屏蔽元素可以含非有限值。

实现复用已有 TE 广播、Select、max、exp 和 sum：

1. 用相应浮点类型的最小有限值替代被屏蔽的 logits，再求每行最大值。全 mask 行仍有有限最大值。
2. 被屏蔽位置在减法前选择该最大值，使其 shifted 值为零；可见位置减去最大值后求指数。
3. 将被屏蔽的指数权重显式置零，再求和。若权重和为零，分母取 1；最终零权重除以 1 得到精确正零。

没有引入新 IR、单独的 shape evaluator、运行时编译或新的后端数学实现。归约内部的 TE stage 由已有生产 lowering 处理；单个 masked_softmax 对应一个普通 primitive/kernel，并非一次内存遍历或融合 attention kernel。

## 接入与身份

- [算子声明](../../contracts/relay_op_contract.json)是唯一注册来源，使用现有 `SoftmaxAttrs` 和生成式注册。生成器原本就支持写入 `TAttrs`，本次移除检查器中过时的“生成注册只能无属性”限制，保留其他 schema/类型/绑定检查。
- [InferType](../../src/relay/type_infer.cc)验证 dtype、axis 与单向广播；[FFI](../../src/relay/op/op_ffi.cc)提供 `kxc.relay.op._make.masked_softmax`；[TE 实现](../../src/relay/op/nn/softmax.cc)由真实 primitive 编译消费者调用。
- [受限形状解析器](../../src/compiler/shape/shape_value_resolver.cc)复用已有广播证明与归约范围检查。编译适用性版本从 12 升至 13，所有 arity/语法/冻结合同/动态单元入口同步接线；ModuleInvocationContract、runtime state 与 kernel ABI 格式不变。
- 静态 TE Program 身份区分归约轴。两个形状相同而 bool 常量内容不同的图可共享同一 kernel 产物，但必须各自绑定常量内容；这两种情况均经过执行验证。

## 证据与三轮 QA

[masked_softmax_llvm_test](../../test/masked_softmax_llvm_test.cpp)同时覆盖合同和真实 LLVM 执行：

| 检查 | 实际范围与结果 |
|---|---|
| 静态数值 | 六种 dtype/shape/axis 配置，涵盖 float32/64、标量 mask、尾轴广播、非尾归约、空外轴和长度 7；全 False、全 True、混合 mask 与正负最大有限值均通过 |
| 屏蔽非有限值 | False 位置放入 NaN/+Inf；另在图及 cache 释放后执行全 False/NaN 输入，结果仍为精确正零 |
| 注意力组合图 | Q `[B,2,S,4]`、转置 K `[B,2,4,T]`、V `[B,2,T,3]`、mask `[S,T]`；返回概率和 context，首个查询行全 False，其余查询部分屏蔽 |
| 静态与有界执行 | 静态 `(1,3,5)`；有界范围 B=1..2、S=1..4、T=1..6，同产物执行 `(1,1,1)`、`(1,3,5)`、`(2,4,6)`、`(2,2,3)` |
| 执行前拒绝 | 错误 dtype/attrs/axis/arity、mask 扩大输入或不可广播、空归约；有界 T 下界为 0 在准备期拒绝；每组注意力的错误 mask 形状均零 kernel 提交、cache 不变 |
| 缓存与旧路径 | 不同归约轴的 artifact key 不同；bool 常量 payload 正确重绑定；持有 session 时清除 graph/cache 仍能运行；普通 Softmax 的全负无穷输入仍为 NaN |
| CUDA 边界 | 真实 lowering 结果交给完整的 synthetic CUDA Target，在 `BindCudaThreads` 被拒绝；没有声称真实 CUDA 编译、启动或数值通过 |

QA A 检查复用边界：继续使用 SoftmaxAttrs、既有生成器、广播规则和归约；只新增显式算子及其消费者。两处 arity 分类及冻结合同入口已同步，真实有界执行通过。

QA B 检查身份和数值边界：归约轴身份、常量重绑定、正负最大有限值、屏蔽 NaN/Inf、空外轴与普通 Softmax 兼容性全部通过；静态和有界错误 shape 均以零提交、cache 不变验证拒绝行为。

QA C 检查生产调用者和拒绝路径：FFI→InferType→生产 primitive→LLVM→RuntimeSession 接通；有界多形状组合 attention、CUDA 调度拒绝以及对应 NLP reference/能力矩阵同步维护。完整回归结果在下节记录。

## 复现与回归

```sh
cmake --build out/build/bounded-llvm -j2
ctest --test-dir out/build/bounded-llvm --output-on-failure -R '^masked_softmax_llvm_test$'
python3 python/tools/check_relay_op_contract.py --root .
python3 python/tools/check_nlp_gpu_validation.py --root .
```

最终核对日期为 2026-09-09。代码冻结后重建三套配置并完整执行，期间没有修改源码或测试：

| 完整门禁 | 结果 |
|---|---|
| 默认 CPU/LLVM | 51/51，42.18 秒 |
| adaptive 及显式真实 MiniMind fixture | 51/51，116.10 秒；两代各 650 次 LLVM 调用、17 个输出逐位一致 |
| bounded 及八组显式 fixture | 66/66，260.62 秒；完整 prefill/decode、greedy、持久状态和请求批处理均实际执行 |
| Python | 280/280，1.12 秒 |
| Relay / Pass 契约与生成物 | 38/38、20/20，freshness 均通过 |
| 架构 | 90 个安装头和 10 个实验头独立编译；284 个文件的 include 方向、59 篇文档链接及 diff 检查通过 |
| 注册与强定义 | 三套 archive 分别 1597、1709、1877 个强 C++ 定义，无重复；generated/manual 注册检查通过 |
| NLP 能力与参考 | fixture/workload 指纹、零行参考、执行证据门禁通过；过时 numeric gate、缺执行证据、CUDA 越级声明和错误零行参考均被拒绝 |

新测试的默认/adaptive Bundle 各有 38 次提交及成功完成，bounded Bundle 各有 54 次；这是测试调用总数，包含静态压力与兼容性用例。`<build>/out/masked_softmax_profile/events.jsonl` 中，`masked_attention` run metadata 的 batch、queries、keys 可与四个子 kernel 事件关联。旧 MiniMind 回归证明兼容性，不能代替新 mask 的模型接线证据。

完整回归使用既有导出 fixture；在仓库根目录运行：

```sh
cmake --build out/build/dev-ninja-cpu -j2
OPENBLAS_NUM_THREADS=1 ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error
cmake --build out/build/adaptive-llvm -j2
KXC_MINIMIND_ADAPTIVE_PREFILL_DIR="$PWD/out/fx_decode_stateful" OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir out/build/adaptive-llvm --output-on-failure --no-tests=error
KXC_MINIMIND_PROJECTION_DIR="$PWD/out/fx_minimind_projection" \
KXC_MINIMIND_HEADS_DIR="$PWD/out/fx_minimind_heads" \
KXC_MINIMIND_ROPE_DIR="$PWD/out/fx_minimind_rope" \
KXC_MINIMIND_GQA_DIR="$PWD/out/fx_minimind_gqa" \
KXC_MINIMIND_ATTENTION_DIR="$PWD/out/fx_minimind_attention" \
KXC_MINIMIND_BOUNDED_PREFILL_DIR="$PWD/out/fx_minimind_bounded_prefill" \
KXC_MINIMIND_BOUNDED_DECODE_DIR="$PWD/out/fx_minimind_bounded_decode" \
KXC_MINIMIND_DECODE_LOOP_DIR="$PWD/out/fx_decode_stateful" OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error
```

本地完整回执保存在 `/tmp/kxc-masked-softmax-final-{default,adaptive,bounded}-lasttest.log`，检查器与 Python 输出分别在 `/tmp/kxc-masked-softmax-final-checks.log`、`/tmp/kxc-masked-softmax-final-python.log`。这些是临时执行记录，长期复现依据是代码、合同、fixture 与上述命令。

## 效果的限制

本次关闭全 mask 行的 CPU 数值缺口，没有做吞吐、延迟或内存峰值优化承诺。零概率与有限 V 相乘的 attention 输出已验证；V 含 NaN/Inf 的处理不在本次合同内。CUDA 归约、半精度、任意数据相关形状、ONNX Attention 导入及实际模型的新 mask 接线仍需各自实现和证据。
