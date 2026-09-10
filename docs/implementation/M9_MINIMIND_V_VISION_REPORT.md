# M9 L2：MiniMind-V 完整视觉链技术报告

2026-09-09。完整 SigLIP2 编码器和 MiniMind-V 投影层已通过 ONNX → Relay → TE/TIR → LLVM → RuntimeSession。两组图像输入均返回 `[1,64,768]` 的视觉特征和投影 token，每次执行 472 个 kernel；对独立参考的最大绝对误差为 `7.15256e-6`。本模块完成固定形状、float32、CPU 视觉阶段。图像 token 插入文本及完整语言模型 prefill/decode 的后续结果见 [联合报告](M9_MINIMIND_V_JOINT_REPORT.md)；变长、多图和 GPU 执行仍待验收。

## 1. 使用的真实模型

- MiniMind-V 源码固定为 [`740d467ece78a0b7d2d976fcb424472095d4a688`](https://github.com/jingyaogong/minimind-v/tree/740d467ece78a0b7d2d976fcb424472095d4a688)，直接使用上游 `MMVisionProjector`，没有改写模型层或缩小网络。
- 视觉配置来自 [`jingyaogong/siglip2-base-p32-256-ve`](https://huggingface.co/jingyaogong/siglip2-base-p32-256-ve/blob/9465d1dc89db6bc6227c5b6b0e0ca9b940325d62/config.json)，revision `9465d1dc89db6bc6227c5b6b0e0ca9b940325d62`，配置 SHA256 `5ad8dda7d55541c7749f9b1cc43fe8eb8c70d8664588d89f710242ce06b3167e`。
- 实际配置为 `SiglipVisionModel`：12 层、hidden 768、12 heads、FFN 3072、patch 32、图像 256×256，直接输出 64 个 patch token。采用该仓库指定的类和配置，不根据 README 中的模型名称另选架构。
- 投影层是 LayerNorm → Linear → GELU → Linear，输入和输出宽度均为 768。构造的视觉模型与投影层共 95,735,040 个参数，其中投影层 1,182,720 个。上游取 `last_hidden_state`，未使用的 pooling head 因此不进入 ONNX 输出依赖图。
- 权重为 seed 0 的完整架构随机初始化；执行 dtype 明确改为 float32。发布配置中的 float16 不是本次支持声明。本次证明编译数值语义，不证明识图质量。
- 版本：torch 2.14.0+cpu、transformers 5.16.1、ONNX 1.22.0、NumPy 2.5.3、LLVM 20.1.2。legacy exporter，opset 17，`dynamo=False`，eager attention。

输入已经是预处理后的 `[1,3,256,256]` float32。上游 processor 的 resize/rescale/normalize、图片解码均在本模块边界外。测试使用归一化范围内的确定性随机输入和全零输入，没有下载训练权重。

## 2. 方法与实现

### 从实际导出补齐 GELU 的两个基础算子

编码器的 `gelu_pytorch_tanh` 导出 12 个 Tanh；上游投影层的精确 GELU 导出 1 个 Erf。新增 canonical Relay `tanh`、`erf`，经过同一 JSON 契约、生成注册、InferType、FFI、TE callback 和生产 primitive lowering。

两个算子均为无属性、单输入、float32、保持 shape 的纯算子。LLVM 使用现有 TIR Call 表达和 ORC host symbol owner，显式生成 `tanhf(float)` / `erff(float)` 调用。没有新增 IR，也没有用指数式或近似多项式代替数学函数。TIR 调用的 arity、参数 dtype 和返回 dtype 在后端声明前检查；算子名称沿既有 graph/unit identity 区分产物。

Python 和 C++ 两侧导入均验证边界。`preserve_shape_values` 暂不接纳这两个新算子；其静态成功不能作为 bounded 或 CUDA 的通过项。

### 用明确的固定输入合同处理 Shape

真实 patch 展平导出了 Shape → Slice → Concat → Reshape。原有 `fold_static_subgraph` 只处理全部输入均为常量的节点，保持不变。

新增的显式导出适配 `fold_fixed_shape_queries` 位于同一常量折叠 owner：要求所有图输入维度固定，移除中间值和输出的调用方 shape 注解，再由 ONNX shape inference 从输入和真实节点推导。仅对已证明的 Shape 调用 ONNX ReferenceEvaluator，使用零 stride 视图提供形状，不分配卷积激活、不执行模型。其余控制链再交给原有常量折叠。符号输入直接拒绝，未知 producer 不借用旧 metadata，控制流子图也拒绝。

该函数只由固定视觉 fixture 显式调用。默认 importer、shape-source、RuntimeSession 都没有隐式 shape 特化或编译。原始 ONNX、导入图、shape proof 和各文件 SHA256 均保留在 fixture receipt 中。原图 703 个节点，显式证明 patch Conv 输出为 `[1,768,8,8]` 后，原有常量折叠消去 219 个节点、物化 502,452 字节，保留 484 个 ONNX 节点；标准编译后是 472 次 kernel 调用。

### 独立参考也需要验证

初次运行原始 ONNX ReferenceEvaluator 时，视觉特征最大差约 5.04。逐层检查发现第一层 Conv 已有 3.794773 的差异。使用 ONNX 中同一组权重直接执行 PyTorch convolution，与导出前 checkpoint 完全一致。

定位到本地 ONNX 1.22 的普通及 [optimized Conv reference](https://github.com/onnx/onnx/blob/v1.22.0/onnx/reference/ops_optimized/op_conv_optimized.py)：两者都把 `auto_pad=VALID` 纳入 SAME padding 分支，并错误使用 batch/channel 尺寸计算 padding。根据 [ONNX Conv 语义](https://onnx.ai/onnx/operators/onnx__Conv.html)，VALID 等价于零 padding。因此仅在单独保存的参考图中将它改写为 `NOTSET` 和显式全零 pads；原始导出和 KXC 导入图保留原语义。没有修改安装的 ONNX 包，也没有使用 KXC 输出作为参考。

改写后的完整参考与 PyTorch 的最大绝对误差为：视觉特征 `4.827976e-6`，投影 token `1.125038e-6`。这项差异定位和等价改写写入 receipt。复查还发现 KXC 原 Conv importer 未处理 `auto_pad`：现在明确接纳 NOTSET/VALID，拒绝未实现的 SAME 模式和 VALID 与显式 pads 混用，避免静默改变语义。

## 3. 达到的效果与验证

| 输入 | 视觉特征最大绝对误差 | 投影 token 最大绝对误差 | kernel 提交 |
|---|---:|---:|---:|
| seed 0 随机归一化像素 | 7.15256e-6 | 1.75834e-6 | 472 |
| 全零归一化像素 | 4.70877e-6 | 1.40071e-6 | 472 |

每个输出的全部 49,152 个元素均检查有限值和 `atol=2e-4, rtol=2e-4`。同一个编译产物和 RuntimeSession 执行两组输入；错误尺寸 `[1,3,255,256]` 和错误 float64 输入均在首个 kernel 前拒绝。profile 中共 944 次提交。

默认 Debug 构建的首次专项运行，编译约 3.44 秒，两次主机执行各约 11.3 秒；该数值包含当前 CPU 执行与观测条件，尚未进行性能优化或统计性基准，不作为实时性/加速比声明。

Tanh/Erf 另有独立 LLVM 数值验证：13 元素尾部、正负饱和区、NaN、±Inf、±0、极小值、标量和零元素张量。类型/FFI/生产 lowering、非法 attrs/arity/dtype、LLVM 错误调用声明和 ONNX opset/输出声明错误也有覆盖。

三轮 QA：第一轮检查实际导出和逐层参考，发现并定位 reference Conv 缺陷；第二轮检查 owner、身份和拒绝边界，补充静态 Shape 的伪 metadata 负例以及 Conv auto_pad 拒绝；第三轮完成三套 CPU/LLVM 回归、全库检查和四份 archive 的单定义核对，并逐一检查视觉 bundle 中的真实提交、完成和失败事件。三套 bundle 均有 6,636 条事件、944 对 host_submit/host_execute；两个错误输入各有明确的 shape/dtype 诊断和 submit_count=0。

最终回归（所有配置都设置 `KXC_MINIMIND_VISION_DIR`，实际执行完整视觉 fixture）：

| 配置 | 结果 | 总时间 |
|---|---:|---:|
| 默认 CPU/LLVM | 52/52 | 69.64 秒 |
| adaptive LLVM，含真实 MiniMind prefill 热替换 | 52/52 | 142.63 秒 |
| bounded LLVM，含原有八组实际模型/阶段 fixture | 67/67 | 289.42 秒 |
| Python 全库 | 307/307 | 1.38 秒 |

Relay 40/40、Pass 20/20，两个生成契约 freshness、NLP 矩阵、include-layer 284 文件、90 个安装头文件与 10 个实验头文件独立编译、61 份文档链接/索引、`git diff --check` 均通过。默认/adaptive/bounded/CUDA archive 分别有 1603/1715/1883/1614 个强 C++ 定义，未发现重复定义。

启用 CUDA 的配置成功构建 runtime archive 和五个专项测试，结果为 **4/5**：device runtime 明确跳过无设备部分，CUDA schedule 为结构验证，runtime profiling 和新算子数值走真实 CPU/LLVM；codegen CUDA 的 source emission/非法 TIR/未绑定 TIR 检查通过，硬件阶段报 `CUDA device is unavailable`，未运行 NVRTC/load/launch 数值。`nvidia-smi` 同时仍报 `Driver/library version mismatch`。这些结果不能算作新视觉算子的 GPU 通过。

复验日志保存于 `/tmp/kxc-l2-final-{default,adaptive,bounded}-{build,ctest,lasttest}.log`、`/tmp/kxc-l2-final-python.log`、`/tmp/kxc-l2-cuda-{build,ctest,lasttest}.log`。专项 4/4 的初始记录另存 `/tmp/kxc-l2-focused-lasttest.log`。

## 4. 复现

源码和配置准备于 `out/minimind-v-source`、`out/minimind-v-config/config.json`；上述 commit/revision 和源码文件 SHA256 由 export receipt 记录。两次导出的原图 SHA256 均为 `0d6bf82e5e6e139e3f555bd001b4d77e7eb17e0a7a905bfb6e24bd6556940b26`。导出与 fixture 工具不隐式下载依赖或权重。

```bash
OPENBLAS_NUM_THREADS=1 out/venv/bin/python python/tools/export_minimind_v_onnx.py \
  --src out/minimind-v-source --vision-config out/minimind-v-config/config.json \
  --config-revision 9465d1dc89db6bc6227c5b6b0e0ca9b940325d62 --out out/minimind_v_vision
OPENBLAS_NUM_THREADS=1 out/venv/bin/python python/tools/make_minimind_vision_fixture.py \
  --export out/minimind_v_vision --out out/fx_minimind_vision
cmake --build out/build/dev-ninja-cpu -j2
KXC_MINIMIND_VISION_DIR=/home/oops/repo/hdkx-aicompiler/out/fx_minimind_vision \
  OPENBLAS_NUM_THREADS=1 ctest --test-dir out/build/dev-ninja-cpu \
  --output-on-failure --no-tests=error -R '^minimind_vision_llvm_test$'
```

未设置 fixture 环境变量时，CTest 明确输出 SKIP；这种退出不算模型数值通过。大模型、参数、参考和 bundle 只写入 `out/`。主要代码为导出/fixture 两个脚本、同一 ONNX fold/importer、生成式 Relay 契约和现有 LLVM 后端；测试入口是 `minimind_vision_llvm_test`。

## 5. 后续边界

本报告保留视觉阶段的独立证据。后续 [图文联合模块](M9_MINIMIND_V_JOINT_REPORT.md)现已验证固定单图的 64 个图像占位 token 替换、视觉/文本 embedding 次序、完整八层 logits/KV 和四步 decode 缓存连续性；多图、图文变长和 GPU 仍待验收。用户排除的新 agent/调度 IR 设计不属于这些工作，两个模块均未引入此类 IR。

CUDA 硬件环境本轮再次确认存在驱动和库版本不匹配，专项硬件门禁失败。本模块目前没有 GPU 数值证据；Tanh/Erf 的 CUDA、视觉多图/多 batch、输入长度变化和 pretrained float16 权重均未开放。全项目目标继续推进。

用户要求修复设备环境后的安装核验、启动文件刷新及 Windows/Tailscale 接入过程见 [GPU 驱动修复报告](GPU_DRIVER_REPAIR_REPORT.md)。本机未重启；后续 Windows 基础 CUDA 专项已通过 5/5，见 [实测报告](GPU_WINDOWS_VALIDATION_REPORT.md)，不替代本视觉模块的 GPU 数值验证。
