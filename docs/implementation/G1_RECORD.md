# G1 第一波整体验收记录

> 状态：已完成（2026-09-07）。本文是[第一波并行计划](WAVE_1.md)第 5 节验收清单的逐项证据记录。代码集成点：`dev` @ merge `796fb9f`（A 线 merge commit；C、B 线经 fast-forward 并入）。

## 集成次序与提交

1. **G0 基线**：`e426f7e`（见 [G0 基线记录](G0_BASELINE.md)）。
2. **C 线 Equal**（4 commits `00d9751`→`0b51d52`）：契约生成注册、InferType/TE/FFI、LLVM 数值、负例与身份。合并后全量 42/42 + 检查器通过（线内证据）。
3. **B 线 importer + Equal 接线**（4+2 commits，含 C 合入 `1d2340c`）：Constant + Mul/Sub/Div/Sqrt + Cast/ReduceMean/Reshape、C++ reifier、Equal→Where ONNX fixture。合并 C 后验证：构建 OK、ctest 43/43、pytest 165/165；二阶段后 pytest 176/176。
4. **A 线观测**（3 commits `219203b`→`e191c05`）：runtime 观察者钩子、profiling 适配器、AsyncOperation 完成回调、bundle 正负例。
5. **dev 合并**：`line-c` → `line-b`（含 C）→ `line-a`（merge commit `796fb9f`），无冲突。

## 验收清单逐项证据（WAVE_1 §5）

- [x] **G0 基线已记录**：默认 CPU/LLVM 42/42、bounded 专项 12/12 + 全量 44/44、全部检查器通过（见 [G0 基线记录](G0_BASELINE.md)）。
- [x] **B 列出的 8 种 ONNX 名称有静态受限入口和真实 LLVM 数值证据**：`Constant` 走既有 ParamTensor/常量表示（不新增数学 kernel，dtype/字节保留）；`Mul/Sub/Div/Sqrt` float32 受限广播；`Cast`（int32/int64→float32）；`ReduceMean`（opset 17 静态 axes、keepdims，fixture 用尾轴避开已知缺陷）；`Reshape`（allowzero=0，shape 仅来自 initializer/Constant）。组合 fixture 与独立参考 bit-exact（`onnx_static_s1_protobuf_reifier_llvm_runtime`，pytest 176 + `onnx_importer_test`）。
- [x] **Equal 形成闭环，bool 结果真正用于 Where**：契约（`onnx_ops: ["Equal"]`）→ InferType（同 dtype int32/int64/float32 + NumPy 广播 → bool）→ TE/TIR EQ → LLVM `FCmpOEQ`/`ICmpEQ` → `Compiler::Compile` → RuntimeSession → ONNX `Equal → Where` 组合 fixture bit-exact（`onnx_equal_where_protobuf_reifier_llvm_runtime`）。
- [x] **至少一个组合图由 ONNX 导入后编译执行并导出真实 runtime bundle**：新增 `test/g1_combined_profiling_test.cpp`（ctest `g1_combined_profiling_test`）。ONNX Equal→Where 导入 → 带 ProfileContext 编译 → RuntimeSession 运行；bundle 中同一 run_id 下 `runtime_session_run`（status ok，timing host_execute）+ `kernel_submit`×2（phase=submit，timing host_submit）+ `kernel_exec`×2（真实 kernel_symbol，timing host_execute）+ `alloc`×2 + 常量快照 `copy`×1，内核与分配事件 parent_span_id 均指向 run span；同 bundle 含 relay/tir 编译期事件；观测开启与关闭输出逐位一致。重复运行可复现。**运行注意**：bundle 目录由 CMake 注入 build 目录内（`out/build/dev-ninja-cpu/g1_bundles/`），因为多个 ProfileContext 共用一个 bundle 目录时最后一次 flush 会覆盖前序事件——用独立子目录规避；`onnx_importer_test` 直接以 env 运行只能得到最后一个 case 的 bundle，这是 bundle 目录语义而非观测缺陷。
- [x] **原有静态 Transformer fixture、视觉参照链、bounded 两种合法形状执行均不退化**：合并后 ctest 45/45（含 exact Transformer、ResNet18 参照、`model_prefill_exact_attention`/`model_decode_external_kv`）；bounded 用例在 `dynamic-graph-stage2-integration` @ `f9f955b` 上独立运行（本波未改动其代码，G0 44/44 仍为当前证据；该分支尚未按计划合入主线，合入属后续集成）。
- [x] **错误 shape/dtype/属性在执行前明确失败；开关关闭时原拒绝路径仍有效**：Equal 五类负例（元数/dtype/广播/attrs/输出声明）、B 线全部拒绝组合（见 B 报告）、A 线校验失败抛原异常且 launch 次数为零、profiling 关闭逐位一致，均有测试钉住。
- [x] **profiling 关闭与开启的输出一致，观测不触发编译、改路由或强制设备同步**：`runtime_profiling_test` + `g1_combined_profiling_test` 双重逐位对照。
- [x] **shared contracts 重新生成后通过检查**：`check_relay_op_contract` 25/25（equal 行 stage=tested、onnx 列 1）、`check_pass_contract` 20/20、`check_nlp_gpu_validation` PASS（negative gates 全过）、`check_include_layers` 270 文件、`check_public_headers --compile` 89+8、`check_docs` 通过、`git diff --check` 干净。pytest 176/176。

名称交集计数（仅用于漏项检查，不是"模型可执行"验收）：既有 8 + B 新增 8（`Constant` `Cast` `Div` `Equal` `Mul` `ReduceMean` `Reshape` `Sqrt` `Sub` 中与模型快照的交集）+ Equal 接线后，importer 可接受的模型名称为 **17/25**；仍缺 `ConstantOfShape` `Erf` `Expand` `Pow` `Shape` `Split` `Squeeze` `Unsqueeze` 8 个，且 `Concat` 仍限两输入、`Gather` 仍限常量索引、多输入 Concat 与动态 Gather 属 S2/S3。

## 确认的已知缺陷（本波不修，留待专项切片）

**reduce_mean keepdims=1 数值错误**（主线预存，B 线发现，G1 复核确认）：

- 现象：keepdims=1 且 size-1 保留轴**后跟非单例轴**时，输出各列塌缩为该保留轴首元素值。实测（临时复现程序，未入库）：输入 `arr[i][j][k]=i*100+j*10+k`，[2,3,4]：
  - `axes=[1] keepdims=1` → 期望 [10,11,12,13,110,...]，实测 [10,10,10,10,110,110,110,110]；
  - `axes=[0] keepdims=1` → 期望 [50,51,52,53,...]，实测 [50,50,50,50,51,...]；
  - `axes=[0,1] keepdims=1` → 期望 [60,61,62,63]，实测 [60,60,60,60]；负轴 `axes=[-2]` 同样受影响。
- 正确配置：尾轴 `axes=[2] keepdims=1` 与任意 `keepdims=0` 均正确；现有 `op_numeric_llvm_test` 的 reduce_mean 用例与 Transformer fixture（尾轴）因此未暴露。
- 层位：TE 归约 lowering/TIR 调度链（`src/te/topi/reduction.cc` 一带），与 importer/Equal/观测改动无关。
- 处置：[OP_SUPPORT_MATRIX](../OP_SUPPORT_MATRIX.md) 已标注；修复需独立切片（TE/TIR 层正例+负例），禁止在本波顺手修改。

## 交接与后续

- A 线交接：诊断引擎 `kernel_launch_overhead` 期待 `compiled_module_run`，本波产出工作台词表 `runtime_session_run`/`kernel_exec`，分析器对齐属后续；异步"提交后未观测完成"语义由 AsyncOperation 单测钉住，真实异步证据留待 CUDA 波次。
- B 线交接：initializer 标量（0-d）在既有路径被记为 [1]，新 Constant 路径已修复、initializer 路径未动；`add`/`matmul` 的 MakeAttrs 不校验空 attrs（预存宽松点）。
- C 线交接：Equal 的 bool 输入、float64/int8 等未纳入（fail-closed）；CUDA 列未动。
- 集成协调：`onnx_importer_test` 仅在配置期发现带 onnx 的 Python 时构建（`-DPython3_EXECUTABLE=<venv>`）；CI 必须用带依赖的配置，否则 ONNX e2e 会被静默跳过（ctest 数从 44 变 45 才是含 e2e 的全量）。
- 运行产物：bundle 与复现程序仅在 `out/` 下，不入库。
