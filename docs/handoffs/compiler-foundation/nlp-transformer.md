# W2 exact-static Transformer operator slice handoff

> 分支：`feature/compiler-foundation-nlp-transformer`
>
> 基线：`525950a`（`baseline/compiler-foundation-w1`）
>
> 终审基线：`952be66`
>
> 状态：终审修复进行中；尚未 push、merge 或取得 production approval。CUDA 仅有 source/local-evidence 状态，不是可复现 GPU CI。

## 1. 本次交付范围

本分支补齐五个高价值、静态、精确的 Transformer operator vertical slices：

1. `gather`（embedding lookup）
2. `where`（三元 select）
3. `nn_layer_norm`
4. binary `concatenate`
5. positive-step `slice`

每个算子均进入机器契约 `test/relay_op_contract.json` 及 generated
`OperatorSpec`，并覆盖 canonical attrs、type inference、Relay-to-TE、FFI、Python
ONNX importer、C++ import-spec reifier、CPU negative tests、条件 LLVM 数值测试和
`test/nlp_validation/transformer_capability_matrix.json`。

本次**不交付或声明**：

- KV cache 或 cache mutation；
- dynamic batching、动态 rank 或动态 extent 执行；
- masked-softmax/all-masked-row 新语义；
- CUDA Gather、LayerNorm、Softmax 或 MatMul reduction 数值支持；
- 完整生产 Transformer、decode scheduler 或 attention kernel。

## 2. 精确算子契约

| 算子 | 静态 exact 子集 | 明确拒绝/扩展 |
|---|---|---|
| `gather` | 通用 Relay 支持 int32/int64 runtime indices；ONNX importer 只接受 initializer-backed 且每值位于 `[-extent,extent-1]` 的 subset | Python 与 C++ reifier 双重验证 Constant payload；动态/OOB ONNX indices 拒绝。KXC-only Relay OOB zero-fill 扩展仍保留；CUDA indirect-load schedule 显式拒绝 |
| `where` | condition 必须为 bool；x/y dtype 完全相同且不 promotion；三输入 NumPy trailing broadcast；branch dtype 受限 | 非 bool mask、branch dtype mismatch、不可广播 shape、未知 attrs fail closed；FFI 保持 undefined/fieldless attrs |
| `nn_layer_norm` | float32 data/scale/bias/Y；float64 mean/variance/sqrt/affine，最终 cast Y；axis suffix 与 affine shape 精确；epsilon 有限正 float32 attr | ONNX 仅接受 `[Y]`、`[Y,""]`、`[Y,"",""]`，spec 只携 Y；非空 Mean/InvStdDev、空 normalized suffix、非 float64 accumulation 拒绝；CUDA reduction 拒绝 |
| `concatenate` | canonical 名仅 `concatenate`；恰好两个输入；显式 axis；同 rank/dtype；非 axis 维相等；axis sum int64-safe；允许 empty | 始终 fresh TE compute/copy，不返回 view/alias；不接受 `concat` alias |
| `slice` | 单一运行时 data 输入；提供的四个 ONNX 控制 tensor 必须统一 int32 或统一 int64，并 canonicalize 为 int64 attrs；数组非空等长；axis 唯一；严格 `step == +1`；允许 empty | zero、negative、`>1` step、mixed-width/dynamic Slice 参数、attribute-form、opset < 10 拒绝；identity/empty 仍 fresh indexed copy |

TE-to-TIR 共享 static gate 同时约束 whole-graph 与 production per-unit lowering：每个
iteration extent `<= INT32_MAX`；element count/flatten max/bytes 以 checked arithmetic
同时受 `INT64_MAX` 与 `size_t` 限制；零元素合法但仍扫描并校验全部 sibling extent。
Flatten、Slice、Concat 不再各自绕过该 contract。

Python ONNX importer 对属性 protobuf 类型严格检查，不再把 FLOAT axis 截断为 INT，
并拒绝 duplicate attrs。Slice 接受 opset >= 10 input form；starts/ends/可选
axes/steps 必须为非空 rank-1 int32/int64 initializer。缺省 axes/steps 会被折叠为
canonical attrs；`[data, starts, ends, "", steps]` 的空 optional axes slot 有测试。

## 3. 组合 fixture

`exact_transformer_operator_slice` 以静态 shape 组合：

```text
embedding table --Gather(token ids)-- LayerNorm --Where(bool select)
                                           |-- Slice(prefix) --|
                                           +-------------------Concatenate
                                                             |
                               Transpose -- MatMul -- Softmax -- MatMul(context)
```

覆盖入口：

- `test/infer_type_test.cpp`：组合 shape/type、whole-graph TIR 和 9 个 per-op lowering unit；
- `test/onnx_import_spec_contract_test.cpp`：dependency-free C++ JSON reifier contract（不是 protobuf E2E）；
- `test/op_numeric_llvm_test.cpp`：手写 Relay 的条件 LLVM numeric source（也不是 importer E2E）；
- `test/generate_onnx_transformer_fixture.py` + `test/onnx_importer_test.cpp`：真实 `ModelProto` 写盘/重载 -> Python importer/serializer -> C++ reifier -> LLVM Compiler -> `RuntimeSession` 数值执行。LLVM CI 的 `onnx_importer_test` 命令实际执行该链路，无文件存在性替代。

这些 fixture 只证明上述 exact-static operator composition，不代表 KV cache、dynamic
batching、causal mask 或完整模型支持。

## 4. Backend 事实

### CPU / dependency-free

以下命令通过：

```bash
cmake --build out/build/w2-cpu --parallel 2
ctest --test-dir out/build/w2-cpu --output-on-failure \
  --no-tests=error --label-regex '(^|;)cpu(;|$)'
```

结果：`38/38` CPU-labelled tests 通过。

另外通过：

```bash
python3 python/tools/check_relay_op_contract.py --root .
python3 python/tools/check_nlp_gpu_validation.py --root .
python3 -m py_compile python/kxc_onnx/importer.py \
  test/onnx_importer_py_test.py python/tools/check_nlp_gpu_validation.py
c++ -std=c++17 -Iinclude -Ithird_party/dlpack/include \
  -DKXC_USE_LLVM=0 -fsyntax-only test/op_numeric_llvm_test.cpp
git diff --check
```

Contract checker：24/24 operators；NLP checker：PASS。

### CUDA

本次终审只执行了 CUDA-enabled **build**：

```bash
cmake --build out/build/w2-cuda --parallel 2 --target codegen_cuda_test
```

目标编译通过，但本次没有运行 CUDA executable，也没有 GPU CI/local-run artifact，
因此不记录设备、driver 或数值 PASS。`codegen_cuda_test` 源码包含非空 1-D
Where/Slice/Concat 的 Compiler+RuntimeSession fixture；capability matrix 只标
`implemented/local-evidence`，不得标 `validated`。同一源码要求多维
Where/Slice/Concat、Gather indirect load 和 LayerNorm/Softmax/MatMul reductions 在
target schedule fail closed；empty-output CUDA 和更广 shape 均不声明。

### LLVM / ONNX Python 依赖状态

本机缺少 LLVM，以及 Python `onnx`/`numpy`/`pytest`，未安装任何依赖。因此：

- `op_numeric_llvm_test` 的 Gather/Where/LayerNorm/Concat/Slice 和组合 fixture 均保留
  条件测试入口；LayerNorm 源码覆盖 FLT_MAX 常量行、巨大 offset、multi-axis、axis 0、prefix-zero，但**没有本地 LLVM 执行绿色记录**；
- Python ONNX pytest 入口完整且 `py_compile` 通过，但**没有本地 ONNX pytest
  执行记录**；
- C++ dependency-free import-spec/reifier tests 已实际运行通过。

CI 的 LLVM job 选择 `op_numeric_llvm_test` 与 `onnx_importer_test`；后者配置为实际执行 protobuf-to-RuntimeSession 小 fixture，但当前尚无本地/本次 CI 绿色记录。CPU job 的 `cpu` label 覆盖 contract/type/reifier negative tests。仓库没有 GPU CI runner，因此 CUDA 只保留 source/local-evidence 级别。

## 5. 原子提交

| Commit | 内容 |
|---|---|
| `1af5734` | exact static Gather/embedding slice |
| `7603745` | exact static Where slice |
| `856ac63` | exact static LayerNorm slice |
| `5d902f7` | exact static binary Concatenate slice |
| `be32700` | exact static positive-step Slice slice |
| `0d38af1` | 最小 exact Transformer composition fixture |
| `1b53ba8` | strict ONNX attrs、bool Constant reification、组合 reifier 与多轴 LayerNorm coverage |
| `299b27e` | CUDA injective validation、Gather/reduction rejection 与 Concat ABI shape 修复 |
| `8f0c664` | C++ reifier 对 Gather/Where 非 canonical attrs 的 fail-closed 修正 |

## 6. 集成注意事项

1. 从 `525950a..299b27e` 按顺序审查/集成，保留前五个 per-operator 原子提交。
2. 不要把 capability matrix 中 reference numeric `validated` 等同于本机 LLVM
   backend validation；每层状态和 gate 必须分别解释。
3. 保持 Gather OOB zero-fill 是 KXC 的显式 Relay 扩展；默认 ONNX Gather 只允许
   initializer-backed、逐值在域内的 subset，Python/C++ 双边均不可放宽。
4. 保持 CUDA Gather/reduction 及 multi-dimensional injective rejection tests；没有
   可复现 GPU CI 前，1-D CUDA 记录也只能是 implemented/local-evidence。
5. 不得据此 handoff 宣称 KV cache、dynamic batching、完整 Transformer 或动态模型
   production support。
