# W2 exact-static Transformer operator slice handoff

> 分支：`feature/compiler-foundation-nlp-transformer`
>
> 基线：`525950a`（`baseline/compiler-foundation-w1`）
>
> 当前验证提交：`8f0c664`
>
> 状态：实现与本地证据完成，尚未 push、merge 或取得 production approval。

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
| `gather` | data rank >= 1；indices 为 int32/int64；axis 规范化；输出 shape 为 `data[:axis] + indices.shape + data[axis+1:]` | 合法负索引规范化；KXC 对 ONNX 合法索引域外值作 typed zero-fill 扩展，并用 lazy `Select` 避免 OOB load；CUDA indirect-load schedule 显式拒绝 |
| `where` | condition 必须为 bool；x/y dtype 完全相同且不 promotion；三输入 NumPy trailing broadcast；branch dtype 为 `{float32,float64,int32,int64,int8,uint8,bool}` | 非 bool mask、branch dtype mismatch、不可广播 shape、未知 attrs fail closed |
| `nn_layer_norm` | 仅 float32 data/scale/bias；axis 指定 normalized suffix；scale/bias shape 必须精确等于 suffix；epsilon 有限且 > 0；累积 dtype 固定 `float32` | rank zero、空 normalized suffix、非 float32 accumulation、非单输出 ONNX 形式拒绝；CUDA nested reduction 显式拒绝 |
| `concatenate` | canonical 名仅 `concatenate`；恰好两个输入；显式 axis；同 rank/dtype；非 axis 维相等；axis sum int64-safe；允许 empty | 始终 fresh TE compute/copy，不返回 view/alias；不接受 `concat` alias |
| `slice` | 单一运行时 data 输入；canonical attrs 顺序 `{starts,ends,axes,steps}`；四数组非空等长；axis 唯一；严格 `step == +1`；overflow-safe Python/ONNX clamp；允许 empty | zero、negative、`>1` step、动态 Slice 参数、attribute-form、opset < 10 拒绝；identity/empty 仍 fresh indexed copy |

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
- `test/onnx_import_spec_contract_test.cpp`：initializer-backed token ids、bool condition、scale/bias/fallback 的 C++ reifier 与 TIR lowering；
- `test/op_numeric_llvm_test.cpp`：条件 LLVM/Compiler/RuntimeSession 数值 fixture。

此 fixture 只证明上述 exact-static operator composition，不代表 KV cache、dynamic
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

本地配置：

```bash
cmake -S . -B out/build/w2-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DKXC_ENABLE_CUDA=ON \
  -DKXC_ENABLE_LLVM=OFF -DKXC_BUILD_RESNET18_IR_DUMP=OFF
cmake --build out/build/w2-cuda --parallel 2 --target codegen_cuda_test
./out/build/w2-cuda/codegen_cuda_test
```

设备：NVIDIA GeForce GTX 1650，SM 7.5，driver `580.159.03`。

`codegen_cuda_test` 全部通过，包括：

- 非空 1-D `Where`、positive-step `Slice`、binary `Concatenate` 分别经
  `Compiler` + `RuntimeSession` 的 device 数值路径；
- `Gather` 的 indirect load 在 `BindCudaThreads` 阶段拒绝；
- LayerNorm/Softmax/MatMul reductions 在 `BindCudaThreads` 阶段拒绝；
- 无 silent CPU fallback。

CUDA 结论仅覆盖上述非空 1-D injective fixtures；不声明 empty-output CUDA 或更广
shape target approval。Concat 验证过程中修复了 `topi::concatenate` 对共享 input
shape `Array` 的误写，确保 output shape 构造不会改变 ABI input shape。

### LLVM / ONNX Python 依赖状态

本机缺少 LLVM，以及 Python `onnx`/`numpy`/`pytest`，未安装任何依赖。因此：

- `op_numeric_llvm_test` 的 Gather/Where/LayerNorm/Concat/Slice 和组合 fixture 均保留
  条件测试入口，源码 syntax-only 通过，但**没有本地 LLVM 执行绿色记录**；
- Python ONNX pytest 入口完整且 `py_compile` 通过，但**没有本地 ONNX pytest
  执行记录**；
- C++ dependency-free import-spec/reifier tests 已实际运行通过。

CI 的 LLVM job 已选择 `op_numeric_llvm_test` 与条件 `onnx_importer_test`；CPU job 的
既有 `cpu` label 会覆盖新增 contract/type/reifier negative tests。仓库当前没有可供
本分支新增结论使用的 GPU CI runner，因此 CUDA 证据是上述本地真实设备记录。

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
3. 保持 Gather OOB zero-fill 是 KXC 的确定性域外扩展，而不是 ONNX 对无效索引的
   标准保证。
4. 保持 CUDA Gather/reduction rejection tests；在真正新增 schedule/backend
   数值证据前不得打开这些 gates。
5. 不得据此 handoff 宣称 KV cache、dynamic batching、完整 Transformer 或动态模型
   production support。
