# M4-D Split 多输出技术报告

状态：已完成（固定两路静态子集）  
日期：2026-09-10

## 目标与范围

本切片补齐 ONNX `Split` 的一条可复现纵向链路，并验证输出顺序、类型、形状和 LLVM RuntimeSession 数值。支持范围明确为：

- ONNX opset >= 2；
- 一个数据输入，或一个数据输入加一个静态 int64 initializer 分段输入；
- 显式或 initializer 提供恰好两个非负分段长度；
- 静态、非负数据 shape，axis 可为负并按 rank 归一化；
- 输入 dtype 为 `float32/float64/int32/int64/int8/uint8/bool`；
- 分段长度之和必须等于输入 axis extent，两个输出声明必须与推导结果完全一致。

动态 split lengths、三路以上输出、空或缺失输出名、任意未证明的 shape 控制值和 CUDA 专项执行不在本切片范围内。

## 实现方法

Relay 层新增 `SplitAttrs(axis, sections)`，并在 `contracts/relay_op_contract.json` 中声明唯一的 `split` 条目。生成器重新产生 contract include 和 registration，注册使用 `FInferType` 与 `FRelayToTEMulti`，没有新增 registry 或运行时 owner。

类型推导要求一个 Tensor 输入和两个静态分段，校验 dtype、rank、axis、int64 溢出和分段总长度，返回按声明顺序排列的两个 TensorType。TE lowering 为每个分段建立独立的 indexed copy，沿 axis 加上累计 offset，输出命名为 `T_split_0` 与 `T_split_1`。构造时复制 shape 容器，避免两个输出共享可变 shape 节点。

Python importer 将 ONNX `Split` 映射到 canonical `split`。它把静态 `split` 属性或 initializer 输入规范化为 attrs，并只把数据输入交给 Relay。C++ spec reifier 同步解析 `axis/sections`，只构造一次 Split Call，再按 output 顺序绑定 `TupleGetItem(call, 0/1)`。因此一个生产 primitive 同时产生两个稳定叶子值，避免重复计算或输出错绑。

## 验证结果

- `python/tools/check_relay_op_contract.py --root .`：**41/41 operators passed**，包含 schema、InferType、multi-output lowering、FFI、ONNX mapping、TIR 和 backend 测试引用。
- `infer_type_test`：Split 负 axis 推导为 `[2,2]` 与 `[2,4]`，并通过 sections 总长度错误、三路 sections 和动态 axis extent 负例。
- `operator_compilation_test`：生产 Split 经过 `Compiler::Compile` 和 LLVM RuntimeSession，输出 shape 为 `[1]`、`[3]`，数值和输出顺序逐元素相等。
- `onnx_importer_test`：临时 JSON import spec 经 C++ reifier → Relay → LLVM → RuntimeSession；`[2,6]` 输入切为 `[2,2]`/ `[2,4]`，结果与独立切片参考 `[0,1,6,7]` 和 `[2,3,4,5,8,9,10,11]` 完全一致。
- Python importer：`PYTHONPATH=python uv run --with onnx --with numpy --with pytest -- python -m pytest -q test/onnx_importer_py_test.py`，**327 passed**，覆盖属性形式、initializer 形式、输出顺序以及输出数、总长度、动态控制输入和 dtype 负例。
- C++ 构建：`kxc_runtime`、`infer_type_test`、`operator_compilation_test`、`onnx_importer_test` 均成功编译并运行。

## 效果与后续边界

Split 现在成为声明、类型、lowering、FFI、ONNX importer、C++ reifier、LLVM 执行和负例检查均闭合的生产算子；旧的单输出 ONNX 节点和 LayerNormalization optional output 规则没有放宽。下一步若需要更广 ONNX 覆盖，应单独设计动态/多于两路分段的 shape 与 ABI 合同，不能把本切片的静态两路保证推广到任意 Split 图。

