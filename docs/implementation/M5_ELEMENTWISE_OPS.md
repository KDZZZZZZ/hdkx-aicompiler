# M5：Equal、Pow、Erf 与 MiniMind 逐元素缺口

第一波 C 线已经完成 Equal 的声明、InferType、TE/TIR、LLVM 数值、负例，并由 B 线接通 Equal→Where 的 ONNX 组合；Constant 也已由第一波导入。MiniMind 的实际静态折叠图仍需要 Pow、Mul、Div、Sqrt、ReduceMean、Reshape、Expand、Neg、Sigmoid、Unsqueeze 等语义，dynamic_axes 图还会保留更多 shape 控制算子。导入器里“有名字”不等于该 dtype、广播、属性和 LLVM 执行都已验证。

本模块的后续工作按 M9 的真实 prefill/decode inventory 排序：先补 L1a 实际用到的逐元素/静态变形算子，再独立处理 Pow/Erf；不为了凑算子数量放开任意 dtype、广播或外部函数。MiniMind-O 的音频算子不属于本模块。

> 状态：Equal 第一波已完成；Pow/Erf 和 MiniMind 实际缺口待实施。第一波证据见 [G1](G1_RECORD.md)，当前分派见 [WAVE_2](WAVE_2.md)。现有能力以 [OP_SUPPORT_MATRIX.md](../OP_SUPPORT_MATRIX.md) 和 [PROJECT_GOAL.md](../PROJECT_GOAL.md) §2.2 为准。

## 本模块要做的模块

| 模块 | 当前情况 | 计划结果 |
|---|---|---|
| Equal | Relay/LLVM/ONNX/Where 闭环已验收 | 只维护回归和能力矩阵，不重复注册 |
| L1a 算子 | OP_TODO 的静态折叠口径列出 Expand、Neg、Pow、Sigmoid、Unsqueeze 等缺口 | 每个实际节点有 dtype/广播/attrs/LLVM/负例证据 |
| Pow | LLVM 数学调用和边界仍未闭环 | float32 受限指数子集，独立数值误差合同 |
| Erf | 主要服务后续 GELU/视觉链，MiniMind 当前未必使用 | 只有真实 inventory 命中后才进入 L1；否则保持 deferred |
| shape/control | dynamic_axes 下的 Shape/ConstantOfShape 等不是纯逐元素 | 交给 M3/M4，不在此模块偷做 shape VM |

## S1：Equal 的具体行为（已完成）

例如 `A=[1,2,3]`、`B=[1,0,3]`，结果应为 `[true,false,true]`。广播允许时，小输入可以按已证明的轴关系参与比较；不兼容 shape 或未支持 dtype 必须拒绝。ONNX 语义采用逐元素比较和多向广播，见[官方 Equal 定义](https://onnx.ai/onnx/operators/onnx__Equal.html)。

首版以静态精确 CPU/LLVM 为目标，明确列出支持的同类型 int32/int64/float32 输入；bool 输入仅在验证通过后纳入。输出沿用仓库已有 bool 存储约定。浮点 NaN、正负零和普通值的比较遵守选定语义，不能用整数位相等代替数值相等。

## S1 实施步骤（第一波已完成，以下作为回归基线）

1. 在 [relay_op_contract.json](../../contracts/relay_op_contract.json) 增加 canonical `equal`，声明固定二元输入、单输出、无 attrs、纯计算和确定性。优先使用现有 generated subset。
2. 复用现有广播 shape 工具实现 InferType。输出逻辑 dtype 为 bool，输入 dtype 不匹配或 shape 不可广播时立即报错。
3. TE callback 生成已有 TIR EQ 表达式。检查比较结果的逻辑 bool 与 NDArray/kernel buffer 的物理表示是否一致；需要转换时复用已有 cast，不另定义 bool ABI。
4. 接入真实 LowerPrimitiveUnit、LLVM 编译和 RuntimeSession。新增 op 在 ONNX 接入前也已有生产编译消费者，不能只直接调用 TE callback 测试。
5. 更新 FFI、生成文件和注册锚点检查，保证 static archive 中有且只有一个注册来源。
6. 对浮点特殊值、空张量、标量和合法广播做数值比较；把 bool 结果继续用于 Where，验证真实 buffer 的读写约定。
7. 将已验证输入子集和错误规则交给 B 线。导入尚未合入时 onnx_ops 保持与真实 mapping 一致；B 接入 Equal 后共同提交合同变化。

## S2：Pow

先选模型实际需要的 float32 输入和指数子集。沿用现有二元广播计算，在既有 backend math-call 分派中增加明确的参数数量、类型和函数对应，不增加任意外部函数执行入口。

核对 LLVM declaration、JIT 符号/内建支持及实际数值；不能只看生成 IR 中出现了一个名字。测试包含零指数、负底数的整数指数、合法分数指数，以及按声明处理的非有限结果。禁止用未经等价证明的 `exp(b*log(a))` 覆盖一般 Pow。

S2 单独形成 JSON/Relay/TE/backend/LLVM/ONNX 纵向切片。CUDA 仅在对应发射与真实执行验证完成后更新该后端状态。

## S3：Erf

先为模型需要的 float32 建立明确精度与函数调用合同。优先使用工具链已经可用的实现，但必须验证声明和 JIT 可解析性；如需近似算法，要显式记录误差目标和适用区间，不能偷偷替换数学语义。

测试零点、正负对称、饱和区和模型数值区间，与独立高精度参考对比；再用真实 Gelu 组合中的小图验证误差传播。该测试只证明组合结果，不表示整个 Transformer 已支持。

## 修改位置与所有权

| 位置 | 改动 |
|---|---|
| [contracts](../../contracts/relay_op_contract.json)、[generator](../../python/tools/generate_relay_op_contract.py)、generated 文件 | 唯一声明及自动注册 |
| [op.h](../../include/kxc/relay/op.h)、[type_infer.cc](../../src/relay/type_infer.cc)、[tensor math](../../src/relay/op/tensor/math.cc)、[op_ffi.cc](../../src/relay/op/op_ffi.cc) | 类型、计算和公开构造入口 |
| [TE elementwise](../../src/te/topi/elemwise.cc)及既有广播工具 | 纯计算 DAG，不选择 target 或分配 runtime 内存 |
| [LLVM codegen](../../src/codegen/llvm/codegen_llvm.cc)、[LLVM JIT](../../src/codegen/llvm/llvm_jit.cc) | Pow/Erf 后续切片的明确调用支持 |
| [operator_compilation_test](../../test/operator_compilation_test.cpp)、[op_numeric_llvm_test](../../test/op_numeric_llvm_test.cpp)、[infer_type_test](../../test/infer_type_test.cpp) | 完整生产执行和边界验证 |

共享 contract、public op 头和测试文件由集成负责人安排合并顺序。C 线不同时编辑 B 线的 importer；有需要的映射变更作为明确交接。

## 生成与验证

```bash
python3 python/tools/generate_relay_op_contract.py \
  --matrix contracts/relay_op_contract.json \
  --output src/relay/generated/relay_op_contract.inc \
  --registration-output src/relay/generated/relay_op_registration.cc
python3 python/tools/check_relay_op_contract.py --root .
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error \
  -R 'relay_op_contract|registry_test|infer_type_test|operator_compilation_test|op_numeric_llvm_test|compiler_identity_test'
```

完成相应构建后执行上述测试，再跑公共检查及 LLVM 全量。不能手改 generated 文件。

- [x] Equal 有单一注册、正确 dtype/广播推导、真实生产 lowering 和 LLVM 数值结果。
- [x] bool 结果能够直接被 Where 消费；空张量和物理 buffer 表示正确。
- [x] 非法元数、类型、广播在 backend launch 前拒绝。
- [x] ONNX 接线与合同一致，C/B 两线的联合 fixture 通过。
- [ ] Pow/Erf 各自独立标记实现、编译和数值层结果，不能用 Equal 验证替代。

## 身份和风险

新 op 名、attrs、类型和常量进入已有 semantic key；backend 调用规则变化进入相应 backend/pipeline 身份。仅增加新算子条目不意味着无条件提升整个合同格式版本，只有实际兼容性变化才升级。

三项设计检查：复用现有 TIR EQ/广播/调用分派；不产生第二 registry 或 bool ABI；每项新入口都有真实编译运行者和负例。第一波验收只要求 S1 及其 ONNX 交接，不把 S2/S3 顺手加入同一个 PR。
