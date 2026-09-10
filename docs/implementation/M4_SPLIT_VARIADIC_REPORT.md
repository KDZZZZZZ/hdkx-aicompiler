# M4-D：静态 Split 多路输出扩展技术报告

日期：2026-09-10  
状态：已完成限定范围实现并通过 CPU/LLVM 回归

## 目标

ONNX/Relay `Split` 原先只允许固定两路静态分段。该限制会阻断常见的三路以上拆分，也使多输出 ABI 的顺序保证只能由双输出样例覆盖。本次把支持范围扩展为**静态常量分段、至少两路、输出数量与分段数量相等**，同时保留无法证明时的执行前拒绝。

## 实现方法

1. **声明和身份契约**：`OperatorSpec.output_arity` 继续表示固定输出叶子数量；值为 `0` 表示由类型推导和 lowering 校验的 variadic 多输出。Split 的生成契约改为 `output_arity=0`，并重新生成 Relay 注册代码。固定输出算子仍执行原有精确 arity 检查。
2. **ONNX/Python 入口**：Split 节点现在要求至少两个非空输出名，且数量必须等于静态 `sections`；分段输入仍只接受可证明的 int64 initializer。错误消息明确指出少于两路、动态分段或数量不匹配的原因。
3. **Relay 类型推导**：保留负 axis 归一化、非负 section 和总长度检查；为任意数量的 sections 生成同顺序的 tuple tensor fields。单路 Split 仍拒绝。
4. **TE/lowering 与重建**：Split 仍降低为一个多输出 primitive，按 sections 顺序写入各输出 buffer；JSON spec reifier 和 FFI 构造器保留同一 canonical attrs，并按 tuple field 数量绑定 `TupleGetItem`。没有新增第二套 ABI。
5. **可验证性边界**：动态分段长度、动态 partition axis、零长度/负长度、section 总和不等于输入 extent、输出数量不匹配及少于两路都在导入或执行前失败。CUDA 设备执行不属于本模块验收范围。

## 验证

- Python：`test/onnx_importer_py_test.py -k split`，7 passed。
- C++ InferType：三路 `{1,2,3}` 类型推导、单路拒绝、动态轴拒绝和原有双路合同通过。
- C++ production lowering：单个 `{1,1,2}` Split primitive 返回三个输出，shape 和数值顺序通过。
- C++ ONNX JSON → Relay → LLVM → RuntimeSession：输入 `[2,6]` 按 `{1,3,2}` 产生 `left/middle/right` 三个输出，shape 与独立切片参考逐值一致。
- Relay 契约：`check_relay_op_contract.py` 报告 `41 checked, 41 passed`；生成文件 freshness 检查通过。
- 代码质量：`git diff --check` 通过。

## 效果和剩余限制

Split 的静态多输出链路已从仅双路扩展到任意两路以上，输出顺序、tuple 类型、单 primitive ABI 和 RuntimeSession 结果由三路端到端样例共同覆盖。该变化没有放开任意动态 ONNX Split：动态 lengths、动态轴和数据相关分段仍需后续 shape/value 合同，不能由本次静态实现推断支持。

