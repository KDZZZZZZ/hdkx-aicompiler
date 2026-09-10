# M3 技术报告：定容 prefill 的 int64 索引算术

> 状态：**已完成本切片（2026-09-08）**。这份报告只覆盖 M3 中的 ONNX `Add` int64 导入边界；它不替代 M3 的 bounded attention 联合验收。

## 目标

定容 MiniMind 导出在 `position + arange` 路径上使用 int64 索引算术。原有 Python ONNX importer 的二元算术推导只接受 float32，因此这条形状/位置控制链会在导入前被拒绝。

## 方法

1. 为 ONNX `Add` 单独建立 `ADD_DTYPES = {float32, int64}` 边界。
2. 保持 `Mul`、`Sub`、`Div` 和 `Sqrt` 的既有 float32-only 合同，避免顺带宣称未验证的整数除法或截断语义。
3. 让推导结果继承两个同 dtype 输入的 dtype；Relay 的既有 `AddInferType`、TOPI broadcast add 和 LLVM lowering 已支持同 dtype int64，因此不新增算子注册或第二套算术实现。
4. 为 int64 正例和 int32 负例增加 importer 测试，确保支持范围明确且 fail-closed。

## 改动

- `python/kxc_onnx/importer.py`
  - 增加 `ADD_DTYPES`。
  - `Add` 使用独立 dtype 白名单并保留结果 dtype；其他算术路径保持原约束。
- `test/onnx_importer_py_test.py`
  - 增加 int64 广播 Add 正例。
  - 增加未验证 int32 Add 拒绝例。

## 验证与效果

执行：

```bash
PYTHONPATH=python out/venv/bin/python -m pytest -q test/onnx_importer_py_test.py
```

结果：**233 passed**。`git diff --check` 通过。

这使 `position + arange` 的 int64 `Add` 能通过 Python importer 的类型/形状推导，并继续进入已有 Relay → TE → LLVM 路径；未验证的整数 `Mul`/`Sub`/`Div`/`Sqrt` 仍会在导入期拒绝。

## 后续边界

本切片还没有宣称定容 prefill 端到端完成。剩余工作是让当前定容图实际经过生产 importer、动态 state/extent owner 和 LLVM 运行，并补 M3 bounded attention 与 M2 的联合证据。
