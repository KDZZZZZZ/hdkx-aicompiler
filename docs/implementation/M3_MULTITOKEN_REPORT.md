# M3 技术报告：单计划多 token 当前长度与 prefill→decode 交接

日期：2026-09-10

本模块把真实 MiniMind decode 图的当前 token 轴 `C` 纳入同一份 bounded LLVM 计划。它验证了一个计划可以连续接收 `C=1/2/3` 的输入，并消费四 token prefill 产生的外部 K/V 后执行两 token decode。K/V 仍由调用方数组提供；本报告不把它描述成 RuntimeSession 持久状态追加。

## 方法

- 新增 `make_minimind_bounded_multitoken_fixture.py`。它从实际八层 MiniMind ONNX decode/prefill 导出生成动态参考、四组 B/C 输入和完整 17 路输出参考，同时保存导出 SHA-256、op 计数和交接元数据。
- 扩展受限 shape resolver：两个不同直接 shape symbol 的加法只作为切片端点的暂存证明；`P+C-C` 可证明为残余轴或常量。无法投影到受准备算子的路径仍 fail-closed。
- 扩展 prepared `slice` 的 Relay/InferType/TE ABI。动态窗口使用三个输入：静态位置表、提供起点 extent 的 anchor、提供窗口长度 extent 的 anchor；`window_size=-2` 和 `window_extent_axis` 进入 canonical attrs。TE 只读取 runtime extent，不读取控制 payload。
- 在 bounded lowering 中把 slice 的两个 anchor 标成 metadata-only 输入，仍由同一 `DynamicUnitShapeContract` 提供 runtime extent 顺序、上界和输出形状检查；slice schema 升至 5，bounded applicability 升至 14，unit shape contract 升至 9。

## 效果与证据

在 `out/build/bounded-llvm/minimind_bounded_decode_llvm_test` 中设置 `KXC_MINIMIND_BOUNDED_MULTITOKEN_DIR=out/fx_minimind_bounded_multitoken` 后：

- 一份 782-call LLVM 计划执行四组 `(B,P,C)`：`(1,4,1)`、`(1,4,2)`、`(2,4,3)`、`(3,4,1)`；17 路 logits/K/V 的最大绝对误差为 `1.83731e-05`。
- 同一计划消费四 token prefill 的 16 路 K/V，执行两 token decode；结果与 ONNX `ReferenceEvaluator` 对齐。
- `C=4` 在 RuntimeSession launch 前被拒绝，profile kernel/alloc/copy 计数和 primitive cache 保持不变。
- `restricted_symbolic_shape_test` 四项通过；Relay op contract 检查 41/41 通过。

复现命令：

```bash
PYTHONPATH=python uv run --with onnx --with numpy -- python3 \
  python/tools/make_minimind_bounded_multitoken_fixture.py \
  --onnx out/minimind_onnx_bounded/minimind_decode.onnx \
  --prefill out/minimind_onnx_bounded/minimind_prefill.onnx \
  --out out/fx_minimind_bounded_multitoken

cmake --build out/build/bounded-llvm --target minimind_bounded_decode_llvm_test -j2
KXC_MINIMIND_BOUNDED_MULTITOKEN_DIR="$PWD/out/fx_minimind_bounded_multitoken" \
  ./out/build/bounded-llvm/minimind_bounded_decode_llvm_test
```

## 边界

当前证据把外部 K/V 的 `P` 固定为代表值 4，以隔离本模块的 `C` 多 token 证明；可变 `P` 与持久 KV capacity/valid extent 属于 M2/M3 state contract，尚未由这份报告宣称完成。CUDA 设备执行仍需在线 Windows GPU 主机，当前本地测试只覆盖 LLVM。
