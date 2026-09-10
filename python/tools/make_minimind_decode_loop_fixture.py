#!/usr/bin/env python3
"""生成定容 KV 多步 decode 的验收 fixture（G2 第 2 项）。

与单步 fixture 的关键区别：定容 decode 图的 past 按 `capacity` 定长，历史长度
不进形状，而是由运行时的 `position` 与 `attention_mask` 决定。因此**同一个编译
产物可以服务任意步**——这正是第 2 项要求而静态导出给不了的性质。

本脚本用 ONNX 参考实现把整个 prefill → N 步 decode 的循环跑一遍，逐步落盘：
每步的 token、extent、以及该步的 logits 参考值。C++ 测试据此驱动同一个产物跑
同样的循环，逐步比较。

无效容量区一律填一个**有限的小哨兵**。掩码是加性的 `(1 - mask) * -1e9`，只能
压住 1e9 量级的分数：实测填充值 <= 1e6 时输出与精确长度 past 逐位相同，
>= 1e9 时掩码失效、有效位也会被一起压掉。哨兵取非零值是为了让「无效区泄漏」
可被发现——若图偷偷读了无效槽位，logits 会明显偏离参考值。

目录布局：

    graph.txt                 decode_capacity
    capacity.txt              容量 N
    sentinel.txt              无效区填充值
    layout.txt                "batch kv_heads head_dim layers"
    decode_capacity.json      导入规范
    decode_capacity.params    参数负载
    prefill.json/.params       静态 prefill 的导入规范和参数
    prefill_input_ids.bin      固定 prefill 输入
    prefill_reference_logits.bin  完整 prefill logits 参考
    export_receipt.txt        定容 decode ONNX 的 sha256 回执
    seed_<name>.bin           prefill 产出的初始 cache（extent 条有效）
    seed_extent.txt           初始 extent
    steps.txt                 每行 "step token_id extent_before"
    prefill_logits.bin        prefill 最后一个位置的 logits（greedy 初始采样）
    sampling.txt              采样策略（当前为 greedy_argmax）
    ref_step<i>_logits.bin    该步的 logits 参考
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from kxc_onnx import import_onnx_model, save_imported_model  # noqa: E402
from kxc_onnx.fold import fold_static_subgraph  # noqa: E402


def _evaluate(model: onnx.ModelProto, feeds: dict[str, np.ndarray]):
    from onnx.reference import ReferenceEvaluator

    folded, report = fold_static_subgraph(model)
    names = [output.name for output in folded.graph.output]
    return names, ReferenceEvaluator(folded).run(None, feeds), report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4, help="decode 步数（>= 3）")
    parser.add_argument("--sentinel", type=float, default=7.0,
                        help="无效容量区填充值；必须 <= 1e6，否则加性掩码压不住")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if abs(args.sentinel) > 1e6:
        raise SystemExit("sentinel 超出加性掩码可压制的量级（<= 1e6）")
    if args.steps < 3:
        raise SystemExit("G2 第 2 项要求至少 3 步 decode")

    decode = onnx.load(str(args.onnx / "minimind_decode_capacity.onnx"))
    prefill = onnx.load(str(args.onnx / "minimind_prefill_static.onnx"))
    args.out.mkdir(parents=True, exist_ok=True)
    export_digest = hashlib.sha256(
        (args.onnx / "minimind_decode_capacity.onnx").read_bytes()
    ).hexdigest()
    (args.out / "export_receipt.txt").write_text(
        f"sha256:minimind_decode_capacity.onnx:{export_digest}\n"
    )
    prefill_digest = hashlib.sha256(
        (args.onnx / "minimind_prefill_static.onnx").read_bytes()
    ).hexdigest()
    (args.out / "prefill_export_receipt.txt").write_text(
        f"sha256:minimind_prefill_static.onnx:{prefill_digest}\n"
    )

    shapes = {value.name: [int(d.dim_value) for d in value.type.tensor_type.shape.dim]
              for value in decode.graph.input}
    capacity = shapes["past_k_0"][1]
    batch = shapes["input_ids"][0]
    if batch != 1:
        raise SystemExit("G2 greedy fixture currently requires batch=1")
    vocab = int([int(d.dim_value) for d in decode.graph.output[0].type.tensor_type.shape.dim][-1])

    imported = import_onnx_model(decode)
    save_imported_model(imported, args.out / "decode_capacity.json",
                        args.out / "decode_capacity.params")
    save_imported_model(import_onnx_model(prefill), args.out / "prefill.json",
                        args.out / "prefill.params")
    (args.out / "graph.txt").write_text("decode_capacity\n")
    (args.out / "capacity.txt").write_text(f"{capacity}\n")
    (args.out / "sentinel.txt").write_text(f"{args.sentinel!r}\n")
    layers = sum(1 for name in shapes if name.startswith("past_k_"))
    kv_heads, head_dim = shapes["past_k_0"][2], shapes["past_k_0"][3]
    (args.out / "layout.txt").write_text(
        f"{batch} {kv_heads} {head_dim} {layers}\n")

    # 1) prefill 产出初始 cache
    rng = np.random.default_rng(args.seed)
    prefill_len = [int(d.dim_value) for d in prefill.graph.input[0].type.tensor_type.shape.dim][1]
    prefill_input = rng.integers(0, vocab, size=(batch, prefill_len), dtype=np.int64)
    (args.out / "prefill_input_ids.bin").write_bytes(prefill_input.tobytes())
    names, values, _ = _evaluate(prefill, {"input_ids": prefill_input})
    seed_cache = {name: np.ascontiguousarray(np.asarray(value, np.float32))
                  for name, value in zip(names, values) if name.startswith("present_")}
    prefill_logits = np.ascontiguousarray(np.asarray(dict(zip(names, values))["logits"],
                                                        np.float32))
    if prefill_logits.ndim != 3 or prefill_logits.shape[0] != batch:
        raise SystemExit("prefill logits 必须是 [batch, seq, vocab]")
    (args.out / "prefill_reference_logits.bin").write_bytes(prefill_logits.tobytes())
    (args.out / "prefill_logits.bin").write_bytes(
        np.ascontiguousarray(prefill_logits[:, -1, :]).tobytes()
    )
    (args.out / "sampling.txt").write_text("greedy_argmax\n")
    extent = seed_cache["present_k_0"].shape[1]
    if extent + args.steps > capacity:
        raise SystemExit(f"capacity {capacity} 容不下 prefill {extent} + {args.steps} 步")
    for name, value in seed_cache.items():
        (args.out / f"seed_{name}.bin").write_bytes(value.tobytes())
    (args.out / "seed_extent.txt").write_text(f"{extent}\n")

    # 2) 定容 cache：有效区来自 prefill，其余填哨兵
    cache: dict[str, np.ndarray] = {}
    for name, value in seed_cache.items():
        past = np.full((batch, capacity, value.shape[2], value.shape[3]),
                       args.sentinel, dtype=np.float32)
        past[:, :extent] = value
        cache["past_" + name[len("present_"):]] = past

    # 3) 逐步 decode：同一张图、同一份 past 形状，只有 position/mask 随 extent 变。
    #    token 由上一步 logits 的最后位置做确定性 argmax，形成可复验的 host greedy loop。
    step_lines = []
    next_token = int(np.argmax(prefill_logits[:, -1, :], axis=-1)[0])
    for step in range(args.steps):
        token = next_token
        mask = np.zeros((batch, capacity + 1), dtype=np.float32)
        mask[:, :extent] = 1.0
        mask[:, capacity] = 1.0          # 新 token 恒在下标 capacity
        feeds = {"input_ids": np.full((batch, 1), token, dtype=np.int64),
                 "position": np.asarray([extent], dtype=np.int64),
                 "attention_mask": mask}
        feeds.update(cache)
        names, values, _ = _evaluate(decode, feeds)
        outputs = dict(zip(names, values))
        logits = np.ascontiguousarray(np.asarray(outputs["logits"], np.float32))
        (args.out / f"ref_step{step}_logits.bin").write_bytes(logits.tobytes())
        step_lines.append(f"{step} {token} {extent}")
        next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])

        # 新 K/V 恒在 present 的下标 capacity；写回 cache 的 extent 槽位
        for name, value in outputs.items():
            if not name.startswith("present_"):
                continue
            fresh = np.asarray(value, np.float32)[:, capacity:capacity + 1]
            cache["past_" + name[len("present_"):]][:, extent:extent + 1] = fresh
        # Independent full-capacity state evidence, in the locked interleaved
        # K/V order. Includes the untouched invalid region after every append.
        for layer in range(layers):
            for kind_index, kind in enumerate(("k", "v")):
                state = np.ascontiguousarray(cache[f"past_{kind}_{layer}"])
                (args.out / f"ref_step{step}_state_{2 * layer + kind_index}.bin").write_bytes(state.tobytes())
        extent += 1

    (args.out / "steps.txt").write_text("\n".join(step_lines) + "\n")
    print(f"定容 decode 图：{len(decode.graph.node)} 节点，capacity {capacity}，"
          f"Relay {len(imported.function.nodes)} 节点 / {len(imported.params)} 参数")
    print(f"prefill 种子 extent {seed_cache['present_k_0'].shape[1]}，"
          f"{args.steps} 步 decode，哨兵 {args.sentinel}，写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
