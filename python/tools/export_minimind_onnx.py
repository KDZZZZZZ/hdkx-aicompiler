#!/usr/bin/env python3
"""把 MiniMind 导出为 ONNX，用于生成目标模型算子清单。

导出两张图，对应自回归推理的两个阶段：
  prefill  input_ids [B, S]        -> logits + 8 组 (k, v)
  decode   input_ids [B, 1] + past -> logits + 8 组 (k, v)

模型权重随机初始化：本脚本只关心图结构与算子集合，不关心数值。
用法见 --help；产物默认落在 out/minimind_onnx/。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO / "out" / "minimind"
DEFAULT_OUT = REPO / "out" / "minimind_onnx"
SEED = 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def minimind_commit(src: Path) -> str | None:
    """记录 MiniMind 源码 commit；非 git 检出时如实返回 None。"""
    try:
        return subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def load_model_module(src: Path):
    if not (src / "model" / "model_minimind.py").exists():
        sys.exit(f"找不到 MiniMind 源码：{src}\n先克隆 https://github.com/jingyaogong/minimind.git 到该路径。")
    sys.path.insert(0, str(src))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
    return MiniMindConfig, MiniMindForCausalLM


class Prefill(nn.Module):
    """无 past 的首次前向，返回 logits 和全部 KV。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        out = self.model(input_ids, use_cache=True)
        flat = []
        for k, v in out.past_key_values:
            flat += [k, v]
        return (out.logits, *flat)


class Decode(nn.Module):
    """带 past 的单步前向。past 以扁平张量列表传入，两个一组。"""

    def __init__(self, model, num_layers):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(self, input_ids, *flat_past):
        past = [(flat_past[2 * i], flat_past[2 * i + 1]) for i in range(self.num_layers)]
        out = self.model(input_ids, past_key_values=past, use_cache=True)
        flat = []
        for k, v in out.past_key_values:
            flat += [k, v]
        return (out.logits, *flat)


def export(stage, wrapper, args, names_in, names_out, dyn, path, opset):
    print(f"  导出 {stage} -> {path.name}", flush=True)
    torch.onnx.export(
        wrapper,
        args,
        str(path),
        input_names=names_in,
        output_names=names_out,
        dynamic_axes=dyn,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="MiniMind 仓库路径")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="ONNX 输出目录")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=16, help="prefill 的样例序列长度")
    ap.add_argument("--past", type=int, default=16, help="decode 的样例已缓存长度")
    ap.add_argument("--max-pos", type=int, default=2048, help="RoPE 表长度，只影响常量大小")
    ap.add_argument("--flash", action="store_true", help="保留 SDPA 融合路径（默认关闭以导出显式 attention）")
    ap.add_argument("--static", action="store_true", help="不声明 dynamic_axes，导出固定形状图（用于对照）")
    a = ap.parse_args()

    MiniMindConfig, MiniMindForCausalLM = load_model_module(a.src)

    cfg = MiniMindConfig(
        hidden_size=768,
        num_hidden_layers=8,
        use_moe=False,
        flash_attn=a.flash,
        max_position_embeddings=a.max_pos,
        dropout=0.0,
    )
    torch.manual_seed(SEED)
    model = MiniMindForCausalLM(cfg).eval()

    L, KV, HD, V = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, cfg.vocab_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MiniMind: {L} 层 / hidden {cfg.hidden_size} / heads {cfg.num_attention_heads}"
          f" / kv_heads {KV} / head_dim {HD} / vocab {V} / 参数 {n_params/1e6:.2f}M")
    print(f"flash_attn={cfg.flash_attn}  opset={a.opset}")

    a.out.mkdir(parents=True, exist_ok=True)
    kv_names = sum(([f"past_k_{i}", f"past_v_{i}"] for i in range(L)), [])
    present_names = sum(([f"present_k_{i}", f"present_v_{i}"] for i in range(L)), [])
    written: list[Path] = []

    with torch.no_grad():
        # ---- prefill ----
        ids = torch.randint(0, V, (a.batch, a.seq), dtype=torch.long)
        dyn = {"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}}
        for n in present_names:
            dyn[n] = {0: "batch", 1: "total"}
        export("prefill", Prefill(model), (ids,), ["input_ids"],
               ["logits"] + present_names, None if a.static else dyn,
               a.out / ("minimind_prefill_static.onnx" if a.static else "minimind_prefill.onnx"), a.opset)
        written.append(a.out / ("minimind_prefill_static.onnx" if a.static else "minimind_prefill.onnx"))

        # ---- decode ----
        ids1 = torch.randint(0, V, (a.batch, 1), dtype=torch.long)
        past = []
        for _ in range(L):
            past += [torch.randn(a.batch, a.past, KV, HD), torch.randn(a.batch, a.past, KV, HD)]
        dyn = {"input_ids": {0: "batch"}, "logits": {0: "batch"}}
        for n in kv_names:
            dyn[n] = {0: "batch", 1: "past"}
        for n in present_names:
            dyn[n] = {0: "batch", 1: "total"}
        export("decode", Decode(model, L), (ids1, *past), ["input_ids"] + kv_names,
               ["logits"] + present_names, None if a.static else dyn,
               a.out / ("minimind_decode_static.onnx" if a.static else "minimind_decode.onnx"), a.opset)
        written.append(a.out / ("minimind_decode_static.onnx" if a.static else "minimind_decode.onnx"))

    commit = minimind_commit(a.src)
    meta = {
        "model": "jingyaogong/minimind (MiniMindForCausalLM)",
        "minimind_source": {"path": str(a.src), "commit": commit},
        "seed": SEED,
        "params_M": round(n_params / 1e6, 2),
        "config": {k: getattr(cfg, k) for k in
                   ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                    "head_dim", "intermediate_size", "vocab_size", "hidden_act", "rms_norm_eps",
                    "rope_theta", "max_position_embeddings", "tie_word_embeddings", "flash_attn")},
        "export": {"opset": a.opset, "torch": torch.__version__, "onnx": pkg_version("onnx"),
                   "dynamo": False, "static_shapes": a.static,
                   "batch": a.batch, "prefill_seq": a.seq, "decode_past": a.past},
        "kv_layout": "[batch, seq, kv_heads, head_dim]",
        "sha256": {p.name: sha256_file(p) for p in written},
    }
    (a.out / ("export_metadata_static.json" if a.static else "export_metadata.json")).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"完成，产物在 {a.out}")


if __name__ == "__main__":
    main()
