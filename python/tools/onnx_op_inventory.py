#!/usr/bin/env python3
"""统计 ONNX 图的算子清单，并与仓库当前的算子覆盖面对照。

输出 docs/OP_TODO.md 使用的表格格式，另附三个表面的差距：
  contracts/relay_op_contract.json   契约
  python/kxc_onnx/importer.py        ONNX 导入映射
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import onnx
from onnx import AttributeProto

REPO = Path(__file__).resolve().parents[2]


def attr_summary(node, limit=48):
    parts = []
    for at in node.attribute:
        if at.type == AttributeProto.INT:
            v = str(at.i)
        elif at.type == AttributeProto.FLOAT:
            v = f"{at.f:g}"
        elif at.type == AttributeProto.INTS:
            v = "[" + ", ".join(str(x) for x in at.ints) + "]"
        elif at.type == AttributeProto.STRING:
            v = at.s.decode("utf-8", "replace")
        elif at.type == AttributeProto.TENSOR:
            dims = "x".join(str(d) for d in at.t.dims) or "标量"
            v = f"Tensor({dims})"
        else:
            v = AttributeProto.AttributeType.Name(at.type)
        parts.append(f"`{at.name}: {v}`")
    s = "、".join(parts)
    return (s[:limit] + "…") if len(s) > limit else (s or "无")


def scan(path: Path):
    m = onnx.load(str(path))
    nodes = list(m.graph.node)
    counts = Counter(n.op_type for n in nodes)
    nin, nout, attrs = defaultdict(set), defaultdict(set), {}
    for n in nodes:
        nin[n.op_type].add(len(n.input))
        nout[n.op_type].add(len(n.output))
        if n.op_type not in attrs and n.attribute:
            attrs[n.op_type] = attr_summary(n)
    opset = {i.domain or "ai.onnx": i.version for i in m.opset_import}
    return counts, nin, nout, attrs, opset, len(nodes)


def repo_surfaces():
    contract = json.loads((REPO / "contracts" / "relay_op_contract.json").read_text(encoding="utf-8"))
    relay_ops = sorted(contract["operators"].keys())
    src = (REPO / "python" / "kxc_onnx" / "importer.py").read_text(encoding="utf-8")
    onnx_mapped = sorted(set(re.findall(r'"([A-Z][A-Za-z0-9]*)"\s*:', src)))
    return relay_ops, onnx_mapped


def fmt_set(s):
    return "、".join(str(x) for x in sorted(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", type=Path, help="一个或多个 .onnx 文件")
    ap.add_argument("--label", action="append", default=None, help="与 models 一一对应的阶段名")
    a = ap.parse_args()

    labels = a.label or [p.stem for p in a.models]
    per, union = {}, Counter()
    meta = {}
    for lbl, p in zip(labels, a.models):
        counts, nin, nout, attrs, opset, total = scan(p)
        per[lbl] = counts
        union.update(counts)
        meta[lbl] = (nin, nout, attrs, opset, total)

    print("# 目标模型算子清单（自动生成）\n")
    for lbl in labels:
        nin, nout, attrs, opset, total = meta[lbl]
        print(f"- `{lbl}`：{total} 个节点，{len(per[lbl])} 种算子，opset "
              + "、".join(f"{k}={v}" for k, v in opset.items()))
    print()

    hdr = "| ONNX 算子 | " + " | ".join(labels) + " | 输入数 | 输出数 | 示例属性 |"
    sep = "|---|" + "---:|" * len(labels) + "---|---:|---|"
    print(hdr); print(sep)
    for op in sorted(union):
        cells = " | ".join(str(per[l].get(op, 0) or "—") for l in labels)
        nin = set(); nout = set(); ex = "无"
        for l in labels:
            if op in per[l]:
                nin |= meta[l][0][op]; nout |= meta[l][1][op]
                ex = meta[l][2].get(op, ex) if ex == "无" else ex
        print(f"| `{op}` | {cells} | {fmt_set(nin)} | {fmt_set(nout)} | {ex} |")

    relay_ops, onnx_mapped = repo_surfaces()
    model_ops = set(union)
    reachable = model_ops & set(onnx_mapped)
    missing = sorted(model_ops - set(onnx_mapped))

    print(f"\n## 与仓库三个表面的差距\n")
    print(f"- 模型算子种类：**{len(model_ops)}**")
    print(f"- Relay 契约算子：{len(relay_ops)}（`contracts/relay_op_contract.json`）")
    print(f"- ONNX 导入映射：{len(onnx_mapped)}（`python/kxc_onnx/importer.py`）")
    print(f"- **可达交集：{len(reachable)}/{len(model_ops)}** —— {fmt_set(reachable) or '无'}")
    print(f"- 缺失 {len(missing)} 种：{fmt_set(missing)}")
    unused = sorted(set(onnx_mapped) - model_ops)
    print(f"- 导入器已映射但本模型不用：{fmt_set(unused) or '无'}")


if __name__ == "__main__":
    main()
