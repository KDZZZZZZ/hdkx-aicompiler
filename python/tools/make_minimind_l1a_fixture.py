#!/usr/bin/env python3
"""生成 L1a/L1b 验收 fixture：导入规范、参数、输入与 ONNX 参考输出。

产出目录供 `test/minimind_l1a_llvm_test.cpp` 通过 `KXC_MINIMIND_IMPORT_DIR`
消费。权重按模型规模可达数百 MB，因此产物不入库，也不做构建硬依赖。

参考值用 ONNX 自己的 ReferenceEvaluator 求得，与被测的 importer→Relay→LLVM
路径完全独立，因此比较是真实的交叉验证而不是自证。

decode 的 past 不用随机值，而是**先跑一遍 prefill 参考、取它的 present**：
prefill 的 `present_*[1,16,4,96]` 与 decode 的 `past_*[1,16,4,96]` 形状精确对接，
decode 的 present 再长成 `[1,17,4,96]`。这样 fixture 是一次真实的自回归续接，
而不是两张互不相干的图各测各的。

目录布局（两张图同构，测试只按清单读，不硬编码任何签名）：

    graph.txt          单行，prefill | decode
    <graph>.json       导入规范
    <graph>.params     参数负载
    inputs.txt         每行 "name dtype d0 d1 ..."，顺序即调用顺序
    in_<name>.bin      每个输入的原始字节
    outputs.txt        每行 "name d0 d1 ..."（一律 float32）
    ref_<name>.bin     每个输出的参考字节
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from kxc_onnx import import_onnx_model, save_imported_model  # noqa: E402
from kxc_onnx.fold import fold_static_subgraph  # noqa: E402

ELEMENT_TYPES = {1: "float32", 7: "int64"}


def _signature(values) -> list[tuple[str, str, list[int]]]:
    return [
        (value.name,
         ELEMENT_TYPES.get(value.type.tensor_type.elem_type, "?"),
         [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim])
        for value in values
    ]


def _evaluate(model: onnx.ModelProto, feeds: dict[str, np.ndarray]):
    """在折叠后的图上求参考值；折叠只是消掉静态子图，不改变语义。"""
    from onnx.reference import ReferenceEvaluator

    folded, report = fold_static_subgraph(model)
    names = [output.name for output in folded.graph.output]
    return names, ReferenceEvaluator(folded).run(None, feeds), report


def _prefill_feeds(model: onnx.ModelProto, seed: int) -> dict[str, np.ndarray]:
    inputs = _signature(model.graph.input)
    vocab = int(_signature(model.graph.output)[0][2][-1])
    name, _, shape = inputs[0]
    rng = np.random.default_rng(seed)
    return {name: rng.integers(0, vocab, size=tuple(shape), dtype=np.int64)}


def _decode_feeds(onnx_dir: Path, decode: onnx.ModelProto,
                  seed: int) -> dict[str, np.ndarray]:
    """decode 的 past 取自 prefill 的 present，构成一次真实的自回归续接。"""
    prefill = onnx.load(str(onnx_dir / "minimind_prefill_static.onnx"))
    names, values, _ = _evaluate(prefill, _prefill_feeds(prefill, seed))
    present = {name: np.ascontiguousarray(np.asarray(value, dtype=np.float32))
               for name, value in zip(names, values) if name.startswith("present_")}

    feeds: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed + 1)
    for name, dtype, shape in _signature(decode.graph.input):
        if name.startswith("past_"):
            source = present["present_" + name[len("past_"):]]
            if list(source.shape) != shape:
                raise SystemExit(
                    f"prefill present {source.shape} does not match decode past "
                    f"{shape} for '{name}'")
            feeds[name] = source
        elif dtype == "int64":
            vocab = int(_signature(decode.graph.output)[0][2][-1])
            feeds[name] = rng.integers(0, vocab, size=tuple(shape), dtype=np.int64)
        else:
            raise SystemExit(f"unexpected decode input '{name}' of dtype {dtype}")
    return feeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True,
                        help="含 minimind_{prefill,decode}_static.onnx 的目录")
    parser.add_argument("--out", type=Path, required=True, help="fixture 输出目录")
    parser.add_argument("--graph", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--seed", type=int, default=0, help="token id 生成种子")
    args = parser.parse_args()

    source = args.onnx / f"minimind_{args.graph}_static.onnx"
    model = onnx.load(str(source))
    args.out.mkdir(parents=True, exist_ok=True)

    imported = import_onnx_model(model)
    save_imported_model(imported, args.out / f"{args.graph}.json",
                        args.out / f"{args.graph}.params")
    (args.out / "graph.txt").write_text(args.graph + "\n")

    feeds = (_prefill_feeds(model, args.seed) if args.graph == "prefill"
             else _decode_feeds(args.onnx, model, args.seed))

    # 输入按图声明顺序落盘：测试据此原样构造调用参数，顺序即 ABI。
    input_manifest = []
    for name, dtype, shape in _signature(model.graph.input):
        array = np.ascontiguousarray(feeds[name])
        (args.out / f"in_{name}.bin").write_bytes(array.tobytes())
        input_manifest.append(f"{name} {dtype} " + " ".join(str(d) for d in shape))
    (args.out / "inputs.txt").write_text("\n".join(input_manifest) + "\n")

    names, values, report = _evaluate(model, feeds)
    output_manifest = []
    for name, value in zip(names, values):
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        (args.out / f"ref_{name}.bin").write_bytes(array.tobytes())
        output_manifest.append(f"{name} " + " ".join(str(d) for d in array.shape))
    (args.out / "outputs.txt").write_text("\n".join(output_manifest) + "\n")

    print(f"[{args.graph}] 源图 {len(model.graph.node)} 节点 -> 折叠掉 "
          f"{report.folded_nodes}，实算 {report.remaining_nodes}")
    print(f"[{args.graph}] Relay 节点 {len(imported.function.nodes)}，"
          f"参数 {len(imported.params)}")
    print(f"[{args.graph}] 输入 {len(input_manifest)} 个，输出 "
          f"{len(output_manifest)} 个，写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
