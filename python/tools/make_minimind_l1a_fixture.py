#!/usr/bin/env python3
"""生成 L1a 验收 fixture：导入规范、参数、输入与 ONNX 参考输出。

产出目录供 `test/minimind_l1a_llvm_test.cpp` 通过 `KXC_MINIMIND_IMPORT_DIR`
消费。权重按模型规模可达数百 MB，因此产物不入库，也不做构建硬依赖。

参考值用 ONNX 自己的 ReferenceEvaluator 求得，与被测的 importer→Relay→LLVM
路径完全独立，因此比较是真实的交叉验证而不是自证。
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True,
                        help="含 minimind_prefill_static.onnx 的目录")
    parser.add_argument("--out", type=Path, required=True, help="fixture 输出目录")
    parser.add_argument("--seed", type=int, default=0, help="token id 生成种子")
    args = parser.parse_args()

    source = args.onnx / "minimind_prefill_static.onnx"
    model = onnx.load(str(source))
    args.out.mkdir(parents=True, exist_ok=True)

    imported = import_onnx_model(model)
    save_imported_model(imported, args.out / "prefill.json", args.out / "prefill.params")

    batch, seq = imported.function.inputs[0].shape
    vocab = int(imported.function.outputs[0].shape[-1])
    rng = np.random.default_rng(args.seed)
    token_ids = rng.integers(0, vocab, size=(batch, seq), dtype=np.int64)
    (args.out / "input_ids.bin").write_bytes(np.ascontiguousarray(token_ids).tobytes())

    # 参考路径独立于被测路径：折叠后的图直接交给 ONNX 参考实现求值。
    folded, report = fold_static_subgraph(model)
    from onnx.reference import ReferenceEvaluator

    values = ReferenceEvaluator(folded).run(None, {"input_ids": token_ids})
    names = [output.name for output in folded.graph.output]

    manifest = []
    for name, value in zip(names, values):
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        (args.out / f"ref_{name}.bin").write_bytes(array.tobytes())
        manifest.append(f"{name} " + " ".join(str(dim) for dim in array.shape))
    (args.out / "outputs.txt").write_text("\n".join(manifest) + "\n")

    print(f"源图 {len(model.graph.node)} 节点 -> 折叠掉 {report.folded_nodes}，"
          f"实算 {report.remaining_nodes}")
    print(f"Relay 节点 {len(imported.function.nodes)}，参数 {len(imported.params)}")
    print(f"输出 {len(manifest)} 个，fixture 写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
