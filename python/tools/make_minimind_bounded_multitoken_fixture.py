#!/usr/bin/env python3
"""Import MiniMind decode with a bounded current-token axis and generate references.

The fixture keeps the decode graph's current token length as ``C``.  Only the
C++ restricted preparation may prove and compile it; this tool performs ONNX
import and reference execution for independent cases plus one prefill→decode
handoff.  It intentionally does not allocate or mutate a K/V runtime state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from kxc_onnx import import_onnx_model, save_imported_model  # noqa: E402
from kxc_onnx.fold import fold_static_subgraph  # noqa: E402


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--prefill", type=Path, help="matching prefill export; defaults to the sibling export")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = onnx.load(args.onnx)
    source_node_count = len(raw.graph.node)
    model, _ = fold_static_subgraph(raw)
    names = ["logits"] + [f"present_{kind}_{layer}" for layer in range(8) for kind in "kv"]
    counts = Counter(node.op_type for node in model.graph.node)
    if ([value.name for value in model.graph.input] != ["input_ids"] + [f"past_{kind}_{layer}" for layer in range(8) for kind in "kv"] or
            [value.name for value in model.graph.output] != names or
            counts["Sigmoid"] != 8 or counts["Softmax"] != 8 or counts["MatMul"] != 73):
        raise ValueError("fixture requires the actual eight-layer MiniMind decode export")
    table = next(t for t in model.graph.initializer if t.name == "model.model.embed_tokens.weight")
    if list(table.dims) != [6400, 768]:
        raise ValueError("fixture requires vocabulary 6400 and hidden width 768")
    shapes = [["B", "C", 6400]] + [["B", "T", 4, 96] for _ in names[1:]]
    dynamic = helper.make_model(helper.make_graph(
        list(model.graph.node), "minimind_actual_bounded_decode",
        [helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, ["B", "C"])] +
        [helper.make_tensor_value_info(f"past_{kind}_{layer}", onnx.TensorProto.FLOAT, ["B","P",4,96])
         for layer in range(8) for kind in "kv"],
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [None] * len(shape))
         for name, shape in zip(names, shapes)], initializer=list(model.graph.initializer)),
        opset_imports=list(model.opset_import), ir_version=model.ir_version)
    onnx.checker.check_model(dynamic)
    representative = onnx.ModelProto()
    representative.CopyFrom(dynamic)
    for value in list(representative.graph.input) + list(representative.graph.output):
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param in {"B", "P", "C", "T"}:
                extent = {"B": 1, "P": 4, "C": 2, "T": 6}[dim.dim_param]
                dim.ClearField("dim_param")
                dim.dim_value = extent
    onnx.checker.check_model(representative)
    imported = import_onnx_model(representative, fold_constants=False, preserve_shape_values=True)
    args.out.mkdir(parents=True, exist_ok=True)
    onnx.save(dynamic, args.out / "decode_multi_dynamic.onnx")
    onnx.save(representative, args.out / "decode_multi_representative.onnx")
    save_imported_model(imported, args.out / "decode_multi.json", args.out / "decode_multi.params")
    (args.out / "export_receipt.txt").write_text(sha256(args.out / "decode_multi_dynamic.onnx") + "\n")
    evaluator = ReferenceEvaluator(dynamic)
    rng = np.random.default_rng(20260909)
    # Keep external K/V at the representative extent.  C remains polymorphic;
    # variable P belongs to the separate persistent KV/state contract.
    cases = [(1, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 1)]
    files = ["decode_multi_dynamic.onnx", "decode_multi_representative.onnx", "decode_multi.json",
             "decode_multi.params", "export_receipt.txt", "cases.txt"]
    for case, (batch, sequence, current) in enumerate(cases):
        tokens = rng.integers(0, 6400, size=(batch, current), dtype=np.int64)
        filename = f"input_{case}.bin"
        (args.out / filename).write_bytes(tokens.tobytes())
        files.append(filename)
        feed = {"input_ids": tokens}
        for state, name in enumerate(value.name for value in dynamic.graph.input if value.name != "input_ids"):
            array = rng.normal(0,0.1,size=(batch,sequence,4,96)).astype(np.float32)
            feed[name] = array
            filename = f"past_{case}_{state}.bin"
            (args.out / filename).write_bytes(array.tobytes())
            files.append(filename)
        for output, array in enumerate(evaluator.run(None, feed)):
            expected = (batch, current, 6400) if output == 0 else (batch, sequence+current, 4, 96)
            if array.shape != expected or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"invalid ONNX reference output {names[output]}")
            filename = f"ref_{case}_{output}.bin"
            (args.out / filename).write_bytes(array.tobytes())
            files.append(filename)
        print(f"ONNX reference B={batch}, P={sequence}, C={current}: logits and 16 KV outputs", flush=True)
    (args.out / "cases.txt").write_text("".join(f"{batch} {sequence} {current}\n" for batch, sequence, current in cases))

    # A single compiled decode graph also consumes a multi-token prefill's
    # present K/V and returns two decode rows.  This is the model-level
    # prefill→decode evidence; the C++ test deliberately keeps the K/V arrays
    # external so it cannot be mistaken for persistent state ownership.
    prefill_path = args.prefill or args.onnx.with_name("minimind_prefill.onnx")
    prefill, _ = fold_static_subgraph(onnx.load(prefill_path))
    prefill_table = next(t for t in prefill.graph.initializer if t.name == table.name)
    if prefill_table.SerializeToString() != table.SerializeToString():
        raise ValueError("prefill and decode embedding weights must match")
    prefill_ids = rng.integers(0, 6400, size=(1, 4), dtype=np.int64)
    prefill_outputs = ReferenceEvaluator(prefill).run(None, {"input_ids": prefill_ids})
    (args.out / "multitoken_prefill_ids.bin").write_bytes(prefill_ids.tobytes())
    files.append("multitoken_prefill_ids.bin")
    for output, array in enumerate(prefill_outputs):
        filename = f"multitoken_prefill_ref_{output}.bin"
        (args.out / filename).write_bytes(array.tobytes())
        files.append(filename)
    decode_ids = rng.integers(0, 6400, size=(1, 2), dtype=np.int64)
    decode_feed = {"input_ids": decode_ids}
    for index, value in enumerate(dynamic.graph.input):
        if index:
            decode_feed[value.name] = prefill_outputs[index]
    decode_outputs = evaluator.run(None, decode_feed)
    (args.out / "multitoken_decode_ids.bin").write_bytes(decode_ids.tobytes())
    files.append("multitoken_decode_ids.bin")
    for output, array in enumerate(decode_outputs):
        filename = f"multitoken_decode_ref_{output}.bin"
        (args.out / filename).write_bytes(array.tobytes())
        files.append(filename)
    receipt = {
        "source_onnx": str(args.onnx), "source_sha256": sha256(args.onnx),
        "source_node_count": source_node_count, "folded_node_count": len(model.graph.node),
        "stage": "full_bounded_multitoken_decode", "layers": 8, "input_names": [value.name for value in dynamic.graph.input],
        "output_names": names, "output_shapes": shapes, "operator_counts": dict(counts),
        "relay_nodes": len(imported.function.nodes), "constant_count": len(imported.params),
        "representative": {"B": 1, "P": 4, "C": 2},
        "bounds": {"B": [1, 3], "P": [4, 4], "C": [1, 3]}, "cases": cases,
        "seed": 20260909, "reference": "onnx.reference.ReferenceEvaluator(full dynamic decode)",
        "prefill_source_sha256": sha256(prefill_path), "handoff": {"prefill_tokens": 4, "decode_tokens": 2,
        "state_owner": "external test arrays; no RuntimeSession state mutation"},
        "onnx_version": onnx.__version__, "numpy_version": np.__version__,
        "sha256": {name: sha256(args.out / name) for name in files},
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"actual full decode: {len(imported.function.nodes)} Relay source calls, "
          f"{len(imported.params)} constants -> {args.out}")


if __name__ == "__main__":
    main()
