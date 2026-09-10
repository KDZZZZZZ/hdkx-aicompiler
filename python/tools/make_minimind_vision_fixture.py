#!/usr/bin/env python3
"""Build the complete fixed-shape L2 vision import and independent references."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from kxc_onnx import import_onnx_model, save_imported_model
from kxc_onnx.fold import fold_fixed_shape_queries, fold_static_subgraph
from export_minimind_onnx import sha256_file


def make_reference_model(raw: onnx.ModelProto):
    """Use equivalent explicit zero padding for ONNX 1.22's VALID Conv bug."""
    reference_model = onnx.ModelProto()
    reference_model.CopyFrom(raw)
    adaptations = []
    for node in reference_model.graph.node:
        if node.op_type != "Conv":
            continue
        attrs = {a.name: a for a in node.attribute}
        if "auto_pad" in attrs and attrs["auto_pad"].s == b"VALID":
            if "pads" in attrs:
                raise ValueError("VALID reference adapter requires absent explicit pads")
            rank = len(attrs["kernel_shape"].ints)
            attrs["auto_pad"].s = b"NOTSET"
            node.attribute.append(onnx.helper.make_attribute("pads", [0] * (2 * rank)))
            adaptations.append({"node": node.name, "VALID_to_explicit_zero_padding": rank})
    onnx.checker.check_model(reference_model)
    return reference_model, adaptations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.export / "export_metadata.json").read_text())
    raw_path = args.export / "minimind_v_vision.onnx"
    if sha256_file(raw_path) != metadata["graph_sha256"]:
        raise ValueError("vision ONNX hash differs from the export receipt")
    for name in ["pixel_values", "vision_features", "visual_tokens"]:
        if sha256_file(args.export / f"{name}.npy") != metadata["arrays"][name]["sha256"]:
            raise ValueError(f"vision {name} differs from the export receipt")
    raw = onnx.load(raw_path)
    normalized, proof = fold_fixed_shape_queries(raw)
    normalized, folded = fold_static_subgraph(normalized)
    normalized = onnx.shape_inference.infer_shapes(normalized, strict_mode=True)
    imported = import_onnx_model(normalized, fold_constants=False)
    args.out.mkdir(parents=True, exist_ok=True)
    save_imported_model(imported, args.out / "vision.json", args.out / "vision.params")
    onnx.save(normalized, args.out / "vision.onnx")
    # ONNX 1.22's optimized and ordinary Conv references both incorrectly treat
    # VALID as SAME padding. Use the equivalent explicit zero-pad form; retain
    # this separate reference graph and its checksum rather than patching ONNX.
    reference_model, reference_adaptations = make_reference_model(raw)
    onnx.save(reference_model, args.out / "vision_reference.onnx")
    evaluator = ReferenceEvaluator(reference_model)
    outputs = ["vision_features", "visual_tokens"]
    if [v.name for v in raw.graph.output] != outputs:
        raise ValueError("vision outputs must retain encoder and projector order")
    comparisons = {}
    for case, pixels in enumerate([np.load(args.export / "pixel_values.npy"),
                                  np.zeros((1, 3, 256, 256), dtype=np.float32)]):
        pixels.tofile(args.out / f"input_{case}.bin")
        reference = evaluator.run(None, {"pixel_values": pixels})
        for name, value in zip(outputs, reference):
            if value.shape != (1, 64, 768) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"unexpected {name} reference signature or non-finite values")
            value.tofile(args.out / f"ref_{case}_{name}.bin")
            if case == 0:
                original = np.load(args.export / f"{name}.npy")
                np.testing.assert_allclose(value, original, rtol=2e-4, atol=2e-4)
                comparisons[name] = float(np.max(np.abs(value - original)))
    receipt = {"export": metadata, "fixed_shape_proof": proof,
               "reference_adaptations": reference_adaptations,
               "fold": asdict(folded), "onnx_nodes": dict(sorted(Counter(
                   node.op_type for node in normalized.graph.node).items())),
               "torch_onnx_max_abs_error": comparisons,
               "cases": ["seeded normalized pixels", "zero normalized pixels"],
               "artifacts": {p.name: sha256_file(p) for p in sorted(args.out.iterdir())
                             if p.is_file() and p.name != "receipt.json"}}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in
                      ["fixed_shape_proof", "onnx_nodes", "torch_onnx_max_abs_error"]}, indent=2))


if __name__ == "__main__":
    main()
