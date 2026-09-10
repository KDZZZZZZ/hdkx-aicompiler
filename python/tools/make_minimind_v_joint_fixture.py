#!/usr/bin/env python3
"""Import full MiniMind-V and verify independent ONNX prefill/decode references."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from kxc_onnx import import_onnx_model, save_imported_model
from kxc_onnx.fold import fold_fixed_shape_queries, fold_static_subgraph
from export_minimind_onnx import sha256_file
from export_minimind_v_joint_onnx import (OUTPUTS, profile_for_image_count,
                                          validate_image_inputs)
from make_minimind_vision_fixture import make_reference_model


def compare(actual, expected) -> float:
    if actual.shape != expected.shape:
        raise ValueError("joint-model reference shapes must match exactly")
    if actual.dtype != np.float32 or expected.dtype != np.float32:
        raise ValueError("joint-model references must retain float32")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("joint-model reference has non-finite values")
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    return float(np.max(np.abs(actual - expected)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.export / "export_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    raw_profile = metadata.get("profile")
    if not isinstance(raw_profile, dict):
        raise ValueError("the export must declare a fixed image profile")
    image_count = int(raw_profile.get("image_count", 1))
    profile = profile_for_image_count(image_count)
    # Accept the original one-image receipt (which predates image_count and
    # image_starts) while requiring every newer profile to be canonical.
    for key, expected in profile.items():
        if image_count == 1 and key in ("image_count", "image_starts") and key not in raw_profile:
            continue
        if raw_profile.get(key) != expected:
            raise ValueError("the fixture must retain the complete, locked MiniMind-V profile")
    if metadata["parameter_count"] != 159647232:
        raise ValueError("the fixture must retain the complete, locked MiniMind-V profile")
    steps = metadata["decode_steps"]
    if not isinstance(steps, int) or not 3 <= steps <= 12:
        raise ValueError("decode steps must fit the fixed capacity")
    expected_references = {f"case{case}_{stage}.npz" for case in range(3)
                           for stage in ["prefill", *(f"step{step}" for step in range(steps))]}
    if set(metadata["reference_files"]) != expected_references:
        raise ValueError("export receipt must cover every prefill and decode reference file")
    for name, expected in metadata["reference_files"].items():
        if Path(name).name != name or sha256_file(args.export / name) != expected:
            raise ValueError(f"reference file differs from export receipt: {name}")
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"export_receipt_sha256": sha256_file(metadata_path), "export": metadata,
               "stages": {}, "cases": []}
    common = []
    for stage in ["prefill", "decode_capacity"]:
        raw_path = args.export / f"{stage}.onnx"
        if sha256_file(raw_path) != metadata["stages"][stage]["sha256"]:
            raise ValueError(f"{stage} ONNX differs from export receipt")
        raw = onnx.load(raw_path)
        if [value.name for value in raw.graph.output] != OUTPUTS:
            raise ValueError(f"{stage} must return logits and all 16 K/V outputs in order")
        normalized, proof = fold_fixed_shape_queries(raw) if stage == "prefill" else (raw, [])
        normalized, folded = fold_static_subgraph(normalized)
        normalized = onnx.shape_inference.infer_shapes(normalized, strict_mode=True)
        imported = import_onnx_model(normalized, fold_constants=False)
        save_imported_model(imported, args.out / f"{stage}.json", args.out / f"{stage}.params")
        common.extend([f"{stage}.json", f"{stage}.params"])
        stage_receipt = {"fixed_shape_proof": proof, "fold": asdict(folded),
                         "relay_nodes": len(imported.function.nodes),
                         "relay_ops": dict(sorted(Counter(node.op_name for node in imported.function.nodes).items()))}
        del imported, normalized
        reference_model, adaptations = make_reference_model(raw)
        onnx.save(reference_model, args.out / f"{stage}_reference.onnx")
        stage_receipt["reference_adaptations"] = adaptations
        receipt["stages"][stage] = stage_receipt
        del reference_model, raw
        gc.collect()
        print(f"{stage}: {stage_receipt['relay_nodes']} imported Relay nodes", flush=True)
    prefill = ReferenceEvaluator(str(args.out / "prefill_reference.onnx"))
    decode = ReferenceEvaluator(str(args.out / "decode_capacity_reference.onnx"))
    capacity = profile["capacity"]
    sequence_length = profile["sequence_length"]
    sentinel = float(metadata["sentinel"])
    if not np.isfinite(sentinel) or abs(sentinel) > 1e6:
        raise ValueError("the additive mask requires a finite bounded sentinel")
    for case in range(3):
        root = args.out / f"case{case}"
        root.mkdir(exist_ok=True)
        for name in common:
            destination = root / name
            if destination.exists():
                if destination.samefile(args.out / name):
                    continue
                destination.unlink()
            os.link(args.out / name, destination)
        for name, value in {"graph.txt": "decode_capacity", "capacity.txt": str(capacity),
                            "layout.txt": "1 4 96 8", "sentinel.txt": str(sentinel),
                            "sampling.txt": "greedy_argmax",
                            "seed_extent.txt": str(profile["sequence_length"]),
                            "image_profile.txt": " ".join(str(profile[key]) for key in
                                ("image_marker", "image_start", "image_tokens", "image_count",
                                 "vocab_size", "sequence_length")),
                            "prefill_export_receipt.txt": "sha256:prefill.onnx:" + metadata["stages"]["prefill"]["sha256"],
                            "export_receipt.txt": "sha256:decode_capacity.onnx:" + metadata["stages"]["decode_capacity"]["sha256"]}.items():
            (root / name).write_text(value + "\n")
        with np.load(args.export / f"case{case}_prefill.npz") as torch_reference:
            ids, pixels = torch_reference["input_ids"], torch_reference["pixel_values"]
            validate_image_inputs(ids, pixels, profile)
            ids.tofile(root / "prefill_input_ids.bin")
            pixels.tofile(root / "pixel_values.bin")
            values = prefill.run(None, {"input_ids": ids, "pixel_values": pixels})
            differences = {name: compare(value, torch_reference[name]) for name, value in zip(OUTPUTS, values)}
        values[0].tofile(root / "prefill_reference_logits.bin")
        np.ascontiguousarray(values[0][:, -1]).tofile(root / "prefill_logits.bin")
        cache = {}
        for name, value in zip(OUTPUTS[1:], values[1:]):
            if value.shape != (1, sequence_length, 4, 96):
                raise ValueError(f"invalid full-model prefill cache shape for {name}")
            value.tofile(root / f"seed_{name}.bin")
            padded = np.full((1, capacity, 4, 96), sentinel, dtype=np.float32)
            padded[:, :sequence_length] = value
            cache[name.replace("present_", "past_", 1)] = padded
        next_token = int(values[0][:, -1].argmax(-1)[0])
        lines, decode_errors = [], []
        for step in range(steps):
            extent = sequence_length + step
            mask = np.zeros((1, capacity + 1), dtype=np.float32)
            mask[:, :extent] = 1
            mask[:, capacity] = 1
            inputs = {"input_ids": np.asarray([[next_token]], dtype=np.int64),
                      "position": np.asarray([extent], dtype=np.int64), "attention_mask": mask, **cache}
            outputs = decode.run(None, inputs)
            with np.load(args.export / f"case{case}_step{step}.npz") as torch_reference:
                np.testing.assert_array_equal(inputs["input_ids"], torch_reference["input_ids"])
                np.testing.assert_array_equal(inputs["position"], torch_reference["position"])
                errors = {"logits": compare(outputs[0], torch_reference["logits"])}
                for index, (name, value) in enumerate(zip(OUTPUTS[1:], outputs[1:])):
                    compact = np.concatenate((value[:, :extent], value[:, capacity:capacity + 1]), axis=1)
                    errors[name] = compare(compact, torch_reference[name])
                    target = cache[name.replace("present_", "past_", 1)]
                    target[:, extent:extent + 1] = value[:, capacity:capacity + 1]
                    target.tofile(root / f"ref_step{step}_state_{index}.bin")
            outputs[0].tofile(root / f"ref_step{step}_logits.bin")
            lines.append(f"{step} {next_token} {extent}")
            next_token = int(outputs[0][:, -1].argmax(-1)[0])
            decode_errors.append(errors)
        (root / "steps.txt").write_text("\n".join(lines) + "\n")
        receipt["cases"].append({"case": case, "prefill_onnx_vs_upstream": differences,
                                 "decode_onnx_vs_upstream": decode_errors})
        print(f"case {case}: independent ONNX/upstream references agree for prefill and {steps} decode steps", flush=True)
    receipt["artifacts"] = {str(path.relative_to(args.out)): sha256_file(path)
                            for path in sorted(args.out.rglob("*")) if path.is_file() and path.name != "receipt.json"}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
