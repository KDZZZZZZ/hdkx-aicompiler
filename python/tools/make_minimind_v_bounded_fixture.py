#!/usr/bin/env python3
"""Import the MiniMind-V slot export and write the bounded CPU/LLVM fixture.

The vision stage keeps its fixed [1,3,256,256] contract and the explicit
fixed-shape Shape proof. The slot prefill keeps its S-dynamic shape controls
for C++ restricted preparation; only the static visual slot table is folded.
Independent ONNX ReferenceEvaluator runs cross-check every exported stage
against the unmodified upstream references before any binary is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from kxc_onnx import import_onnx_model, save_imported_model  # noqa: E402
from kxc_onnx.fold import fold_fixed_shape_queries, fold_static_subgraph  # noqa: E402
from export_minimind_onnx import sha256_file  # noqa: E402
from minimind_v_slots import (CAPACITY, IMAGE_TOKENS, LAYOUTS, MARKER,  # noqa: E402
                                            SENTINEL, SEQUENCE_MAX, SLOT_IMAGES, SLOT_ROWS,
                                            VOCAB, slot_ids)
from export_minimind_v_joint_onnx import OUTPUTS  # noqa: E402
from make_minimind_v_joint_fixture import compare  # noqa: E402
from make_minimind_vision_fixture import make_reference_model  # noqa: E402

REPRESENTATIVE_S = 8


def slot_prefill_model(folded: onnx.ModelProto) -> onnx.ModelProto:
    """Declare the S-dynamic slot boundary over the folded exporter nodes."""
    inputs = [helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, [1, "S"]),
              helper.make_tensor_value_info("visual_slots", onnx.TensorProto.FLOAT, [SLOT_ROWS, 768])]
    shapes = [[1, "S", VOCAB]] + [[1, "S", 4, 96] for _ in OUTPUTS[1:]]
    outputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
               for name, shape in zip(OUTPUTS, shapes)]
    model = helper.make_model(helper.make_graph(list(folded.graph.node), "minimind_v_slot_prefill",
                                                inputs, outputs, initializer=list(folded.graph.initializer)),
                              opset_imports=list(folded.opset_import), ir_version=folded.ir_version)
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.export / "export_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    profile = metadata["profile"]
    expected_profile = {"format": "minimind_v.slot_prefill.v1", "vocab_size": VOCAB, "image_marker": MARKER,
                        "image_tokens": IMAGE_TOKENS, "slot_images": SLOT_IMAGES, "slot_rows": SLOT_ROWS,
                        "sequence_bounds": [1, SEQUENCE_MAX], "capacity": CAPACITY, "layouts": LAYOUTS}
    if any(profile.get(key) != value for key, value in expected_profile.items()):
        raise ValueError("the fixture must retain the locked MiniMind-V slot profile")
    if metadata["parameter_count"] != 159647232 or metadata["sentinel"] != SENTINEL:
        raise ValueError("the fixture must retain the complete MiniMind-V architecture and sentinel")
    steps = metadata["decode_steps"]
    for name, expected in metadata["reference_files"].items():
        if Path(name).name != name or sha256_file(args.export / name) != expected:
            raise ValueError(f"reference file differs from export receipt: {name}")
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"export_receipt_sha256": sha256_file(metadata_path), "export": metadata, "stages": {}, "cases": []}
    evaluators = {}
    for stage in ["vision", "prefill_slots", "decode_capacity"]:
        raw_path = args.export / f"{stage}.onnx"
        if sha256_file(raw_path) != metadata["stages"][stage]["sha256"]:
            raise ValueError(f"{stage} ONNX differs from export receipt")
        raw = onnx.load(raw_path)
        stage_receipt = {"source_nodes": len(raw.graph.node)}
        if stage == "vision":
            normalized, proof = fold_fixed_shape_queries(raw)
            normalized, folded = fold_static_subgraph(normalized)
            normalized = onnx.shape_inference.infer_shapes(normalized, strict_mode=True)
            imported = import_onnx_model(normalized, fold_constants=False)
            stage_receipt["fixed_shape_proof"] = proof
            reference_model, adaptations = make_reference_model(raw)
            stage_receipt["reference_adaptations"] = adaptations
        elif stage == "prefill_slots":
            if [value.name for value in raw.graph.input] != ["input_ids", "visual_slots"]:
                raise ValueError("slot prefill must consume token ids then visual slots")
            folded_model, folded = fold_static_subgraph(raw)
            reference_model = slot_prefill_model(folded_model)
            representative = onnx.ModelProto()
            representative.CopyFrom(reference_model)
            for value in list(representative.graph.input) + list(representative.graph.output):
                for dim in value.type.tensor_type.shape.dim:
                    if dim.dim_param == "S":
                        dim.ClearField("dim_param")
                        dim.dim_value = REPRESENTATIVE_S
            onnx.checker.check_model(representative)
            imported = import_onnx_model(representative, fold_constants=False, preserve_shape_values=True)
            stage_receipt["representative_S"] = REPRESENTATIVE_S
            onnx.save(reference_model, args.out / "prefill_slots_dynamic.onnx")
        else:
            normalized, folded = fold_static_subgraph(raw)
            normalized = onnx.shape_inference.infer_shapes(normalized, strict_mode=True)
            imported = import_onnx_model(normalized, fold_constants=False)
            reference_model = raw
        save_imported_model(imported, args.out / f"{stage}.json", args.out / f"{stage}.params")
        stage_receipt.update({"fold": asdict(folded), "relay_nodes": len(imported.function.nodes),
                              "relay_ops": dict(sorted(Counter(node.op_name for node in imported.function.nodes).items()))})
        receipt["stages"][stage] = stage_receipt
        evaluators[stage] = ReferenceEvaluator(reference_model)
        del imported, raw
        gc.collect()
        print(f"{stage}: {stage_receipt['relay_nodes']} imported Relay nodes", flush=True)
    for case in range(len(LAYOUTS)):
        root = args.out / f"case{case}"
        root.mkdir(exist_ok=True)
        with np.load(args.export / f"case{case}_prefill.npz") as reference:
            ids, slots_ids = reference["input_ids"], reference["slot_ids"]
            pixels, visual = reference["pixel_values"], reference["visual_tokens"]
            remapped, images = slot_ids(ids)
            np.testing.assert_array_equal(remapped, slots_ids)
            sequence = int(ids.shape[1])
            vision_errors = []
            for index in range(images):
                (tokens,) = evaluators["vision"].run(None, {"pixel_values": pixels[index:index + 1]})
                vision_errors.append(compare(tokens[0], visual[index]))
            slots = np.full((SLOT_ROWS, 768), SENTINEL, dtype=np.float32)
            slots[:images * IMAGE_TOKENS] = visual.reshape(-1, 768)
            values = evaluators["prefill_slots"].run(None, {"input_ids": slots_ids, "visual_slots": slots})
            prefill_errors = {name: compare(value, reference[name]) for name, value in zip(OUTPUTS, values)}
            ids.tofile(root / "input_ids.bin")
            slots_ids.tofile(root / "slot_ids.bin")
            pixels.tofile(root / "pixel_values.bin")
            visual.astype(np.float32).tofile(root / "visual_tokens.bin")
            for output, name in enumerate(OUTPUTS):
                reference[name].tofile(root / f"prefill_ref_{output}.bin")
        cache = {}
        for name, value in zip(OUTPUTS[1:], values[1:]):
            padded = np.full((1, CAPACITY, 4, 96), SENTINEL, dtype=np.float32)
            padded[:, :sequence] = value
            cache[name.replace("present_", "past_", 1)] = padded
        next_token = int(values[0][:, -1].argmax(-1)[0])
        lines, decode_errors = [], []
        for step in range(steps):
            extent = sequence + step
            mask = np.zeros((1, CAPACITY + 1), dtype=np.float32)
            mask[:, :extent] = 1
            mask[:, CAPACITY] = 1
            inputs = {"input_ids": np.asarray([[next_token]], dtype=np.int64),
                      "position": np.asarray([extent], dtype=np.int64), "attention_mask": mask, **cache}
            outputs = evaluators["decode_capacity"].run(None, inputs)
            with np.load(args.export / f"case{case}_step{step}.npz") as reference:
                np.testing.assert_array_equal(inputs["input_ids"], reference["input_ids"])
                errors = {"logits": compare(outputs[0], reference["logits"])}
                reference["logits"].tofile(root / f"ref_step{step}_logits.bin")
                for index, (name, value) in enumerate(zip(OUTPUTS[1:], outputs[1:])):
                    compact = np.concatenate((value[:, :extent], value[:, CAPACITY:CAPACITY + 1]), axis=1)
                    errors[name] = compare(compact, reference[name])
                    target = cache[name.replace("present_", "past_", 1)]
                    target[:, extent:extent + 1] = value[:, CAPACITY:CAPACITY + 1]
                    target.tofile(root / f"ref_step{step}_state_{index}.bin")
            lines.append(f"{step} {next_token} {extent}")
            next_token = int(outputs[0][:, -1].argmax(-1)[0])
            decode_errors.append(errors)
        (root / "steps.txt").write_text("\n".join(lines) + "\n")
        (root / "case.txt").write_text(f"{images} {sequence}\n")
        receipt["cases"].append({"case": case, "images": images, "sequence": sequence,
                                 "vision_onnx_vs_upstream": vision_errors,
                                 "prefill_onnx_vs_upstream": prefill_errors,
                                 "decode_onnx_vs_upstream": decode_errors})
        print(f"case {case}: N={images} S={sequence} ONNX references agree with upstream", flush=True)
    (args.out / "profile.txt").write_text(" ".join(str(value) for value in (
        VOCAB, MARKER, IMAGE_TOKENS, SLOT_IMAGES, SEQUENCE_MAX, CAPACITY, SENTINEL, len(LAYOUTS), steps)) + "\n")
    (args.out / "export_receipt.txt").write_text(
        "sha256:prefill_slots.onnx:" + metadata["stages"]["prefill_slots"]["sha256"] + "\n")
    receipt["artifacts"] = {str(path.relative_to(args.out)): sha256_file(path)
                            for path in sorted(args.out.rglob("*")) if path.is_file() and path.name != "receipt.json"}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({name: value["relay_nodes"] for name, value in receipt["stages"].items()}))


if __name__ == "__main__":
    main()
