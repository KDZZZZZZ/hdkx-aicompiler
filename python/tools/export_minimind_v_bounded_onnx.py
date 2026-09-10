#!/usr/bin/env python3
"""Export MiniMind-V with runtime image placement for one bounded prefill artifact.

The fixed joint export bakes the upstream Python marker scan into the graph, so
every image count and text layout needs its own artifact. Here the scan moves to
an explicit host step (`slot_ids`): each run of 64 image markers becomes the
indices ``6400 + 64 * k + [0, 64)`` into a capacity-shaped ``visual_slots``
buffer of three images. The prefill gathers from
``Concat(embed_tokens.weight, visual_slots)``, which has a static
``[6592, 768]`` shape, so only the text length S is a symbolic axis.

Three stages come from one seed-0 model: a fixed per-image vision chain, the
S-dynamic slot prefill and a capacity decode. References come from the
unmodified upstream MiniMindVLM over four image counts and layouts. Weights are
random float32; this is a compiler acceptance fixture, not a quality claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import warnings

import numpy as np
import onnx
import torch
import torch.nn.functional as F

from export_minimind_onnx import minimind_commit, sha256_file
from export_minimind_v_joint_onnx import (OUTPUTS, SOURCE_FILES, JointCapacityDecode, _compare,
                                          _outputs, build_vlm_pair, load_locked_inputs)
from importlib.metadata import version
from minimind_v_slots import (CAPACITY, IMAGE_TOKENS, LAYOUTS, MARKER, SENTINEL, SEQUENCE_MAX,
                              SLOT_IMAGES, SLOT_ROWS, VOCAB, layout_ids, slot_ids)


class SlotEmbedding(torch.nn.Module):
    """Look token rows up in the text table extended by the visual slots."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.slots = None

    def forward(self, ids):
        return F.embedding(ids, torch.cat((self.base.weight, self.slots), dim=0))


class SlotPrefill(torch.nn.Module):
    def __init__(self, model, language_forward):
        super().__init__()
        self.model = model
        self.language_forward = language_forward
        self.embedding = SlotEmbedding(model.model.embed_tokens)

    def forward(self, input_ids, visual_slots):
        original = self.model.model.embed_tokens
        self.embedding.slots = visual_slots
        self.model.model.embed_tokens = self.embedding
        try:
            return _outputs(self.language_forward(self.model, input_ids, use_cache=True))
        finally:
            self.model.model.embed_tokens = original
            self.embedding.slots = None


class VisionStage(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        features = self.model.vision_encoder(pixel_values=pixel_values).last_hidden_state
        return self.model.vision_proj(features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="clean MiniMind-V source checkout")
    parser.add_argument("--vision-config", type=Path, required=True)
    parser.add_argument("--config-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if not 3 <= args.steps <= CAPACITY - SEQUENCE_MAX:
        raise ValueError("the capacity must hold the longest prefill and every decode step")
    config_data = load_locked_inputs(args.src, args.vision_config)
    args.out.mkdir(parents=True, exist_ok=True)
    profile = {"format": "minimind_v.slot_prefill.v1", "batch": 1, "vocab_size": VOCAB,
               "image_marker": MARKER, "image_tokens": IMAGE_TOKENS, "slot_images": SLOT_IMAGES,
               "slot_rows": SLOT_ROWS, "sequence_bounds": [1, SEQUENCE_MAX], "capacity": CAPACITY,
               "layouts": LAYOUTS}
    metadata = {"source_url": "https://github.com/jingyaogong/minimind-v",
                "source_commit": minimind_commit(args.src),
                "source_files": {name: sha256_file(args.src / "model" / name) for name in SOURCE_FILES},
                "vision_config_revision": args.config_revision,
                "vision_config_sha256": sha256_file(args.vision_config), "vision_config": config_data,
                "packages": {name: version(name) for name in ["torch", "transformers", "onnx", "numpy"]},
                "profile": profile, "seed": 0, "opset": 17, "dynamo": False,
                "weights": "full architecture, random float32 seed=0; no pretrained quality claim",
                "decode_steps": args.steps, "sentinel": SENTINEL, "patches": {}, "cases": []}
    with tempfile.TemporaryDirectory(prefix="kxc_minimind_v_bounded_") as temporary:
        model, upstream, adapted_module = build_vlm_pair(args.src, config_data, Path(temporary), metadata)
        metadata["parameter_count"] = sum(value.numel() for value in model.parameters())
        language_forward = adapted_module.MiniMindForCausalLM.forward
        vision = VisionStage(model).eval()
        prefill = SlotPrefill(model, language_forward).eval()
        decode = JointCapacityDecode(model, language_forward).eval()
        rng = np.random.default_rng(20260910)
        samples = {}
        for case, layout in enumerate(LAYOUTS):
            ids = layout_ids(rng, layout)
            slots_ids, images = slot_ids(ids)
            if images != len(layout) - 1:
                raise ValueError("layout generator and host slot scan disagree")
            pixels = rng.uniform(-1, 1, (images, 3, 256, 256)).astype(np.float32)
            tokens = torch.from_numpy(ids)
            captured = {}
            hook = upstream.vision_proj.register_forward_hook(
                lambda module, inputs, output: captured.update(visual=output.detach().clone()))
            with torch.no_grad():
                if images == 0:
                    exact = upstream(tokens, use_cache=True)
                else:
                    image = torch.from_numpy(pixels if images > 1 else pixels[:1]).unsqueeze(0)
                    exact = upstream(tokens, pixel_values={"pixel_values": image if images > 1 else image[0]},
                                     use_cache=True)
            hook.remove()
            visual = (captured["visual"].reshape(images, IMAGE_TOKENS, -1) if images
                      else torch.zeros((0, IMAGE_TOKENS, 768)))
            slots = torch.full((SLOT_ROWS, 768), SENTINEL)
            slots[:images * IMAGE_TOKENS] = visual.reshape(-1, 768)
            vision_worst = 0.0
            with torch.no_grad():
                # Upstream batches all images through one encoder call; the
                # per-image stage may round differently, never beyond 1e-5.
                for index in range(images):
                    per_image = vision(torch.from_numpy(pixels[index:index + 1]))
                    torch.testing.assert_close(per_image[0], visual[index], rtol=1e-5, atol=1e-5)
                    vision_worst = max(vision_worst, (per_image[0] - visual[index]).abs().max().item())
                adapted = prefill(torch.from_numpy(slots_ids), slots)
            exact_values = _outputs(exact)
            comparisons = {"images": images, "sequence": int(ids.shape[1]),
                           "per_image_vision_vs_upstream_batch": vision_worst,
                           "prefill_upstream_vs_slot_adapter": _compare(adapted, exact_values), "decode": []}
            np.savez(args.out / f"case{case}_prefill.npz", input_ids=ids, slot_ids=slots_ids,
                     pixel_values=pixels, visual_tokens=visual.numpy(),
                     **{name: value.detach().numpy() for name, value in zip(OUTPUTS, exact_values)})
            sequence = int(ids.shape[1])
            cache = [torch.full((1, CAPACITY, 4, 96), SENTINEL) for _ in range(16)]
            for target, source in zip(cache, exact_values[1:]):
                target[:, :sequence] = source
            for step in range(args.steps):
                extent = sequence + step
                token = exact.logits[:, -1].argmax(-1).reshape(1, 1)
                position = torch.tensor([extent], dtype=torch.int64)
                mask = torch.zeros((1, CAPACITY + 1))
                mask[:, :extent] = 1
                mask[:, CAPACITY] = 1
                with torch.no_grad():
                    capacity_values = decode(token, position, mask, *cache)
                    exact = upstream(token, past_key_values=exact.past_key_values, use_cache=True)
                compact = (capacity_values[0], *(torch.cat((value[:, :extent], value[:, CAPACITY:CAPACITY + 1]), dim=1)
                                                 for value in capacity_values[1:]))
                comparisons["decode"].append(_compare(compact, _outputs(exact)))
                np.savez(args.out / f"case{case}_step{step}.npz", input_ids=token.numpy(),
                         position=position.numpy(),
                         **{name: value.detach().numpy() for name, value in zip(OUTPUTS, _outputs(exact))})
                if step == 0 and "decode" not in samples:
                    samples["decode"] = (token, position, mask, *(value.clone() for value in cache))
                for target, value in zip(cache, capacity_values[1:]):
                    target[:, extent:extent + 1] = value[:, CAPACITY:CAPACITY + 1]
            if images and "vision" not in samples:
                samples["vision"] = (torch.from_numpy(pixels[:1]),)
            if images == SLOT_IMAGES:
                samples["prefill"] = (torch.from_numpy(slots_ids), slots)
            metadata["cases"].append(comparisons)
            print(f"case {case}: N={images} S={sequence} upstream/adapter prefill and "
                  f"{args.steps} decode steps checked", flush=True)
        dynamic = {"input_ids": {1: "S"}, **{name: {1: "S"} for name in OUTPUTS}}
        stages = [("vision", vision, samples["vision"], ["pixel_values"], ["visual_tokens"], None),
                  ("prefill_slots", prefill, samples["prefill"], ["input_ids", "visual_slots"], OUTPUTS, dynamic),
                  ("decode_capacity", decode, samples["decode"], ["input_ids", "position", "attention_mask"] +
                   [f"past_{kind}_{layer}" for layer in range(8) for kind in ("k", "v")], OUTPUTS, None)]
        metadata["stages"] = {}
        for stage, wrapper, inputs, input_names, output_names, axes in stages:
            path = args.out / f"{stage}.onnx"
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                torch.onnx.export(wrapper, inputs, path, input_names=input_names, output_names=output_names,
                                  dynamic_axes=axes, opset_version=17, dynamo=False, do_constant_folding=True)
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            metadata["stages"][stage] = {"sha256": sha256_file(path),
                "nodes": dict(sorted(Counter(node.op_type for node in graph.graph.node).items())),
                "warnings": sorted(set(str(warning.message) for warning in captured_warnings))}
            del graph
            print(f"exported {stage}", flush=True)
    reference_names = [f"case{case}_{stage}.npz" for case in range(len(LAYOUTS))
                       for stage in ["prefill", *(f"step{step}" for step in range(args.steps))]]
    metadata["reference_files"] = {name: sha256_file(args.out / name) for name in reference_names}
    (args.out / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"parameter_count": metadata["parameter_count"], "cases": metadata["cases"],
                      "stages": {name: value["sha256"] for name, value in metadata["stages"].items()}}, indent=2))


if __name__ == "__main__":
    main()
