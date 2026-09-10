#!/usr/bin/env python3
"""Export full MiniMind-V prefill and capacity decode with upstream references.

The explicit profile is a fixed one-or-more-image layout. Each image contributes
64 marker tokens and two separator tokens separate adjacent images. TorchScript
specializes the upstream Python marker scan to the selected profile. Every
consumer must validate that layout before invoking the compiled prefill graph.
Weights are deterministic random float32 values; this is a compiler acceptance
fixture, not a pretrained model or a language-quality evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import onnx
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from export_minimind_onnx import minimind_commit, sha256_file


def profile_for_image_count(image_count: int) -> dict:
    """Return a fixed marker/pixel profile for one or more images."""
    if image_count < 1:
        raise ValueError("image_count must be positive")
    image_tokens = 64
    prefix = 2
    separator = 2
    starts = [prefix + index * (image_tokens + separator)
              for index in range(image_count)]
    sequence_length = prefix + image_count * image_tokens + (image_count - 1) * separator + 2
    capacity = sequence_length + 12
    if capacity > 2048:
        raise ValueError("the image profile and decode capacity must fit the locked 2048 positions")
    return {
        "format": "minimind_v.fixed_multi_image.v1" if image_count > 1
                  else "minimind_v.fixed_single_image.v1",
        "batch": 1,
        "sequence_length": sequence_length,
        "image_start": starts[0],
        "image_starts": starts,
        "image_tokens": image_tokens,
        "image_count": image_count,
        "image_marker": 12,
        "vocab_size": 6400,
        "capacity": capacity,
        "pixel_shape": ([1, 3, 256, 256] if image_count == 1
                        else [1, image_count, 3, 256, 256]),
    }


PROFILE = profile_for_image_count(1)
OUTPUTS = ["logits"] + [f"present_{kind}_{layer}" for layer in range(8) for kind in ("k", "v")]


def validate_image_inputs(ids: np.ndarray, pixels: np.ndarray,
                          profile: dict | None = None) -> None:
    profile = PROFILE if profile is None else profile
    sequence_length = int(profile["sequence_length"])
    image_count = int(profile["image_count"])
    expected_pixels = tuple(profile["pixel_shape"])
    if ids.dtype != np.int64 or ids.shape != (1, sequence_length):
        raise ValueError(f"VLM input_ids require int64 [1,{sequence_length}]")
    if (pixels.dtype != np.float32 or pixels.shape != expected_pixels or
            not np.isfinite(pixels).all()):
        raise ValueError(f"VLM pixels require finite float32 {list(expected_pixels)}")
    markers = np.zeros((1, sequence_length), dtype=bool)
    starts = profile.get("image_starts", [profile["image_start"]])
    for start in starts:
        markers[:, int(start):int(start) + int(profile["image_tokens"])] = True
    if (ids < 0).any() or (ids >= int(profile["vocab_size"])).any() or not np.array_equal(
            ids == int(profile["image_marker"]), markers):
        raise ValueError("VLM tokens violate the fixed image-slot profile")
    if len(starts) != image_count:
        raise ValueError("VLM image marker count does not match the pixel profile")


def _module(source: Path, name: str):
    directory = source / "model"
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
                                                submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(name + ".model_vlm")


def _outputs(result):
    return (result.logits, *(value for pair in result.past_key_values for value in pair))


def _arrays(values):
    return {name: value.detach().numpy() for name, value in zip(OUTPUTS, values)}


def _compare(actual, expected) -> float:
    if len(actual) != len(expected):
        raise ValueError("VLM comparison must include the same complete output list")
    worst = 0.0
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-4)
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError("VLM reference contains non-finite values")
        worst = max(worst, (left - right).abs().max().item())
    return worst


class JointPrefill(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, pixel_values):
        return _outputs(self.model(input_ids, pixel_values={"pixel_values": pixel_values}, use_cache=True))


class JointCapacityDecode(torch.nn.Module):
    def __init__(self, model, language_forward):
        super().__init__()
        self.model = model
        self.language_forward = language_forward

    def forward(self, input_ids, position, attention_mask, *flat_past):
        # At nonzero history, upstream MiniMind-V uses only this same backbone.
        # The existing explicit capacity adapter obtains RoPE positions from the
        # runtime extent; it does not use the capacity as the history length.
        return _outputs(self.language_forward(
            self.model, input_ids, attention_mask=attention_mask,
            past_key_values=list(zip(flat_past[::2], flat_past[1::2])),
            use_cache=True, position=position))


SOURCE_FILES = ["__init__.py", "model_minimind.py", "model_vlm.py"]


def load_locked_inputs(src: Path, vision_config: Path) -> dict:
    """Require the clean upstream checkout and the full SigLIP2 configuration."""
    top = subprocess.check_output(["git", "-C", str(src), "rev-parse", "--show-toplevel"], text=True).strip()
    if Path(top).resolve() != src.resolve():
        raise ValueError("--src must be the original MiniMind-V checkout root")
    subprocess.run(["git", "-C", str(src), "diff", "--exit-code", "HEAD", "--", "model"],
                   check=True, stdout=subprocess.DEVNULL)
    config_data = json.loads(vision_config.read_text())
    for key, value in dict(hidden_size=768, image_size=256, intermediate_size=3072,
                           num_attention_heads=12, num_channels=3, num_hidden_layers=12,
                           patch_size=32, hidden_act="gelu_pytorch_tanh").items():
        if config_data.get(key) != value:
            raise ValueError(f"full MiniMind-V vision requires {key}={value!r}")
    return config_data


def build_vlm_pair(src: Path, config_data: dict, adapted: Path, metadata: dict):
    """Build the patched export model and the unmodified upstream twin.

    Both share the same seed-0 weights; ``metadata`` records the patches.
    """
    shutil.copytree(src / "model", adapted / "model", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ["minimind_noninplace_mask.patch", "minimind_capacity_kv.patch"]:
        patch = Path(__file__).parent / name
        subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch.resolve())],
                       cwd=adapted, check=True)
        metadata["patches"][name] = sha256_file(patch)
    metadata["adapted_source_files"] = {name: sha256_file(adapted / "model" / name) for name in SOURCE_FILES}
    upstream_module = _module(src.resolve(), "_kxc_vlm_upstream")
    adapted_module = _module(adapted, "_kxc_vlm_export")
    config_kwargs = dict(hidden_size=768, num_hidden_layers=8, use_moe=False,
                         flash_attn=False, dropout=0.0, max_position_embeddings=2048)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = adapted_module.MiniMindVLM(adapted_module.VLMConfig(**config_kwargs),
               vision_model_path=str(adapted / "no_pretrained_weights")).float().eval()
    vision_config = SiglipVisionConfig.from_dict(config_data)
    vision_config._attn_implementation = "eager"
    model.vision_encoder = SiglipVisionModel(vision_config).float().eval()
    upstream = upstream_module.MiniMindVLM(upstream_module.VLMConfig(**config_kwargs),
                   vision_model_path=str(adapted / "no_pretrained_weights")).float().eval()
    upstream.vision_encoder = model.vision_encoder
    upstream.load_state_dict(model.state_dict(), strict=True)
    return model, upstream, adapted_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="clean MiniMind-V source checkout")
    parser.add_argument("--vision-config", type=Path, required=True)
    parser.add_argument("--config-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--image-count", type=int, default=1,
                        help="fixed number of image slots in the text profile")
    args = parser.parse_args()
    profile = profile_for_image_count(args.image_count)
    if not 3 <= args.steps <= 12:
        raise ValueError(f"the fixed capacity must hold the {profile['sequence_length']}-token prefill and 3..12 decode steps")
    config_data = load_locked_inputs(args.src, args.vision_config)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {"source_url": "https://github.com/jingyaogong/minimind-v",
                "source_commit": minimind_commit(args.src),
                "source_files": {name: sha256_file(args.src / "model" / name) for name in SOURCE_FILES},
                "vision_config_revision": args.config_revision,
                "vision_config_sha256": sha256_file(args.vision_config), "vision_config": config_data,
                "packages": {name: version(name) for name in ["torch", "transformers", "onnx", "numpy"]},
                "profile": profile, "seed": 0, "opset": 17, "dynamo": False,
                "weights": "full architecture, random float32 seed=0; no pretrained quality claim",
                "decode_steps": args.steps, "sentinel": 7.0, "patches": {}, "cases": []}
    with tempfile.TemporaryDirectory(prefix="kxc_minimind_v_export_") as temporary:
        adapted = Path(temporary)
        model, upstream, adapted_module = build_vlm_pair(args.src, config_data, adapted, metadata)
        metadata["parameter_count"] = sum(value.numel() for value in model.parameters())
        metadata["language_config"] = {name: getattr(model.config, name) for name in
            ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
             "head_dim", "intermediate_size", "vocab_size", "rms_norm_eps", "max_position_embeddings"]}
        prefill = JointPrefill(model).eval()
        decode = JointCapacityDecode(model, adapted_module.MiniMindForCausalLM.forward).eval()
        rng = np.random.default_rng(0)
        ids = rng.integers(13, profile["vocab_size"],
                           (1, profile["sequence_length"]), dtype=np.int64)
        for start in profile["image_starts"]:
            ids[:, start:start + profile["image_tokens"]] = profile["image_marker"]
        pixels = rng.uniform(-1, 1, tuple(profile["pixel_shape"])).astype(np.float32)
        changed_ids = ids.copy()
        changed_ids[:, :profile["image_start"]] = rng.integers(
            13, profile["vocab_size"], (1, profile["image_start"]), dtype=np.int64)
        changed_ids[:, profile["image_starts"][-1] + profile["image_tokens"]:] = rng.integers(
            13, profile["vocab_size"],
            (1, profile["sequence_length"] - profile["image_starts"][-1] - profile["image_tokens"]),
            dtype=np.int64)
        cases = [(ids, pixels), (ids, np.zeros_like(pixels)), (changed_ids, pixels)]
        saved_logits = []
        for case, (case_ids, case_pixels) in enumerate(cases):
            validate_image_inputs(case_ids, case_pixels, profile)
            tokens, image = torch.from_numpy(case_ids), torch.from_numpy(case_pixels)
            checkpoints = {}
            hooks = [upstream.vision_proj.register_forward_hook(
                lambda module, inputs, output: checkpoints.update(visual_tokens=output.detach().clone())),
                upstream.model.layers[0].register_forward_pre_hook(
                lambda module, inputs: checkpoints.update(mixed_embeddings=inputs[0].detach().clone()))]
            with torch.no_grad():
                exact = upstream(tokens, pixel_values={"pixel_values": image}, use_cache=True)
                adapted_prefill = prefill(tokens, image)
            for hook in hooks:
                hook.remove()
            exact_values = _outputs(exact)
            comparisons = {"prefill_upstream_vs_adapter": _compare(adapted_prefill, exact_values), "decode": []}
            embedding = upstream.model.embed_tokens(tokens).detach()
            mixed = checkpoints["mixed_embeddings"]
            visual_tokens = checkpoints["visual_tokens"].view(
                profile["batch"], profile["image_count"], profile["image_tokens"], -1)
            cursor = 0
            for image_index, start in enumerate(profile["image_starts"]):
                torch.testing.assert_close(mixed[:, cursor:start], embedding[:, cursor:start],
                                           rtol=0, atol=0)
                end = start + profile["image_tokens"]
                torch.testing.assert_close(mixed[:, start:end],
                                           visual_tokens[:, image_index],
                                           rtol=0, atol=0)
                cursor = end
            torch.testing.assert_close(mixed[:, cursor:], embedding[:, cursor:], rtol=0, atol=0)
            np.savez(args.out / f"case{case}_prefill.npz", input_ids=case_ids, pixel_values=case_pixels,
                     **_arrays(exact_values), **{name: value.numpy() for name, value in checkpoints.items()})
            saved_logits.append(exact.logits.detach().numpy())
            cache = [torch.full((1, profile["capacity"], 4, 96), 7.0) for _ in range(16)]
            for target, source in zip(cache, exact_values[1:]):
                target[:, :profile["sequence_length"]] = source
            for step in range(args.steps):
                extent = profile["sequence_length"] + step
                token = exact.logits[:, -1].argmax(-1).reshape(1, 1)
                position = torch.tensor([extent], dtype=torch.int64)
                mask = torch.zeros((1, profile["capacity"] + 1))
                mask[:, :extent] = 1
                mask[:, profile["capacity"]] = 1
                with torch.no_grad():
                    capacity_values = decode(token, position, mask, *cache)
                    exact = upstream(token, past_key_values=exact.past_key_values, use_cache=True)
                compact = (capacity_values[0], *(torch.cat((value[:, :extent], value[:, profile["capacity"]:profile["capacity"] + 1]), dim=1)
                            for value in capacity_values[1:]))
                comparisons["decode"].append(_compare(compact, _outputs(exact)))
                np.savez(args.out / f"case{case}_step{step}.npz", input_ids=token.numpy(),
                         position=position.numpy(), **_arrays(_outputs(exact)))
                if case == 0 and step == 0:
                    decode_args = (token, position, mask, *(value.clone() for value in cache))
                for target, value in zip(cache, capacity_values[1:]):
                    target[:, extent:extent + 1] = value[:, profile["capacity"]:profile["capacity"] + 1]
            metadata["cases"].append(comparisons)
            print(f"case {case}: upstream/adapter prefill and {args.steps} decode steps checked", flush=True)
        metadata["image_effect_max_abs"] = float(np.max(np.abs(saved_logits[0] - saved_logits[1])))
        metadata["text_effect_max_abs"] = float(np.max(np.abs(saved_logits[0] - saved_logits[2])))
        if min(metadata["image_effect_max_abs"], metadata["text_effect_max_abs"]) <= 0:
            raise ValueError("both image and text must affect full-model logits")
        stages = [
            ("prefill", prefill, (torch.from_numpy(ids), torch.from_numpy(pixels)), ["input_ids", "pixel_values"]),
            ("decode_capacity", decode, decode_args, ["input_ids", "position", "attention_mask"] +
             [f"past_{kind}_{layer}" for layer in range(8) for kind in ("k", "v")])]
        metadata["stages"] = {}
        for stage, wrapper, inputs, input_names in stages:
            path = args.out / f"{stage}.onnx"
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                torch.onnx.export(wrapper, inputs, path, input_names=input_names, output_names=OUTPUTS,
                                  opset_version=17, dynamo=False, do_constant_folding=True)
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            metadata["stages"][stage] = {"sha256": sha256_file(path),
                "nodes": dict(sorted(Counter(node.op_type for node in graph.graph.node).items())),
                "warnings": sorted(set(str(warning.message) for warning in captured))}
            del graph
            print(f"exported complete {stage}", flush=True)
    reference_names = [f"case{case}_{stage}.npz" for case in range(3)
                       for stage in ["prefill", *(f"step{step}" for step in range(args.steps))]]
    metadata["reference_files"] = {name: sha256_file(args.out / name) for name in reference_names}
    (args.out / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({name: metadata[name] for name in
        ["parameter_count", "image_effect_max_abs", "text_effect_max_abs", "stages"]}, indent=2))


if __name__ == "__main__":
    main()
