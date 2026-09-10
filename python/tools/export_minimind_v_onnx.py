#!/usr/bin/env python3
"""Export the actual MiniMind-V SigLIP2 encoder and upstream vision projector.

Uses the full published architecture with deterministic random float32 weights.
The input is already preprocessed [1,3,256,256]; image decoding/resizing and text
token insertion are outside this graph. No model weights or network access are
required. Source/config revisions and output checksums are recorded for review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from importlib.metadata import version

import numpy as np
import onnx
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from export_minimind_onnx import minimind_commit, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--vision-config", type=Path, required=True)
    parser.add_argument("--config-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config_data = json.loads(args.vision_config.read_text())
    required = dict(hidden_size=768, image_size=256, intermediate_size=3072,
                    num_attention_heads=12, num_channels=3, num_hidden_layers=12,
                    patch_size=32, hidden_act="gelu_pytorch_tanh")
    for key, value in required.items():
        if config_data.get(key) != value:
            raise ValueError(f"MiniMind-V L2 requires {key}={value!r}")
    sys.path.insert(0, str(args.src.resolve()))
    from model.model_vlm import MMVisionProjector

    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = SiglipVisionConfig.from_dict(config_data)
    config._attn_implementation = "eager"

    class VisionChain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SiglipVisionModel(config).float().eval()
            self.projector = MMVisionProjector(768, 768).float().eval()

        def forward(self, pixel_values):
            features = self.encoder(pixel_values=pixel_values).last_hidden_state
            return features, self.projector(features)

    model = VisionChain().eval()
    pixels = torch.rand(1, 3, 256, 256) * 2 - 1
    checkpoints = {}
    encoder = getattr(model.encoder, "vision_model", model.encoder)
    def capture(name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            checkpoints[name] = value.detach().clone().numpy()
        return hook
    hooks = [module.register_forward_hook(capture(name)) for name, module in [
        ("patch_embedding", encoder.embeddings.patch_embedding),
        ("embeddings", encoder.embeddings),
        ("layer0", encoder.encoder.layers[0])]]
    with torch.no_grad():
        references = model(pixels)
    for hook in hooks:
        hook.remove()
    args.out.mkdir(parents=True, exist_ok=True)
    graph_path = args.out / "minimind_v_vision.onnx"
    torch.onnx.export(model, (pixels,), graph_path, input_names=["pixel_values"],
                      output_names=["vision_features", "visual_tokens"],
                      opset_version=17, dynamo=False, do_constant_folding=True)
    graph = onnx.load(graph_path)
    onnx.checker.check_model(graph)
    arrays = {"pixel_values": pixels.numpy(),
              **{name: value.numpy() for name, value in
                 zip(["vision_features", "visual_tokens"], references)}}
    for name, array in arrays.items():
        np.save(args.out / f"{name}.npy", array)
    for name, array in checkpoints.items():
        np.save(args.out / f"checkpoint_{name}.npy", array)
    (args.out / "vision_config.json").write_bytes(args.vision_config.read_bytes())
    metadata = {
        "source_url": "https://github.com/jingyaogong/minimind-v",
        "source_commit": minimind_commit(args.src),
        "source_files": {name: sha256_file(args.src / "model" / name)
                         for name in ["model_vlm.py", "model_minimind.py"]},
        "vision_config_repo": "jingyaogong/siglip2-base-p32-256-ve",
        "vision_config_revision": args.config_revision,
        "vision_config_sha256": sha256_file(args.vision_config),
        "vision_config": config_data,
        "packages": {name: version(name) for name in ["torch", "transformers", "onnx", "numpy"]},
        "weights": "random initialization, seed=0, float32; no quality claim",
        "seed": 0, "opset": 17, "dynamo": False, "attention": "eager",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "projector_parameter_count": sum(p.numel() for p in model.projector.parameters()),
        "graph_sha256": sha256_file(graph_path),
        "onnx_initializers_elements": sum(int(np.prod(t.dims)) for t in graph.graph.initializer),
        "onnx_nodes": dict(sorted(Counter(n.op_type for n in graph.graph.node).items())),
        "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype),
                          "sha256": sha256_file(args.out / f"{name}.npy")}
                   for name, array in arrays.items()},
        "boundary": "preprocessed float32 pixels to 64 vision features and projected visual tokens; no text insertion or decoder",
    }
    (args.out / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in
                      ["source_commit", "parameter_count", "graph_sha256", "onnx_nodes"]}, indent=2))


if __name__ == "__main__":
    main()
