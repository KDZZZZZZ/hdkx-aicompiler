#!/usr/bin/env python3
"""Extract actual MiniMind RMSNorm/QKV and generate variable-shape ONNX references.

The importer receives an explicit concrete representative. The C++ production
test admits its B/S axes through RestrictedSymbolicShapeAdapter; this fixture
does not claim that the static ONNX importer accepts a whole dynamic model.
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    stage_options = parser.add_mutually_exclusive_group()
    stage_options.add_argument("--heads", action="store_true",
                        help="retain actual head reshape controls and Q/K normalization")
    stage_options.add_argument("--gqa", action="store_true",
                        help="extract the actual K/V repeat-interleave shape-control subgraphs")
    stage_options.add_argument("--rope", action="store_true",
                        help="include head normalization and actual rotary Q/K transforms")
    stage_options.add_argument("--attention", action="store_true",
                        help="include actual causal self-attention and output projection")
    args = parser.parse_args()
    rotary = args.rope or args.attention
    model, _ = fold_static_subgraph(onnx.load(args.onnx))
    nodes = list(model.graph.node)

    def unique(suffix: str):
        matches = [node for node in nodes if node.name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected one actual export node ending in {suffix!r}, found {len(matches)}")
        return matches[0]

    prefix = f"/layers.{args.layer}/"
    cast = unique(prefix + "input_layernorm/Cast")
    projections = [unique(prefix + f"self_attn/{name}_proj/MatMul") for name in "qkv"]
    output_nodes = ([unique(prefix + "self_attn/" + name) for name in
                     ["q_norm/Mul_1", "k_norm/Mul_1", "Reshape_2"]]
                    if args.heads or rotary else projections)
    if args.gqa:
        output_nodes = [unique(prefix + "self_attn/" + name) for name in ["Reshape_4", "Reshape_6"]]
    if rotary:
        output_nodes = [unique(prefix + "self_attn/" + name) for name in ["Cast_2", "Cast_5", "Reshape_2"]]
    if args.attention:
        output_nodes = [unique(prefix + "self_attn/" + name) for name in
                        ["o_proj/MatMul", "Cast_5", "Reshape_2"]]
    stem = "attention" if args.attention else ("rope" if args.rope else ("gqa" if args.gqa else ("heads" if args.heads else "projection")))
    activation = cast.input[0]
    input_names = [f"present_k_{args.layer}", f"present_v_{args.layer}"] if args.gqa else [activation]
    input_shape = ["B", "S", 4, 96] if args.gqa else ["B", "S", 768]
    input_shapes = [input_shape for _ in input_names]
    if rotary:
        input_names += ["/model/model/Slice_output_0", "/model/model/Slice_1_output_0"]
        input_shapes += [["S", 96], ["S", 96]]
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    weights = [initializers[node.input[1]] for node in projections]
    if [list(weight.dims) for weight in weights] != [[768, 768], [768, 384], [768, 384]]:
        raise ValueError("fixture requires actual MiniMind hidden=768 and KV width=384 weights")
    producers = {name: index for index, node in enumerate(nodes) for name in node.output}
    selected, constants = set(), set()
    pending = [node.output[0] for node in output_nodes]
    while pending:
        name = pending.pop()
        if name in input_names:
            continue
        if name in initializers:
            constants.add(name)
            continue
        if name not in producers:
            raise ValueError(f"projection depends on an unexpected external input {name!r}")
        index = producers[name]
        if index not in selected:
            selected.add(index)
            pending.extend(value for value in nodes[index].input if value)
    selected_nodes = [node for index, node in enumerate(nodes) if index in selected]
    expected = Counter(Cast=1, Pow=1, ReduceMean=1, Add=1, Sqrt=1, Div=1, Mul=2, MatMul=3)
    if args.heads or rotary:
        expected.update(Cast=2, Pow=2, ReduceMean=2, Add=2, Sqrt=2, Div=2, Mul=4,
                        Shape=2, Gather=2, Unsqueeze=6, Concat=3, Reshape=3)
    if rotary:
        expected.update(Shape=2, Gather=2, Div=2, Cast=6, Unsqueeze=6, Slice=4,
                        Neg=2, Concat=2, Mul=4, Add=2)
    if args.gqa:
        expected = Counter(Shape=10, Gather=8, Unsqueeze=18, Concat=4, Reshape=4,
                           ConstantOfShape=2, Mul=4, Equal=2, Where=2, Expand=2)
    if args.attention:
        expected = Counter(Cast=10, Pow=3, ReduceMean=3, Add=6, Sqrt=3, Div=6, Mul=14,
            Shape=14, Gather=12, MatMul=6, Unsqueeze=34, Concat=11, Reshape=8, Slice=4,
            Neg=2, Transpose=4, ConstantOfShape=3, Equal=2, Where=2, Expand=2, Trilu=1, Softmax=1)
    if Counter(node.op_type for node in selected_nodes) != expected:
        raise ValueError("actual MiniMind projection subgraph drifted from its operator contract")

    output_shapes = ([["B", "S", 8, 96]] * 2 if args.gqa else
        [["B", "S", weight.dims[1] // 96, 96] if args.heads or rotary else ["B", "S", weight.dims[1]]
         for weight in weights])
    if args.attention:
        output_shapes = [["B", "S", 768], ["B", "S", 4, 96], ["B", "S", 4, 96]]
    dynamic = helper.make_model(helper.make_graph(
        selected_nodes, "minimind_actual_" + stem if args.gqa or rotary else "minimind_actual_rmsnorm_qkv",
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
         for name, shape in zip(input_names, input_shapes)],
        [helper.make_tensor_value_info(node.output[0], onnx.TensorProto.FLOAT,
                                      shape) for node, shape in zip(output_nodes, output_shapes)],
        initializer=[tensor for tensor in model.graph.initializer if tensor.name in constants]),
        opset_imports=list(model.opset_import), ir_version=model.ir_version)
    onnx.checker.check_model(dynamic)
    representative = onnx.ModelProto()
    representative.CopyFrom(dynamic)
    for value in list(representative.graph.input) + list(representative.graph.output):
        for dimension in value.type.tensor_type.shape.dim:
            if dimension.dim_param in {"B", "S"}:
                extent = {"B": 1, "S": 4}[dimension.dim_param]
                dimension.ClearField("dim_param")
                dimension.dim_value = extent
    onnx.checker.check_model(representative)
    imported = import_onnx_model(representative, fold_constants=False,
                                 preserve_shape_values=args.heads or args.gqa or rotary)

    args.out.mkdir(parents=True, exist_ok=True)
    onnx.save(dynamic, args.out / f"{stem}_dynamic.onnx")
    onnx.save(representative, args.out / f"{stem}_representative.onnx")
    (args.out / "export_receipt.txt").write_text(sha256(args.out / f"{stem}_dynamic.onnx") + "\n")
    save_imported_model(imported, args.out / f"{stem}.json", args.out / f"{stem}.params")
    evaluator = ReferenceEvaluator(dynamic)
    rng = np.random.default_rng(20260908)
    cases = [(1, 1), (1, 4), (2, 3), (3, 8)]
    input_files = []
    rotary_tables = []
    if rotary:
        for name in input_names[1:]:
            source = nodes[producers[name]].input[0]
            rotary_tables.append(onnx.numpy_helper.to_array(initializers[source]))
    for index, (batch, sequence) in enumerate(cases):
        feeds = {}
        for argument, name in enumerate(input_names):
            shape = [{"B": batch, "S": sequence}.get(dim, dim) for dim in input_shapes[argument]]
            data = (np.ascontiguousarray(rotary_tables[argument - 1][index:index + sequence])
                    if rotary and argument > 0 else rng.standard_normal(shape, dtype=np.float32))
            filename = f"input_{index}_{argument}.bin" if args.gqa or rotary else f"input_{index}.bin"
            (args.out / filename).write_bytes(data.tobytes())
            input_files.append(filename)
            feeds[name] = data
        for output, array in enumerate(evaluator.run(None, feeds)):
            (args.out / f"ref_{index}_{output}.bin").write_bytes(np.asarray(array, dtype=np.float32).tobytes())
    (args.out / "cases.txt").write_text("".join(f"{batch} {sequence}\n" for batch, sequence in cases))
    files = [f"{stem}_dynamic.onnx", f"{stem}_representative.onnx", f"{stem}.json", f"{stem}.params", "cases.txt", "export_receipt.txt"]
    files += input_files
    files += [f"ref_{i}_{j}.bin" for i in range(len(cases)) for j in range(len(output_nodes))]
    receipt = {
        "source_onnx": str(args.onnx), "source_sha256": sha256(args.onnx), "layer": args.layer,
        "stage": stem, "input_name": input_names[0], "input_names": input_names,
        "input_shapes": input_shapes,
        "output_names": [node.output[0] for node in output_nodes],
        "source_nodes": [node.name for node in selected_nodes], "operator_counts": dict(expected),
        "representative": [1, 4, *input_shape[2:]], "cases": cases,
        "seed": 20260908, "reference": "onnx.reference.ReferenceEvaluator(dynamic subgraph)",
        "onnx_version": onnx.__version__, "numpy_version": np.__version__,
        "relay_nodes": len(imported.function.nodes), "constant_count": len(imported.params),
        "sha256": {name: sha256(args.out / name) for name in files},
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"actual MiniMind layer {args.layer}: {len(selected_nodes)} ONNX nodes, "
          f"{len(imported.function.nodes)} Relay source calls, {len(imported.params)} constants, "
          f"four dynamic ONNX references -> {args.out}")


if __name__ == "__main__":
    main()
