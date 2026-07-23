#!/usr/bin/env python3
"""Generate the real-protobuf exact Transformer importer/runtime fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from kxc_onnx import import_onnx, save_imported_model


def build_model() -> onnx.ModelProto:
    initializers = [
        helper.make_tensor("token_ids", TensorProto.INT64, [2], [0, 2]),
        helper.make_tensor("condition", TensorProto.BOOL, [2, 1], [True, False]),
        helper.make_tensor("fallback", TensorProto.FLOAT, [1, 2], [0.0, 0.0]),
        helper.make_tensor("scale", TensorProto.FLOAT, [2], [1.0, 1.0]),
        helper.make_tensor("bias", TensorProto.FLOAT, [2], [0.0, 0.0]),
        helper.make_tensor("starts", TensorProto.INT64, [1], [0]),
        helper.make_tensor("ends", TensorProto.INT64, [1], [1]),
        helper.make_tensor("axes", TensorProto.INT64, [1], [0]),
        helper.make_tensor("steps", TensorProto.INT64, [1], [1]),
    ]
    nodes = [
        helper.make_node("Gather", ["embedding_table", "token_ids"], ["embedded"],
                         name="embedding", axis=0),
        helper.make_node("LayerNormalization", ["embedded", "scale", "bias"],
                         ["normalized", "", ""], name="norm", axis=-1,
                         epsilon=1e-5, stash_type=1),
        helper.make_node("Where", ["condition", "normalized", "fallback"],
                         ["selected"], name="select"),
        helper.make_node("Slice", ["selected", "starts", "ends", "axes", "steps"],
                         ["prefix"], name="prefix"),
        helper.make_node("Concat", ["prefix", "selected"], ["sequence"],
                         name="sequence", axis=0),
        helper.make_node("Transpose", ["sequence"], ["keys"],
                         name="keys", perm=[1, 0]),
        helper.make_node("MatMul", ["sequence", "keys"], ["scores"], name="scores"),
        helper.make_node("Softmax", ["scores"], ["weights"], name="weights", axis=-1),
        helper.make_node("MatMul", ["weights", "sequence"], ["context"],
                         name="context_node"),
    ]
    graph = helper.make_graph(
        nodes,
        "exact_transformer_protobuf_e2e",
        [helper.make_tensor_value_info(
            "embedding_table", TensorProto.FLOAT, [4, 2])],
        [helper.make_tensor_value_info("context", TensorProto.FLOAT, [3, 2])],
        initializer=initializers,
        value_info=[
            helper.make_tensor_value_info("keys", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [3, 3]),
        ],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    args = parser.parse_args()

    model = build_model()
    onnx.checker.check_model(model)
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.onnx)

    # Parse the bytes back through onnx.load, then run the production Python
    # importer/serializer rather than constructing the JSON spec by hand.
    imported = import_onnx(args.onnx)
    assert [node.op_name for node in imported.function.nodes] == [
        "gather", "nn_layer_norm", "where", "slice", "concatenate",
        "transpose", "matmul", "softmax", "matmul",
    ]
    assert imported.function.nodes[1].outputs == ["normalized"]
    assert imported.function.nodes[1].attrs["accumulation_dtype"] == "float64"
    save_imported_model(imported, args.json, args.params)


if __name__ == "__main__":
    main()
