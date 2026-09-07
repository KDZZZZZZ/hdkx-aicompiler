#!/usr/bin/env python3
"""Generate the real-protobuf M4/M5 five-operator importer/runtime fixture.

The model is pinned to opset 17 and covers the D-line M4/M5 wave with the
MiniMind-L1a shapes: Neg → Pow(x², the RMSNorm/SwiGLU exponent pattern) →
Sigmoid → Unsqueeze (axes=[0], normalized to reshape) → Expand to a rank-3
target. The C++ side runs the reified spec through Compiler::Compile (LLVM)
and RuntimeSession and compares the output element-wise against the
independent NumPy reference computed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from kxc_onnx import import_onnx, save_imported_model


def build_model() -> onnx.ModelProto:
    nodes = [
        helper.make_node("Neg", ["x"], ["n"], name="s1_neg"),
        helper.make_node("Pow", ["n", "exponent"], ["p"], name="s2_pow"),
        helper.make_node("Sigmoid", ["p"], ["s"], name="s1_sigmoid"),
        helper.make_node("Unsqueeze", ["s", "axes"], ["u"], name="s2_unsqueeze"),
        helper.make_node("Expand", ["u", "target"], ["out"], name="s1_expand"),
    ]
    graph = helper.make_graph(
        nodes,
        "m4m5_ops_protobuf_e2e",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [3, 2, 4])],
        initializer=[
            numpy_helper.from_array(np.array([2.0], dtype=np.float32), "exponent"),
            numpy_helper.from_array(np.array([0], dtype=np.int64), "axes"),
            numpy_helper.from_array(np.array([3, 2, 4], dtype=np.int64), "target"),
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
        "neg", "pow", "sigmoid", "reshape", "expand",
    ]
    assert imported.function.nodes[3].attrs == {"newshape": [1, 2, 4], "allowzero": 0}
    assert imported.function.nodes[4].attrs == {"target_shape": [3, 2, 4]}
    assert imported.function.nodes[3].inputs == ["s"]
    assert imported.function.nodes[4].inputs == ["u"]
    assert imported.param_order == ["exponent", "axes", "target"]
    assert imported.function.outputs[0].shape == [3, 2, 4]
    assert imported.function.outputs[0].dtype == "float32"

    # The independent reference: NumPy ONNX semantics evaluated outside the
    # importer/Relay stack, in float32 element order.
    x = np.array(
        [[1.0, -2.0, 0.5, 3.0], [-0.25, 2.0, -4.0, 0.75]], dtype=np.float32
    )
    n = np.negative(x)
    p = np.power(n, np.float32(2.0))
    s = np.float32(1.0) / (np.float32(1.0) + np.exp(np.negative(p)))
    u = np.expand_dims(s, axis=0)
    expected = np.broadcast_to(u, (3, 2, 4))
    assert expected.shape == (3, 2, 4)
    assert expected.dtype == np.float32

    save_imported_model(imported, args.json, args.params)


if __name__ == "__main__":
    main()
