#!/usr/bin/env python3
"""Generate the real-protobuf Equal→Where composition importer/runtime fixture.

The model is pinned to opset 17 and covers the M4 wiring of the M5 Equal op:
two float32 graph inputs feed an Equal node (NumPy multidirectional broadcast,
bool output), whose condition selects between two float32 Where branches with
provably different values. The C++ side runs the reified spec through
Compiler::Compile (LLVM) and RuntimeSession and compares the output bit-exactly
against the independent NumPy reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from kxc_onnx import import_onnx, save_imported_model


def build_model() -> onnx.ModelProto:
    nodes = [
        helper.make_node("Equal", ["a", "b"], ["cond"], name="s1_equal"),
        helper.make_node(
            "Constant", [], ["const_x"], name="const_x",
            value=helper.make_tensor("x", TensorProto.FLOAT, [1, 3],
                                     [10.0, 20.0, 30.0]),
        ),
        helper.make_node(
            "Constant", [], ["const_y"], name="const_y",
            value=helper.make_tensor("y", TensorProto.FLOAT, [], [-1.0]),
        ),
        helper.make_node("Where", ["cond", "const_x", "const_y"], ["out"],
                         name="s1_where"),
    ]
    graph = helper.make_graph(
        nodes,
        "equal_where_protobuf_e2e",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [3]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])],
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
    assert [node.op_name for node in imported.function.nodes] == ["equal", "where"]
    assert all(node.attrs == {} for node in imported.function.nodes)
    assert [node.inputs for node in imported.function.nodes] == [
        ["a", "b"],
        ["cond", "const_x", "const_y"],
    ]
    assert imported.function.nodes[0].outputs == ["cond"]
    assert imported.param_order == ["const_x", "const_y"]
    assert imported.function.outputs[0].shape == [2, 3]
    assert imported.function.outputs[0].dtype == "float32"

    # The independent reference: NumPy ONNX semantics evaluated outside the
    # importer/Relay stack. The comparison values are exactly representable and
    # Where only selects between them, so the C++ check can require bit-exactness.
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([1.0, 0.0, 3.0], dtype=np.float32)
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y = np.float32(-1.0)
    expected = np.where(a == b, x, y)
    assert expected.tolist() == [[10.0, -1.0, 30.0], [-1.0, -1.0, -1.0]]

    save_imported_model(imported, args.json, args.params)


if __name__ == "__main__":
    main()
