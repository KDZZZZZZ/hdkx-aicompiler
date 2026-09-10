#!/usr/bin/env python3
"""Generate the real-protobuf static S1 arithmetic importer/runtime fixture.

The model is pinned to opset 17 and covers exactly the S1 subset: Constant,
Mul, Sub, Div, Sqrt, Cast (int64 to float32), ReduceMean (static axes
attribute form), and Reshape (Constant shape input, allowzero=0).

ReduceMean reduces over the trailing axis with keepdims=1: the pre-existing
TE/TIR lowering collapses values when a kept size-1 axis is followed by
non-singleton axes (see the wave report), so the fixture uses the verified
trailing-axis form.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from kxc_onnx import import_onnx, save_imported_model


def build_model() -> onnx.ModelProto:
    nodes = [
        helper.make_node(
            "Constant", [], ["const_scale"], name="const_scale",
            value=helper.make_tensor("scale", TensorProto.FLOAT, [4],
                                     [1.0, 2.0, 0.5, 4.0]),
        ),
        helper.make_node("Mul", ["x", "const_scale"], ["scaled"], name="s1_mul"),
        helper.make_node(
            "Constant", [], ["const_sub"], name="const_sub",
            value=helper.make_tensor("sub", TensorProto.FLOAT, [], [0.25]),
        ),
        helper.make_node("Sub", ["scaled", "const_sub"], ["shifted"], name="s1_sub"),
        helper.make_node(
            "Constant", [], ["const_div"], name="const_div",
            value=helper.make_tensor("div", TensorProto.FLOAT, [], [2.0]),
        ),
        helper.make_node("Div", ["shifted", "const_div"], ["quotient"], name="s1_div"),
        helper.make_node("Sqrt", ["quotient"], ["rooted"], name="s1_sqrt"),
        helper.make_node("Cast", ["idx"], ["scale3"], name="s1_cast",
                         to=TensorProto.FLOAT),
        helper.make_node("Mul", ["rooted", "scale3"], ["weighted"], name="s1_mul2"),
        helper.make_node("ReduceMean", ["weighted"], ["pooled"], name="s1_reduce_mean",
                         axes=[2], keepdims=1),
        helper.make_node(
            "Constant", [], ["const_shape"], name="const_shape",
            value=helper.make_tensor("shape", TensorProto.INT64, [1], [6]),
        ),
        helper.make_node("Reshape", ["pooled", "const_shape"], ["out"],
                         name="s1_reshape"),
    ]
    graph = helper.make_graph(
        nodes,
        "static_s1_protobuf_e2e",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("idx", TensorProto.INT64, [1]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [6])],
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
        "mul", "subtract", "divide", "sqrt", "cast", "mul", "reduce_mean",
        "reshape",
    ]
    assert imported.function.nodes[4].attrs == {"to": 0}
    assert imported.function.nodes[6].attrs == {"axes": [2], "keepdims": 1}
    assert imported.function.nodes[7].attrs == {"newshape": [6], "allowzero": 0}
    assert [node.inputs for node in imported.function.nodes] == [
        ["x", "const_scale"],
        ["scaled", "const_sub"],
        ["shifted", "const_div"],
        ["quotient"],
        ["idx"],
        ["rooted", "scale3"],
        ["weighted"],
        ["pooled"],
    ]
    assert imported.function.outputs[0].shape == [6]
    assert imported.function.outputs[0].dtype == "float32"
    # Every Constant output becomes a param; the Reshape shape tensor keeps its
    # real bytes in the serialized payload even though attrs carry the resolved
    # target shape.
    assert imported.param_order == [
        "const_scale", "const_sub", "const_div", "const_shape",
    ]
    save_imported_model(imported, args.json, args.params)


if __name__ == "__main__":
    main()
