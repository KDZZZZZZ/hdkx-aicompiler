from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from kxc_onnx import (
    UnsupportedONNXOpError,
    import_onnx,
    import_onnx_model,
    save_imported_model,
    to_json_dict,
)


def test_resnet18_imports_mvp_ops_and_params():
    imported = import_onnx(Path("resnet18.onnx"), default_batch=1)

    assert len(imported.function.nodes) == 49
    assert len(imported.params) == 42
    assert imported.param_order[0] == "fc.weight"
    assert imported.function.inputs[0].name == "input"
    assert imported.function.inputs[0].shape == [1, 3, 224, 224]
    assert imported.function.outputs[0].shape == [1, 1000]
    assert {node.op_name for node in imported.function.nodes} == {
        "nn_conv2d",
        "nn_relu",
        "nn_max_pool2d",
        "add",
        "nn_global_avg_pool2d",
        "nn_flatten",
        "nn_gemm",
    }

    first_conv = imported.params["onnx::Conv_193"]
    assert first_conv.shape == [64, 3, 7, 7]
    assert first_conv.dtype == "float32"
    assert len(first_conv.data) == 64 * 3 * 7 * 7 * 4
    assert any(b != 0 for b in first_conv.data[:256])

    first_node = imported.function.nodes[0]
    assert first_node.name == "/conv1/Conv"
    assert first_node.op_name == "nn_conv2d"
    assert first_node.inputs == ["input", "onnx::Conv_193", "onnx::Conv_194"]
    assert first_node.attrs["channels"] == 64
    assert first_node.attrs["kernel_size"] == [7, 7]
    assert first_node.attrs["strides"] == [2, 2]
    assert first_node.attrs["pads"] == [3, 3, 3, 3]


def test_resnet18_serialization_records_param_offsets(tmp_path):
    imported = import_onnx("resnet18.onnx", default_batch=1)
    json_path = tmp_path / "resnet18.import.json"
    params_path = tmp_path / "resnet18.params.bin"

    save_imported_model(imported, json_path, params_path)
    metadata = onnx_import_metadata(json_path)
    param_data = params_path.read_bytes()

    assert len(metadata["params"]) == 42
    assert len(param_data) == sum(len(imported.params[name].data) for name in imported.param_order)
    assert metadata["param_order"] == imported.param_order

    fc_weight_meta = next(x for x in metadata["params"] if x["name"] == "fc.weight")
    fc_weight = imported.params["fc.weight"]
    assert fc_weight_meta["shape"] == [1000, 512]
    assert fc_weight_meta["dtype"] == "float32"
    assert fc_weight_meta["nbytes"] == len(fc_weight.data)
    offset = fc_weight_meta["offset"]
    assert param_data[offset : offset + 16] == fc_weight.data[:16]

    as_dict = to_json_dict(imported)
    assert as_dict["format"] == "kxc.onnx_import.v1"
    assert as_dict["function"]["nodes"][-1]["op_name"] == "nn_gemm"


def _model_with_io_shapes(input_shape, output_shape=None):
    output_shape = input_shape if output_shape is None else output_shape
    graph = helper.make_graph(
        [],
        "shape_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=6)


def test_known_zero_dimension_is_preserved():
    imported = import_onnx_model(_model_with_io_shapes([0, 3]))

    assert imported.function.inputs[0].shape == [0, 3]
    assert imported.function.outputs[0].shape == [0, 3]


def _unknown_rank_value_info(name):
    value_info = onnx.ValueInfoProto()
    value_info.name = name
    value_info.type.tensor_type.elem_type = TensorProto.FLOAT
    return value_info


def _model_with_value_infos(input_value_info, output_value_info):
    graph = helper.make_graph([], "rank_test", [input_value_info], [output_value_info])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=6)


def test_unknown_input_rank_is_rejected_with_value_context():
    model = _model_with_value_infos(
        _unknown_rank_value_info("input"),
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [1]),
    )

    with pytest.raises(ValueError, match=r"Unresolved ONNX rank.*tensor 'input'.*no shape field"):
        import_onnx_model(model)


def test_unknown_output_rank_is_rejected_with_value_context():
    model = _model_with_value_infos(
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [1]),
        _unknown_rank_value_info("output"),
    )

    with pytest.raises(ValueError, match=r"Unresolved ONNX rank.*tensor 'output'.*no shape field"):
        import_onnx_model(model)


def test_scalar_shape_field_is_preserved():
    model = _model_with_io_shapes([])

    assert model.graph.input[0].type.tensor_type.HasField("shape")
    assert model.graph.output[0].type.tensor_type.HasField("shape")
    imported = import_onnx_model(model)
    assert imported.function.inputs[0].shape == []
    assert imported.function.outputs[0].shape == []


def test_symbolic_batch_requires_explicit_binding():
    model = _model_with_io_shapes(["batch", 3])

    with pytest.raises(
        ValueError, match=r"tensor 'input'.*axis 0.*dim_param='batch'"
    ):
        import_onnx_model(model)


def test_explicit_symbolic_batch_binding_is_applied_to_inputs_and_outputs():
    imported = import_onnx_model(_model_with_io_shapes(["batch", 3]), default_batch=4)

    assert imported.function.inputs[0].shape == [4, 3]
    assert imported.function.outputs[0].shape == [4, 3]


@pytest.mark.parametrize("shape", [[1, "channels"], [1, None]])
def test_non_batch_unresolved_dimensions_are_rejected_even_with_batch_binding(shape):
    with pytest.raises(ValueError, match=r"tensor 'input'.*axis 1"):
        import_onnx_model(_model_with_io_shapes(shape), default_batch=4)


def test_output_unresolved_dimension_error_includes_value_name_and_dim_param():
    model = _model_with_io_shapes([1, 3], [1, "classes"])

    with pytest.raises(
        ValueError, match=r"tensor 'output'.*axis 1.*dim_param='classes'"
    ):
        import_onnx_model(model)


@pytest.mark.parametrize("default_batch", [0, -1])
def test_explicit_default_batch_must_be_positive(default_batch):
    with pytest.raises(ValueError, match="default_batch must be positive"):
        import_onnx_model(_model_with_io_shapes(["batch", 3]), default_batch=default_batch)


def _static_operator_model(nodes, opset):
    graph = helper.make_graph(
        nodes,
        "static_operator_model",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2, 3]),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 2])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=6
    )


def _matmul_model(a_shape, b_shape, *, a_dtype=TensorProto.FLOAT, b_dtype=TensorProto.FLOAT,
                  output_name="unused", output_shape=(1,)):
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["a", "b"], ["scores"], name="matmul")],
        "matmul_test",
        [helper.make_tensor_value_info("a", a_dtype, a_shape),
         helper.make_tensor_value_info("b", b_dtype, b_shape)],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=6)


@pytest.mark.parametrize(
    ("a_shape", "b_shape", "a_dtype", "b_dtype", "message"),
    [
        ([3], [3, 2], TensorProto.FLOAT, TensorProto.FLOAT, "rank >= 2"),
        ([1, 2, 3], [1, 4, 2], TensorProto.FLOAT, TensorProto.FLOAT, "K dimensions differ"),
        ([2, 2, 3], [3, 3, 2], TensorProto.FLOAT, TensorProto.FLOAT, "leading batch dimensions"),
        ([1, 2, 3], [1, 3, 2], TensorProto.FLOAT, TensorProto.INT32, "matching input dtypes"),
    ],
)
def test_matmul_rejects_invalid_static_input_contract(
    a_shape, b_shape, a_dtype, b_dtype, message
):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(_matmul_model(a_shape, b_shape, a_dtype=a_dtype, b_dtype=b_dtype))


def test_matmul_accepts_static_batch_broadcast():
    imported = import_onnx_model(
        _matmul_model(
            [2, 1, 3, 4],
            [1, 7, 4, 5],
            output_name="scores",
            output_shape=[2, 7, 3, 5],
        )
    )

    assert imported.function.outputs[0].shape == [2, 7, 3, 5]


def test_matmul_infers_prior_matmul_output_for_chains():
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["a", "b"], ["scores"], name="first"),
            helper.make_node("MatMul", ["scores", "c"], ["out"], name="second"),
        ],
        "matmul_chain",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 4]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 4, 5]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 5])],
    )
    assert len(import_onnx_model(helper.make_model(graph)).function.nodes) == 2


def test_matmul_rejects_missing_non_matmul_intermediate_metadata():
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["a", "b"], ["scores"], name="first"),
            helper.make_node("Softmax", ["scores"], ["weights"], name="softmax"),
            helper.make_node("MatMul", ["weights", "c"], ["out"], name="second"),
        ],
        "matmul_missing_metadata",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 4]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 4, 5]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 5])],
    )
    with pytest.raises(ValueError, match=r"input 'weights' metadata is absent or unresolved"):
        import_onnx_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))


def test_matmul_rejects_declared_output_mismatch():
    with pytest.raises(ValueError, match=r"output 'scores' declaration.*does not match inferred"):
        import_onnx_model(
            _matmul_model([1, 2, 3], [1, 3, 4], output_name="scores", output_shape=[1, 2, 5])
        )


def _gather_model(data_shape, indices_shape, *, axis=0, indices_dtype=TensorProto.INT64,
                  output_shape=(1,), output_dtype=TensorProto.FLOAT):
    graph = helper.make_graph(
        [helper.make_node("Gather", ["data", "indices"], ["out"], name="gather", axis=axis)],
        "gather_test",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
         helper.make_tensor_value_info("indices", indices_dtype, indices_shape)],
        [helper.make_tensor_value_info("out", output_dtype, output_shape)],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=6
    )


def test_gather_mapping_and_inferred_output_contract():
    imported = import_onnx_model(
        _gather_model([2, 3, 4], [5, 6], axis=1, output_shape=[2, 5, 6, 4])
    )

    assert [(node.op_name, node.attrs) for node in imported.function.nodes] == [
        ("gather", {"axis": 1}),
    ]
    assert imported.function.outputs[0].shape == [2, 5, 6, 4]
    assert imported.function.outputs[0].dtype == "float32"


def test_gather_rejects_declared_output_mismatch():
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(
            _gather_model([2, 3, 4], [5, 6], axis=1, output_shape=[2, 5, 4])
        )


@pytest.mark.parametrize(
    ("data_shape", "indices_dtype", "axis", "message"),
    [
        ([], TensorProto.INT64, 0, "data rank >= 1"),
        ([2, 3], TensorProto.FLOAT, 0, "int32 or int64 indices"),
        ([2, 3], TensorProto.INT64, 2, "axis 2 is out of range"),
    ],
)
def test_gather_rejects_invalid_static_contract(data_shape, indices_dtype, axis, message):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(
            _gather_model(data_shape, [1], axis=axis, indices_dtype=indices_dtype)
        )


def test_matmul_softmax_transpose_mapping_and_attrs():
    model = _static_operator_model(
        [
            helper.make_node("MatMul", ["a", "b"], ["scores"], name="matmul"),
            helper.make_node("Softmax", ["scores"], ["weights"], name="softmax", axis=1),
            helper.make_node("Transpose", ["weights"], ["out"], name="transpose", perm=[0, 2, 1]),
        ],
        opset=13,
    )

    imported = import_onnx_model(model)

    assert [(node.op_name, node.attrs) for node in imported.function.nodes] == [
        ("matmul", {}),
        ("softmax", {"axis": 1}),
        ("transpose", {"perm": [0, 2, 1]}),
    ]


def _softmax_model(shape, opset):
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["input"], ["output"], name="softmax")],
        "softmax_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=6
    )


@pytest.mark.parametrize("shape", [[2, 3], [2, 3, 4]])
@pytest.mark.parametrize("opset", [1, 11, 12])
def test_softmax_before_opset13_is_rejected(shape, opset):
    with pytest.raises(
        UnsupportedONNXOpError,
        match=rf"Softmax opset {opset}.*flattened-axis.*single-axis",
    ):
        import_onnx_model(_softmax_model(shape, opset))


@pytest.mark.parametrize("shape", [[2, 3], [2, 3, 4]])
def test_softmax_opset13_default_axis_is_relay_last_axis(shape):
    imported = import_onnx_model(_softmax_model(shape, 13))

    assert imported.function.nodes[0].attrs == {"axis": -1}


def test_transpose_empty_perm_uses_relay_default():
    model = _static_operator_model(
        [
            helper.make_node("MatMul", ["a", "b"], ["scores"], name="matmul"),
            helper.make_node("Softmax", ["scores"], ["weights"], name="softmax"),
            helper.make_node("Transpose", ["weights"], ["out"], name="transpose"),
        ],
        opset=13,
    )
    imported = import_onnx_model(model)

    assert imported.function.nodes[1].attrs == {"axis": -1}
    assert imported.function.nodes[2].attrs == {"perm": []}


def test_unsupported_op_error_includes_op_type_and_node_name():
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["output"], name="bad_identity"),
        ],
        "unsupported_identity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=6,
    )

    with pytest.raises(UnsupportedONNXOpError, match="Identity.*bad_identity"):
        import_onnx_model(model, default_batch=1)


def onnx_import_metadata(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
