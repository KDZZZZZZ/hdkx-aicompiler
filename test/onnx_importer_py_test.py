from pathlib import Path

import onnx
import pytest
from onnx import AttributeProto, TensorProto, helper, numpy_helper

from kxc_onnx.fold import ConstantFoldingError, fold_static_subgraph
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
    with pytest.raises(
        ValueError, match=r"default_batch must be a positive integer"
    ):
        import_onnx_model(_model_with_io_shapes(["batch", 3]), default_batch=default_batch)


def _static_operator_model(nodes, opset, output_shape=(1, 2, 2)):
    graph = helper.make_graph(
        nodes,
        "static_operator_model",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2, 3]),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(output_shape))],
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
            # Relu 不在导入器的逐算子推导链里，其输出也没有 value_info，
            # 因此 weights 的元数据确实无从解析。Softmax 已可推导，不再适合
            # 用来构造这个负例。
            helper.make_node("Relu", ["scores"], ["weights"], name="relu"),
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
                  indices_values=None, dynamic_indices=False, node_outputs=("out",),
                  output_shape=(1,), output_dtype=TensorProto.FLOAT):
    value_count = 1
    for extent in indices_shape:
        value_count *= extent
    if indices_values is None:
        indices_values = [0] * value_count
    graph_inputs = [helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape)]
    initializers = []
    if dynamic_indices:
        graph_inputs.append(
            helper.make_tensor_value_info("indices", indices_dtype, indices_shape)
        )
    else:
        initializers.append(
            helper.make_tensor(
                "indices", indices_dtype, indices_shape, list(indices_values)
            )
        )
    graph = helper.make_graph(
        [helper.make_node("Gather", ["data", "indices"], node_outputs,
                          name="gather", axis=axis)],
        "gather_test",
        graph_inputs,
        [helper.make_tensor_value_info("out", output_dtype, output_shape)],
        initializer=initializers,
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


def test_gather_rejects_non_integer_or_duplicate_axis_attribute():
    with pytest.raises(ValueError, match="axis.*exact INT type"):
        import_onnx_model(_gather_model([2, 3], [1], axis=0.5))

    model = _gather_model([2, 3], [1], axis=0)
    model.graph.node[0].attribute.extend([helper.make_attribute("axis", 0)])
    with pytest.raises(ValueError, match="duplicate attribute 'axis'"):
        import_onnx_model(model)


def test_gather_rejects_unknown_attribute():
    model = _gather_model([2, 3], [1], axis=0)
    model.graph.node[0].attribute.extend([helper.make_attribute("unknown", 1)])
    with pytest.raises(ValueError, match="unsupported attribute"):
        import_onnx_model(model)


def test_gather_rejects_declared_output_mismatch():
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(
            _gather_model([2, 3, 4], [5, 6], axis=1, output_shape=[2, 5, 4])
        )


def test_gather_accepts_runtime_indices():
    """运行时索引是 embedding 查表的形态：导入期只定形状/dtype/axis 合同。

    索引值域交给已 lower 的 GatherCompute 守卫——负索引按 ONNX 语义折回，
    越界经 Select 取零且不形成越界 Load。
    """
    imported = import_onnx_model(_gather_model(
        [2, 3], [1], axis=1, dynamic_indices=True, output_shape=[2, 1]
    ))

    assert imported.function.nodes[0].op_name == "gather"
    assert imported.function.nodes[0].inputs == ["data", "indices"]
    assert imported.function.outputs[0].shape == [2, 1]


def test_gather_rejects_empty_output_name():
    with pytest.raises(ValueError, match="exactly one non-empty output"):
        import_onnx_model(_gather_model(
            [2, 3], [1], axis=1, node_outputs=("",), output_shape=[2, 1]
        ))


@pytest.mark.parametrize("indices_dtype", [TensorProto.INT32, TensorProto.INT64])
def test_gather_accepts_constant_index_domain_boundaries(indices_dtype):
    imported = import_onnx_model(_gather_model(
        [2, 3], [2], axis=1, indices_dtype=indices_dtype,
        indices_values=[-3, 2], output_shape=[2, 2]
    ))

    assert imported.function.nodes[0].inputs == ["data", "indices"]


@pytest.mark.parametrize("bad_index", [3, -4, -(2**63)])
def test_gather_rejects_constant_out_of_domain_indices(bad_index):
    with pytest.raises(ValueError, match="outside the ONNX domain"):
        import_onnx_model(_gather_model(
            [2, 3], [1], axis=1, indices_values=[bad_index], output_shape=[2, 1]
        ))


def test_gather_accepts_empty_constant_indices():
    imported = import_onnx_model(_gather_model(
        [2, 3], [0], axis=1, indices_values=[], output_shape=[2, 0]
    ))

    assert imported.function.outputs[0].shape == [2, 0]


@pytest.mark.parametrize(
    ("data_shape", "indices_dtype", "axis", "message"),
    [
        ([], TensorProto.INT64, 0, "data rank >= 1"),
        ([2, 3], TensorProto.FLOAT, 0, "int32 or int64 indices"),
        ([2, 3], TensorProto.INT64, 2, "axis 2 is out of range"),
        ([-1, 3], TensorProto.INT64, 1, "non-negative static dimensions"),
    ],
)
def test_gather_rejects_invalid_static_contract(data_shape, indices_dtype, axis, message):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(
            _gather_model(data_shape, [1], axis=axis, indices_dtype=indices_dtype)
        )


def _concat_model(
    lhs_shape,
    rhs_shape,
    output_shape,
    *,
    axis=0,
    lhs_dtype=TensorProto.FLOAT,
    rhs_dtype=None,
    output_dtype=None,
    node_inputs=("lhs", "rhs"),
    include_axis=True,
):
    rhs_dtype = lhs_dtype if rhs_dtype is None else rhs_dtype
    output_dtype = lhs_dtype if output_dtype is None else output_dtype
    attrs = {"axis": axis} if include_axis else {}
    graph = helper.make_graph(
        [helper.make_node("Concat", node_inputs, ["out"], name="concat", **attrs)],
        "concat_test",
        [
            helper.make_tensor_value_info("lhs", lhs_dtype, lhs_shape),
            helper.make_tensor_value_info("rhs", rhs_dtype, rhs_shape),
            helper.make_tensor_value_info("extra", lhs_dtype, lhs_shape),
        ],
        [helper.make_tensor_value_info("out", output_dtype, output_shape)],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=6
    )


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "axis", "output_shape"),
    [
        ([2, 2], [2, 3], 1, [2, 5]),
        ([2, 2], [2, 3], -1, [2, 5]),
    ],
)
def test_concat_maps_positive_and_negative_axis(
    lhs_shape, rhs_shape, axis, output_shape
):
    imported = import_onnx_model(_concat_model(lhs_shape, rhs_shape, output_shape, axis=axis))

    assert [(node.op_name, node.attrs) for node in imported.function.nodes] == [
        ("concatenate", {"axis": axis}),
    ]
    assert imported.function.outputs[0].shape == output_shape


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "output_shape"),
    [
        ([2, 0], [2, 3], [2, 3]),
        ([2, 0], [2, 0], [2, 0]),
    ],
)
def test_concat_preserves_empty_side_and_empty_output(lhs_shape, rhs_shape, output_shape):
    imported = import_onnx_model(_concat_model(lhs_shape, rhs_shape, output_shape, axis=1))

    assert imported.function.outputs[0].shape == output_shape


def test_concat_bool_dtype_maps_and_infers_output():
    imported = import_onnx_model(
        _concat_model([1, 2], [1, 1], [1, 3], axis=1, lhs_dtype=TensorProto.BOOL)
    )

    assert imported.function.outputs[0].dtype == "bool"


def test_concat_rejects_non_integer_axis_attribute():
    with pytest.raises(ValueError, match="axis.*exact INT type"):
        import_onnx_model(_concat_model([2, 2], [2, 3], [2, 5], axis=1.5))


def test_concat_requires_explicit_axis():
    with pytest.raises(ValueError, match="exactly the axis attribute"):
        import_onnx_model(_concat_model([2, 2], [2, 3], [2, 5], include_axis=False))


def test_concat_rejects_rank_zero_and_out_of_range_axis():
    with pytest.raises(ValueError, match="rank >= 1"):
        import_onnx_model(_concat_model([], [], [], axis=0))
    with pytest.raises(ValueError, match="axis 2 is out of range"):
        import_onnx_model(_concat_model([2, 2], [2, 3], [2, 5], axis=2))


@pytest.mark.parametrize("node_inputs", [(), ("lhs", ""), ("",)])
def test_concat_rejects_missing_inputs(node_inputs):
    with pytest.raises(ValueError, match="exactly two non-empty inputs"):
        import_onnx_model(_concat_model([2, 2], [2, 3], [2, 5], node_inputs=node_inputs))


@pytest.mark.parametrize("shape, axis", [([1, 68, 768], 0), ([2, 3], -1), ([2, 0], 1)])
def test_singleton_concat_uses_existing_binary_copy_semantics(shape, axis):
    model = _concat_model(shape, shape, shape, axis=axis, node_inputs=("lhs",))
    original = model.SerializeToString()
    imported = import_onnx_model(model, fold_constants=False)
    node, = imported.function.nodes
    assert node.op_name == "concatenate" and node.inputs[0] == "lhs"
    empty = imported.params[node.inputs[1]]
    expected_shape = list(shape)
    expected_shape[axis] = 0
    assert empty.shape == expected_shape and empty.dtype == "float32" and empty.data == b""
    assert imported.function.outputs[0].shape == shape
    assert model.SerializeToString() == original
    with pytest.raises(UnsupportedONNXOpError, match="Singleton Concat.*static"):
        import_onnx_model(model, fold_constants=False, preserve_shape_values=True)


@pytest.mark.parametrize("axis", [2, -3, 1.5])
def test_singleton_concat_does_not_erase_invalid_axis(axis):
    with pytest.raises(ValueError, match="axis"):
        import_onnx_model(_concat_model([2, 3], [2, 3], [2, 3], axis=axis,
                                       node_inputs=("lhs",)), fold_constants=False)


@pytest.mark.parametrize("preserve_shape_values", [False, True])
@pytest.mark.parametrize("axis", [1, -1])
def test_variadic_concat_preserves_order_and_source(axis, preserve_shape_values):
    import numpy as np
    from onnx.reference import ReferenceEvaluator
    from kxc_onnx.importer import _binary_concats

    model = _concat_model([2, 2], [2, 3], [2, 7], axis=axis,
                          node_inputs=("lhs", "rhs", "extra"))
    original = model.SerializeToString()
    feeds = {"lhs": np.arange(4, dtype=np.float32).reshape(2, 2),
             "rhs": np.arange(6, dtype=np.float32).reshape(2, 3) + 20,
             "extra": np.arange(4, dtype=np.float32).reshape(2, 2) + 100}
    expected = ReferenceEvaluator(model).run(None, feeds)[0]
    normalized = _binary_concats(model)
    actual = ReferenceEvaluator(normalized).run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)
    assert model.SerializeToString() == original
    imported = import_onnx_model(model, preserve_shape_values=preserve_shape_values)
    assert [node.inputs for node in imported.function.nodes] == [
        ["lhs", "rhs"], ["out__kxc_concat_0", "extra"]]
    assert all(node.op_name == "concatenate" and node.attrs == {"axis": axis}
               for node in imported.function.nodes)
    assert imported.function.outputs[0].shape == [2, 7]
    assert to_json_dict(imported) == to_json_dict(import_onnx_model(
        model, preserve_shape_values=preserve_shape_values))


def test_variadic_concat_keeps_empty_operands_and_checks_final_declaration():
    model = _concat_model([2, 0], [2, 3], [2, 3], axis=1,
                          node_inputs=("lhs", "rhs", "extra", "lhs"))
    imported = import_onnx_model(model)
    assert len(imported.function.nodes) == 3
    assert imported.function.outputs[0].shape == [2, 3]
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 4
    with pytest.raises(ValueError, match="does not match inferred"):
        import_onnx_model(model)


@pytest.mark.parametrize("failure", ["dtype", "rank", "nonaxis", "missing", "empty", "attrs"])
def test_variadic_concat_validates_late_operands(failure):
    model = _concat_model([2, 2], [2, 3], [2, 7], axis=1,
                          node_inputs=("lhs", "rhs", "extra"))
    if failure == "dtype":
        model.graph.input[2].type.tensor_type.elem_type = TensorProto.INT64
    elif failure == "rank":
        model.graph.input[2].CopyFrom(helper.make_tensor_value_info("extra", TensorProto.FLOAT, [4]))
    elif failure == "nonaxis":
        model.graph.input[2].type.tensor_type.shape.dim[0].dim_value = 3
    elif failure == "missing":
        # This invalid input used to collide with the first generated temporary.
        model.graph.node[0].input[2] = "out__kxc_concat_0"
    elif failure == "empty":
        model.graph.node[0].input[2] = ""
    else:
        model.graph.node[0].attribute.append(helper.make_attribute("unexpected", 1))
    message = {"dtype": "matching input dtypes", "rank": "input ranks must match",
               "nonaxis": "non-axis dimensions", "missing": "unresolved prior input",
               "empty": "Concat requires nonempty", "attrs": "exactly the axis"}[failure]
    with pytest.raises(ValueError, match=message):
        import_onnx_model(model, fold_constants=False)


def test_variadic_concat_reserves_original_value_metadata_and_node_names():
    from kxc_onnx.importer import _binary_concats

    model = _concat_model([2, 2], [2, 3], [2, 7], axis=1,
                          node_inputs=("lhs", "rhs", "extra"))
    model.graph.value_info.append(helper.make_tensor_value_info(
        "out__kxc_concat_0", TensorProto.FLOAT, [99]))
    model.graph.node.append(helper.make_node("Relu", ["out"], ["positive"], name="concat_part0"))
    normalized = _binary_concats(model)
    assert normalized.graph.node[0].output[0] == "out__kxc_concat_0_"
    assert normalized.graph.node[0].name == "concat_part0_"
    imported = import_onnx_model(model, fold_constants=False)
    assert imported.function.outputs[0].shape == [2, 7]


def test_variadic_concat_checks_overflow_in_late_operand():
    model = _concat_model([1, (1 << 63) - 2], [1, 1], [1, 1], axis=1,
                          node_inputs=("lhs", "rhs", "extra"))
    with pytest.raises(ValueError, match="axis extent sum overflows int64"):
        import_onnx_model(model, fold_constants=False)


@pytest.mark.parametrize("fold_constants", [False, True])
@pytest.mark.parametrize("op", ["Concat", "Add", "Constant"])
def test_custom_domain_cannot_be_reinterpreted_as_standard_onnx(op, fold_constants):
    initializers = [helper.make_tensor(name, TensorProto.FLOAT, [1], [1.0])
                    for name in ["a", "b", "c"]]
    inputs = {"Concat": ["a", "b", "c"], "Add": ["a", "b"], "Constant": []}[op]
    attrs = {"axis": 0} if op == "Concat" else ({"value": initializers[0]} if op == "Constant" else {})
    model = helper.make_model(helper.make_graph(
        [helper.make_node(op, inputs, ["out"], domain="vendor.private", **attrs)],
        "custom_domain", [], [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
        initializer=initializers),
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("vendor.private", 1)])
    with pytest.raises(UnsupportedONNXOpError, match="Unsupported ONNX domain 'vendor.private'"):
        import_onnx_model(model, fold_constants=fold_constants)


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "rhs_dtype", "message"),
    [
        ([2, 2], [2, 3], TensorProto.INT32, "matching input dtypes"),
        ([2, 2], [2, 3, 1], TensorProto.FLOAT, "input ranks must match"),
        ([2, 2], [3, 3], TensorProto.FLOAT, "non-axis dimensions must exactly match"),
    ],
)
def test_concat_rejects_dtype_rank_and_nonaxis_mismatches(
    lhs_shape, rhs_shape, rhs_dtype, message
):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(
            _concat_model(lhs_shape, rhs_shape, [2, 5], axis=1, rhs_dtype=rhs_dtype)
        )


def test_concat_rejects_declared_output_mismatch():
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(_concat_model([2, 2], [2, 3], [2, 4], axis=1))


def test_concat_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="Unsupported ONNX tensor dtype"):
        import_onnx_model(
            _concat_model(
                [2, 2], [2, 3], [2, 5], axis=1, lhs_dtype=TensorProto.FLOAT16
            )
        )


def _split_model(
    data_shape,
    output_shapes,
    *,
    axis=0,
    sections=(1, 1),
    use_input=False,
    output_names=("left", "right"),
    data_dtype=TensorProto.FLOAT,
):
    inputs = [helper.make_tensor_value_info("data", data_dtype, data_shape)]
    initializers = []
    node_inputs = ["data"]
    attrs = {"axis": axis}
    if use_input:
        node_inputs.append("sections")
        initializers.append(
            helper.make_tensor("sections", TensorProto.INT64, [len(sections)], list(sections))
        )
    else:
        attrs["split"] = list(sections)
    graph = helper.make_graph(
        [helper.make_node("Split", node_inputs, list(output_names), name="split", **attrs)],
        "split_test",
        inputs,
        [
            helper.make_tensor_value_info(name, data_dtype, shape)
            for name, shape in zip(output_names, output_shapes)
        ],
        initializer=initializers,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=6
    )


def test_split_maps_two_outputs_and_preserves_order():
    imported = import_onnx_model(
        _split_model([2, 6], [[2, 2], [2, 4]], axis=1, sections=(2, 4))
    )

    assert [(node.op_name, node.inputs, node.attrs) for node in imported.function.nodes] == [
        ("split", ["data"], {"axis": 1, "sections": [2, 4]}),
    ]
    assert [output.name for output in imported.function.outputs] == ["left", "right"]
    assert [output.shape for output in imported.function.outputs] == [[2, 2], [2, 4]]


def test_split_accepts_constant_sections_input():
    imported = import_onnx_model(
        _split_model([6], [[2], [4]], sections=(2, 4), use_input=True)
    )

    assert imported.function.nodes[0].op_name == "split"
    assert imported.function.nodes[0].inputs == ["data"]
    assert imported.function.nodes[0].attrs == {"axis": 0, "sections": [2, 4]}


def test_split_maps_three_outputs_and_preserves_order():
    imported = import_onnx_model(
        _split_model([2, 6], [[2, 1], [2, 3], [2, 2]], axis=1,
                     sections=(1, 3, 2), output_names=("first", "middle", "last"))
    )

    assert imported.function.nodes[0].attrs == {"axis": 1, "sections": [1, 3, 2]}
    assert [output.name for output in imported.function.outputs] == [
        "first", "middle", "last"
    ]
    assert [output.shape for output in imported.function.outputs] == [
        [2, 1], [2, 3], [2, 2]
    ]


@pytest.mark.parametrize("failure", ["outputs", "sum", "dynamic", "dtype"])
def test_split_rejects_unsupported_contracts(failure):
    if failure == "outputs":
        model = _split_model([4], [[1], [1], [2]], sections=(1, 1),
                             output_names=("a", "b", "c"))
        message = "output count must equal the number of sections"
    elif failure == "sum":
        model = _split_model([4], [[1], [2]], sections=(1, 2))
        message = "sections must sum to the input axis extent"
    elif failure == "dynamic":
        model = _split_model([4], [[1], [3]], sections=(1, 3), use_input=True)
        model.graph.initializer.clear()
        model.graph.input.append(
            helper.make_tensor_value_info("sections", TensorProto.INT64, [2])
        )
        message = "must be a static initializer or Constant node output"
    else:
        model = _split_model([4], [[1], [3]], sections=(1, 3), data_dtype=TensorProto.FLOAT16)
        message = "Unsupported ONNX tensor dtype"
    with pytest.raises(ValueError, match=message):
        import_onnx_model(model)


def _where_model(condition_shape, x_shape, y_shape, *, condition_dtype=TensorProto.BOOL,
                 x_dtype=TensorProto.FLOAT, y_dtype=TensorProto.FLOAT,
                 output_shape=(1,), output_dtype=TensorProto.FLOAT):
    graph = helper.make_graph(
        [helper.make_node("Where", ["condition", "x", "y"], ["out"], name="where")],
        "where_test",
        [helper.make_tensor_value_info("condition", condition_dtype, condition_shape),
         helper.make_tensor_value_info("x", x_dtype, x_shape),
         helper.make_tensor_value_info("y", y_dtype, y_shape)],
        [helper.make_tensor_value_info("out", output_dtype, output_shape)],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=6
    )


def test_where_mapping_and_joint_broadcast_output_contract():
    imported = import_onnx_model(
        _where_model([2, 1, 1], [], [1, 3, 4], output_shape=[2, 3, 4])
    )

    assert [(node.op_name, node.attrs) for node in imported.function.nodes] == [
        ("where", {}),
    ]
    assert imported.function.outputs[0].shape == [2, 3, 4]
    assert imported.function.outputs[0].dtype == "float32"


def test_where_rejects_attributes():
    model = _where_model([2, 1], [2, 3], [2, 3], output_shape=[2, 3])
    model.graph.node[0].attribute.extend([helper.make_attribute("axis", 0)])
    with pytest.raises(ValueError, match="does not support attributes"):
        import_onnx_model(model)


def test_where_preserves_broadcast_zero_extent():
    imported = import_onnx_model(
        _where_model([0, 1], [1, 3], [0, 3], output_shape=[0, 3])
    )

    assert imported.function.outputs[0].shape == [0, 3]


def test_where_rejects_declared_output_mismatch():
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(
            _where_model([2, 1], [], [1, 3], output_shape=[2, 2])
        )


def test_where_rejects_unsupported_initializer_branch_dtype():
    graph = helper.make_graph(
        [helper.make_node("Where", ["condition", "x", "y"], ["out"], name="where")],
        "where_float16_initializer_test",
        [helper.make_tensor_value_info("condition", TensorProto.BOOL, [2, 1])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])],
        initializer=[
            helper.make_tensor("x", TensorProto.FLOAT16, [], [1.0]),
            helper.make_tensor("y", TensorProto.FLOAT16, [1, 3], [1.0, 2.0, 3.0]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=6)

    with pytest.raises(ValueError, match="branch dtypes"):
        import_onnx_model(model)


@pytest.mark.parametrize(
    ("condition_shape", "x_shape", "y_shape", "condition_dtype", "x_dtype", "y_dtype", "message"),
    [
        ([2, 1], [], [1, 3], TensorProto.UINT8, TensorProto.FLOAT, TensorProto.FLOAT, "bool condition"),
        ([2, 1], [], [1, 3], TensorProto.BOOL, TensorProto.FLOAT, TensorProto.INT32, "matching x/y dtypes"),
        ([2, 2], [2, 3], [2, 3], TensorProto.BOOL, TensorProto.FLOAT, TensorProto.FLOAT, "incompatible broadcast dimensions"),
    ],
)
def test_where_rejects_invalid_static_contract(
    condition_shape, x_shape, y_shape, condition_dtype, x_dtype, y_dtype, message
):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(
            _where_model(condition_shape, x_shape, y_shape,
                         condition_dtype=condition_dtype, x_dtype=x_dtype, y_dtype=y_dtype)
        )


def _equal_model(a_shape, b_shape, *, a_dtype=TensorProto.FLOAT, b_dtype=TensorProto.FLOAT,
                 output_shape=(2, 3), output_dtype=TensorProto.BOOL, attrs=None,
                 opset=17):
    nodes = [helper.make_node("Equal", ["a", "b"], ["out"], name="equal")]
    if attrs:
        nodes[0].attribute.extend(
            helper.make_attribute(name, value) for name, value in attrs.items()
        )
    graph = helper.make_graph(
        nodes,
        "equal_test",
        [helper.make_tensor_value_info("a", a_dtype, a_shape),
         helper.make_tensor_value_info("b", b_dtype, b_shape)],
        [helper.make_tensor_value_info("out", output_dtype, output_shape)],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=6
    )


def test_equal_mapping_and_broadcast_output_contract():
    imported = import_onnx_model(
        _equal_model([2, 1], [], output_shape=[2, 1])
    )

    assert [(node.op_name, node.attrs) for node in imported.function.nodes] == [
        ("equal", {})
    ]
    assert imported.function.outputs[0].shape == [2, 1]
    assert imported.function.outputs[0].dtype == "bool"


@pytest.mark.parametrize("dtype", [TensorProto.INT32, TensorProto.INT64, TensorProto.FLOAT])
def test_equal_accepts_each_supported_same_dtype(dtype):
    imported = import_onnx_model(
        _equal_model([2, 3], [3], a_dtype=dtype, b_dtype=dtype,
                     output_shape=[2, 3])
    )

    assert imported.function.nodes[0].op_name == "equal"
    assert imported.function.outputs[0].dtype == "bool"


def test_equal_feeds_where_as_condition():
    graph = helper.make_graph(
        [
            helper.make_node("Equal", ["a", "b"], ["cond"], name="s1_equal"),
            helper.make_node("Where", ["cond", "x", "y"], ["out"], name="s1_where"),
        ],
        "equal_where_test",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, [3]),
         helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3]),
         helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=6
    )

    imported = import_onnx_model(model)

    assert [(node.op_name, node.attrs) for node in imported.function.nodes] == [
        ("equal", {}),
        ("where", {}),
    ]
    assert imported.function.outputs[0].shape == [2, 3]
    assert imported.function.outputs[0].dtype == "float32"


def test_equal_rejects_attributes():
    model = _equal_model([2, 3], [3], attrs={"axis": 0})
    with pytest.raises(ValueError, match="does not support attributes"):
        import_onnx_model(model)


@pytest.mark.parametrize(
    ("a_shape", "b_shape", "a_dtype", "b_dtype", "message"),
    [
        ([2, 3], [2, 3], TensorProto.INT32, TensorProto.FLOAT,
         "matching input dtypes"),
        ([2, 3], [2, 3], TensorProto.DOUBLE, TensorProto.DOUBLE,
         "same-dtype int32, int64, or float32"),
        ([2, 3], [2, 3], TensorProto.BOOL, TensorProto.BOOL,
         "same-dtype int32, int64, or float32"),
        ([2, 3], [2, 4], TensorProto.FLOAT, TensorProto.FLOAT,
         "incompatible broadcast dimensions"),
    ],
)
def test_equal_rejects_invalid_static_contract(
    a_shape, b_shape, a_dtype, b_dtype, message
):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(
            _equal_model(a_shape, b_shape, a_dtype=a_dtype, b_dtype=b_dtype,
                         output_shape=[2, 3])
        )


def test_equal_rejects_declared_output_not_bool():
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(
            _equal_model([2, 3], [3], output_shape=[2, 3],
                         output_dtype=TensorProto.FLOAT)
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
        # 缺省 perm 在 ONNX 与 Relay 里都是逆序：[1,2,2] -> [2,2,1]。
        output_shape=(2, 2, 1),
    )
    imported = import_onnx_model(model)

    assert imported.function.nodes[1].attrs == {"axis": -1}
    assert imported.function.nodes[2].attrs == {"perm": []}
    assert imported.function.outputs[0].shape == [2, 2, 1]


def test_unsupported_op_error_includes_op_type_and_node_name():
    graph = helper.make_graph(
        [
            helper.make_node("Trilu", ["input"], ["output"], name="bad_trilu", upper=1),
        ],
        "unsupported_trilu",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=6,
    )

    with pytest.raises(UnsupportedONNXOpError, match="Trilu.*bad_trilu"):
        import_onnx_model(model, default_batch=1)


def _layer_normalization_model(
    data_shape=(2, 3, 4),
    scale_shape=(3, 4),
    bias_shape=(3, 4),
    *,
    data_dtype=TensorProto.FLOAT,
    scale_dtype=TensorProto.FLOAT,
    bias_dtype=TensorProto.FLOAT,
    output_shape=None,
    output_dtype=TensorProto.FLOAT,
    axis=1,
    epsilon=1e-5,
    stash_type=None,
    extra_attrs=None,
    node_inputs=None,
    node_outputs=None,
    opset=17,
):
    node_inputs = ["data", "scale", "bias"] if node_inputs is None else node_inputs
    node_outputs = ["out"] if node_outputs is None else node_outputs
    attrs = {"axis": axis, "epsilon": epsilon}
    if stash_type is not None:
        attrs["stash_type"] = stash_type
    if extra_attrs is not None:
        attrs.update(extra_attrs)
    graph = helper.make_graph(
        [helper.make_node("LayerNormalization", node_inputs, node_outputs,
                          name="layer_norm", **attrs)],
        "layer_norm_test",
        [helper.make_tensor_value_info("data", data_dtype, data_shape),
         helper.make_tensor_value_info("scale", scale_dtype, scale_shape),
         helper.make_tensor_value_info("bias", bias_dtype, bias_shape)],
        [helper.make_tensor_value_info(
            "out", output_dtype, data_shape if output_shape is None else output_shape)],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=6
    )


def test_layer_normalization_maps_exact_static_float32_contract():
    imported = import_onnx_model(_layer_normalization_model(axis=-2, epsilon=0.125))

    assert [(node.op_name, node.attrs) for node in imported.function.nodes] == [
        ("nn_layer_norm", {
            "axis": -2,
            "epsilon": pytest.approx(0.125),
            "accumulation_dtype": "float64",
        }),
    ]
    assert imported.function.outputs[0].shape == [2, 3, 4]
    assert imported.function.outputs[0].dtype == "float32"


@pytest.mark.parametrize("opset", [1, 16])
def test_layer_normalization_before_opset17_is_rejected(opset):
    with pytest.raises(UnsupportedONNXOpError, match=rf"LayerNormalization opset {opset}.*>= 17"):
        import_onnx_model(_layer_normalization_model(opset=opset))


@pytest.mark.parametrize(
    ("node_inputs", "message"),
    [
        (["data", "scale"], "exactly three non-empty inputs"),
        (["data", "scale", "bias", "extra"], "exactly three non-empty inputs"),
        (["data", "scale", ""], "exactly three non-empty inputs"),
    ],
)
def test_layer_normalization_rejects_missing_or_extra_inputs(node_inputs, message):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(_layer_normalization_model(node_inputs=node_inputs))


@pytest.mark.parametrize("node_outputs", [["out"], ["out", ""], ["out", "", ""]])
def test_layer_normalization_accepts_omitted_optional_output_slots(node_outputs):
    imported = import_onnx_model(_layer_normalization_model(node_outputs=node_outputs))

    assert imported.function.nodes[0].outputs == ["out"]


@pytest.mark.parametrize(
    ("node_outputs", "message"),
    [
        ([], "requires Y and at most two empty optional output slots"),
        ([""], "requires Y and at most two empty optional output slots"),
        (["out", "mean"], "non-empty Mean or InvStdDev"),
        (["out", "", "inv_std"], "non-empty Mean or InvStdDev"),
        (["out", "", "", ""], "requires Y and at most two empty optional output slots"),
    ],
)
def test_layer_normalization_rejects_requested_or_malformed_optional_outputs(
    node_outputs, message
):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(_layer_normalization_model(node_outputs=node_outputs))


@pytest.mark.parametrize("stash_type", [0, 2])
def test_layer_normalization_rejects_non_float32_stash_type(stash_type):
    with pytest.raises(ValueError, match="stash_type default/1"):
        import_onnx_model(_layer_normalization_model(stash_type=stash_type))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"axis": 1.5}, "axis.*exact INT type"),
        ({"epsilon": 1}, "epsilon.*exact FLOAT type"),
        ({"stash_type": 1.0}, "stash_type.*exact INT type"),
    ],
)
def test_layer_normalization_rejects_malformed_attribute_types(kwargs, message):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(_layer_normalization_model(**kwargs))


def test_layer_normalization_rejects_unsupported_attribute():
    with pytest.raises(ValueError, match="unsupported attribute"):
        import_onnx_model(_layer_normalization_model(extra_attrs={"unsupported": 1}))


def test_layer_normalization_rejects_non_float32_input_dtype():
    with pytest.raises(ValueError, match="requires float32 data, scale, and bias"):
        import_onnx_model(_layer_normalization_model(scale_dtype=TensorProto.INT32))


@pytest.mark.parametrize(
    ("data_shape", "scale_shape", "bias_shape", "message"),
    [
        ([], (), (), "data rank >= 1"),
        ((2, 3, 4), (4,), (3, 4), "scale and bias shapes must exactly equal"),
        ((2, 0, 4), (0, 4), (0, 4), "normalized suffix dimensions must be > 0"),
    ],
)
def test_layer_normalization_rejects_invalid_static_shapes(
    data_shape, scale_shape, bias_shape, message
):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(_layer_normalization_model(data_shape, scale_shape, bias_shape))


@pytest.mark.parametrize("axis", [3, -4])
def test_layer_normalization_rejects_out_of_range_axis(axis):
    with pytest.raises(ValueError, match="axis .* is out of range"):
        import_onnx_model(_layer_normalization_model(axis=axis))


@pytest.mark.parametrize("epsilon", [0.0, -1e-5, float("inf"), float("nan")])
def test_layer_normalization_rejects_nonpositive_or_nonfinite_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon must be finite and > 0"):
        import_onnx_model(_layer_normalization_model(epsilon=epsilon))


@pytest.mark.parametrize(
    ("output_shape", "output_dtype"),
    [
        ((2, 3, 5), TensorProto.FLOAT),
        ((2, 3, 4), TensorProto.DOUBLE),
    ],
)
def test_layer_normalization_rejects_declared_output_mismatch(output_shape, output_dtype):
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(
            _layer_normalization_model(output_shape=output_shape, output_dtype=output_dtype)
        )


def _slice_model(data_shape=(2, 3), *, starts=(-2,), ends=(99,), axes=(-1,), steps=(1,),
                 parameter_dtype=TensorProto.INT64, parameter_dtypes=None,
                 output_shape=(2, 2), output_dtype=TensorProto.FLOAT,
                 opset=13, dynamic_params=False):
    initializer = []
    parameter_inputs = []
    parameter_dtypes = {} if parameter_dtypes is None else parameter_dtypes
    values = {"starts": starts, "ends": ends, "axes": axes, "steps": steps}
    for name, value in values.items():
        if value is None:
            continue
        dtype = parameter_dtypes.get(name, parameter_dtype)
        if dynamic_params:
            parameter_inputs.append(helper.make_tensor_value_info(name, dtype, [len(value)]))
        else:
            initializer.append(helper.make_tensor(name, dtype, [len(value)], list(value)))
    node_inputs = ["data", "starts", "ends"]
    if axes is not None:
        node_inputs.append("axes")
    elif steps is not None:
        node_inputs.append("")
    if steps is not None:
        node_inputs.append("steps")
    graph = helper.make_graph(
        [helper.make_node("Slice", node_inputs, ["out"], name="slice")], "slice_test",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape)] + parameter_inputs,
        [helper.make_tensor_value_info("out", output_dtype, output_shape)], initializer=initializer,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=6)


def test_slice_initializer_mapping_clamping_and_int32_int64():
    for dtype in (TensorProto.INT32, TensorProto.INT64):
        imported = import_onnx_model(_slice_model(parameter_dtype=dtype))
        assert [(node.op_name, node.inputs, node.attrs) for node in imported.function.nodes] == [
            ("slice", ["data"], {"starts": [-2], "ends": [99], "axes": [-1], "steps": [1]})
        ]
        assert imported.function.outputs[0].shape == [2, 2]


def test_slice_rejects_mixed_control_initializer_integer_widths():
    with pytest.raises(ValueError, match="one consistent int32 or int64 dtype"):
        import_onnx_model(_slice_model(
            parameter_dtype=TensorProto.INT32,
            parameter_dtypes={"ends": TensorProto.INT64},
        ))


def test_slice_omitted_axes_and_steps_are_canonicalized():
    imported = import_onnx_model(_slice_model(starts=(0, -99), ends=(1, 99), axes=None,
                                               steps=None, output_shape=(1, 3)))
    assert imported.function.nodes[0].attrs == {
        "starts": [0, -99], "ends": [1, 99], "axes": [0, 1], "steps": [1, 1]
    }


def test_slice_steps_can_use_an_empty_optional_axes_slot():
    imported = import_onnx_model(_slice_model(starts=(0, 0), ends=(2, 3), axes=None,
                                               steps=(1, 1), output_shape=(2, 3)))
    assert imported.function.nodes[0].attrs == {
        "starts": [0, 0], "ends": [2, 3], "axes": [0, 1], "steps": [1, 1]
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"opset": 9}, "opset 9"),
        ({"dynamic_params": True}, "must be a static initializer"),
        ({"data_shape": (), "output_shape": ()}, "data rank >= 1"),
        ({"output_dtype": TensorProto.DOUBLE}, "declaration.*does not match inferred"),
        ({"starts": (), "ends": (), "axes": (), "steps": (), "output_shape": (2, 3)}, "nonempty rank-1"),
        ({"starts": (0,), "ends": (1, 2), "axes": (0,), "steps": (1,)}, "lengths must match"),
        ({"starts": (0, 0), "ends": (1, 1), "axes": (0, 0), "steps": (1, 1)}, "unique and in range"),
        ({"axes": (2,)}, "unique and in range"),
        ({"steps": (0,)}, r"equal exactly \+1"),
        ({"steps": (-1,)}, r"equal exactly \+1"),
        ({"steps": (2,)}, r"equal exactly \+1"),
    ],
)
def test_slice_rejects_exact_static_contract_violations(kwargs, message):
    with pytest.raises((ValueError, UnsupportedONNXOpError), match=message):
        import_onnx_model(_slice_model(**kwargs))


def test_slice_int64_min_is_clamped_without_overflow():
    imported = import_onnx_model(_slice_model(starts=(-2**63,), ends=(2**63 - 1,),
                                               output_shape=(2, 3)))
    assert imported.function.outputs[0].shape == [2, 3]


def onnx_import_metadata(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _s1_model(nodes, graph_inputs, outputs, initializers=(), value_infos=(), opset=17):
    graph = helper.make_graph(
        nodes, "s1_static_test", list(graph_inputs), list(outputs),
        initializer=list(initializers), value_info=list(value_infos),
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)


def _constant_model(tensor, *, output_name="c", attrs=None, value_infos=(),
                    extra_nodes=(), opset=17, output_dtype=TensorProto.FLOAT,
                    output_shape=None):
    attrs = {"value": tensor} if attrs is None else attrs
    nodes = [helper.make_node("Constant", [], [output_name], name="constant", **attrs)]
    nodes.extend(extra_nodes)
    if output_shape is None:
        output_shape = list(tensor.dims) if tensor is not None else []
    return _s1_model(nodes, [],
                     [helper.make_tensor_value_info(output_name, output_dtype, output_shape)],
                     value_infos=value_infos, opset=opset)


def _arith_model(op, a_shape, b_shape, *, a_dtype=TensorProto.FLOAT, b_dtype=TensorProto.FLOAT,
                 out_shape=None, out_dtype=TensorProto.FLOAT, attrs=None,
                 node_inputs=("a", "b"), opset=17, b_is_initializer=False):
    nodes = [helper.make_node(op, list(node_inputs), ["out"], name="arith", **(attrs or {}))]
    initializers = []
    graph_inputs = [helper.make_tensor_value_info("a", a_dtype, a_shape)]
    if b_is_initializer:
        initializers.append(helper.make_tensor("b", b_dtype, b_shape, [1] * max(1, _size(b_shape))))
    else:
        graph_inputs.append(helper.make_tensor_value_info("b", b_dtype, b_shape))
    outputs = [helper.make_tensor_value_info("out", out_dtype, out_shape if out_shape is not None else a_shape)]
    return _s1_model(nodes, graph_inputs, outputs, initializers=initializers, opset=opset)


def _size(shape):
    count = 1
    for dim in shape:
        count *= dim
    return count


def test_constant_normalizes_to_param_and_preserves_dtype():
    tensor = helper.make_tensor("ignored", TensorProto.INT64, [2], [7, -3])
    imported = import_onnx_model(_constant_model(tensor, output_dtype=TensorProto.INT64))

    assert imported.function.nodes == []
    assert imported.param_order == ["c"]
    constant = imported.params["c"]
    assert constant.shape == [2]
    assert constant.dtype == "int64"
    assert constant.data == numpy_helper.to_array(tensor).tobytes()
    assert imported.function.outputs[0].shape == [2]
    assert imported.function.outputs[0].dtype == "int64"


def test_constant_preserves_scalar_and_zero_size_semantics():
    imported = import_onnx_model(_constant_model(
        helper.make_tensor("ignored", TensorProto.FLOAT, [], [1.5])))
    assert imported.params["c"].shape == []
    assert imported.params["c"].dtype == "float32"

    imported = import_onnx_model(_constant_model(
        helper.make_tensor("ignored", TensorProto.FLOAT, [0], [])))
    assert imported.params["c"].shape == [0]
    assert imported.params["c"].data == b""


def test_constant_downstream_consumers_use_the_param_name():
    tensor = helper.make_tensor("ignored", TensorProto.FLOAT, [4], [1, 2, 4, 8])
    graph = _s1_model(
        [helper.make_node("Constant", [], ["c"], name="constant",
                          value=helper.make_tensor("ignored", TensorProto.FLOAT, [4], [1, 2, 4, 8])),
         helper.make_node("Mul", ["x", "c"], ["out"], name="mul")],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 4])],
    )
    imported = import_onnx_model(graph)

    assert [node.op_name for node in imported.function.nodes] == ["mul"]
    assert imported.function.nodes[0].inputs == ["x", "c"]
    assert imported.params["c"].data == numpy_helper.to_array(tensor).tobytes()
    assert imported.param_order == ["c"]


def test_constant_rejects_shortcut_sparse_and_multi_attribute_forms():
    with pytest.raises(ValueError, match="only supports the dense 'value' attribute"):
        import_onnx_model(_constant_model(None, attrs={"value_float": 1.0}))
    with pytest.raises(ValueError, match="only supports the dense 'value' attribute"):
        import_onnx_model(_constant_model(None, attrs={"sparse_value": onnx.SparseTensorProto()}))
    with pytest.raises(ValueError, match="exactly one value attribute"):
        import_onnx_model(_constant_model(None, attrs={"value_int": 1, "value_float": 2.0}))
    with pytest.raises(ValueError, match="exactly one value attribute"):
        import_onnx_model(_constant_model(None, attrs={}))


def test_constant_rejects_non_tensor_value_attribute():
    model = _constant_model(None, attrs={})
    attr = AttributeProto()
    attr.name = "value"
    attr.type = AttributeProto.INT
    attr.i = 3
    model.graph.node[0].attribute.append(attr)
    with pytest.raises(ValueError, match="'value' must have exact TENSOR type"):
        import_onnx_model(model)


def test_constant_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="unsupported dtype 'float16'"):
        import_onnx_model(_constant_model(
            helper.make_tensor("ignored", TensorProto.FLOAT16, [1], [1.0])))


def test_constant_rejects_duplicate_output_names():
    tensor = helper.make_tensor("ignored", TensorProto.FLOAT, [1], [1.0])
    model = _s1_model(
        [helper.make_node("Constant", [], ["c"], name="first", value=tensor),
         helper.make_node("Constant", [], ["c"], name="second", value=tensor)],
        [], [helper.make_tensor_value_info("c", TensorProto.FLOAT, [1])],
    )
    with pytest.raises(ValueError, match="conflicts with an existing graph input"):
        import_onnx_model(model)


def test_constant_rejects_initializer_and_graph_input_name_conflicts():
    tensor = helper.make_tensor("ignored", TensorProto.FLOAT, [1], [1.0])
    model = _s1_model(
        [helper.make_node("Constant", [], ["weight"], name="constant", value=tensor)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("weight", TensorProto.FLOAT, [1])],
        initializers=[helper.make_tensor("weight", TensorProto.FLOAT, [1], [2.0])],
    )
    with pytest.raises(ValueError, match="conflicts with an existing graph input"):
        import_onnx_model(model)

    model = _s1_model(
        [helper.make_node("Constant", [], ["x"], name="constant", value=tensor)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
    )
    with pytest.raises(ValueError, match="conflicts with an existing graph input"):
        import_onnx_model(model)



def _identity_model(*, source="w", alias="w_alias", extra_nodes=(), source_is_initializer=True,
                    node_inputs=None, node_outputs=None):
    """Identity aliasing a shared initializer, the shape torch.onnx.export emits."""
    # 用 is None 区分"未指定"与"显式空列表"，否则零输入的负例构造不出来。
    nodes = [helper.make_node("Identity",
                              [source] if node_inputs is None else list(node_inputs),
                              [alias] if node_outputs is None else list(node_outputs),
                              name="ident")]
    nodes.extend(extra_nodes)
    initializers = ([helper.make_tensor(source, TensorProto.FLOAT, [2], [1.5, 2.5])]
                    if source_is_initializer else [])
    graph_inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])]
    if not source_is_initializer:
        nodes.insert(0, helper.make_node("Mul", ["x", "x"], [source], name="produce"))
    return _s1_model(nodes, graph_inputs,
                     [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])]
                     if extra_nodes else
                     [helper.make_tensor_value_info(alias, TensorProto.FLOAT, [2])],
                     initializers=initializers)


def test_identity_aliases_a_deduped_initializer_into_a_param():
    imported = import_onnx_model(_identity_model())

    # 透传不产生 Relay 节点，也不产生拷贝 kernel。
    assert imported.function.nodes == []
    assert imported.param_order == ["w", "w_alias"]
    assert imported.params["w_alias"].shape == [2]
    assert imported.params["w_alias"].dtype == "float32"
    assert imported.params["w_alias"].data == imported.params["w"].data
    assert imported.params["w_alias"].name == "w_alias"


def test_identity_alias_is_consumable_by_downstream_nodes():
    imported = import_onnx_model(_identity_model(
        extra_nodes=[helper.make_node("Mul", ["x", "w_alias"], ["out"], name="mul")]))

    assert [node.op_name for node in imported.function.nodes] == ["mul"]
    assert imported.function.nodes[0].inputs == ["x", "w_alias"]
    assert imported.params["w_alias"].data == imported.params["w"].data


def test_identity_rejects_aliasing_a_computed_value():
    with pytest.raises(UnsupportedONNXOpError, match="is not an initializer or Constant"):
        import_onnx_model(_identity_model(source="t", source_is_initializer=False))


def test_identity_rejects_name_collisions():
    model = _s1_model(
        [helper.make_node("Identity", ["w"], ["x"], name="ident")],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        initializers=[helper.make_tensor("w", TensorProto.FLOAT, [2], [1.0, 2.0])],
    )
    with pytest.raises(ValueError, match="collides with an existing"):
        import_onnx_model(model)


@pytest.mark.parametrize("inputs,outputs", [([], ["a"]), (["w", "w"], ["a"]), (["w"], [""])])
def test_identity_rejects_malformed_arity(inputs, outputs):
    with pytest.raises(ValueError, match="Identity node"):
        import_onnx_model(_identity_model(node_inputs=inputs, node_outputs=outputs))


def test_identity_rejects_unresolved_input():
    model = _s1_model(
        [helper.make_node("Identity", ["missing"], ["a"], name="ident")],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2])],
    )
    with pytest.raises(ValueError, match="unresolved input"):
        import_onnx_model(model)


def test_constant_rejects_declared_value_info_mismatch():
    model = _constant_model(
        helper.make_tensor("ignored", TensorProto.FLOAT, [2], [1.0, 2.0]),
        value_infos=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [3])],
    )
    with pytest.raises(ValueError, match=r"output 'c' declaration.*does not match inferred"):
        import_onnx_model(model)


@pytest.mark.parametrize("op", ["Mul", "Sub", "Div"])
def test_arithmetic_maps_with_trailing_broadcast(op):
    imported = import_onnx_model(_arith_model(op, [2, 1, 3], [1, 4, 1], out_shape=[2, 4, 3]))

    assert [(node.op_name, node.attrs) for node in imported.function.nodes] == [
        ({"Mul": "mul", "Sub": "subtract", "Div": "divide"}[op], {}),
    ]
    assert imported.function.outputs[0].shape == [2, 4, 3]
    assert imported.function.outputs[0].dtype == "float32"


def test_add_accepts_int64_index_arithmetic():
    imported = import_onnx_model(
        _arith_model(
            "Add", [1, 3], [3], a_dtype=TensorProto.INT64,
            b_dtype=TensorProto.INT64, out_shape=[1, 3],
            out_dtype=TensorProto.INT64,
        )
    )

    assert [(node.op_name, node.attrs, node.inputs) for node in imported.function.nodes] == [
        ("add", {}, ["a", "b"]),
    ]
    assert imported.function.outputs[0].shape == [1, 3]
    assert imported.function.outputs[0].dtype == "int64"


def test_add_rejects_unverified_integer_dtypes():
    with pytest.raises(ValueError, match="requires float32 or int64 inputs"):
        import_onnx_model(
            _arith_model(
                "Add", [2, 2], [2, 2], a_dtype=TensorProto.INT32,
                b_dtype=TensorProto.INT32, out_dtype=TensorProto.INT32,
            )
        )


@pytest.mark.parametrize("op", ["Mul", "Sub", "Div"])
def test_arithmetic_rejects_attributes(op):
    with pytest.raises(ValueError, match="does not support attributes"):
        import_onnx_model(_arith_model(op, [2, 2], [2, 2], attrs={"axis": 0}))


@pytest.mark.parametrize("op", ["Mul", "Sub", "Div"])
def test_arithmetic_rejects_non_float32_inputs(op):
    with pytest.raises(ValueError, match="requires float32 inputs in the static S1 subset"):
        import_onnx_model(_arith_model(op, [2, 2], [2, 2],
                                       a_dtype=TensorProto.INT32, b_dtype=TensorProto.INT32))


@pytest.mark.parametrize("op", ["Mul", "Sub", "Div"])
def test_arithmetic_rejects_incompatible_broadcast(op):
    with pytest.raises(ValueError, match="incompatible broadcast dimensions"):
        import_onnx_model(_arith_model(op, [2, 3], [2, 4], out_shape=[2, 4]))


def test_arithmetic_preserves_zero_extent_broadcast():
    imported = import_onnx_model(_arith_model("Mul", [0, 3], [1, 3], out_shape=[0, 3]))

    assert imported.function.outputs[0].shape == [0, 3]


def test_arithmetic_rejects_declared_output_mismatch():
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(_arith_model("Mul", [2, 3], [2, 3], out_shape=[2, 4]))


def test_sqrt_maps_and_preserves_shape():
    imported = import_onnx_model(_s1_model(
        [helper.make_node("Sqrt", ["a"], ["out"], name="sqrt")],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])],
    ))

    assert [(node.op_name, node.attrs, node.inputs) for node in imported.function.nodes] == [
        ("sqrt", {}, ["a"]),
    ]
    assert imported.function.outputs[0].shape == [2, 3]


def test_sqrt_rejects_non_float32_and_attributes():
    with pytest.raises(ValueError, match="requires float32 input in the static S1 subset"):
        import_onnx_model(_s1_model(
            [helper.make_node("Sqrt", ["a"], ["out"], name="sqrt")],
            [helper.make_tensor_value_info("a", TensorProto.INT64, [2])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
        ))
    model = _s1_model(
        [helper.make_node("Sqrt", ["a"], ["out"], name="sqrt")],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
    )
    model.graph.node[0].attribute.extend([helper.make_attribute("axis", 0)])
    with pytest.raises(ValueError, match="does not support attributes"):
        import_onnx_model(model)


def test_arithmetic_rejects_missing_input_value():
    with pytest.raises(ValueError, match="input 'ghost' metadata is absent or unresolved"):
        import_onnx_model(_arith_model("Mul", [2, 3], [2, 3], node_inputs=("a", "ghost")))


def test_cast_int_to_float32_maps_to_relay_code_zero():
    for dtype in (TensorProto.INT32, TensorProto.INT64):
        imported = import_onnx_model(_s1_model(
            [helper.make_node("Cast", ["a"], ["out"], name="cast", to=TensorProto.FLOAT)],
            [helper.make_tensor_value_info("a", dtype, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])],
        ))

        assert [(node.op_name, node.attrs) for node in imported.function.nodes] == [
            ("cast", {"to": 0}),
        ]
        assert imported.function.outputs[0].dtype == "float32"
        assert imported.function.outputs[0].shape == [2, 3]


def test_cast_rejects_non_float32_targets():
    for dtype, name in ((TensorProto.INT32, "int32"), (TensorProto.BOOL, "bool"),
                        (TensorProto.DOUBLE, "float64")):
        with pytest.raises(ValueError, match="supports only to=float32"):
            import_onnx_model(_s1_model(
                [helper.make_node("Cast", ["a"], ["out"], name="cast", to=dtype)],
                [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2])],
                [helper.make_tensor_value_info("out", dtype, [2])],
            ))


def test_cast_accepts_the_identity_float32_conversion():
    """MiniMind 的 RMSNorm `.float()` 在已是 float32 的图上导出成恒等 Cast。"""
    imported = import_onnx_model(_s1_model(
        [helper.make_node("Cast", ["a"], ["out"], name="cast", to=TensorProto.FLOAT)],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
    ))

    assert imported.function.nodes[0].op_name == "cast"
    assert imported.function.outputs[0].dtype == "float32"


def test_cast_rejects_unsupported_source_dtypes():
    for dtype in (TensorProto.BOOL, TensorProto.INT8):
        with pytest.raises(ValueError, match="supports only int32/int64 to float32"):
            import_onnx_model(_s1_model(
                [helper.make_node("Cast", ["a"], ["out"], name="cast", to=TensorProto.FLOAT)],
                [helper.make_tensor_value_info("a", dtype, [2])],
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
            ))


def test_cast_rejects_missing_or_malformed_to_attribute():
    with pytest.raises(ValueError, match="requires exactly the 'to' attribute"):
        import_onnx_model(_s1_model(
            [helper.make_node("Cast", ["a"], ["out"], name="cast")],
            [helper.make_tensor_value_info("a", TensorProto.INT64, [2])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
        ))
    with pytest.raises(ValueError, match="'to' must have exact INT type"):
        import_onnx_model(_s1_model(
            [helper.make_node("Cast", ["a"], ["out"], name="cast", to=1.5)],
            [helper.make_tensor_value_info("a", TensorProto.INT64, [2])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
        ))


def test_cast_rejects_unsupported_dtype_enum_and_extra_attrs():
    with pytest.raises(ValueError, match="unsupported 'to' dtype enum 999"):
        import_onnx_model(_s1_model(
            [helper.make_node("Cast", ["a"], ["out"], name="cast", to=999)],
            [helper.make_tensor_value_info("a", TensorProto.INT64, [2])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
        ))
    model = _s1_model(
        [helper.make_node("Cast", ["a"], ["out"], name="cast", to=TensorProto.FLOAT)],
        [helper.make_tensor_value_info("a", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
    )
    model.graph.node[0].attribute.extend([helper.make_attribute("extra", 1)])
    with pytest.raises(ValueError, match="requires exactly the 'to' attribute"):
        import_onnx_model(model)


def test_cast_rejects_declared_output_mismatch():
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(_s1_model(
            [helper.make_node("Cast", ["a"], ["out"], name="cast", to=TensorProto.FLOAT)],
            [helper.make_tensor_value_info("a", TensorProto.INT64, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 4])],
        ))


def test_reduce_mean_maps_axes_keepdims_and_negative_axis():
    imported = import_onnx_model(_s1_model(
        [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce", axes=[1, -1],
                          keepdims=1)],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 1, 1])],
    ))

    assert [(node.op_name, node.attrs) for node in imported.function.nodes] == [
        ("reduce_mean", {"axes": [1, -1], "keepdims": 1}),
    ]
    assert imported.function.outputs[0].shape == [2, 1, 1]


def test_reduce_mean_keepdims_zero_prunes_axes():
    imported = import_onnx_model(_s1_model(
        [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce", axes=[-2],
                          keepdims=0)],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 4])],
    ))

    assert imported.function.nodes[0].attrs == {"axes": [-2], "keepdims": 0}
    assert imported.function.outputs[0].shape == [2, 4]


def test_reduce_mean_absent_or_empty_axes_reduce_all():
    imported = import_onnx_model(_s1_model(
        [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce", keepdims=1)],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])],
    ))
    assert imported.function.nodes[0].attrs == {"axes": [], "keepdims": 1}
    assert imported.function.outputs[0].shape == [1, 1]

    model = _s1_model(
        [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce")],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])],
    )
    attr = AttributeProto()
    attr.name = "axes"
    attr.type = AttributeProto.INTS
    model.graph.node[0].attribute.append(attr)
    imported = import_onnx_model(model)
    assert imported.function.outputs[0].shape == [1, 1]


def test_reduce_mean_opset18_input_form_is_rejected():
    with pytest.raises(UnsupportedONNXOpError, match=r"ReduceMean opset 18.*input-form"):
        import_onnx_model(_s1_model(
            [helper.make_node("ReduceMean", ["a", "axes"], ["out"], name="reduce")],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
             helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
            opset=18,
        ))


def test_reduce_mean_rejects_malformed_and_out_of_range_attrs():
    with pytest.raises(ValueError, match="keepdims must be 0 or 1"):
        import_onnx_model(_s1_model(
            [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce",
                              axes=[0], keepdims=2)],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])],
        ))
    with pytest.raises(ValueError, match="must have exact INTS type"):
        import_onnx_model(_s1_model(
            [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce", axes=0)],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])],
        ))
    with pytest.raises(ValueError, match="duplicate"):
        import_onnx_model(_s1_model(
            [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce",
                              axes=[1, 1], keepdims=1)],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 1])],
        ))
    with pytest.raises(ValueError, match="axis 2 is out of range for rank 2"):
        import_onnx_model(_s1_model(
            [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce",
                              axes=[2], keepdims=1)],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])],
        ))


def test_reduce_mean_rejects_non_float32_and_zero_extent_axes():
    with pytest.raises(ValueError, match="requires float32 input in the static S1 subset"):
        import_onnx_model(_s1_model(
            [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce",
                              axes=[0], keepdims=1)],
            [helper.make_tensor_value_info("a", TensorProto.INT64, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])],
        ))
    with pytest.raises(ValueError, match="zero-extent axis"):
        import_onnx_model(_s1_model(
            [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce",
                              axes=[1], keepdims=1)],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 0])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 1])],
        ))


def test_reduce_mean_rejects_unsupported_attrs_and_declared_mismatch():
    with pytest.raises(ValueError, match="unsupported attribute"):
        import_onnx_model(_s1_model(
            [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce",
                              axes=[0], keepdims=1, noop_with_empty_axes=0)],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])],
        ))
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(_s1_model(
            [helper.make_node("ReduceMean", ["a"], ["out"], name="reduce",
                              axes=[0], keepdims=1)],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])],
        ))


def test_reshape_resolves_initializer_shape_to_proven_target():
    imported = import_onnx_model(_s1_model(
        [helper.make_node("Reshape", ["a", "shape"], ["out"], name="reshape")],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [6, 4])],
        initializers=[helper.make_tensor("shape", TensorProto.INT64, [2], [6, 4])],
    ))

    assert [(node.op_name, node.inputs, node.attrs) for node in imported.function.nodes] == [
        ("reshape", ["a"], {"newshape": [6, 4], "allowzero": 0}),
    ]
    assert imported.function.outputs[0].shape == [6, 4]


def test_reshape_resolves_zero_and_minus_one_from_constant_shape():
    imported = import_onnx_model(_s1_model(
        [helper.make_node("Constant", [], ["shape"], name="shape_const",
                          value=helper.make_tensor("ignored", TensorProto.INT64, [3],
                                                   [0, -1, 2])),
         helper.make_node("Reshape", ["a", "shape"], ["out"], name="reshape")],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [3, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [3, 2, 2])],
    ))

    assert imported.function.nodes[0].attrs == {"newshape": [3, 2, 2], "allowzero": 0}
    assert imported.param_order == ["shape"]


def test_reshape_scalar_target_from_empty_shape_tensor():
    imported = import_onnx_model(_s1_model(
        [helper.make_node("Reshape", ["a", "shape"], ["out"], name="reshape")],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [])],
        initializers=[helper.make_tensor("shape", TensorProto.INT64, [0], [])],
    ))

    assert imported.function.nodes[0].attrs == {"newshape": [], "allowzero": 0}
    assert imported.function.outputs[0].shape == []


def test_reshape_rejects_ambiguous_or_inconsistent_targets():
    def reshape_model(raw_shape, data_shape=(2, 3), output_shape=(6,), shape_dtype=TensorProto.INT64):
        return _s1_model(
            [helper.make_node("Reshape", ["a", "shape"], ["out"], name="reshape")],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, list(data_shape))],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(output_shape))],
            initializers=[helper.make_tensor("shape", shape_dtype, [len(raw_shape)],
                                             list(raw_shape))],
        )

    with pytest.raises(ValueError, match="element count mismatch"):
        import_onnx_model(reshape_model([7]))
    with pytest.raises(ValueError, match="cannot prove a unique target shape"):
        import_onnx_model(reshape_model([0, -1], data_shape=(0, 3), output_shape=[0]))
    with pytest.raises(ValueError, match="allows at most one -1 dimension"):
        import_onnx_model(reshape_model([-1, -1]))
    with pytest.raises(ValueError, match="only supports positive, 0, and -1"):
        import_onnx_model(reshape_model([-2]))
    with pytest.raises(ValueError, match="must be int64"):
        import_onnx_model(reshape_model([6], shape_dtype=TensorProto.INT32))
    with pytest.raises(ValueError, match="must be a static initializer or Constant"):
        import_onnx_model(_s1_model(
            [helper.make_node("Reshape", ["a", "shape"], ["out"], name="reshape")],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
             helper.make_tensor_value_info("shape", TensorProto.INT64, [1])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [6])],
        ))


def test_reshape_rejects_allowzero_and_extra_attrs():
    with pytest.raises(ValueError, match="requires allowzero=0"):
        import_onnx_model(_s1_model(
            [helper.make_node("Reshape", ["a", "shape"], ["out"], name="reshape",
                              allowzero=1)],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [0, 3])],
            initializers=[helper.make_tensor("shape", TensorProto.INT64, [2], [0, 3])],
        ))
    with pytest.raises(ValueError, match="unsupported attribute"):
        import_onnx_model(_s1_model(
            [helper.make_node("Reshape", ["a", "shape"], ["out"], name="reshape",
                              unknown=1)],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [6])],
            initializers=[helper.make_tensor("shape", TensorProto.INT64, [1], [6])],
        ))


def test_reshape_rejects_declared_output_mismatch():
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(_s1_model(
            [helper.make_node("Reshape", ["a", "shape"], ["out"], name="reshape")],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [7])],
            initializers=[helper.make_tensor("shape", TensorProto.INT64, [1], [6])],
        ))


def test_reshape_int64_overflow_is_rejected():
    huge = (1 << 63) - 1
    with pytest.raises(ValueError, match="overflows int64"):
        import_onnx_model(_s1_model(
            [helper.make_node("Reshape", ["a", "shape"], ["out"], name="reshape")],
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [huge, huge])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
            initializers=[helper.make_tensor("shape", TensorProto.INT64, [1], [-1])],
        ))


# ---------------------------------------------------------------------------
# M4/M5 wave: Neg / Sigmoid (fieldless float32 unary ops, opset >= 13 form).
# ---------------------------------------------------------------------------


def _unary_math_model(op, shape, *, dtype=TensorProto.FLOAT,
                      output_dtype=TensorProto.FLOAT, output_shape=None, opset=17):
    return _s1_model(
        [helper.make_node(op, ["a"], ["out"], name=f"s1_{op.lower()}")],
        [helper.make_tensor_value_info("a", dtype, shape)],
        [helper.make_tensor_value_info("out", output_dtype, output_shape or shape)],
        opset=opset,
    )


@pytest.mark.parametrize("op,relay_op", [("Neg", "neg"), ("Sigmoid", "sigmoid"), ("Tanh", "tanh"), ("Erf", "erf")])
def test_unary_math_maps_and_preserves_shape(op, relay_op):
    imported = import_onnx_model(_unary_math_model(op, [2, 3]))

    assert [(node.op_name, node.attrs, node.inputs) for node in imported.function.nodes] == [
        (relay_op, {}, ["a"]),
    ]
    assert imported.function.outputs[0].shape == [2, 3]
    assert imported.function.outputs[0].dtype == "float32"


@pytest.mark.parametrize("op", ["Neg", "Sigmoid", "Tanh", "Erf"])
def test_unary_math_rejects_pre_opset13(op):
    with pytest.raises(UnsupportedONNXOpError, match="opset >= 13 form is required"):
        import_onnx_model(_unary_math_model(op, [2], opset=12))


@pytest.mark.parametrize("op", ["Neg", "Sigmoid", "Tanh", "Erf"])
def test_unary_math_rejects_non_float32(op):
    with pytest.raises(ValueError, match="requires float32 input in the M4/M5 static subset"):
        import_onnx_model(_unary_math_model(op, [2], dtype=TensorProto.INT64))


@pytest.mark.parametrize("op", ["Neg", "Sigmoid", "Tanh", "Erf"])
def test_unary_math_rejects_attributes(op):
    model = _unary_math_model(op, [2])
    model.graph.node[0].attribute.extend([helper.make_attribute("axis", 0)])
    with pytest.raises(ValueError, match="does not support attributes"):
        import_onnx_model(model)


@pytest.mark.parametrize("op", ["Neg", "Sigmoid", "Tanh", "Erf"])
def test_unary_math_rejects_declared_output_mismatch(op):
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(_unary_math_model(op, [2, 3], output_shape=[2, 4]))


@pytest.mark.parametrize("op", ["Tanh", "Erf"])
def test_vision_math_does_not_admit_unproven_shape_source_mode(op):
    with pytest.raises(UnsupportedONNXOpError, match="static import contract"):
        import_onnx_model(_unary_math_model(op, [2, 3]), preserve_shape_values=True)


@pytest.mark.parametrize("padding", ["VALID", "NOTSET", "SAME_UPPER", "SAME_LOWER"])
def test_conv_auto_padding_is_explicit_or_rejected(padding):
    import numpy as np
    model = helper.make_model(helper.make_graph([
        helper.make_node("Conv", ["x", "w"], ["y"], auto_pad=padding, strides=[2, 2]),
    ], "conv_padding", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 5, 5])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])],
        initializer=[numpy_helper.from_array(np.ones((1, 1, 2, 2), np.float32), name="w")]),
        opset_imports=[helper.make_opsetid("", 17)])
    if padding.startswith("SAME"):
        with pytest.raises(UnsupportedONNXOpError, match="requires explicit pads"):
            import_onnx_model(model)
    else:
        imported = import_onnx_model(model)
        assert imported.function.nodes[0].attrs["pads"] == [0, 0, 0, 0]
        if padding == "VALID":
            model.graph.node[0].attribute.append(helper.make_attribute("pads", [0, 0, 0, 0]))
            with pytest.raises(ValueError, match="cannot be combined"):
                import_onnx_model(model)


# ---------------------------------------------------------------------------
# M5 S2: Pow (fieldless float32 binary broadcast, opset >= 13 form).
# ---------------------------------------------------------------------------


def _pow_model(a_shape, b_shape, *, a_dtype=TensorProto.FLOAT, b_dtype=TensorProto.FLOAT,
               output_shape=(2, 3), output_dtype=TensorProto.FLOAT, opset=17):
    return _s1_model(
        [helper.make_node("Pow", ["a", "b"], ["out"], name="s2_pow")],
        [helper.make_tensor_value_info("a", a_dtype, a_shape),
         helper.make_tensor_value_info("b", b_dtype, b_shape)],
        [helper.make_tensor_value_info("out", output_dtype, output_shape)],
        opset=opset,
    )


def test_pow_maps_with_trailing_broadcast():
    imported = import_onnx_model(_pow_model([2, 1], [1, 3]))

    assert [(node.op_name, node.attrs, node.inputs) for node in imported.function.nodes] == [
        ("pow", {}, ["a", "b"]),
    ]
    assert imported.function.outputs[0].shape == [2, 3]
    assert imported.function.outputs[0].dtype == "float32"


def test_pow_rejects_pre_opset13():
    with pytest.raises(UnsupportedONNXOpError, match="opset >= 13 form is required"):
        import_onnx_model(_pow_model([2], [2], opset=12))


def test_pow_rejects_attributes():
    model = _pow_model([2, 3], [2, 3])
    model.graph.node[0].attribute.extend([helper.make_attribute("axis", 0)])
    with pytest.raises(ValueError, match="does not support attributes"):
        import_onnx_model(model)


@pytest.mark.parametrize(
    ("a_shape", "b_shape", "a_dtype", "b_dtype", "message"),
    [
        ([2, 3], [2, 3], TensorProto.INT32, TensorProto.INT32,
         "requires same-dtype float32 inputs in the M4/M5 static subset"),
        ([2, 3], [2, 3], TensorProto.DOUBLE, TensorProto.DOUBLE,
         "requires same-dtype float32 inputs in the M4/M5 static subset"),
        ([2, 3], [2, 3], TensorProto.FLOAT, TensorProto.INT64,
         "requires same-dtype float32 inputs in the M4/M5 static subset"),
        ([2, 3], [2, 4], TensorProto.FLOAT, TensorProto.FLOAT,
         "incompatible broadcast dimensions"),
    ],
)
def test_pow_rejects_invalid_static_contract(a_shape, b_shape, a_dtype, b_dtype, message):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(_pow_model(a_shape, b_shape, a_dtype=a_dtype, b_dtype=b_dtype))


def test_pow_rejects_declared_output_mismatch():
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(_pow_model([2, 3], [3], output_shape=[2, 4]))


# ---------------------------------------------------------------------------
# M4/M5: Expand (constant target shape control input, broadcast_to rules).
# ---------------------------------------------------------------------------


def _expand_model(data_shape, target, *, dtype=TensorProto.FLOAT,
                  output_shape=None, output_dtype=None, opset=17):
    return _s1_model(
        [helper.make_node("Expand", ["a", "shape"], ["out"], name="s1_expand")],
        [helper.make_tensor_value_info("a", dtype, data_shape)],
        [helper.make_tensor_value_info("out", output_dtype or dtype,
                                       output_shape or target)],
        initializers=[helper.make_tensor("shape", TensorProto.INT64, [len(target)],
                                         list(target))],
        opset=opset,
    )


def test_expand_maps_constant_target_and_preserves_dtype():
    imported = import_onnx_model(_expand_model([2, 1], [2, 3]))

    assert [(node.op_name, node.attrs, node.inputs) for node in imported.function.nodes] == [
        ("expand", {"target_shape": [2, 3]}, ["a"]),
    ]
    assert imported.function.outputs[0].shape == [2, 3]
    assert imported.function.outputs[0].dtype == "float32"


def test_expand_accepts_rank_raise_from_vector():
    imported = import_onnx_model(_expand_model([3], [2, 3]))

    assert imported.function.nodes[0].attrs == {"target_shape": [2, 3]}
    assert imported.function.outputs[0].shape == [2, 3]


def test_expand_rejects_dynamic_shape_input():
    # 目标 shape 来自前一个节点的输出（非 initializer/Constant）必须拒绝：
    # 动态 shape 输入由 M3 的形状值切片承接。
    graph = _s1_model(
        [helper.make_node("Expand", ["a", "dyn_shape"], ["out"], name="s1_expand")],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 1]),
         helper.make_tensor_value_info("dyn_shape", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])],
    )
    with pytest.raises(ValueError, match="must be a static initializer or Constant node output"):
        import_onnx_model(graph)


def test_expand_rejects_pre_opset13():
    with pytest.raises(UnsupportedONNXOpError, match="opset >= 13 form is required"):
        import_onnx_model(_expand_model([2, 1], [2, 3], opset=12))


def test_expand_rejects_attributes():
    model = _expand_model([2, 1], [2, 3])
    model.graph.node[0].attribute.extend([helper.make_attribute("axis", 0)])
    with pytest.raises(ValueError, match="does not support attributes"):
        import_onnx_model(model)


def test_expand_rejects_non_int64_shape_input():
    graph = _s1_model(
        [helper.make_node("Expand", ["a", "shape"], ["out"], name="s1_expand")],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 1])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])],
        initializers=[helper.make_tensor("shape", TensorProto.INT32, [2], [2, 3])],
    )
    with pytest.raises(ValueError, match="must be int64"):
        import_onnx_model(graph)


def test_expand_rejects_negative_target_dimensions():
    with pytest.raises(ValueError, match="must be non-negative"):
        import_onnx_model(_expand_model([2, 1], [-2, 3], output_shape=[2, 3]))


@pytest.mark.parametrize(
    ("data_shape", "target", "message"),
    [
        ([2, 3, 4], [3, 4], "must not exceed"),
        ([3], [2, 4], "must be 1 or equal to the target dimension"),
        ([2, 3], [2, 4], "must be 1 or equal to the target dimension"),
    ],
)
def test_expand_rejects_incompatible_contract(data_shape, target, message):
    with pytest.raises(ValueError, match=message):
        import_onnx_model(_expand_model(data_shape, target))


def test_expand_rejects_declared_output_mismatch():
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(_expand_model([2, 1], [2, 3], output_shape=[2, 4]))


# ---------------------------------------------------------------------------
# M4 S2: Unsqueeze normalized to the existing reshape (no new canonical op).
# ---------------------------------------------------------------------------


def _unsqueeze_model(data_shape, axes, *, dtype=TensorProto.FLOAT,
                     output_shape=None, opset=17):
    return _s1_model(
        [helper.make_node("Unsqueeze", ["a", "axes"], ["out"], name="s2_unsqueeze")],
        [helper.make_tensor_value_info("a", dtype, data_shape)],
        [helper.make_tensor_value_info("out", dtype, output_shape or [])],
        initializers=[helper.make_tensor("axes", TensorProto.INT64, [len(axes)],
                                         list(axes))],
        opset=opset,
    )


@pytest.mark.parametrize(
    ("data_shape", "axes", "expected"),
    [
        ([2, 3], [0], [1, 2, 3]),
        ([2, 3], [-1], [2, 3, 1]),
        ([3], [0, 2], [1, 3, 1]),
        ([2, 3, 4], [-5, 2], [1, 2, 1, 3, 4]),
    ],
)
def test_unsqueeze_normalizes_to_reshape(data_shape, axes, expected):
    imported = import_onnx_model(_unsqueeze_model(data_shape, axes, output_shape=expected))

    assert [(node.op_name, node.attrs, node.inputs) for node in imported.function.nodes] == [
        ("reshape", {"newshape": expected, "allowzero": 0}, ["a"]),
    ]
    assert imported.function.outputs[0].shape == expected
    assert imported.function.outputs[0].dtype == "float32"


def test_unsqueeze_rejects_pre_opset13():
    with pytest.raises(UnsupportedONNXOpError,
                       match="axes-input opset >= 13 form is required"):
        import_onnx_model(_unsqueeze_model([2, 3], [0], output_shape=[1, 2, 3], opset=12))


def test_unsqueeze_rejects_dynamic_axes_input():
    graph = _s1_model(
        [helper.make_node("Unsqueeze", ["a", "dyn_axes"], ["out"], name="s2_unsqueeze")],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("dyn_axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 3])],
    )
    with pytest.raises(ValueError, match="must be a static initializer or Constant node output"):
        import_onnx_model(graph)


def test_unsqueeze_rejects_duplicate_axes():
    with pytest.raises(ValueError, match="axes must be unique after normalization"):
        import_onnx_model(_unsqueeze_model([3], [0, -3], output_shape=[1, 1, 3]))


def test_unsqueeze_rejects_out_of_range_axis():
    with pytest.raises(ValueError, match="axis 3 is out of range for output rank 2"):
        import_onnx_model(_unsqueeze_model([2], [3], output_shape=[1, 2]))


def test_unsqueeze_rejects_non_int64_and_attributes():
    graph = _s1_model(
        [helper.make_node("Unsqueeze", ["a", "axes"], ["out"], name="s2_unsqueeze")],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 3])],
        initializers=[helper.make_tensor("axes", TensorProto.INT32, [1], [0])],
    )
    with pytest.raises(ValueError, match="must be int64"):
        import_onnx_model(graph)
    model = _unsqueeze_model([2, 3], [0], output_shape=[1, 2, 3])
    model.graph.node[0].attribute.extend([helper.make_attribute("axis", 0)])
    with pytest.raises(ValueError, match="does not support attributes"):
        import_onnx_model(model)


def test_unsqueeze_rejects_declared_output_mismatch():
    with pytest.raises(ValueError, match=r"output 'out' declaration.*does not match inferred"):
        import_onnx_model(_unsqueeze_model([2, 3], [0], output_shape=[2, 3, 1]))

# ---------------------------------------------------------------- 常量折叠

def _fold_model(nodes, inputs, outputs, initializers=(), opset=17):
    graph = helper.make_graph(nodes, "fold_test", inputs, outputs,
                              initializer=list(initializers))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                             ir_version=8)


def test_folding_evaluates_the_static_mask_subgraph():
    """ConstantOfShape -> Trilu 是导出器留下的纯静态 mask 构造。

    这两个算子都没有 Relay 映射，也不需要有：折叠后它们整体消失，只留下一个
    物化的常量，被保留的实算节点消费。
    """
    nodes = [
        helper.make_node("Constant", [], ["s"], name="shape_const",
                         value=helper.make_tensor("v", TensorProto.INT64, [2], [2, 2])),
        helper.make_node("Constant", [], ["k"], name="k_const",
                         value=helper.make_tensor("v", TensorProto.INT64, [], [1])),
        helper.make_node("ConstantOfShape", ["s"], ["filled"], name="cos",
                         value=helper.make_tensor("v", TensorProto.FLOAT, [1], [-3.0])),
        helper.make_node("Trilu", ["filled", "k"], ["mask"], name="trilu", upper=1),
        helper.make_node("Add", ["x", "mask"], ["out"], name="add"),
    ]
    model = _fold_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 2])],
    )
    folded, report = fold_static_subgraph(model)

    assert report.folded_nodes == 4
    assert report.remaining_nodes == 1
    assert [node.op_type for node in folded.graph.node] == ["Add"]
    assert report.materialized == ["mask"]
    mask = numpy_helper.to_array(
        next(i for i in folded.graph.initializer if i.name == "mask"))
    assert mask.tolist() == [[0.0, -3.0], [0.0, 0.0]]

    # 未映射的 ConstantOfShape/Trilu 折叠后不再出现，整图可导入。
    imported = import_onnx_model(model)
    assert [node.op_name for node in imported.function.nodes] == ["add"]


def test_folding_leaves_graph_input_dependent_nodes_alone():
    nodes = [
        helper.make_node("Constant", [], ["c"], name="c",
                         value=helper.make_tensor("v", TensorProto.FLOAT, [2], [1.0, 2.0])),
        helper.make_node("Mul", ["x", "c"], ["out"], name="mul"),
    ]
    model = _fold_model(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
    )
    folded, report = fold_static_subgraph(model)

    assert report.folded_nodes == 1 and report.remaining_nodes == 1
    assert [node.op_type for node in folded.graph.node] == ["Mul"]


def test_folding_keeps_nodes_that_produce_graph_outputs():
    """图输出必须由节点产出，不能被折成 initializer。"""
    nodes = [
        helper.make_node("Constant", [], ["c"], name="c",
                         value=helper.make_tensor("v", TensorProto.FLOAT, [2], [1.0, 2.0])),
        helper.make_node("Sqrt", ["c"], ["out"], name="sqrt"),
    ]
    model = _fold_model(
        nodes, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
    )
    folded, report = fold_static_subgraph(model)

    assert "Sqrt" in [node.op_type for node in folded.graph.node]
    assert report.materialized == ["c"]


def test_folding_refuses_to_exceed_the_byte_budget():
    nodes = [
        helper.make_node("Constant", [], ["s"], name="s",
                         value=helper.make_tensor("v", TensorProto.INT64, [2], [256, 256])),
        helper.make_node("ConstantOfShape", ["s"], ["big"], name="cos",
                         value=helper.make_tensor("v", TensorProto.FLOAT, [1], [1.0])),
        helper.make_node("Add", ["x", "big"], ["out"], name="add"),
    ]
    model = _fold_model(
        nodes, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [256, 256])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [256, 256])],
    )
    with pytest.raises(ConstantFoldingError, match="over the .*-byte budget"):
        fold_static_subgraph(model, byte_budget=1024)


def test_folding_can_be_disabled_and_then_the_raw_op_is_rejected():
    nodes = [
        helper.make_node("Constant", [], ["s"], name="s",
                         value=helper.make_tensor("v", TensorProto.INT64, [2], [2, 2])),
        helper.make_node("ConstantOfShape", ["s"], ["mask"], name="cos",
                         value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.0])),
        helper.make_node("Add", ["x", "mask"], ["out"], name="add"),
    ]
    model = _fold_model(
        nodes, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 2])],
    )
    assert [n.op_name for n in import_onnx_model(model).function.nodes] == ["add"]
    with pytest.raises(UnsupportedONNXOpError, match="ConstantOfShape"):
        import_onnx_model(model, fold_constants=False)


def test_folding_only_materializes_the_frontier():
    """纯中间静态值不进 initializer，只物化被保留节点消费的那一层。"""
    nodes = [
        helper.make_node("Constant", [], ["c"], name="c",
                         value=helper.make_tensor("v", TensorProto.FLOAT, [2], [4.0, 9.0])),
        helper.make_node("Sqrt", ["c"], ["mid"], name="sqrt"),
        helper.make_node("Sqrt", ["mid"], ["root"], name="sqrt2"),
        helper.make_node("Mul", ["x", "root"], ["out"], name="mul"),
    ]
    model = _fold_model(
        nodes, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
    )
    folded, report = fold_static_subgraph(model)

    assert report.materialized == ["root"]
    names = {i.name for i in folded.graph.initializer}
    assert "mid" not in names and "c" not in names


def _head_shape_source_model():
    return _s1_model([
        helper.make_node("Constant", [], ["ib"], value=helper.make_tensor("v", TensorProto.INT64, [], [-3])),
        helper.make_node("Shape", ["x"], ["shape"], name="shape"),
        helper.make_node("Gather", ["shape", "ib"], ["b"], axis=0),
        helper.make_node("Gather", ["shape", "is"], ["s"], axis=0),
        helper.make_node("Unsqueeze", ["b", "axes"], ["bv"]),
        helper.make_node("Unsqueeze", ["s", "axes"], ["sv"]),
        helper.make_node("Concat", ["bv", "sv", "h", "d"], ["target"], axis=0),
        helper.make_node("Reshape", ["x", "target"], ["heads"]),
    ], [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])],
       [helper.make_tensor_value_info("heads", TensorProto.FLOAT, [1, 4, 2, 4])],
       initializers=[helper.make_tensor("is", TensorProto.INT32, [], [-2]),
                     helper.make_tensor("axes", TensorProto.INT64, [1], [0]),
                     helper.make_tensor("h", TensorProto.INT64, [1], [2]),
                     helper.make_tensor("d", TensorProto.INT64, [1], [4])])


def test_shape_source_preserves_scalar_ranks_and_original_shape_dependencies():
    import numpy as np
    from onnx.reference import ReferenceEvaluator

    model = _head_shape_source_model()
    folded, _ = fold_static_subgraph(model)
    assert next(list(t.dims) for t in folded.graph.initializer if t.name == "ib") == []
    data = np.arange(32, dtype=np.float32).reshape(1, 4, 8)
    original = ReferenceEvaluator(model).run(None, {"x": data})[0]
    actual = ReferenceEvaluator(folded).run(None, {"x": data})[0]
    np.testing.assert_array_equal(original, actual)
    assert actual.shape == (1, 4, 2, 4)

    imported = import_onnx_model(model, preserve_shape_values=True)
    assert to_json_dict(imported)["format"] == "kxc.onnx_shape_source.v1"
    assert imported.params["ib"].shape == imported.params["is"].shape == []
    assert imported.params["is"].dtype == "int32"
    assert len(imported.function.nodes) == 9
    assert sum(n.op_name == "concatenate" for n in imported.function.nodes) == 3
    assert sum(n.op_name == "unsqueeze" for n in imported.function.nodes) == 2
    target = imported.function.nodes[-1]
    assert (target.op_name, target.inputs, target.attrs) == ("reshape_dynamic", ["x", "target"], {})
    assert imported.function.outputs[0].shape == [1, 4, 2, 4]
    # The default static import does not silently opt into source semantics.
    with pytest.raises(ValueError, match="must be a static initializer or Constant"):
        import_onnx_model(model)


def _gqa_shape_source_model():
    initializers = [helper.make_tensor(f"i{axis}", TensorProto.INT64, [], [axis]) for axis in range(4)]
    initializers += [helper.make_tensor("axis0", TensorProto.INT64, [1], [0]),
                     helper.make_tensor("axis3", TensorProto.INT64, [1], [3]),
                     helper.make_tensor("repeat_vector", TensorProto.INT64, [1], [2]),
                     helper.make_tensor("repeat", TensorProto.INT64, [], [2]),
                     helper.make_tensor("flat", TensorProto.INT64, [1], [-1]),
                     helper.make_tensor("negative", TensorProto.INT64, [], [-1])]
    nodes = [helper.make_node("Shape", ["x"], ["shape"])]
    for axis in range(4):
        nodes += [helper.make_node("Gather", ["shape", f"i{axis}"], [f"d{axis}"], axis=0),
                  helper.make_node("Unsqueeze", [f"d{axis}", "axis0"], [f"v{axis}"])]
    nodes += [
        helper.make_node("Concat", ["v0", "v1", "v2", "repeat_vector", "v3"], ["target"], axis=0),
        helper.make_node("Reshape", ["target", "flat"], ["flat_target"]),
        helper.make_node("Shape", ["flat_target"], ["length"]),
        helper.make_node("ConstantOfShape", ["length"], ["ones"], name="fill",
                         value=helper.make_tensor("fill", TensorProto.INT64, [1], [1])),
        helper.make_node("Mul", ["ones", "negative"], ["marker"]),
        helper.make_node("Equal", ["flat_target", "marker"], ["condition"]),
        helper.make_node("Where", ["condition", "ones", "flat_target"], ["selected"]),
        helper.make_node("Unsqueeze", ["x", "axis3"], ["input5"]),
        helper.make_node("Expand", ["input5", "selected"], ["expanded"], name="expand"),
        helper.make_node("Mul", ["d2", "repeat"], ["heads"]),
        helper.make_node("Unsqueeze", ["heads", "axis0"], ["head_vector"]),
        helper.make_node("Concat", ["v0", "v1", "head_vector", "v3"], ["output_shape"], axis=0),
        helper.make_node("Reshape", ["expanded", "output_shape"], ["out"]),
    ]
    return _s1_model(nodes, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 4, 3])], initializers=initializers)


def test_gqa_shape_source_retains_controls_and_unresolved_intermediate_extents():
    import numpy as np
    from onnx.reference import ReferenceEvaluator

    model = _gqa_shape_source_model()
    data = np.arange(24, dtype=np.float32).reshape(1, 4, 2, 3)
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, {"x": data})[0],
                                  np.repeat(data, 2, axis=2))
    source = import_onnx_model(model, preserve_shape_values=True, fold_constants=False)
    assert to_json_dict(source)["format"] == "kxc.onnx_shape_source.v1"
    fill = next(node for node in source.function.nodes if node.name == "fill")
    assert fill.op_name == "constant_of_shape" and fill.attrs == {"dtype_code": 2, "value": 1}
    expand = next(node for node in source.function.nodes if node.name == "expand")
    assert (expand.op_name, expand.inputs, expand.attrs) == ("expand_dynamic", ["input5", "selected"], {})
    assert sum(node.op_name == "where" for node in source.function.nodes) == 1
    assert sum(node.op_name == "reshape_dynamic" for node in source.function.nodes) == 2
    assert source.function.outputs[0].shape == [1, 4, 4, 3]


@pytest.mark.parametrize("dtype,shape,value", [
    (TensorProto.DOUBLE, [1], 1.0), (TensorProto.INT64, [], 1),
    (TensorProto.INT64, [1], 2**53 + 1),
])
def test_gqa_source_rejects_unrepresentable_or_wrong_fill(dtype, shape, value):
    model = _gqa_shape_source_model()
    fill = next(node for node in model.graph.node if node.name == "fill")
    fill.attribute[0].t.CopyFrom(helper.make_tensor("bad", dtype, shape, [value]))
    # Test the importer boundary directly; ONNX's strict inference may reject
    # a changed fill dtype even before this boundary is reached.
    from kxc_onnx.importer import _shape_source_fill_attrs
    with pytest.raises(ValueError, match="exactly representable int64"):
        _shape_source_fill_attrs(fill)


def test_gqa_source_rejects_fill_attrs_and_expand_attrs():
    from kxc_onnx.importer import _shape_source_fill_attrs
    model = _gqa_shape_source_model()
    fill = next(node for node in model.graph.node if node.name == "fill")
    fill.attribute.add().CopyFrom(fill.attribute[0])
    with pytest.raises(ValueError, match="explicit tensor value"):
        _shape_source_fill_attrs(fill)
    model = _gqa_shape_source_model()
    expand = next(node for node in model.graph.node if node.name == "expand")
    expand.attribute.append(helper.make_attribute("unexpected", 1))
    with pytest.raises(ValueError, match="Expand requires"):
        import_onnx_model(model, preserve_shape_values=True)


def test_shape_source_concat_generated_names_cannot_shadow_initializers():
    model = _head_shape_source_model()
    model.graph.initializer.append(helper.make_tensor("target__kxc_concat_0", TensorProto.INT64, [], [7]))
    imported = import_onnx_model(model, preserve_shape_values=True)
    outputs = [v for node in imported.function.nodes for v in node.outputs]
    assert "target__kxc_concat_0" not in outputs
    assert len(outputs) == len(set(outputs))
    assert imported.function.nodes[-1].inputs == ["x", "target"]


@pytest.mark.parametrize("attribute", ["shape_slice", "allowzero"])
def test_shape_source_rejects_control_attributes_outside_its_subset(attribute):
    model = _head_shape_source_model()
    if attribute == "shape_slice":
        model.graph.node[1].attribute.append(helper.make_attribute("start", 0))
    else:
        model.graph.node[-1].attribute.append(helper.make_attribute("allowzero", 1))
    with pytest.raises(ValueError, match="Shape"):
        import_onnx_model(model, preserve_shape_values=True)


def test_shape_source_requires_an_explicit_concrete_representative():
    model = _head_shape_source_model()
    dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    dim.ClearField("dim_value")
    dim.dim_param = "B"
    with pytest.raises(ValueError, match="Unresolved ONNX dimension"):
        import_onnx_model(model, preserve_shape_values=True)


def test_shape_import_full_dimensions_keeps_static_format():
    model = _s1_model([helper.make_node("Shape", ["x"], ["shape"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 0, 3])],
        [helper.make_tensor_value_info("shape", TensorProto.INT64, [3])])
    imported = import_onnx_model(model)
    assert to_json_dict(imported)["format"] == "kxc.onnx_import.v1"
    assert imported.function.nodes[0].op_name == "shape_of"
    assert imported.function.outputs[0].shape == [3]


def test_shape_source_flag_must_be_explicit_bool():
    with pytest.raises(ValueError, match="preserve_shape_values must be bool"):
        import_onnx_model(_head_shape_source_model(), preserve_shape_values=1)


def _rope_shape_source_model():
    constants = [helper.make_tensor(name, TensorProto.INT64, shape, [value])
        for name, shape, value in [("last", [], -1), ("two", [], 2), ("axis0", [1], 0),
                                  ("axis3", [1], -1), ("zero", [1], 0),
                                  ("end", [1], 2**63-1), ("step", [1], 1)]]
    nodes = [
        helper.make_node("Shape", ["x"], ["shape"]),
        helper.make_node("Gather", ["shape", "last"], ["width"], axis=0),
        helper.make_node("Div", ["width", "two"], ["half"]),
        helper.make_node("Cast", ["half"], ["cast"], to=TensorProto.INT64),
        helper.make_node("Unsqueeze", ["cast", "axis0"], ["halfv"]),
        helper.make_node("Slice", ["x", "halfv", "end", "axis3", "step"], ["tail"], name="tail"),
        helper.make_node("Neg", ["tail"], ["negative"]),
        helper.make_node("Slice", ["x", "zero", "halfv", "axis3", "step"], ["head"], name="head"),
        helper.make_node("Concat", ["negative", "head"], ["rotated"], axis=-1),
        helper.make_node("Mul", ["x", "cos"], ["xc"]),
        helper.make_node("Mul", ["rotated", "sin"], ["rs"]),
        helper.make_node("Add", ["xc", "rs"], ["out"]),
    ]
    return _s1_model(nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1,4,2,6]),
         helper.make_tensor_value_info("cos", TensorProto.FLOAT, [4,1,6]),
         helper.make_tensor_value_info("sin", TensorProto.FLOAT, [4,1,6])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1,4,2,6])], constants)


def test_rope_source_preserves_proved_slice_controls_and_int64_cast():
    import numpy as np
    from onnx.reference import ReferenceEvaluator
    model = _rope_shape_source_model()
    x = np.arange(48, dtype=np.float32).reshape(1,4,2,6) / 17
    c = np.cos(np.arange(24, dtype=np.float32)).reshape(4,1,6)
    s = np.sin(np.arange(24, dtype=np.float32)).reshape(4,1,6)
    expected = x*c + np.concatenate((-x[...,3:], x[...,:3]), axis=-1)*s
    np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, {"x":x,"cos":c,"sin":s})[0], expected)
    source = import_onnx_model(model, preserve_shape_values=True, fold_constants=False)
    assert to_json_dict(source)["format"] == "kxc.onnx_shape_source.v1"
    slices = [node for node in source.function.nodes if node.op_name == "slice"]
    assert len(slices) == 2 and all(len(node.inputs) == 5 and node.attrs == {} for node in slices)
    assert slices[0].inputs[1] == slices[1].inputs[2] == "halfv"
    assert next(node for node in source.function.nodes if node.op_name == "cast").attrs == {"to":2}
    assert "halfv" not in source.params
    assert source.function.outputs[0].shape == [1,4,2,6]
    with pytest.raises(ValueError, match="float32"):
        import_onnx_model(model, fold_constants=False)


@pytest.mark.parametrize("change", ["attrs", "missing_step", "int32", "scalar"])
def test_rope_source_rejects_invalid_slice_source_controls(change):
    model = _rope_shape_source_model()
    node = next(node for node in model.graph.node if node.name == "tail")
    if change == "attrs":
        node.attribute.append(helper.make_attribute("unexpected", 1))
    elif change == "missing_step":
        del node.input[-1]
    else:
        tensor = next(tensor for tensor in model.graph.initializer if tensor.name == "step")
        tensor.CopyFrom(helper.make_tensor("step", TensorProto.INT32 if change == "int32" else TensorProto.INT64,
                                          [1] if change == "int32" else [], [1]))
    with pytest.raises((ValueError, onnx.shape_inference.InferenceError), match="Slice|slice"):
        import_onnx_model(model, preserve_shape_values=True, fold_constants=False)


def test_unresolved_slice_does_not_bypass_source_cast_dtype_boundary():
    model = _rope_shape_source_model()
    del model.graph.node[6:]
    model.graph.node.append(helper.make_node("Cast", ["tail"], ["out"], to=TensorProto.INT64))
    model.graph.output[0].CopyFrom(helper.make_tensor_value_info("out", TensorProto.INT64, [1,4,2,3]))
    with pytest.raises(ValueError, match="Shape-source Cast"):
        import_onnx_model(model, preserve_shape_values=True, fold_constants=False)


def _causal_mask_source_model():
    nodes = [
        helper.make_node("Shape", ["x"], ["shape"]),
        helper.make_node("Gather", ["shape", "axis"], ["sequence"], axis=0),
        helper.make_node("Concat", ["sequence", "sequence"], ["target"], axis=0),
        helper.make_node("ConstantOfShape", ["target"], ["fill"],
            value=helper.make_tensor("value", TensorProto.FLOAT, [1], [-float("inf")])),
        helper.make_node("Trilu", ["fill", "k"], ["mask"], upper=1),
    ]
    graph = helper.make_graph(nodes, "dynamic_causal_mask",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 6])],
        [helper.make_tensor_value_info("mask", TensorProto.FLOAT, [4, 4])],
        initializer=[helper.make_tensor("axis", TensorProto.INT64, [1], [1]),
                     helper.make_tensor("k", TensorProto.INT64, [], [1])])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_causal_mask_source_preserves_infinity_and_unresolved_shape():
    import json
    from kxc_onnx.spec import to_json_dict
    source = import_onnx_model(_causal_mask_source_model(), preserve_shape_values=True)
    fill = next(node for node in source.function.nodes if node.op_name == "constant_of_shape")
    triangle = next(node for node in source.function.nodes if node.op_name == "trilu")
    assert fill.attrs == {"dtype_code": 0, "value_bits": 0xff800000}
    assert triangle.inputs == ["fill"] and triangle.attrs == {"upper": 1, "k": 1}
    encoded = json.dumps(to_json_dict(source), allow_nan=False)
    assert "Infinity" not in encoded and "NaN" not in encoded


def _prefill_entry_source_model():
    nodes = [
        helper.make_node("Shape", ["ids"], ["dimensions"]),
        helper.make_node("Gather", ["dimensions", "one"], ["length"], axis=0),
        helper.make_node("Slice", ["positions", "zero", "length", "zero", "one"], ["prefix"]),
        helper.make_node("Unsqueeze", ["prefix", "zero"], ["position"]),
        helper.make_node("Gather", ["vocabulary", "ids"], ["embedding"], axis=0),
        helper.make_node("Add", ["position", "embedding"], ["hidden"]),
        helper.make_node("Sigmoid", ["hidden"], ["activation"]),
        helper.make_node("Pow", ["activation", "power"], ["square"]),
        helper.make_node("ReduceMean", ["square"], ["mean"], axes=[-1], keepdims=1),
        helper.make_node("Sqrt", ["mean"], ["output"]),
        helper.make_node("Shape", ["output"], ["output_shape"]),
    ]
    graph = helper.make_graph(nodes, "prefill_entry",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1]),
         helper.make_tensor_value_info("output_shape", TensorProto.INT64, [3])],
        initializer=[helper.make_tensor("zero", TensorProto.INT64, [1], [0]),
                     helper.make_tensor("one", TensorProto.INT64, [1], [1]),
                     helper.make_tensor("power", TensorProto.FLOAT, [], [2.0]),
                     helper.make_tensor("positions", TensorProto.FLOAT, [8, 4], list(range(32))),
                     helper.make_tensor("vocabulary", TensorProto.FLOAT, [7, 4], list(range(28)))])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_prefill_entry_keeps_prefix_and_downstream_unknown_extents():
    model = _prefill_entry_source_model()
    source = import_onnx_model(model, preserve_shape_values=True, fold_constants=False)
    nodes = {node.outputs[0]: node for node in source.function.nodes}
    assert nodes["prefix"].op_name == "slice"
    assert nodes["prefix"].inputs == ["positions", "zero", "length", "zero", "one"]
    assert nodes["prefix"].attrs == {} and "length" not in source.params
    assert nodes["position"].op_name == "unsqueeze" and nodes["position"].attrs == {"axes": [0]}
    assert nodes["activation"].op_name == "sigmoid" and nodes["square"].op_name == "pow"
    assert nodes["mean"].op_name == "reduce_mean" and nodes["output"].op_name == "sqrt"
    assert nodes["output_shape"].op_name == "shape_of"
    assert nodes["embedding"].inputs == ["vocabulary", "ids"]


def _decode_window_source_model():
    nodes = [
        helper.make_node("Shape", ["past"], ["past_shape"]),
        helper.make_node("Gather", ["past_shape", "one"], ["length"], axis=0),
        helper.make_node("Add", ["length", "one"], ["total"]),
        helper.make_node("Slice", ["positions", "length", "total", "zero", "one"], ["window"]),
        helper.make_node("Concat", ["past", "token"], ["present"], axis=1),
        helper.make_node("Shape", ["present"], ["present_shape"]),
        helper.make_node("Gather", ["present_shape", "one"], ["present_length"], axis=0),
        helper.make_node("Sub", ["present_length", "one"], ["previous"]),
    ]
    graph = helper.make_graph(nodes, "decode_window",
        [helper.make_tensor_value_info("past", TensorProto.FLOAT, [1, 4, 4]),
         helper.make_tensor_value_info("token", TensorProto.FLOAT, [1, 1, 4])],
        [helper.make_tensor_value_info("window", TensorProto.FLOAT, [1, 4]),
         helper.make_tensor_value_info("present", TensorProto.FLOAT, [1, 5, 4]),
         helper.make_tensor_value_info("previous", TensorProto.INT64, [1])],
        initializer=[helper.make_tensor("zero", TensorProto.INT64, [1], [0]),
                     helper.make_tensor("one", TensorProto.INT64, [1], [1]),
                     helper.make_tensor("positions", TensorProto.FLOAT, [9, 4], list(range(36)))])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_decode_window_keeps_add_sub_and_actual_shape_sources():
    source = import_onnx_model(_decode_window_source_model(), preserve_shape_values=True, fold_constants=False)
    nodes = {node.outputs[0]: node for node in source.function.nodes}
    assert nodes["window"].inputs == ["positions", "length", "total", "zero", "one"]
    assert nodes["total"].op_name == "add" and nodes["previous"].op_name == "subtract"
    assert nodes["previous"].inputs == ["present_length", "one"] and nodes["previous"].attrs == {}
    assert nodes["present_shape"].op_name == "shape_of"
    assert not {"length", "total", "previous", "present_length"} & source.params.keys()


@pytest.mark.parametrize("op_type", ["Add", "Sub"])
def test_decode_shape_arithmetic_still_rejects_unknown_attrs(op_type):
    model = _decode_window_source_model()
    node = next(node for node in model.graph.node if node.op_type == op_type)
    node.attribute.append(helper.make_attribute("extra", 1))
    with pytest.raises((ValueError, onnx.shape_inference.InferenceError), match="attribute|Attribute"):
        import_onnx_model(model, preserve_shape_values=True, fold_constants=False)


def test_int64_subtract_still_requires_explicit_shape_source_mode():
    graph = helper.make_graph([helper.make_node("Sub", ["a", "b"], ["out"])], "integer_subtract",
        [helper.make_tensor_value_info(name, TensorProto.INT64, [1]) for name in ("a", "b")],
        [helper.make_tensor_value_info("out", TensorProto.INT64, [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    with pytest.raises(ValueError, match="float32"):
        import_onnx_model(model, fold_constants=False)


@pytest.mark.parametrize("op_type", ["Sigmoid", "Pow", "Sqrt", "ReduceMean", "Unsqueeze"])
def test_deferred_prefill_math_still_rejects_unknown_attrs(op_type):
    model = _prefill_entry_source_model()
    node = next(node for node in model.graph.node if node.op_type == op_type)
    node.attribute.append(helper.make_attribute("extra", 1))
    with pytest.raises((ValueError, onnx.shape_inference.InferenceError), match="attribute|Attribute"):
        import_onnx_model(model, preserve_shape_values=True, fold_constants=False)


@pytest.mark.parametrize("value,bits", [(0.0, 0), (-0.0, 0x80000000),
    (float("inf"), 0x7f800000), (-float("inf"), 0xff800000), (1.25, 0x3fa00000)])
def test_float_shape_fill_uses_exact_ieee_json_payload(value, bits):
    from kxc_onnx.importer import _shape_source_fill_attrs
    node = helper.make_node("ConstantOfShape", ["shape"], ["fill"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [1], [value]))
    assert _shape_source_fill_attrs(node) == {"dtype_code": 0, "value_bits": bits}


def test_float_shape_fill_rejects_nan():
    from kxc_onnx.importer import _shape_source_fill_attrs
    node = helper.make_node("ConstantOfShape", ["shape"], ["fill"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [1], [float("nan")]))
    with pytest.raises(ValueError):
        _shape_source_fill_attrs(node)


@pytest.mark.parametrize("upper,k", [(0, -2), (1, 1), (1, -(2**63)), (0, 2**63-1)])
def test_trilu_static_scalar_diagonal(upper, k):
    model = _causal_mask_source_model()
    model.graph.node[-1].attribute[0].i = upper
    model.graph.initializer[-1].CopyFrom(helper.make_tensor("k", TensorProto.INT64, [], [k]))
    imported = import_onnx_model(model, preserve_shape_values=True)
    assert imported.function.nodes[-1].attrs == {"upper": upper, "k": k}


@pytest.mark.parametrize("mode", ["vector", "int32", "runtime", "upper", "attrs"])
def test_trilu_rejects_unproved_diagonal_or_attrs(mode):
    from kxc_onnx.importer import _trilu_attrs
    model = _causal_mask_source_model()
    imported = import_onnx_model(model, preserve_shape_values=True)
    node = model.graph.node[-1]
    params = dict(imported.params)
    if mode in {"vector", "int32"}:
        from kxc_onnx.spec import ParamTensor
        old = params["k"]
        params["k"] = ParamTensor("k", [1] if mode == "vector" else [],
            "int32" if mode == "int32" else "int64", old.data)
    elif mode == "runtime":
        params.pop("k")
    elif mode == "upper":
        node.attribute[0].i = 2
    else:
        node.attribute.append(helper.make_attribute("extra", 1))
    with pytest.raises(ValueError, match="Trilu"):
        _trilu_attrs(node, params, 17)


@pytest.mark.parametrize("constant_data,k", [(False, None), (False, -2), (True, 1)])
def test_trilu_static_import_input_and_initializer_metadata(constant_data, k):
    inputs = [] if constant_data else [helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3])]
    initializers = ([helper.make_tensor("data", TensorProto.FLOAT, [2, 3], list(range(6)))]
                    if constant_data else [])
    arguments = ["data"]
    if k is not None:
        arguments.append("k")
        initializers.append(helper.make_tensor("k", TensorProto.INT64, [], [k]))
    graph = helper.make_graph([helper.make_node("Trilu", arguments, ["output"])], "static_triangle", inputs,
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    imported = import_onnx_model(model)
    assert imported.function.nodes[0].op_name == "trilu"
    assert imported.function.nodes[0].inputs == ["data"]
    assert imported.function.nodes[0].attrs == {"upper": 1, "k": 0 if k is None else k}
