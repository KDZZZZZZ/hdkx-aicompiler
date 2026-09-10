import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from kxc_onnx.fold import ConstantFoldingError, fold_fixed_shape_queries, fold_static_subgraph


def shape_model(shape=(2, 3), *, op="Relu", query_attrs=None, stale=False):
    nodes = [helper.make_node(op, ["x"], ["hidden"], domain="custom" if op == "Unknown" else ""),
             helper.make_node("Shape", ["hidden"], ["result"], name="query", **(query_attrs or {}))]
    model = helper.make_model(helper.make_graph(nodes, "shape", [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)], [
        helper.make_tensor_value_info("result", TensorProto.INT64, [None])], value_info=[
        helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [200, 300])
    ] if stale else []), opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("custom", 1)])
    return model


def test_fixed_shape_proof_ignores_stale_metadata_and_is_explicit():
    model = shape_model(stale=True)
    original = model.SerializeToString()
    default, _ = fold_static_subgraph(model)
    assert any(node.op_type == "Shape" for node in default.graph.node)
    normalized, proof = fold_fixed_shape_queries(model)
    assert proof == [{"node": "query", "input": "hidden", "shape": [2, 3], "value": [2, 3]}]
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(ReferenceEvaluator(normalized).run(None, {"x": x})[0],
                                  ReferenceEvaluator(model).run(None, {"x": x})[0])
    assert model.SerializeToString() == original


@pytest.mark.parametrize("shape,attrs,expected", [
    ((2, 3, 4), {"start": -2, "end": 99}, [3, 4]),
    ((2, 0, 4), {"start": 2, "end": 1}, []),
    ((), {}, []),
])
def test_fixed_shape_uses_onnx_slice_and_scalar_semantics(shape, attrs, expected):
    normalized, proof = fold_fixed_shape_queries(shape_model(shape, query_attrs=attrs))
    assert proof[0]["value"] == expected
    result = ReferenceEvaluator(normalized).run(None, {"x": np.zeros(shape, np.float32)})[0]
    assert result.dtype == np.int64
    assert result.tolist() == expected


def test_fixed_shape_rejects_symbolic_inputs_despite_concrete_annotations():
    with pytest.raises(ConstantFoldingError, match="concrete"):
        fold_fixed_shape_queries(shape_model(("B", 3), stale=True))


def test_unknown_producer_cannot_borrow_a_declared_shape():
    normalized, proof = fold_fixed_shape_queries(shape_model(op="Unknown", stale=True))
    assert not proof
    assert normalized.graph.node[-1].op_type == "Shape"


def test_fixed_shape_unlocks_existing_constant_folder_without_evaluating_data():
    model = shape_model()
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info("result_data", TensorProto.FLOAT, [6]))
    model.graph.node.extend([
        helper.make_node("ReduceProd", ["result"], ["elements"], keepdims=1),
        helper.make_node("Reshape", ["hidden", "elements"], ["result_data"]),
    ])
    normalized, _ = fold_fixed_shape_queries(model)
    normalized, report = fold_static_subgraph(normalized)
    assert [node.op_type for node in normalized.graph.node] == ["Relu", "Reshape"]
    assert report.materialized_bytes == 8
    x = np.array([[-2, -1, 0], [1, 2, 3]], np.float32)
    np.testing.assert_array_equal(ReferenceEvaluator(normalized).run(None, {"x": x})[0],
                                  np.maximum(x, 0).reshape(6))
