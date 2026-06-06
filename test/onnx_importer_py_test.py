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
