from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, ModelProto, TensorProto, numpy_helper

from .spec import (
    ImportedONNXModel,
    ParamTensor,
    RelayFunctionSpec,
    RelayNodeSpec,
    TensorSpec,
)


ONNX_TO_RELAY = {
    "Conv": "nn_conv2d",
    "Relu": "nn_relu",
    "MaxPool": "nn_max_pool2d",
    "Add": "add",
    "GlobalAveragePool": "nn_global_avg_pool2d",
    "Flatten": "nn_flatten",
    "Gemm": "nn_gemm",
    "MatMul": "matmul",
    "Softmax": "softmax",
    "Transpose": "transpose",
}


class UnsupportedONNXOpError(RuntimeError):
    pass


def onnx_dtype_to_kxc(dtype: int) -> str:
    mapping = {
        TensorProto.FLOAT: "float32",
        TensorProto.DOUBLE: "float64",
        TensorProto.INT64: "int64",
        TensorProto.INT32: "int32",
        TensorProto.INT8: "int8",
        TensorProto.UINT8: "uint8",
        TensorProto.BOOL: "bool",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported ONNX tensor dtype: {dtype}")
    return mapping[dtype]


def import_onnx(
    model_path: str | Path, default_batch: int | None = None
) -> ImportedONNXModel:
    model_path = Path(model_path)
    model = onnx.load(str(model_path))
    return import_onnx_model(model, default_batch=default_batch, base_dir=model_path.parent)


def import_onnx_model(
    model: ModelProto,
    default_batch: int | None = None,
    base_dir: str | Path | None = None,
) -> ImportedONNXModel:
    if default_batch is not None and (
        not isinstance(default_batch, int)
        or isinstance(default_batch, bool)
        or default_batch <= 0
    ):
        raise ValueError("default_batch must be a positive integer when explicitly supplied")

    graph = model.graph
    base_dir_path = Path(base_dir) if base_dir is not None else None
    opset_version = next(
        (int(opset.version) for opset in model.opset_import if opset.domain in {"", "ai.onnx"}),
        1,
    )

    params: dict[str, ParamTensor] = {}
    param_order: list[str] = []
    for initializer in graph.initializer:
        array = numpy_helper.to_array(
            initializer, base_dir=str(base_dir_path) if base_dir_path is not None else ""
        )
        array = np.ascontiguousarray(array)
        param_order.append(initializer.name)
        params[initializer.name] = ParamTensor(
            name=initializer.name,
            shape=[int(dim) for dim in array.shape],
            dtype=str(array.dtype),
            data=array.tobytes(order="C"),
        )

    value_info_by_name = {
        value_info.name: value_info
        for value_info in list(graph.input) + list(graph.output) + list(graph.value_info)
    }

    inputs = [
        _tensor_spec_from_value_info(value_info, default_batch)
        for value_info in graph.input
        if value_info.name not in params
    ]
    outputs = [_tensor_spec_from_value_info(value_info, default_batch) for value_info in graph.output]

    nodes: list[RelayNodeSpec] = []
    available_values = {x.name for x in inputs} | set(params)
    for node in graph.node:
        if node.op_type not in ONNX_TO_RELAY:
            raise UnsupportedONNXOpError(
                f"Unsupported ONNX op '{node.op_type}' in node '{node.name or '<unnamed>'}'"
            )
        if len(node.output) != 1:
            raise ValueError(
                f"ONNX node '{node.name or node.op_type}' has {len(node.output)} outputs; "
                "only single-output nodes are supported in the static-shape MVP"
            )

        missing = [name for name in node.input if name and name not in available_values]
        if missing:
            raise ValueError(
                f"ONNX node '{node.name or node.op_type}' has missing input(s): {missing}"
            )

        relay_inputs = [name for name in node.input if name]
        relay_outputs = [name for name in node.output]
        nodes.append(
            RelayNodeSpec(
                name=node.name or f"{node.op_type}_{len(nodes)}",
                op_name=ONNX_TO_RELAY[node.op_type],
                inputs=relay_inputs,
                outputs=relay_outputs,
                attrs=_convert_attrs(node, params, value_info_by_name, opset_version),
            )
        )
        available_values.update(relay_outputs)

    return ImportedONNXModel(
        function=RelayFunctionSpec(inputs=inputs, outputs=outputs, nodes=nodes),
        params=params,
        param_order=param_order,
    )


def _tensor_spec_from_value_info(
    value_info: onnx.ValueInfoProto, default_batch: int | None
) -> TensorSpec:
    tensor_type = value_info.type.tensor_type
    shape: list[int] = []
    for axis, dim in enumerate(tensor_type.shape.dim):
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif axis == 0 and default_batch is not None:
            shape.append(default_batch)
        else:
            message = (
                f"Unresolved ONNX dimension for tensor '{value_info.name}' at axis {axis}"
            )
            if dim.dim_param:
                message += f" (dim_param='{dim.dim_param}')"
            raise ValueError(message)
    return TensorSpec(
        name=value_info.name,
        shape=shape,
        dtype=onnx_dtype_to_kxc(tensor_type.elem_type),
    )


def _attribute_value(attr: onnx.AttributeProto) -> Any:
    if attr.type == AttributeProto.INT:
        return int(attr.i)
    if attr.type == AttributeProto.FLOAT:
        return float(attr.f)
    if attr.type == AttributeProto.INTS:
        return [int(x) for x in attr.ints]
    if attr.type == AttributeProto.FLOATS:
        return [float(x) for x in attr.floats]
    if attr.type == AttributeProto.STRING:
        return attr.s.decode("utf-8")
    raise ValueError(f"Unsupported ONNX attribute type for '{attr.name}': {attr.type}")


def _attrs_by_name(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: _attribute_value(attr) for attr in node.attribute}


def _list_attr(attrs: dict[str, Any], name: str, default: list[int]) -> list[int]:
    value = attrs.get(name, default)
    return [int(x) for x in value]


def _int_attr(attrs: dict[str, Any], name: str, default: int) -> int:
    return int(attrs.get(name, default))


def _float_attr(attrs: dict[str, Any], name: str, default: float) -> float:
    return float(attrs.get(name, default))


def _convert_attrs(
    node: onnx.NodeProto,
    params: dict[str, ParamTensor],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    opset_version: int,
) -> dict[str, Any]:
    attrs = _attrs_by_name(node)
    if node.op_type == "Conv":
        if len(node.input) < 2 or node.input[1] not in params:
            raise ValueError(f"Conv node '{node.name or '<unnamed>'}' is missing weight initializer")
        weight = params[node.input[1]]
        kernel_size = _list_attr(attrs, "kernel_shape", weight.shape[2:4])
        return {
            "strides": _list_attr(attrs, "strides", [1, 1]),
            "pads": _list_attr(attrs, "pads", [0, 0, 0, 0]),
            "dilations": _list_attr(attrs, "dilations", [1, 1]),
            "group": _int_attr(attrs, "group", 1),
            "channels": int(weight.shape[0]),
            "kernel_size": kernel_size,
            "data_layout": "NCHW",
            "kernel_layout": "OIHW",
            "out_layout": "",
            "out_dtype": "",
        }
    if node.op_type == "MaxPool":
        return {
            "strides": _list_attr(attrs, "strides", [1, 1]),
            "pads": _list_attr(attrs, "pads", [0, 0, 0, 0]),
            "dilations": _list_attr(attrs, "dilations", [1, 1]),
            "pool_size": _list_attr(attrs, "kernel_shape", [1, 1]),
            "layout": "NCHW",
            "ceil_mode": bool(_int_attr(attrs, "ceil_mode", 0)),
        }
    if node.op_type == "Flatten":
        return {"axis": _int_attr(attrs, "axis", 1)}
    if node.op_type == "Gemm":
        return {
            "alpha": _float_attr(attrs, "alpha", 1.0),
            "beta": _float_attr(attrs, "beta", 1.0),
            "transA": _int_attr(attrs, "transA", 0),
            "transB": _int_attr(attrs, "transB", 0),
        }
    if node.op_type == "Softmax":
        return {"axis": _int_attr(attrs, "axis", 1 if opset_version < 13 else -1)}
    if node.op_type == "Transpose":
        return {"perm": _list_attr(attrs, "perm", [])}
    if node.op_type in {"Relu", "Add", "GlobalAveragePool", "MatMul"}:
        return {}
    raise UnsupportedONNXOpError(
        f"Unsupported ONNX op '{node.op_type}' in node '{node.name or '<unnamed>'}'"
    )
