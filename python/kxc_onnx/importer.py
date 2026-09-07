from __future__ import annotations

import math
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


WHERE_BRANCH_DTYPES = {"float32", "float64", "int32", "int64", "int8", "uint8", "bool"}
CONCATENATE_DTYPES = WHERE_BRANCH_DTYPES
SLICE_DTYPES = WHERE_BRANCH_DTYPES
INT64_MAX = (1 << 63) - 1

# S1 static subset: arithmetic (Mul/Sub/Div) and Sqrt only accept float32 so the
# truncating-division and widening semantics of integer ONNX inputs are never claimed.
ARITHMETIC_DTYPES = {"float32"}
# M5 Equal subset (pinned by the C-line InferType contract): both inputs must carry
# the same dtype from this set and the broadcast result is bool. bool/float64 and
# mixed-dtype comparisons are rejected with the canonical diagnostics.
EQUAL_DTYPES = {"int32", "int64", "float32"}
# Relay cast dtype codes as understood by CastDTypeFromCode in the C++ reifier.
RELAY_CAST_DTYPE_CODES = {
    "float32": 0,
    "int32": 1,
    "int64": 2,
    "float64": 3,
    "bool": 4,
    "int8": 5,
    "uint8": 6,
}
# S1 Cast subset: only the conversions the target models need are opened.
CAST_SUPPORTED_CONVERSIONS = {("int32", "float32"), ("int64", "float32")}
# ONNX TensorProto dtype enum values accepted by the Cast boundary.
ONNX_CAST_DTYPE_NAMES = {
    TensorProto.FLOAT: "float32",
    TensorProto.DOUBLE: "float64",
    TensorProto.INT64: "int64",
    TensorProto.INT32: "int32",
    TensorProto.INT8: "int8",
    TensorProto.UINT8: "uint8",
    TensorProto.BOOL: "bool",
}
CONSTANT_SUPPORTED_DTYPES = WHERE_BRANCH_DTYPES
# ONNX Constant-13 shortcut/sparse attributes that the dense-value S1 subset rejects.
CONSTANT_DENSE_VALUE_ONLY_ATTRS = {
    "sparse_value",
    "value_float",
    "value_floats",
    "value_int",
    "value_ints",
    "value_string",
    "value_strings",
}


ONNX_TO_RELAY = {
    "Concat": "concatenate",
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
    "Gather": "gather",
    "Where": "where",
    "Equal": "equal",
    "LayerNormalization": "nn_layer_norm",
    "Slice": "slice",
    "Mul": "mul",
    "Sub": "subtract",
    "Div": "divide",
    "Sqrt": "sqrt",
    "Cast": "cast",
    "ReduceMean": "reduce_mean",
    "Reshape": "reshape",
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
    output_declarations: dict[str, list[onnx.ValueInfoProto]] = {}
    for value_info in list(graph.output) + list(graph.value_info):
        output_declarations.setdefault(value_info.name, []).append(value_info)

    inputs = [
        _tensor_spec_from_value_info(value_info, default_batch)
        for value_info in graph.input
        if value_info.name not in params
    ]
    outputs = [_tensor_spec_from_value_info(value_info, default_batch) for value_info in graph.output]

    nodes: list[RelayNodeSpec] = []
    input_specs = {spec.name: spec for spec in inputs}
    inferred_static_specs: dict[str, TensorSpec] = {}
    available_values = {x.name for x in inputs} | set(params)
    for node in graph.node:
        if node.op_type == "Constant":
            _import_constant_node(
                node, input_specs, params, param_order, available_values,
                output_declarations, default_batch,
            )
            continue
        if node.op_type not in ONNX_TO_RELAY:
            raise UnsupportedONNXOpError(
                f"Unsupported ONNX op '{node.op_type}' in node '{node.name or '<unnamed>'}'"
            )
        if node.op_type == "LayerNormalization":
            node_name = node.name or "<unnamed>"
            if not 1 <= len(node.output) <= 3 or not node.output[0]:
                raise ValueError(
                    f"LayerNormalization node '{node_name}' requires Y and at most two empty optional output slots"
                )
            if any(node.output[1:]):
                raise ValueError(
                    f"LayerNormalization node '{node_name}' does not support non-empty Mean or InvStdDev outputs"
                )
        elif len(node.output) != 1:
            raise ValueError(
                f"ONNX node '{node.name or node.op_type}' has {len(node.output)} outputs; "
                "only single-output nodes are supported in the static-shape MVP"
            )

        if node.op_type == "Slice":
            node_name = node.name or "<unnamed>"
            if opset_version < 10:
                raise UnsupportedONNXOpError(
                    f"Unsupported ONNX Slice opset {opset_version} in node "
                    f"'{node_name}': input-form opset >= 10 is required"
                )
            if not 3 <= len(node.input) <= 5 or not all(node.input[:3]):
                raise ValueError(
                    f"Slice node '{node_name}' requires non-empty data/starts/ends and at most optional axes/steps"
                )
            if not node.output[0]:
                raise ValueError(f"Slice node '{node_name}' requires exactly one non-empty output")
            if node.input[0] not in available_values:
                raise ValueError(f"Slice node '{node_name}' has unresolved data input")
            inferred_static_specs[node.output[0]] = _infer_slice_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "Concat":
            node_name = node.name or "<unnamed>"
            if len(node.input) != 2 or not all(node.input):
                raise ValueError(f"Concat node '{node_name}' requires exactly two non-empty inputs")
            if not node.output[0]:
                raise ValueError(f"Concat node '{node_name}' requires exactly one non-empty output")
            missing = [name for name in node.input if name not in available_values]
            if missing:
                raise ValueError(
                    f"Concat node '{node_name}' has unresolved prior input(s): {missing}"
                )
            inferred_static_specs[node.output[0]] = _infer_concatenate_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "LayerNormalization":
            node_name = node.name or "<unnamed>"
            if opset_version < 17:
                raise UnsupportedONNXOpError(
                    f"Unsupported ONNX LayerNormalization opset {opset_version} in node "
                    f"'{node_name}': opset >= 17 is required"
                )
            if len(node.input) != 3 or not all(node.input):
                raise ValueError(
                    f"LayerNormalization node '{node_name}' requires exactly three non-empty inputs"
                )
            missing = [name for name in node.input if name not in available_values]
            if missing:
                raise ValueError(
                    f"LayerNormalization node '{node_name}' has unresolved prior input(s): {missing}"
                )
            inferred_static_specs[node.output[0]] = _infer_layer_normalization_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "MatMul":
            inferred_static_specs[node.output[0]] = _infer_matmul_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "Gather":
            node_name = node.name or "<unnamed>"
            if len(node.input) != 2 or not all(node.input):
                raise ValueError(
                    f"Gather node '{node_name}' requires exactly two non-empty inputs"
                )
            if not node.output[0]:
                raise ValueError(
                    f"Gather node '{node_name}' requires exactly one non-empty output"
                )
            inferred_static_specs[node.output[0]] = _infer_gather_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type in {"Mul", "Sub", "Div"}:
            inferred_static_specs[node.output[0]] = _infer_binary_arithmetic_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "Sqrt":
            inferred_static_specs[node.output[0]] = _infer_sqrt_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "Cast":
            inferred_static_specs[node.output[0]] = _infer_cast_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "ReduceMean":
            inferred_static_specs[node.output[0]] = _infer_reduce_mean_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch, opset_version,
            )
        if node.op_type == "Reshape":
            inferred_static_specs[node.output[0]] = _infer_reshape_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "Where":
            inferred_static_specs[node.output[0]] = _infer_where_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "Equal":
            inferred_static_specs[node.output[0]] = _infer_equal_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )

        missing = [name for name in node.input if name and name not in available_values]
        if missing:
            raise ValueError(
                f"ONNX node '{node.name or node.op_type}' has missing input(s): {missing}"
            )

        relay_inputs = ([node.input[0]] if node.op_type in {"Slice", "Reshape"}
                        else [name for name in node.input if name])
        relay_outputs = ([node.output[0]] if node.op_type == "LayerNormalization"
                         else [name for name in node.output])
        nodes.append(
            RelayNodeSpec(
                name=node.name or f"{node.op_type}_{len(nodes)}",
                op_name=ONNX_TO_RELAY[node.op_type],
                inputs=relay_inputs,
                outputs=relay_outputs,
                attrs=_convert_attrs(node, params, value_info_by_name, opset_version,
                                     input_specs, inferred_static_specs),
            )
        )
        available_values.update(relay_outputs)

    return ImportedONNXModel(
        function=RelayFunctionSpec(inputs=inputs, outputs=outputs, nodes=nodes),
        params=params,
        param_order=param_order,
    )


def _resolve_static_input(
    op_type: str,
    node_name: str,
    name: str,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    default_batch: int | None,
) -> TensorSpec:
    if name in inferred_specs:
        return inferred_specs[name]
    if name in params:
        param = params[name]
        return TensorSpec(name=name, shape=param.shape, dtype=param.dtype)
    if name in input_specs:
        return input_specs[name]
    value_info = value_info_by_name.get(name)
    if value_info is None:
        raise ValueError(
            f"{op_type} node '{node_name}' input '{name}' metadata is absent or unresolved"
        )
    try:
        return _tensor_spec_from_value_info(value_info, default_batch)
    except ValueError as error:
        raise ValueError(
            f"{op_type} node '{node_name}' input '{name}' metadata is unresolved: {error}"
        ) from error


def _validate_declared_output(
    op_type: str,
    node_name: str,
    result: TensorSpec,
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> None:
    for declared in output_declarations.get(result.name, []):
        try:
            declared_spec = _tensor_spec_from_value_info(declared, default_batch)
        except ValueError as error:
            raise ValueError(
                f"{op_type} node '{node_name}' output '{result.name}' metadata is unresolved: {error}"
            ) from error
        if declared_spec.shape != result.shape or declared_spec.dtype != result.dtype:
            raise ValueError(
                f"{op_type} node '{node_name}' output '{result.name}' declaration "
                f"{declared_spec.shape}/{declared_spec.dtype} does not match inferred "
                f"{result.shape}/{result.dtype}"
            )


def _import_constant_node(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    param_order: list[str],
    available_values: set[str],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> None:
    """Normalize an ONNX Constant node into the existing ParamTensor constant path.

    The S1 subset only accepts the dense Tensor ``value`` attribute; every shortcut
    (``value_float``...) and sparse form is rejected. Output names must be unique
    and must not collide with initializers or graph inputs.
    """
    node_name = node.name or "<unnamed>"
    if len(node.output) != 1 or not node.output[0]:
        raise ValueError(
            f"Constant node '{node_name}' requires exactly one non-empty output name"
        )
    attr_names = [attr.name for attr in node.attribute]
    if len(attr_names) != len(set(attr_names)):
        raise ValueError(
            f"Constant node '{node_name}' has duplicate attribute names"
        )
    if not attr_names:
        raise ValueError(
            f"Constant node '{node_name}' requires exactly one value attribute; "
            "none was provided"
        )
    if len(attr_names) > 1:
        raise ValueError(
            f"Constant node '{node_name}' requires exactly one value attribute; "
            f"got {sorted(attr_names)}"
        )
    if attr_names != ["value"]:
        raise ValueError(
            f"Constant node '{node_name}' only supports the dense 'value' attribute "
            f"in the static S1 subset; got {sorted(attr_names)}"
        )
    value_attr = node.attribute[0]
    if value_attr.type != AttributeProto.TENSOR:
        raise ValueError(
            f"Constant node '{node_name}' attribute 'value' must have exact TENSOR type"
        )
    array = numpy_helper.to_array(value_attr.t)
    # np.ascontiguousarray promotes 0-d arrays to shape (1,); scalars must stay scalar.
    if array.ndim > 0:
        array = np.ascontiguousarray(array)
    dtype = str(array.dtype)
    if dtype not in CONSTANT_SUPPORTED_DTYPES:
        raise ValueError(
            f"Constant node '{node_name}' has unsupported dtype '{dtype}'"
        )
    name = node.output[0]
    if name in params or name in input_specs or name in available_values:
        raise ValueError(
            f"Constant node '{node_name}' output name '{name}' conflicts with an "
            "existing graph input, initializer, or produced value"
        )
    param_order.append(name)
    params[name] = ParamTensor(
        name=name,
        shape=[int(dim) for dim in array.shape],
        dtype=dtype,
        data=array.tobytes(order="C"),
    )
    available_values.add(name)
    _validate_declared_output(
        "Constant", node_name,
        TensorSpec(name=name, shape=params[name].shape, dtype=dtype),
        output_declarations, default_batch,
    )


def _infer_binary_arithmetic_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    op_type = node.op_type
    node_name = node.name or "<unnamed>"
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(
            f"{op_type} node '{node_name}' requires exactly two non-empty inputs"
        )
    lhs, rhs = (
        _resolve_static_input(op_type, node_name, name, input_specs, params,
                              inferred_specs, value_info_by_name, default_batch)
        for name in node.input
    )
    if lhs.dtype not in ARITHMETIC_DTYPES or rhs.dtype not in ARITHMETIC_DTYPES:
        raise ValueError(
            f"{op_type} node '{node_name}' requires float32 inputs in the static "
            f"S1 subset; got {lhs.dtype} and {rhs.dtype}"
        )
    result = TensorSpec(
        name=node.output[0],
        shape=_broadcast_shapes(op_type, node_name, lhs.shape, rhs.shape),
        dtype="float32",
    )
    _validate_declared_output(op_type, node_name, result, output_declarations, default_batch)
    return result


def _infer_sqrt_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    node_name = node.name or "<unnamed>"
    if len(node.input) != 1 or not node.input[0]:
        raise ValueError(f"Sqrt node '{node_name}' requires exactly one non-empty input")
    data = _resolve_static_input("Sqrt", node_name, node.input[0], input_specs, params,
                                 inferred_specs, value_info_by_name, default_batch)
    if data.dtype not in ARITHMETIC_DTYPES:
        raise ValueError(
            f"Sqrt node '{node_name}' requires float32 input in the static S1 "
            f"subset; got {data.dtype}"
        )
    result = TensorSpec(name=node.output[0], shape=list(data.shape), dtype="float32")
    _validate_declared_output("Sqrt", node_name, result, output_declarations, default_batch)
    return result


def _cast_target_dtype(node: onnx.NodeProto) -> str:
    """Resolve the Cast ``to`` attribute to a kxc dtype name under the S1 boundary."""
    node_name = node.name or "<unnamed>"
    attrs = _attrs_by_name(node)
    if set(attrs) != {"to"}:
        raise ValueError(
            f"Cast node '{node_name}' requires exactly the 'to' attribute"
        )
    to = _int_attr(attrs, "to", -1)
    if to not in ONNX_CAST_DTYPE_NAMES:
        raise ValueError(
            f"Cast node '{node_name}' has unsupported 'to' dtype enum {to}"
        )
    target = ONNX_CAST_DTYPE_NAMES[to]
    if target != "float32":
        raise ValueError(
            f"Cast node '{node_name}' supports only to=float32 in the static S1 "
            f"subset; got to={target}"
        )
    return target


def _infer_cast_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    node_name = node.name or "<unnamed>"
    if len(node.input) != 1 or not node.input[0]:
        raise ValueError(f"Cast node '{node_name}' requires exactly one non-empty input")
    data = _resolve_static_input("Cast", node_name, node.input[0], input_specs, params,
                                 inferred_specs, value_info_by_name, default_batch)
    target = _cast_target_dtype(node)
    if (data.dtype, target) not in CAST_SUPPORTED_CONVERSIONS:
        raise ValueError(
            f"Cast node '{node_name}' supports only int32/int64 to float32 in the "
            f"static S1 subset; got {data.dtype} to {target}"
        )
    result = TensorSpec(name=node.output[0], shape=list(data.shape), dtype=target)
    _validate_declared_output("Cast", node_name, result, output_declarations, default_batch)
    return result


def _reduce_mean_attrs(node: onnx.NodeProto, opset_version: int) -> dict[str, Any]:
    """Validate ReduceMean attributes for the opset-17 static subset.

    The axes-attribute form (opset <= 17) is required; the opset-18 input form
    is never opened. Absent or empty axes mean "reduce all axes".
    """
    node_name = node.name or "<unnamed>"
    if opset_version >= 18:
        raise UnsupportedONNXOpError(
            f"Unsupported ONNX ReduceMean opset {opset_version} in node "
            f"'{node_name}': the input-form opset >= 18 is not supported; "
            "the static axes-attribute form (opset <= 17) is required"
        )
    attrs = _attrs_by_name(node)
    unsupported = set(attrs) - {"axes", "keepdims"}
    if unsupported:
        raise ValueError(
            f"ReduceMean node '{node_name}' has unsupported attribute(s): "
            f"{sorted(unsupported)}"
        )
    axes = _list_attr(attrs, "axes", [])
    keepdims = _int_attr(attrs, "keepdims", 1)
    if keepdims not in (0, 1):
        raise ValueError(
            f"ReduceMean node '{node_name}' keepdims must be 0 or 1; got {keepdims}"
        )
    if len(axes) != len(set(axes)):
        raise ValueError(
            f"ReduceMean node '{node_name}' axes must not contain duplicates"
        )
    return {"axes": axes, "keepdims": keepdims}


def _infer_reduce_mean_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
    opset_version: int,
) -> TensorSpec:
    node_name = node.name or "<unnamed>"
    attrs = _reduce_mean_attrs(node, opset_version)
    if len(node.input) != 1 or not node.input[0]:
        raise ValueError(
            f"ReduceMean node '{node_name}' requires exactly one non-empty input"
        )
    data = _resolve_static_input("ReduceMean", node_name, node.input[0], input_specs,
                                 params, inferred_specs, value_info_by_name, default_batch)
    if data.dtype not in ARITHMETIC_DTYPES:
        raise ValueError(
            f"ReduceMean node '{node_name}' requires float32 input in the static "
            f"S1 subset; got {data.dtype}"
        )
    if any(dim < 0 for dim in data.shape):
        raise ValueError(
            f"ReduceMean node '{node_name}' requires non-negative static dimensions"
        )
    rank = len(data.shape)
    normalized: list[int] = []
    for axis in attrs["axes"]:
        resolved = axis + rank if axis < 0 else axis
        if resolved < 0 or resolved >= rank:
            raise ValueError(
                f"ReduceMean node '{node_name}' axis {axis} is out of range for rank {rank}"
            )
        normalized.append(resolved)
    if not normalized:
        # ONNX opset <= 17: absent or empty axes reduce all axes.
        normalized = list(range(rank))
    for axis in set(normalized):
        if data.shape[axis] == 0:
            raise ValueError(
                f"ReduceMean node '{node_name}' reduces over zero-extent axis {axis}; "
                "the result is undefined and is rejected in the static S1 subset"
            )
    output_shape: list[int] = []
    for index, dim in enumerate(data.shape):
        if index in normalized:
            if attrs["keepdims"]:
                output_shape.append(1)
        else:
            output_shape.append(dim)
    result = TensorSpec(
        name=node.output[0], shape=output_shape, dtype="float32",
    )
    _validate_declared_output(
        "ReduceMean", node_name, result, output_declarations, default_batch
    )
    return result


def _reshape_attrs(
    node: onnx.NodeProto, params: dict[str, ParamTensor], input_shape: list[int]
) -> dict[str, Any]:
    """Resolve the Reshape constant shape input into a proven target shape.

    The shape input must be an int64 rank-1 initializer or Constant node output.
    With allowzero=0 the 0/-1 dimensions are resolved against the data shape and
    the canonical attrs carry the fully resolved target shape.
    """
    node_name = node.name or "<unnamed>"
    attrs = _attrs_by_name(node)
    unsupported = set(attrs) - {"allowzero"}
    if unsupported:
        raise ValueError(
            f"Reshape node '{node_name}' has unsupported attribute(s): {sorted(unsupported)}"
        )
    allowzero = _int_attr(attrs, "allowzero", 0)
    if allowzero != 0:
        raise ValueError(
            f"Reshape node '{node_name}' requires allowzero=0 in the static S1 subset"
        )
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(
            f"Reshape node '{node_name}' requires exactly two non-empty inputs"
        )
    shape_name = node.input[1]
    shape_param = params.get(shape_name)
    if shape_param is None:
        raise ValueError(
            f"Reshape node '{node_name}' shape input '{shape_name}' must be a static "
            "initializer or Constant node output"
        )
    if shape_param.dtype != "int64":
        raise ValueError(
            f"Reshape node '{node_name}' shape input '{shape_name}' must be int64 "
            f"(Reshape-14 tensor(int64)); got {shape_param.dtype}"
        )
    if len(shape_param.shape) != 1:
        raise ValueError(
            f"Reshape node '{node_name}' shape input '{shape_name}' must be rank-1; "
            f"got rank {len(shape_param.shape)}"
        )
    if len(shape_param.data) != 8 * shape_param.shape[0]:
        raise ValueError(
            f"Reshape node '{node_name}' shape initializer '{shape_name}' byte size "
            "is invalid"
        )
    raw_shape: list[int] = [
        int(value) for value in np.frombuffer(shape_param.data, dtype="<i8")
    ]
    return _resolve_reshape_target(node_name, raw_shape, input_shape, allowzero)


def _infer_reshape_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    node_name = node.name or "<unnamed>"
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(
            f"Reshape node '{node_name}' requires exactly two non-empty inputs"
        )
    data = _resolve_static_input("Reshape", node_name, node.input[0], input_specs, params,
                                 inferred_specs, value_info_by_name, default_batch)
    if any(dim < 0 for dim in data.shape):
        raise ValueError(
            f"Reshape node '{node_name}' requires non-negative static input dimensions"
        )
    attrs = _reshape_attrs(node, params, data.shape)
    result = TensorSpec(
        name=node.output[0], shape=list(attrs["newshape"]), dtype=data.dtype,
    )
    _validate_declared_output(
        "Reshape", node_name, result, output_declarations, default_batch
    )
    return result


def _resolve_reshape_target(
    node_name: str, raw_shape: list[int], input_shape: list[int], allowzero: int
) -> dict[str, Any]:
    """Resolve 0/-1 dimensions of a raw newshape against the data shape.

    Returns the canonical attrs carrying the fully resolved target shape; every
    ambiguous case (zero known product with -1, multiple -1, mismatched element
    counts) is rejected instead of guessed.
    """
    infer_index: int | None = None
    resolved: list[int] = []
    known_product = 1
    for index, dim in enumerate(raw_shape):
        if dim == -1:
            if infer_index is not None:
                raise ValueError(
                    f"Reshape node '{node_name}' allows at most one -1 dimension"
                )
            infer_index = len(resolved)
            resolved.append(-1)
        elif dim == 0 and not allowzero:
            if index >= len(input_shape):
                raise ValueError(
                    f"Reshape node '{node_name}' 0-dim copy index exceeds input rank"
                )
            resolved.append(input_shape[index])
            known_product *= input_shape[index]
        elif dim > 0:
            resolved.append(dim)
            known_product *= dim
        else:
            raise ValueError(
                f"Reshape node '{node_name}' only supports positive, 0, and -1 "
                f"dimensions; got {dim}"
            )
    input_product = 1
    for dim in input_shape:
        input_product *= dim
    if infer_index is not None:
        if known_product == 0:
            raise ValueError(
                f"Reshape node '{node_name}' cannot prove a unique target shape for "
                "-1: the known dimensions have zero elements"
            )
        if input_product % known_product != 0:
            raise ValueError(
                f"Reshape node '{node_name}' element count {input_product} is not "
                f"divisible by the known dimensions product {known_product}"
            )
        resolved[infer_index] = input_product // known_product
    elif input_product != known_product:
        raise ValueError(
            f"Reshape node '{node_name}' element count mismatch: input has "
            f"{input_product} elements but the target shape has {known_product}"
        )
    for dim in resolved:
        if dim > INT64_MAX:
            raise ValueError(
                f"Reshape node '{node_name}' resolved dimension {dim} overflows int64"
            )
    return {"newshape": resolved, "allowzero": allowzero}


def _infer_matmul_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    node_name = node.name or "<unnamed>"
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(f"MatMul node '{node_name}' requires exactly two non-empty inputs")

    left, right = (
        _resolve_static_input("MatMul", node_name, name, input_specs, params,
                              inferred_specs, value_info_by_name, default_batch)
        for name in node.input
    )
    if len(left.shape) < 2 or len(right.shape) < 2:
        raise ValueError(f"MatMul node '{node_name}' requires both inputs to have rank >= 2")
    if left.dtype != right.dtype:
        raise ValueError(
            f"MatMul node '{node_name}' requires matching input dtypes; "
            f"got {left.dtype} and {right.dtype}"
        )
    if left.shape[-1] != right.shape[-2]:
        raise ValueError(
            f"MatMul node '{node_name}' K dimensions differ: "
            f"{left.shape[-1]} != {right.shape[-2]}"
        )

    batch: list[int] = []
    for left_dim, right_dim in zip(reversed(left.shape[:-2]), reversed(right.shape[:-2])):
        if left_dim != right_dim and left_dim != 1 and right_dim != 1:
            raise ValueError(
                f"MatMul node '{node_name}' has incompatible leading batch dimensions: "
                f"{left.shape[:-2]} and {right.shape[:-2]}"
            )
        batch.append(left_dim if right_dim == 1 else right_dim)
    batch.extend(reversed(left.shape[:-2][: len(left.shape[:-2]) - len(right.shape[:-2])]))
    batch.extend(reversed(right.shape[:-2][: len(right.shape[:-2]) - len(left.shape[:-2])]))
    result = TensorSpec(
        name=node.output[0],
        shape=list(reversed(batch)) + [left.shape[-2], right.shape[-1]],
        dtype=left.dtype,
    )

    _validate_declared_output("MatMul", node_name, result, output_declarations, default_batch)
    return result


def _infer_gather_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    node_name = node.name or "<unnamed>"
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(f"Gather node '{node_name}' requires exactly two non-empty inputs")
    data, indices = (
        _resolve_static_input("Gather", node_name, name, input_specs, params,
                              inferred_specs, value_info_by_name, default_batch)
        for name in node.input
    )
    if not data.shape:
        raise ValueError(f"Gather node '{node_name}' requires data rank >= 1")
    if any(dim < 0 for dim in data.shape + indices.shape):
        raise ValueError(
            f"Gather node '{node_name}' requires non-negative static dimensions"
        )
    if indices.dtype not in {"int32", "int64"}:
        raise ValueError(
            f"Gather node '{node_name}' requires int32 or int64 indices; got {indices.dtype}"
        )
    if node.input[1] not in params:
        raise ValueError(
            f"Gather node '{node_name}' requires initializer-backed constant indices"
        )
    attrs = _attrs_by_name(node)
    unsupported = set(attrs) - {"axis"}
    if unsupported:
        raise ValueError(
            f"Gather node '{node_name}' has unsupported attribute(s): {sorted(unsupported)}"
        )
    axis = _int_attr(attrs, "axis", 0)
    if axis < 0:
        axis += len(data.shape)
    if axis < 0 or axis >= len(data.shape):
        raise ValueError(f"Gather node '{node_name}' axis {axis} is out of range")
    if indices.dtype == "int32" and data.shape[axis] > np.iinfo(np.int32).max:
        raise ValueError(
            f"Gather node '{node_name}' int32 indices cannot address axis extent > INT32_MAX"
        )
    index_param = params[node.input[1]]
    dtype = np.dtype("<i4" if index_param.dtype == "int32" else "<i8")
    values = np.frombuffer(index_param.data, dtype=dtype)
    expected_values = math.prod(index_param.shape)
    if values.size != expected_values:
        raise ValueError(
            f"Gather node '{node_name}' indices initializer byte size is invalid"
        )
    extent = data.shape[axis]
    for value in values:
        index = int(value)
        if index < -extent or index >= extent:
            raise ValueError(
                f"Gather node '{node_name}' constant index {index} is outside "
                f"the ONNX domain [{-extent}, {extent - 1}]"
            )
    result = TensorSpec(
        name=node.output[0],
        shape=data.shape[:axis] + indices.shape + data.shape[axis + 1 :],
        dtype=data.dtype,
    )
    _validate_declared_output("Gather", node_name, result, output_declarations, default_batch)
    return result


def _slice_initializer_values(
    node_name: str, name: str, params: dict[str, ParamTensor]
) -> tuple[list[int], str]:
    param = params.get(name)
    if param is None:
        raise ValueError(
            f"Slice node '{node_name}' parameter '{name}' must be a static initializer"
        )
    if param.dtype not in {"int32", "int64"} or len(param.shape) != 1 or not param.shape or param.shape[0] <= 0:
        raise ValueError(
            f"Slice node '{node_name}' parameter '{name}' must be a nonempty rank-1 int32/int64 initializer"
        )
    dtype = np.dtype("<i4" if param.dtype == "int32" else "<i8")
    values = np.frombuffer(param.data, dtype=dtype)
    if values.size != param.shape[0]:
        raise ValueError(f"Slice node '{node_name}' initializer '{name}' byte size is invalid")
    return [int(value) for value in values], param.dtype


def _clamp_positive_step_endpoint(endpoint: int, dim: int) -> int:
    if endpoint < 0:
        return 0 if endpoint < -dim else endpoint + dim
    return min(endpoint, dim)


def _slice_attrs(node: onnx.NodeProto, params: dict[str, ParamTensor]) -> dict[str, list[int]]:
    node_name = node.name or "<unnamed>"
    if node.attribute:
        raise ValueError(f"Slice node '{node_name}' does not support attributes in input form")
    starts, starts_dtype = _slice_initializer_values(node_name, node.input[1], params)
    ends, ends_dtype = _slice_initializer_values(node_name, node.input[2], params)
    provided_dtypes = {starts_dtype, ends_dtype}
    if len(starts) != len(ends):
        raise ValueError(f"Slice node '{node_name}' starts and ends lengths must match")
    if len(node.input) >= 4 and node.input[3]:
        axes, axes_dtype = _slice_initializer_values(node_name, node.input[3], params)
        provided_dtypes.add(axes_dtype)
    else:
        axes = list(range(len(starts)))
    if len(node.input) >= 5 and node.input[4]:
        steps, steps_dtype = _slice_initializer_values(node_name, node.input[4], params)
        provided_dtypes.add(steps_dtype)
    else:
        steps = [1] * len(starts)
    if len(provided_dtypes) != 1:
        raise ValueError(
            f"Slice node '{node_name}' control initializers must use one consistent int32 or int64 dtype"
        )
    if not starts or len(axes) != len(starts) or len(steps) != len(starts):
        raise ValueError(f"Slice node '{node_name}' starts, ends, axes, and steps must be nonempty and equal length")
    return {"starts": starts, "ends": ends, "axes": axes, "steps": steps}


def _infer_slice_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    node_name = node.name or "<unnamed>"
    data = _resolve_static_input("Slice", node_name, node.input[0], input_specs, params,
                                 inferred_specs, value_info_by_name, default_batch)
    if data.dtype not in SLICE_DTYPES:
        raise ValueError(f"Slice node '{node_name}' requires dtype in {{float32,float64,int32,int64,int8,uint8,bool}}")
    if not data.shape:
        raise ValueError(f"Slice node '{node_name}' requires data rank >= 1")
    if any(dim < 0 for dim in data.shape):
        raise ValueError(f"Slice node '{node_name}' requires non-negative static data dimensions")
    attrs = _slice_attrs(node, params)
    output_shape = list(data.shape)
    seen: set[int] = set()
    for start, end, axis, step in zip(attrs["starts"], attrs["ends"], attrs["axes"], attrs["steps"]):
        if step != 1:
            raise ValueError(f"Slice node '{node_name}' requires every step to equal exactly +1")
        normalized_axis = axis + len(data.shape) if axis < 0 else axis
        if normalized_axis < 0 or normalized_axis >= len(data.shape) or normalized_axis in seen:
            raise ValueError(f"Slice node '{node_name}' axes must be unique and in range")
        seen.add(normalized_axis)
        dim = output_shape[normalized_axis]
        clamped_start = _clamp_positive_step_endpoint(start, dim)
        clamped_end = _clamp_positive_step_endpoint(end, dim)
        output_shape[normalized_axis] = max(clamped_end - clamped_start, 0)
    result = TensorSpec(name=node.output[0], shape=output_shape, dtype=data.dtype)
    _validate_declared_output("Slice", node_name, result, output_declarations, default_batch)
    return result


def _infer_concatenate_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    node_name = node.name or "<unnamed>"
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(f"Concat node '{node_name}' requires exactly two non-empty inputs")
    attrs = _attrs_by_name(node)
    if set(attrs) != {"axis"}:
        raise ValueError(f"Concat node '{node_name}' requires exactly the axis attribute")
    axis = _int_attr(attrs, "axis", 0)
    lhs, rhs = (
        _resolve_static_input("Concat", node_name, name, input_specs, params,
                              inferred_specs, value_info_by_name, default_batch)
        for name in node.input
    )
    if lhs.dtype not in CONCATENATE_DTYPES or rhs.dtype not in CONCATENATE_DTYPES:
        raise ValueError(
            f"Concat node '{node_name}' requires dtypes in "
            "{float32,float64,int32,int64,int8,uint8,bool}; "
            f"got {lhs.dtype} and {rhs.dtype}"
        )
    if lhs.dtype != rhs.dtype:
        raise ValueError(
            f"Concat node '{node_name}' requires matching input dtypes; "
            f"got {lhs.dtype} and {rhs.dtype}"
        )
    if not lhs.shape or not rhs.shape:
        raise ValueError(f"Concat node '{node_name}' requires rank >= 1 inputs")
    if len(lhs.shape) != len(rhs.shape):
        raise ValueError(f"Concat node '{node_name}' input ranks must match")
    if axis < 0:
        axis += len(lhs.shape)
    if axis < 0 or axis >= len(lhs.shape):
        raise ValueError(f"Concat node '{node_name}' axis {axis} is out of range")
    for index, (left_dim, right_dim) in enumerate(zip(lhs.shape, rhs.shape)):
        if left_dim < 0 or right_dim < 0:
            raise ValueError(f"Concat node '{node_name}' requires non-negative static dimensions")
        if index != axis and left_dim != right_dim:
            raise ValueError(
                f"Concat node '{node_name}' non-axis dimensions must exactly match"
            )
    if lhs.shape[axis] > INT64_MAX - rhs.shape[axis]:
        raise ValueError(f"Concat node '{node_name}' axis extent sum overflows int64")
    output_shape = list(lhs.shape)
    output_shape[axis] = lhs.shape[axis] + rhs.shape[axis]
    result = TensorSpec(name=node.output[0], shape=output_shape, dtype=lhs.dtype)
    _validate_declared_output("Concat", node_name, result, output_declarations, default_batch)
    return result


def _infer_layer_normalization_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    node_name = node.name or "<unnamed>"
    data, scale, bias = (
        _resolve_static_input("LayerNormalization", node_name, name, input_specs, params,
                              inferred_specs, value_info_by_name, default_batch)
        for name in node.input
    )
    if data.dtype != "float32" or scale.dtype != "float32" or bias.dtype != "float32":
        raise ValueError(
            f"LayerNormalization node '{node_name}' requires float32 data, scale, and bias"
        )
    if not data.shape:
        raise ValueError(f"LayerNormalization node '{node_name}' requires data rank >= 1")
    if any(dim < 0 for dim in data.shape):
        raise ValueError(
            f"LayerNormalization node '{node_name}' requires non-negative static data dimensions"
        )
    attrs = _attrs_by_name(node)
    unsupported = set(attrs) - {"axis", "epsilon", "stash_type"}
    if unsupported:
        raise ValueError(
            f"LayerNormalization node '{node_name}' has unsupported attribute(s): "
            f"{sorted(unsupported)}"
        )
    axis = _int_attr(attrs, "axis", -1)
    if axis < 0:
        axis += len(data.shape)
    if axis < 0 or axis >= len(data.shape):
        raise ValueError(f"LayerNormalization node '{node_name}' axis {axis} is out of range")
    epsilon = _float_attr(attrs, "epsilon", 1e-5)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            f"LayerNormalization node '{node_name}' epsilon must be finite and > 0"
        )
    stash_type = _int_attr(attrs, "stash_type", 1)
    if stash_type != 1:
        raise ValueError(
            f"LayerNormalization node '{node_name}' only supports stash_type default/1 (float32)"
        )
    suffix = data.shape[axis:]
    if any(dim <= 0 for dim in suffix):
        raise ValueError(
            f"LayerNormalization node '{node_name}' normalized suffix dimensions must be > 0"
        )
    if scale.shape != suffix or bias.shape != suffix:
        raise ValueError(
            f"LayerNormalization node '{node_name}' scale and bias shapes must exactly equal "
            "data.shape[axis:]"
        )
    result = TensorSpec(name=node.output[0], shape=list(data.shape), dtype="float32")
    _validate_declared_output(
        "LayerNormalization", node_name, result, output_declarations, default_batch
    )
    return result


def _broadcast_shapes(op_type: str, node_name: str, left: list[int], right: list[int]) -> list[int]:
    result: list[int] = []
    for left_dim, right_dim in zip(reversed(left), reversed(right)):
        if left_dim != right_dim and left_dim != 1 and right_dim != 1:
            raise ValueError(
                f"{op_type} node '{node_name}' has incompatible broadcast dimensions: "
                f"{left} and {right}"
            )
        result.append(right_dim if left_dim == 1 else left_dim)
    longer = left if len(left) > len(right) else right
    result.extend(reversed(longer[: abs(len(left) - len(right))]))
    return list(reversed(result))


def _infer_where_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    node_name = node.name or "<unnamed>"
    if len(node.input) != 3 or not all(node.input):
        raise ValueError(f"Where node '{node_name}' requires exactly three non-empty inputs")
    condition, x, y = (
        _resolve_static_input("Where", node_name, name, input_specs, params,
                              inferred_specs, value_info_by_name, default_batch)
        for name in node.input
    )
    if condition.dtype != "bool":
        raise ValueError(
            f"Where node '{node_name}' requires bool condition; got {condition.dtype}"
        )
    if x.dtype not in WHERE_BRANCH_DTYPES or y.dtype not in WHERE_BRANCH_DTYPES:
        raise ValueError(
            f"Where node '{node_name}' requires branch dtypes in "
            "{float32,float64,int32,int64,int8,uint8,bool}; "
            f"got {x.dtype} and {y.dtype}"
        )
    if x.dtype != y.dtype:
        raise ValueError(
            f"Where node '{node_name}' requires matching x/y dtypes; "
            f"got {x.dtype} and {y.dtype}"
        )
    result = TensorSpec(
        name=node.output[0],
        shape=_broadcast_shapes(
            "Where", node_name,
            _broadcast_shapes("Where", node_name, condition.shape, x.shape), y.shape,
        ),
        dtype=x.dtype,
    )
    _validate_declared_output("Where", node_name, result, output_declarations, default_batch)
    return result


def _infer_equal_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    """Infer the static bool output of an opset-17 Equal node.

    ONNX Equal is fieldless: two inputs, one output, no attributes. Both inputs
    must carry the same dtype from the C-line verified subset (int32, int64,
    float32); the NumPy multidirectional broadcast shape is proven here so the
    result can feed Where as a condition.
    """
    node_name = node.name or "<unnamed>"
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(
            f"Equal node '{node_name}' requires exactly two non-empty inputs"
        )
    lhs, rhs = (
        _resolve_static_input("Equal", node_name, name, input_specs, params,
                              inferred_specs, value_info_by_name, default_batch)
        for name in node.input
    )
    if lhs.dtype not in EQUAL_DTYPES or rhs.dtype not in EQUAL_DTYPES:
        raise ValueError(
            f"Equal node '{node_name}' requires same-dtype int32, int64, or "
            f"float32 inputs; got {lhs.dtype} and {rhs.dtype}"
        )
    if lhs.dtype != rhs.dtype:
        raise ValueError(
            f"Equal node '{node_name}' requires matching input dtypes; "
            f"got {lhs.dtype} and {rhs.dtype}"
        )
    result = TensorSpec(
        name=node.output[0],
        shape=_broadcast_shapes("Equal", node_name, lhs.shape, rhs.shape),
        dtype="bool",
    )
    _validate_declared_output("Equal", node_name, result, output_declarations, default_batch)
    return result


def _tensor_spec_from_value_info(
    value_info: onnx.ValueInfoProto, default_batch: int | None
) -> TensorSpec:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError(
            f"Unresolved ONNX rank for tensor '{value_info.name}': "
            "tensor_type has no shape field"
        )
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
    attrs: dict[str, Any] = {}
    for attr in node.attribute:
        if attr.name in attrs:
            raise ValueError(
                f"ONNX node '{node.name or node.op_type}' has duplicate attribute '{attr.name}'"
            )
        attrs[attr.name] = _attribute_value(attr)
    return attrs


def _list_attr(attrs: dict[str, Any], name: str, default: list[int]) -> list[int]:
    if name not in attrs:
        return list(default)
    value = attrs[name]
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"ONNX attribute '{name}' must have exact INTS type")
    return value


def _int_attr(attrs: dict[str, Any], name: str, default: int) -> int:
    if name not in attrs:
        return default
    value = attrs[name]
    if type(value) is not int:
        raise ValueError(f"ONNX attribute '{name}' must have exact INT type")
    return value


def _float_attr(attrs: dict[str, Any], name: str, default: float) -> float:
    if name not in attrs:
        return default
    value = attrs[name]
    if type(value) is not float:
        raise ValueError(f"ONNX attribute '{name}' must have exact FLOAT type")
    return value


def _convert_attrs(
    node: onnx.NodeProto,
    params: dict[str, ParamTensor],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    opset_version: int,
    input_specs: dict[str, TensorSpec] | None = None,
    inferred_specs: dict[str, TensorSpec] | None = None,
) -> dict[str, Any]:
    input_specs = input_specs or {}
    inferred_specs = inferred_specs or {}
    attrs = _attrs_by_name(node)
    if node.op_type == "Slice":
        if opset_version < 10:
            raise UnsupportedONNXOpError(
                f"Unsupported ONNX Slice opset {opset_version} in node "
                f"'{node.name or '<unnamed>'}': input-form opset >= 10 is required"
            )
        return _slice_attrs(node, params)
    if node.op_type == "Concat":
        attrs = _attrs_by_name(node)
        if set(attrs) != {"axis"}:
            raise ValueError(
                f"Concat node '{node.name or '<unnamed>'}' requires exactly the axis attribute"
            )
        return {"axis": _int_attr(attrs, "axis", 0)}
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
        if opset_version < 13:
            raise UnsupportedONNXOpError(
                f"Unsupported ONNX Softmax opset {opset_version} in node "
                f"'{node.name or '<unnamed>'}': pre-opset-13 flattened-axis semantics "
                "cannot be represented by Relay single-axis softmax"
            )
        return {"axis": _int_attr(attrs, "axis", -1)}
    if node.op_type == "Transpose":
        return {"perm": _list_attr(attrs, "perm", [])}
    if node.op_type == "Gather":
        unsupported = set(attrs) - {"axis"}
        if unsupported:
            raise ValueError(
                f"Gather node '{node.name or '<unnamed>'}' has unsupported attribute(s): "
                f"{sorted(unsupported)}"
            )
        return {"axis": _int_attr(attrs, "axis", 0)}
    if node.op_type == "LayerNormalization":
        node_name = node.name or "<unnamed>"
        if opset_version < 17:
            raise UnsupportedONNXOpError(
                f"Unsupported ONNX LayerNormalization opset {opset_version} in node "
                f"'{node_name}': opset >= 17 is required"
            )
        return {
            "axis": _int_attr(attrs, "axis", -1),
            "epsilon": _float_attr(attrs, "epsilon", 1e-5),
            "accumulation_dtype": "float64",
        }
    if node.op_type == "Where":
        if attrs:
            raise ValueError(
                f"Where node '{node.name or '<unnamed>'}' does not support attributes"
            )
        return {}
    if node.op_type == "Equal":
        if attrs:
            raise ValueError(
                f"Equal node '{node.name or '<unnamed>'}' does not support attributes"
            )
        return {}
    if node.op_type in {"Mul", "Sub", "Div", "Sqrt"}:
        if attrs:
            raise ValueError(
                f"{node.op_type} node '{node.name or '<unnamed>'}' does not support attributes"
            )
        return {}
    if node.op_type == "Cast":
        target = _cast_target_dtype(node)
        return {"to": RELAY_CAST_DTYPE_CODES[target]}
    if node.op_type == "ReduceMean":
        return _reduce_mean_attrs(node, opset_version)
    if node.op_type == "Reshape":
        data = _resolve_static_input("Reshape", node.name or "<unnamed>", node.input[0],
                                     input_specs, params, inferred_specs,
                                     value_info_by_name, None)
        return _reshape_attrs(node, params, data.shape)
    if node.op_type in {"Relu", "Add", "GlobalAveragePool", "MatMul"}:
        return {}
    raise UnsupportedONNXOpError(
        f"Unsupported ONNX op '{node.op_type}' in node '{node.name or '<unnamed>'}'"
    )
