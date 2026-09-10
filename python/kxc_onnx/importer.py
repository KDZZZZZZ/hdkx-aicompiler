from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, ModelProto, TensorProto, numpy_helper

from .fold import ConstantFoldingError, fold_static_subgraph
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
# ONNX Add also carries shape/index values in the capacity-shaped MiniMind
# export (`position + arange`).  Add has no truncating-division or widening
# ambiguity, so the production subset admits exact int64 addition while the
# other arithmetic operators remain float32-only.
ADD_DTYPES = {"float32", "int64"}
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
# ("float32", "float32") 是恒等转换：MiniMind 的 RMSNorm 用 `.float()` 提升精度，
# 在已是 float32 的图上导出成 57 个恒等 Cast。Relay `cast` 对 dtype 不设限，
# 同 dtype 经 topi 落成一次拷贝，语义正确。消除这层拷贝需要值别名改写，
# 属于优化而非正确性，留待后续。
CAST_SUPPORTED_CONVERSIONS = {
    ("int32", "float32"),
    ("int64", "float32"),
    ("float32", "float32"),
}
# M4/M5 subset: Neg/Sigmoid are fieldless float32 unary ops, Pow a fieldless
# float32 binary broadcast op. The multidirectional-broadcast input form
# (opset >= 13) is the declared ONNX boundary for all of them.
M4M5_MATH_MIN_OPSET = 13
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
    "Split": "split",
    "Mul": "mul",
    "Sub": "subtract",
    "Div": "divide",
    "Sqrt": "sqrt",
    "Cast": "cast",
    "ReduceMean": "reduce_mean",
    "ReduceMax": "reduce_max",
    "ReduceMin": "reduce_min",
    "ArgMax": "argmax",
    "Reshape": "reshape",
    "Neg": "neg",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Erf": "erf",
    "Pow": "pow",
    "Expand": "expand",
    "Unsqueeze": "reshape",
    "Shape": "shape_of",
    "ConstantOfShape": "constant_of_shape",
    "Trilu": "trilu",
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
    model_path: str | Path, default_batch: int | None = None,
    fold_constants: bool = True,
    preserve_shape_values: bool = False,
) -> ImportedONNXModel:
    model_path = Path(model_path)
    model = onnx.load(str(model_path))
    return import_onnx_model(model, default_batch=default_batch,
                             base_dir=model_path.parent,
                             fold_constants=fold_constants,
                             preserve_shape_values=preserve_shape_values)


def import_onnx_model(
    model: ModelProto,
    default_batch: int | None = None,
    base_dir: str | Path | None = None,
    fold_constants: bool = True,
    preserve_shape_values: bool = False,
) -> ImportedONNXModel:
    """把 ONNX 模型导入成 Relay 图规格。

    ``fold_constants`` 默认开启：先用 ONNX 参考实现求值掉纯静态子图，再进入
    唯一的节点导入链路。导出器留下的 mask 构造（``ConstantOfShape``/``Trilu``）
    和喂 ``Expand`` 的 ``Shape`` 链因此在导入前消失，不需要为它们新增运行时
    算子。关闭后按原始图导入，未映射算子照常报错。

    ``preserve_shape_values=True`` emits kxc.onnx_shape_source.v1: a concrete
    representative with original shape controls retained for the C++ restricted
    producer. This is a source graph, not an executable/static import. ONNX shape
    inference supplies representative metadata; it does not erase Shape calls.
    """
    if default_batch is not None and (
        not isinstance(default_batch, int)
        or isinstance(default_batch, bool)
        or default_batch <= 0
    ):
        raise ValueError("default_batch must be a positive integer when explicitly supplied")

    if not isinstance(preserve_shape_values, bool):
        raise ValueError("preserve_shape_values must be bool")
    for node in model.graph.node:
        if node.domain not in {"", "ai.onnx"}:
            raise UnsupportedONNXOpError(
                f"Unsupported ONNX domain '{node.domain}' in node '{node.name or node.op_type}'"
            )
    if preserve_shape_values and any(node.op_type in {"Tanh", "Erf"} for node in model.graph.node):
        raise UnsupportedONNXOpError("Tanh/Erf currently require the static import contract")
    if fold_constants:
        model, _fold_report = fold_static_subgraph(model)
    model = _binary_concats(model)
    if preserve_shape_values:
        model = onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True)
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
        if array.ndim > 0:
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
    reserved_values = set(value_info_by_name) | available_values
    reserved_values.update(name for node in graph.node for name in list(node.input) + list(node.output))
    for node in graph.node:
        if node.op_type == "Constant":
            _import_constant_node(
                node, input_specs, params, param_order, available_values,
                output_declarations, default_batch,
            )
            continue
        if node.op_type == "Identity":
            _import_identity_node(
                node, params, param_order, available_values, input_specs,
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
        elif node.op_type == "Split":
            if len(node.output) < 2 or not all(node.output):
                raise ValueError(
                    f"Split node '{node.name or '<unnamed>'}' requires at least two non-empty outputs"
                )
        elif len(node.output) != 1:
            raise ValueError(
                f"ONNX node '{node.name or node.op_type}' has {len(node.output)} outputs; "
                "only single-output nodes are supported in the static-shape MVP"
            )

        if node.op_type == "Shape":
            if len(node.input) != 1 or not node.input[0] or _attrs_by_name(node):
                raise ValueError("Shape requires one input and full-rank semantics without attributes")
            _, rank = _resolve_ranked_input("Shape", node.name, node.input[0], input_specs,
                                            params, inferred_static_specs, value_info_by_name)
            if not 1 <= rank <= 8:
                raise ValueError("Shape requires data rank in [1,8]")
            result = TensorSpec(node.output[0], [rank], "int64")
            _validate_declared_output("Shape", node.name, result, output_declarations, default_batch)
            inferred_static_specs[node.output[0]] = result

        if node.op_type == "ConstantOfShape":
            if not preserve_shape_values:
                raise UnsupportedONNXOpError("ConstantOfShape requires explicit shape-source mode or constant folding")
            fill_attrs = _shape_source_fill_attrs(node)
            if len(node.input) != 1 or not node.input[0]:
                raise ValueError("Shape-source ConstantOfShape requires one input")
            control = _resolve_static_input("ConstantOfShape", node.name, node.input[0],
                input_specs, params, inferred_static_specs, value_info_by_name, default_batch)
            if control.dtype != "int64" or len(control.shape) != 1:
                raise ValueError("Shape-source ConstantOfShape requires an int64 target vector")
            if fill_attrs["dtype_code"] == 2:
                if control.shape != [1]:
                    raise ValueError("Shape-source integer ConstantOfShape requires a one-element target vector")
                result = _tensor_spec_from_value_info(value_info_by_name[node.output[0]], default_batch)
                if result.dtype != "int64" or len(result.shape) != 1 or not 0 <= result.shape[0] <= 16:
                    raise ValueError("Shape-source ConstantOfShape supports int64 control vectors of length <=16")
                inferred_static_specs[node.output[0]] = result
            elif not 1 <= control.shape[0] <= 8:
                raise ValueError("Shape-source float32 ConstantOfShape requires target rank in [1,8]")

        if node.op_type == "Trilu":
            _trilu_attrs(node, params, opset_version)
            if preserve_shape_values and node.input[0] not in params:
                declaration = value_info_by_name.get(node.input[0])
                if declaration is None:
                    raise ValueError("Trilu requires input dtype/rank metadata")
                tensor = declaration.type.tensor_type
                if tensor.elem_type != TensorProto.FLOAT or not tensor.HasField("shape") or len(tensor.shape.dim) < 2:
                    raise ValueError("Trilu requires float32 input of rank >= 2")
            else:
                data = _resolve_static_input("Trilu", node.name, node.input[0], input_specs,
                    params, inferred_static_specs, value_info_by_name, default_batch)
                if data.dtype != "float32" or len(data.shape) < 2:
                    raise ValueError("Trilu requires float32 input of rank >= 2")
                result = TensorSpec(node.output[0], data.shape, data.dtype)
                _validate_declared_output("Trilu", node.name, result, output_declarations, default_batch)
                inferred_static_specs[node.output[0]] = result

        # Unresolved source extents remain ONNX metadata. The C++ producer
        # proves them and checks declared outputs; do not guess from a sample.
        deferred_shape = preserve_shape_values and any(
            name in value_info_by_name and any(not dim.HasField("dim_value")
                for dim in value_info_by_name[name].type.tensor_type.shape.dim)
            for name in node.input if name)
        if deferred_shape and node.op_type in {"Neg", "Sigmoid", "Pow", "Sqrt", "ReduceMean", "ReduceMax", "ReduceMin"}:
            arity = 2 if node.op_type == "Pow" else 1
            if len(node.input) != arity or not all(node.input):
                raise ValueError(f"Shape-source {node.op_type} requires {arity} nonempty inputs")
            if node.op_type in {"Neg", "Sigmoid", "Tanh", "Erf", "Pow"} and opset_version < M4M5_MATH_MIN_OPSET:
                raise UnsupportedONNXOpError(f"Shape-source {node.op_type} requires opset >=13")
            for name in node.input:
                dtype, _ = _resolve_ranked_input(node.op_type, node.name, name, input_specs,
                    params, inferred_static_specs, value_info_by_name)
                if dtype != "float32":
                    raise ValueError(f"Shape-source {node.op_type} requires float32 inputs")

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
            if preserve_shape_values:
                if len(node.input) != 5 or not all(node.input) or node.attribute:
                    raise ValueError("Shape-source Slice requires all five inputs and no attrs")
                for control_name in node.input[1:]:
                    control = _resolve_static_input("Slice", node_name, control_name,
                        input_specs, params, inferred_static_specs, value_info_by_name, default_batch)
                    if control.dtype != "int64" or len(control.shape) != 1 or not 1 <= control.shape[0] <= 16:
                        raise ValueError("Shape-source Slice requires nonempty int64 control vectors of length <=16")
            else:
                inferred_static_specs[node.output[0]] = _infer_slice_spec(
                    node, input_specs, params, inferred_static_specs, value_info_by_name,
                    output_declarations, default_batch,
                )
        if node.op_type == "Concat" and len(node.input) == 1 and preserve_shape_values:
            raise UnsupportedONNXOpError("Singleton Concat currently requires the static import contract")
        if node.op_type == "Concat" and not deferred_shape:
            node_name = node.name or "<unnamed>"
            if len(node.input) == 1 and node.input[0]:
                if node.input[0] not in available_values:
                    raise ValueError(f"Concat node '{node_name}' has unresolved prior input")
                source = _resolve_static_input("Concat", node_name, node.input[0],
                    input_specs, params, inferred_static_specs, value_info_by_name, default_batch)
                attrs = _attrs_by_name(node)
                if set(attrs) != {"axis"}:
                    raise ValueError(f"Concat node '{node_name}' requires exactly the axis attribute")
                axis = _int_attr(attrs, "axis", 0)
                if not source.shape or not -len(source.shape) <= axis < len(source.shape):
                    raise ValueError(f"Concat node '{node_name}' requires rank >= 1 and an in-range axis")
                # The existing binary op's zero-width identity preserves fresh
                # output storage and bit patterns, without a new copy operator.
                empty_shape = list(source.shape)
                empty_shape[axis] = 0
                empty_name = node.output[0] + "__kxc_concat_empty"
                while empty_name in reserved_values:
                    empty_name += "_"
                reserved_values.add(empty_name)
                params[empty_name] = ParamTensor(empty_name, empty_shape, source.dtype, b"")
                param_order.append(empty_name)
                available_values.add(empty_name)
                binary = onnx.NodeProto()
                binary.CopyFrom(node)
                binary.input.append(empty_name)
                node = binary
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
        if node.op_type == "MatMul" and not deferred_shape:
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
        if not deferred_shape and (node.op_type in {"Mul", "Sub", "Div"} or (
            node.op_type == "Add"
            and _inputs_are_resolvable(node, input_specs, params,
                                       inferred_static_specs, value_info_by_name)
        )):
            inferred_static_specs[node.output[0]] = _infer_binary_arithmetic_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch, allow_int64=preserve_shape_values,
            )
        if node.op_type == "Softmax" and not deferred_shape and _inputs_are_resolvable(
            node, input_specs, params, inferred_static_specs, value_info_by_name
        ):
            inferred_static_specs[node.output[0]] = _infer_softmax_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "Transpose" and not deferred_shape and _inputs_are_resolvable(
            node, input_specs, params, inferred_static_specs, value_info_by_name
        ):
            inferred_static_specs[node.output[0]] = _infer_transpose_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "Sqrt" and not deferred_shape:
            inferred_static_specs[node.output[0]] = _infer_sqrt_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch,
            )
        if node.op_type == "Cast" and not deferred_shape:
            inferred_static_specs[node.output[0]] = _infer_cast_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch, allow_shape_control=preserve_shape_values,
            )
        if node.op_type == "ArgMax" and not deferred_shape:
            inferred_static_specs[node.output[0]] = _infer_argmax_spec(
                node, input_specs, params, inferred_static_specs,
                value_info_by_name, output_declarations, default_batch,
                opset_version,
            )
        if node.op_type in {"ReduceMean", "ReduceMax", "ReduceMin"} and not deferred_shape:
            inferred_static_specs[node.output[0]] = _infer_reduce_spec(
                node.op_type, node, input_specs, params, inferred_static_specs,
                value_info_by_name, output_declarations, default_batch,
                opset_version,
            )
        if node.op_type == "Reshape":
            if preserve_shape_values:
                attrs = _attrs_by_name(node)
                if set(attrs) - {"allowzero"} or _int_attr(attrs, "allowzero", 0) != 0:
                    raise ValueError("Shape-source Reshape requires allowzero=0 and no other attrs")
                if len(node.input) != 2 or not all(node.input):
                    raise ValueError("Shape-source Reshape requires data and shape inputs")
                control = _resolve_static_input("Reshape", node.name, node.input[1], input_specs,
                    params, inferred_static_specs, value_info_by_name, default_batch)
                if control.dtype != "int64" or len(control.shape) != 1 or control.shape[0] < 1:
                    raise ValueError("Shape-source Reshape requires a nonempty int64 shape vector")
                declaration = value_info_by_name.get(node.output[0])
                if declaration is None or not declaration.type.tensor_type.HasField("shape"):
                    raise ValueError("Shape-source Reshape requires known-rank metadata")
                dimensions = declaration.type.tensor_type.shape.dim
                if len(dimensions) != control.shape[0]:
                    raise ValueError("Shape-source Reshape output rank differs from control length")
                if all(dim.HasField("dim_value") for dim in dimensions):
                    inferred_static_specs[node.output[0]] = _tensor_spec_from_value_info(declaration, default_batch)
            else:
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
        if node.op_type in {"Neg", "Sigmoid", "Tanh", "Erf"} and not deferred_shape:
            inferred_static_specs[node.output[0]] = _infer_unary_math_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch, opset_version,
            )
        if node.op_type == "Pow" and not deferred_shape:
            inferred_static_specs[node.output[0]] = _infer_pow_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch, opset_version,
            )
        if node.op_type == "Expand":
            if preserve_shape_values:
                if opset_version < 13 or _attrs_by_name(node) or len(node.input) != 2 or not all(node.input):
                    raise ValueError("Shape-source Expand requires opset >=13, two inputs and no attrs")
                dtype, rank = _resolve_ranked_input("Expand", node.name, node.input[0], input_specs,
                    params, inferred_static_specs, value_info_by_name)
                control = _resolve_static_input("Expand", node.name, node.input[1], input_specs,
                    params, inferred_static_specs, value_info_by_name, default_batch)
                if dtype != "float32" or control.dtype != "int64" or control.shape != [rank]:
                    raise ValueError("Shape-source Expand requires float32 data and a same-rank int64 target")
                declaration = value_info_by_name.get(node.output[0])
                if declaration is None or not declaration.type.tensor_type.HasField("shape"):
                    raise ValueError("Shape-source Expand requires known representative output rank")
                output_type = declaration.type.tensor_type
                if onnx_dtype_to_kxc(output_type.elem_type) != dtype or len(output_type.shape.dim) != rank:
                    raise ValueError("Shape-source Expand representative output rank/dtype mismatch")
                # ONNX shape inference does not propagate Where's control values.
                # Keep unresolved intermediate extents for the C++ producer;
                # declared graph outputs still require concrete representatives.
                if all(dimension.HasField("dim_value") for dimension in output_type.shape.dim):
                    inferred_static_specs[node.output[0]] = _tensor_spec_from_value_info(declaration, default_batch)
            else:
                inferred_static_specs[node.output[0]] = _infer_expand_spec(
                    node, input_specs, params, inferred_static_specs, value_info_by_name,
                    output_declarations, default_batch, opset_version,
                )
        if node.op_type == "Unsqueeze" and not deferred_shape:
            inferred_static_specs[node.output[0]] = _infer_unsqueeze_spec(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch, opset_version,
            )
        if node.op_type == "Split":
            split_specs = _infer_split_specs(
                node, input_specs, params, inferred_static_specs, value_info_by_name,
                output_declarations, default_batch, opset_version,
            )
            for spec in split_specs:
                inferred_static_specs[spec.name] = spec

        missing = [name for name in node.input if name and name not in available_values]
        if missing:
            raise ValueError(
                f"ONNX node '{node.name or node.op_type}' has missing input(s): {missing}"
            )

        relay_inputs = ([node.input[0]]
                        if (node.op_type in {"Unsqueeze", "Trilu", "Split"} or
                            (node.op_type == "Slice" and not preserve_shape_values) or
                            (node.op_type in {"Reshape", "Expand"} and not preserve_shape_values))
                        else [name for name in node.input if name])
        relay_outputs = ([node.output[0]] if node.op_type == "LayerNormalization"
                         else [name for name in node.output])
        op_name = ONNX_TO_RELAY[node.op_type]
        if preserve_shape_values and node.op_type == "Reshape":
            op_name, relay_attrs = "reshape_dynamic", {}
        elif preserve_shape_values and node.op_type == "Expand":
            op_name, relay_attrs = "expand_dynamic", {}
        elif preserve_shape_values and node.op_type == "Slice":
            relay_attrs = {}
        elif preserve_shape_values and node.op_type == "Cast":
            target = _cast_target_dtype(node, allow_shape_control=True)
            if deferred_shape:
                dtype = onnx_dtype_to_kxc(value_info_by_name[node.input[0]].type.tensor_type.elem_type)
                if (dtype, target) not in CAST_SUPPORTED_CONVERSIONS and not dtype == target == "int64":
                    raise ValueError("Shape-source Cast supports float32 conversions or int64 identity controls")
            relay_attrs = {"to": RELAY_CAST_DTYPE_CODES[target]}
        elif preserve_shape_values and node.op_type == "ConstantOfShape":
            relay_attrs = _shape_source_fill_attrs(node)
        elif preserve_shape_values and node.op_type == "Unsqueeze":
            if opset_version < M4M5_MATH_MIN_OPSET:
                raise UnsupportedONNXOpError("Shape-source Unsqueeze requires opset >=13")
            _, rank = _resolve_ranked_input("Unsqueeze", node.name, node.input[0], input_specs,
                params, inferred_static_specs, value_info_by_name)
            _unsqueeze_axes(node, params, rank)
            op_name = "unsqueeze"
            relay_attrs = {"axes": _int64_constant_vector("Unsqueeze", node.name, node.input[1], params)}
        else:
            relay_attrs = _convert_attrs(node, params, value_info_by_name, opset_version,
                                        input_specs, inferred_static_specs)
        nodes.append(
            RelayNodeSpec(
                name=node.name or f"{node.op_type}_{len(nodes)}",
                op_name=op_name,
                inputs=relay_inputs,
                outputs=relay_outputs,
                attrs=relay_attrs,
            )
        )
        available_values.update(relay_outputs)

    return ImportedONNXModel(
        function=RelayFunctionSpec(inputs=inputs, outputs=outputs, nodes=nodes),
        params=params,
        param_order=param_order,
        preserve_shape_values=preserve_shape_values,
    )


def _shape_source_fill_attrs(node: onnx.NodeProto) -> dict[str, Any]:
    if (len(node.attribute) != 1 or node.attribute[0].name != "value"
            or node.attribute[0].type != AttributeProto.TENSOR):
        raise ValueError("Shape-source ConstantOfShape requires an explicit tensor value")
    array = numpy_helper.to_array(node.attribute[0].t)
    if array.dtype == np.float32 and array.shape == (1,) and not np.isnan(array[0]):
        # Exact IEEE bits keep +/-inf valid in strict JSON, without a second
        # expression evaluator or nonstandard JSON Infinity tokens.
        return {"dtype_code": RELAY_CAST_DTYPE_CODES["float32"],
                "value_bits": int(array.view(np.uint32)[0])}
    if array.dtype != np.int64 or array.shape != (1,) or abs(int(array[0])) > 2**53:
        raise ValueError("Shape-source ConstantOfShape requires one exactly representable int64 fill")
    return {"dtype_code": RELAY_CAST_DTYPE_CODES["int64"], "value": int(array[0])}


def _trilu_attrs(node: onnx.NodeProto, params: dict[str, ParamTensor], opset: int) -> dict[str, int]:
    if opset < 14:
        raise UnsupportedONNXOpError(f"Unsupported ONNX Trilu opset {opset} in node '{node.name}': requires >=14")
    attrs = _attrs_by_name(node)
    upper = _int_attr(attrs, "upper", 1)
    if set(attrs) - {"upper"} or upper not in {0, 1}:
        raise ValueError("Trilu requires upper 0 or 1 and no other attributes")
    if len(node.input) not in {1, 2} or not node.input[0]:
        raise ValueError("Trilu requires data and an optional constant scalar diagonal")
    k = 0
    if len(node.input) == 2 and node.input[1]:
        diagonal = params.get(node.input[1])
        if diagonal is None or diagonal.dtype != "int64" or diagonal.shape != []:
            raise ValueError("Trilu diagonal must be a static int64 scalar")
        k = int(np.frombuffer(diagonal.data, dtype=np.int64)[0])
    return {"upper": upper, "k": k}


def _binary_concats(model: ModelProto) -> ModelProto:
    """Normalize variadic ONNX Concat to the existing ordered binary Relay op.

    Static and shape-source imports share this conversion. Reserve every value
    mention, including unresolved inputs and metadata, so generated names cannot
    repair an invalid graph by capturing a missing value. No data are evaluated.
    """
    if not any(node.op_type == "Concat" and len(node.input) > 2 for node in model.graph.node):
        return model
    result = ModelProto()
    result.CopyFrom(model)
    used = {v.name for v in list(model.graph.input) + list(model.graph.initializer)
            + list(model.graph.output) + list(model.graph.value_info)}
    used.update(v for node in model.graph.node for v in list(node.input) + list(node.output))
    used_node_names = {node.name for node in model.graph.node}
    nodes = []
    for node in model.graph.node:
        if node.op_type != "Concat" or len(node.input) <= 2:
            nodes.append(node)
            continue
        if len(node.output) != 1 or not node.output[0] or not all(node.input):
            raise ValueError("Concat requires nonempty inputs and one output")
        attrs = _attrs_by_name(node)
        if set(attrs) != {"axis"}:
            raise ValueError("Concat requires exactly the axis attribute")
        axis = _int_attr(attrs, "axis", 0)
        previous = node.input[0]
        for index, value in enumerate(node.input[1:]):
            output = node.output[0]
            if index != len(node.input) - 2:
                output += f"__kxc_concat_{index}"
                while output in used:
                    output += "_"
                used.add(output)
            name = f"{node.name or 'Concat'}_part{index}"
            while name in used_node_names:
                name += "_"
            used_node_names.add(name)
            nodes.append(onnx.helper.make_node("Concat", [previous, value], [output],
                         name=name, domain=node.domain, axis=axis))
            previous = output
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    return result


def _resolve_ranked_input(
    op_type: str, node_name: str, name: str,
    input_specs: dict[str, TensorSpec], params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec], value_info_by_name: dict[str, onnx.ValueInfoProto],
) -> tuple[str, int]:
    """Read dtype and fixed rank without inventing unresolved source extents."""
    for specs in (inferred_specs, params, input_specs):
        if name in specs:
            return specs[name].dtype, len(specs[name].shape)
    value = value_info_by_name.get(name)
    if value is None or not value.type.tensor_type.HasField("shape"):
        raise ValueError(f"{op_type} node '{node_name}' input '{name}' requires known-rank metadata")
    tensor = value.type.tensor_type
    return onnx_dtype_to_kxc(tensor.elem_type), len(tensor.shape.dim)


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


def _import_identity_node(
    node: onnx.NodeProto,
    params: dict[str, ParamTensor],
    param_order: list[str],
    available_values: set[str],
    input_specs: dict[str, TensorSpec],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> None:
    """Normalize an ONNX Identity node into a parameter alias.

    ``torch.onnx.export`` dedupes byte-identical initializers and re-exposes the
    survivors through Identity, so every Identity in the locked MiniMind export
    aliases one initializer under a second weight name. Routing that through the
    existing ParamTensor path keeps the pass-through free: no Relay node and no
    copy kernel is emitted, which is why Identity never appears in the folded
    operator inventory.

    Aliasing a computed value would need either downstream input rewriting or a
    real copy unit; it stays rejected rather than binding a name nothing produces.
    """
    node_name = node.name or "<unnamed>"
    if len(node.input) != 1 or not node.input[0]:
        raise ValueError(
            f"Identity node '{node_name}' requires exactly one non-empty input name"
        )
    if len(node.output) != 1 or not node.output[0]:
        raise ValueError(
            f"Identity node '{node_name}' requires exactly one non-empty output name"
        )
    source, name = node.input[0], node.output[0]
    if source not in available_values:
        raise ValueError(
            f"Identity node '{node_name}' has unresolved input '{source}'"
        )
    if name in params or name in input_specs:
        raise ValueError(
            f"Identity node '{node_name}' output '{name}' collides with an existing "
            "initializer, Constant or graph input"
        )
    if source not in params:
        raise UnsupportedONNXOpError(
            f"Identity node '{node_name}' aliases '{source}', which is not an initializer "
            "or Constant; only parameter aliases are supported in the static-shape MVP"
        )
    aliased = params[source]
    param_order.append(name)
    params[name] = ParamTensor(
        name=name,
        shape=list(aliased.shape),
        dtype=aliased.dtype,
        data=aliased.data,
    )
    available_values.add(name)
    _validate_declared_output(
        "Identity", node_name,
        TensorSpec(name=name, shape=params[name].shape, dtype=params[name].dtype),
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
    allow_int64: bool = False,
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
    allowed_dtypes = ADD_DTYPES if op_type == "Add" else ARITHMETIC_DTYPES
    if allow_int64 and op_type in {"Add", "Sub", "Mul", "Div"}:
        allowed_dtypes = allowed_dtypes | {"int64"}
        if lhs.dtype != rhs.dtype:
            raise ValueError(f"{op_type} shape-source operands require matching dtypes")
    if lhs.dtype not in allowed_dtypes or rhs.dtype not in allowed_dtypes:
        if op_type == "Add":
            raise ValueError(
                f"Add node '{node_name}' requires float32 or int64 inputs in the "
                f"static S1 subset; got {lhs.dtype} and {rhs.dtype}"
            )
        raise ValueError(
            f"{op_type} node '{node_name}' requires float32 inputs in the static "
            f"S1 subset; got {lhs.dtype} and {rhs.dtype}"
        )
    result = TensorSpec(
        name=node.output[0],
        shape=_broadcast_shapes(op_type, node_name, lhs.shape, rhs.shape),
        dtype=lhs.dtype,
    )
    _validate_declared_output(op_type, node_name, result, output_declarations, default_batch)
    return result




def _inputs_are_resolvable(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
) -> bool:
    """节点的每个输入是否都有可用元数据。

    Add/Softmax/Transpose 进入推导链只是为了**给下游填元数据**，本身不消费结论。
    当某个输入的元数据确实拿不到时（例如 resnet18 里 Conv 的输出既无 value_info
    也不在推导链内），它们保持沉默，把报错留给真正需要该元数据的消费者——这与
    加入它们之前的行为一致，不会把原先能导入的图变成失败。
    """
    for name in node.input:
        if not name:
            continue
        if name in inferred_specs or name in params or name in input_specs:
            continue
        if value_info_by_name.get(name) is not None:
            continue
        return False
    return True


def _infer_softmax_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    """Softmax 保形保 dtype；只校验 axis 落在秩内（opset 13 起 axis 不做 coerce）。"""
    node_name = node.name or "<unnamed>"
    if len(node.input) != 1 or not node.input[0]:
        raise ValueError(f"Softmax node '{node_name}' requires exactly one non-empty input")
    data = _resolve_static_input("Softmax", node_name, node.input[0], input_specs, params,
                                 inferred_specs, value_info_by_name, default_batch)
    if data.dtype not in ARITHMETIC_DTYPES:
        raise ValueError(
            f"Softmax node '{node_name}' requires float32 input in the static S1 "
            f"subset; got {data.dtype}"
        )
    attrs = _attrs_by_name(node)
    unsupported = set(attrs) - {"axis"}
    if unsupported:
        raise ValueError(
            f"Softmax node '{node_name}' has unsupported attribute(s): {sorted(unsupported)}"
        )
    rank = len(data.shape)
    axis = _int_attr(attrs, "axis", -1)
    if axis < 0:
        axis += rank
    if rank < 1 or axis < 0 or axis >= rank:
        raise ValueError(f"Softmax node '{node_name}' axis is out of range for rank {rank}")
    result = TensorSpec(name=node.output[0], shape=list(data.shape), dtype="float32")
    _validate_declared_output("Softmax", node_name, result, output_declarations, default_batch)
    return result


def _infer_transpose_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
) -> TensorSpec:
    """Transpose 按 perm 重排轴；perm 必须是秩的一个完整置换，缺省为逆序。"""
    node_name = node.name or "<unnamed>"
    if len(node.input) != 1 or not node.input[0]:
        raise ValueError(f"Transpose node '{node_name}' requires exactly one non-empty input")
    data = _resolve_static_input("Transpose", node_name, node.input[0], input_specs, params,
                                 inferred_specs, value_info_by_name, default_batch)
    attrs = _attrs_by_name(node)
    unsupported = set(attrs) - {"perm"}
    if unsupported:
        raise ValueError(
            f"Transpose node '{node_name}' has unsupported attribute(s): {sorted(unsupported)}"
        )
    rank = len(data.shape)
    perm = [int(axis) for axis in _list_attr(attrs, "perm", list(reversed(range(rank))))]
    if sorted(perm) != list(range(rank)):
        raise ValueError(
            f"Transpose node '{node_name}' perm {perm} is not a permutation of rank {rank}"
        )
    result = TensorSpec(
        name=node.output[0],
        shape=[data.shape[axis] for axis in perm],
        dtype=data.dtype,
    )
    _validate_declared_output("Transpose", node_name, result, output_declarations, default_batch)
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


def _cast_target_dtype(node: onnx.NodeProto, *, allow_shape_control: bool = False) -> str:
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
    if target != "float32" and not (allow_shape_control and target == "int64"):
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
    *, allow_shape_control: bool = False,
) -> TensorSpec:
    node_name = node.name or "<unnamed>"
    if len(node.input) != 1 or not node.input[0]:
        raise ValueError(f"Cast node '{node_name}' requires exactly one non-empty input")
    data = _resolve_static_input("Cast", node_name, node.input[0], input_specs, params,
                                 inferred_specs, value_info_by_name, default_batch)
    target = _cast_target_dtype(node, allow_shape_control=allow_shape_control)
    if (data.dtype, target) not in CAST_SUPPORTED_CONVERSIONS and not (
        allow_shape_control and data.dtype == target == "int64"):
        raise ValueError(
            f"Cast node '{node_name}' supports only int32/int64 to float32 in the "
            f"static S1 subset; got {data.dtype} to {target}"
        )
    result = TensorSpec(name=node.output[0], shape=list(data.shape), dtype=target)
    _validate_declared_output("Cast", node_name, result, output_declarations, default_batch)
    return result


def _reduce_attrs(op_type: str, node: onnx.NodeProto, opset_version: int) -> dict[str, Any]:
    """Validate ReduceMean/ReduceMax/ReduceMin attributes for the opset-17 subset.

    The axes-attribute form (opset <= 17) is required; the opset-18 input form
    is never opened. Absent or empty axes mean "reduce all axes".
    """
    node_name = node.name or "<unnamed>"
    if opset_version >= 18:
        raise UnsupportedONNXOpError(
            f"Unsupported ONNX {op_type} opset {opset_version} in node "
            f"'{node_name}': the input-form opset >= 18 is not supported; "
            "the static axes-attribute form (opset <= 17) is required"
        )
    attrs = _attrs_by_name(node)
    unsupported = set(attrs) - {"axes", "keepdims"}
    if unsupported:
        raise ValueError(
            f"{op_type} node '{node_name}' has unsupported attribute(s): "
            f"{sorted(unsupported)}"
        )
    axes = _list_attr(attrs, "axes", [])
    keepdims = _int_attr(attrs, "keepdims", 1)
    if keepdims not in (0, 1):
        raise ValueError(
            f"{op_type} node '{node_name}' keepdims must be 0 or 1; got {keepdims}"
        )
    if len(axes) != len(set(axes)):
        raise ValueError(
            f"{op_type} node '{node_name}' axes must not contain duplicates"
        )
    return {"axes": axes, "keepdims": keepdims}


def _argmax_attrs(node: onnx.NodeProto, opset_version: int) -> dict[str, Any]:
    """Validate ONNX ArgMax attributes for the opset-17 static subset."""
    node_name = node.name or "<unnamed>"
    if opset_version < 12:
        raise UnsupportedONNXOpError(
            f"Unsupported ONNX ArgMax opset {opset_version} in node '{node_name}': "
            "the select_last_index attribute requires opset >= 12"
        )
    if len(node.input) != 1 or not node.input[0]:
        raise ValueError(
            f"ArgMax node '{node_name}' requires exactly one non-empty input"
        )
    attrs = _attrs_by_name(node)
    unsupported = set(attrs) - {"axis", "keepdims", "select_last_index"}
    if unsupported:
        raise ValueError(
            f"ArgMax node '{node_name}' has unsupported attribute(s): "
            f"{sorted(unsupported)}"
        )
    axis = _int_attr(attrs, "axis", 0)
    keepdims = _int_attr(attrs, "keepdims", 1)
    select_last_index = _int_attr(attrs, "select_last_index", 0)
    if keepdims not in (0, 1) or select_last_index not in (0, 1):
        raise ValueError(
            f"ArgMax node '{node_name}' keepdims/select_last_index must be 0 or 1"
        )
    return {"axis": axis, "keepdims": keepdims,
            "select_last_index": select_last_index}


def _infer_argmax_spec(
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
    attrs = _argmax_attrs(node, opset_version)
    data = _resolve_static_input("ArgMax", node_name, node.input[0], input_specs,
                                 params, inferred_specs, value_info_by_name, default_batch)
    if data.dtype not in ARITHMETIC_DTYPES:
        raise ValueError(
            f"ArgMax node '{node_name}' requires float32 input in the static S1 "
            f"subset; got {data.dtype}"
        )
    if any(dim < 0 for dim in data.shape):
        raise ValueError(
            f"ArgMax node '{node_name}' requires non-negative static dimensions"
        )
    rank = len(data.shape)
    axis = attrs["axis"]
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(
            f"ArgMax node '{node_name}' axis {attrs['axis']} is out of range for rank {rank}"
        )
    output_shape: list[int] = []
    for index, dim in enumerate(data.shape):
        if index == axis:
            if attrs["keepdims"]:
                output_shape.append(1)
        else:
            output_shape.append(dim)
    result = TensorSpec(name=node.output[0], shape=output_shape, dtype="int64")
    _validate_declared_output(
        "ArgMax", node_name, result, output_declarations, default_batch
    )
    return result


def _infer_reduce_spec(
    op_type: str,
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
    attrs = _reduce_attrs(op_type, node, opset_version)
    if len(node.input) != 1 or not node.input[0]:
        raise ValueError(
            f"{op_type} node '{node_name}' requires exactly one non-empty input"
        )
    data = _resolve_static_input(op_type, node_name, node.input[0], input_specs,
                                 params, inferred_specs, value_info_by_name, default_batch)
    if data.dtype not in ARITHMETIC_DTYPES:
        raise ValueError(
            f"{op_type} node '{node_name}' requires float32 input in the static "
            f"S1 subset; got {data.dtype}"
        )
    if any(dim < 0 for dim in data.shape):
        raise ValueError(
            f"{op_type} node '{node_name}' requires non-negative static dimensions"
        )
    rank = len(data.shape)
    normalized: list[int] = []
    for axis in attrs["axes"]:
        resolved = axis + rank if axis < 0 else axis
        if resolved < 0 or resolved >= rank:
            raise ValueError(
                f"{op_type} node '{node_name}' axis {axis} is out of range for rank {rank}"
            )
        normalized.append(resolved)
    if not normalized:
        # ONNX opset <= 17: absent or empty axes reduce all axes.
        normalized = list(range(rank))
    for axis in set(normalized):
        if data.shape[axis] == 0:
            raise ValueError(
                f"{op_type} node '{node_name}' reduces over zero-extent axis {axis}; "
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
        op_type, node_name, result, output_declarations, default_batch
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
    extent = data.shape[axis]
    if node.input[1] in params:
        # 常量索引：在导入期就把每个索引证明在 ONNX 域内，保留比运行时守卫
        # 更强的结论。
        index_param = params[node.input[1]]
        dtype = np.dtype("<i4" if index_param.dtype == "int32" else "<i8")
        values = np.frombuffer(index_param.data, dtype=dtype)
        expected_values = math.prod(index_param.shape)
        if values.size != expected_values:
            raise ValueError(
                f"Gather node '{node_name}' indices initializer byte size is invalid"
            )
        for value in values:
            index = int(value)
            if index < -extent or index >= extent:
                raise ValueError(
                    f"Gather node '{node_name}' constant index {index} is outside "
                    f"the ONNX domain [{-extent}, {extent - 1}]"
                )
    # 运行时索引（embedding 查表就是这一支）：索引值到 launch 时才存在，导入期
    # 无法证明其范围。生产 lowering 的 GatherCompute 已经带守卫——负索引按
    # ONNX 语义折回，越界经 Select 取零且**不形成越界 Load**。因此这里只固定
    # 形状、dtype、axis 与可寻址性合同，值域交给已 lower 的守卫。
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


def _split_sections(
    node: onnx.NodeProto, params: dict[str, ParamTensor], opset_version: int
) -> tuple[int, list[int]]:
    """Resolve static constant Split sections from attrs or an initializer."""
    node_name = node.name or "<unnamed>"
    if opset_version < 2:
        raise UnsupportedONNXOpError(
            f"Unsupported ONNX Split opset {opset_version} in node '{node_name}'"
        )
    attrs = _attrs_by_name(node)
    unsupported = set(attrs) - {"axis", "split"}
    if unsupported:
        raise ValueError(
            f"Split node '{node_name}' has unsupported attribute(s): {sorted(unsupported)}"
        )
    axis = _int_attr(attrs, "axis", 0)
    has_attr = "split" in attrs
    if has_attr and len(node.input) != 1:
        raise ValueError(
            f"Split node '{node_name}' cannot combine split attribute with a second input"
        )
    if has_attr:
        sections = _list_attr(attrs, "split", [])
    elif len(node.input) == 2 and node.input[1]:
        sections = _int64_constant_vector("Split", node_name, node.input[1], params)
    else:
        raise ValueError(
            f"Split node '{node_name}' requires a constant split attribute or int64 initializer input"
        )
    if len(node.input) not in {1, 2} or not node.input[0]:
        raise ValueError(f"Split node '{node_name}' requires one data input")
    if len(sections) < 2:
        raise ValueError(
            f"Split node '{node_name}' requires at least two constant sections"
        )
    if any(type(section) is not int or section < 0 for section in sections):
        raise ValueError(
            f"Split node '{node_name}' sections must be non-negative integers"
        )
    return axis, sections


def _infer_split_specs(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
    opset_version: int,
) -> list[TensorSpec]:
    node_name = node.name or "<unnamed>"
    if len(node.output) < 2 or not all(node.output):
        raise ValueError(
            f"Split node '{node_name}' requires at least two non-empty outputs"
        )
    if len(node.input) not in {1, 2} or not node.input[0]:
        raise ValueError(f"Split node '{node_name}' requires one data input")
    data = _resolve_static_input(
        "Split", node_name, node.input[0], input_specs, params,
        inferred_specs, value_info_by_name, default_batch
    )
    if data.dtype not in CONCATENATE_DTYPES:
        raise ValueError(
            f"Split node '{node_name}' requires dtype in "
            "{float32,float64,int32,int64,int8,uint8,bool}"
        )
    if not data.shape:
        raise ValueError(f"Split node '{node_name}' requires data rank >= 1")
    axis, sections = _split_sections(node, params, opset_version)
    if len(node.output) != len(sections):
        raise ValueError(
            f"Split node '{node_name}' output count must equal the number of sections"
        )
    if axis < 0:
        axis += len(data.shape)
    if axis < 0 or axis >= len(data.shape):
        raise ValueError(f"Split node '{node_name}' axis is out of range")
    if any(dim < 0 for dim in data.shape):
        raise ValueError(
            f"Split node '{node_name}' requires non-negative static data dimensions"
        )
    if sum(sections) != data.shape[axis]:
        raise ValueError(
            f"Split node '{node_name}' sections must sum to the input axis extent"
        )
    outputs: list[TensorSpec] = []
    for name, section in zip(node.output, sections):
        shape = list(data.shape)
        shape[axis] = section
        result = TensorSpec(name=name, shape=shape, dtype=data.dtype)
        _validate_declared_output("Split", node_name, result,
                                  output_declarations, default_batch)
        outputs.append(result)
    return outputs


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


def _infer_unary_math_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
    opset_version: int,
) -> TensorSpec:
    """Infer the static output of the fieldless float32 unary ops (Neg, Sigmoid, Tanh, Erf).

    Both ops keep shape and dtype: one float32 input, one float32 output, no
    attributes. The multidirectional-broadcast input form (opset >= 13) is the
    declared boundary; earlier opsets are rejected with the node name.
    """
    op_type = node.op_type
    node_name = node.name or "<unnamed>"
    if opset_version < M4M5_MATH_MIN_OPSET:
        raise UnsupportedONNXOpError(
            f"Unsupported ONNX {op_type} opset {opset_version} in node "
            f"'{node_name}': the opset >= {M4M5_MATH_MIN_OPSET} form is required"
        )
    if len(node.input) != 1 or not node.input[0]:
        raise ValueError(f"{op_type} node '{node_name}' requires exactly one non-empty input")
    data = _resolve_static_input(op_type, node_name, node.input[0], input_specs, params,
                                 inferred_specs, value_info_by_name, default_batch)
    if data.dtype not in ARITHMETIC_DTYPES:
        raise ValueError(
            f"{op_type} node '{node_name}' requires float32 input in the M4/M5 static "
            f"subset; got {data.dtype}"
        )
    result = TensorSpec(name=node.output[0], shape=list(data.shape), dtype="float32")
    _validate_declared_output(op_type, node_name, result, output_declarations, default_batch)
    return result


def _infer_pow_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
    opset_version: int,
) -> TensorSpec:
    """Infer the static output of a Pow node under the M5 S2 float32 subset.

    Pow is fieldless: two same-dtype float32 inputs (base and exponent) with
    multidirectional trailing-axis broadcast. Integer and float64 inputs are
    rejected; the ONNX boundary is the opset >= 13 broadcast form.
    """
    node_name = node.name or "<unnamed>"
    if opset_version < M4M5_MATH_MIN_OPSET:
        raise UnsupportedONNXOpError(
            f"Unsupported ONNX Pow opset {opset_version} in node "
            f"'{node_name}': the opset >= {M4M5_MATH_MIN_OPSET} form is required"
        )
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(f"Pow node '{node_name}' requires exactly two non-empty inputs")
    lhs, rhs = (
        _resolve_static_input("Pow", node_name, name, input_specs, params,
                              inferred_specs, value_info_by_name, default_batch)
        for name in node.input
    )
    if lhs.dtype not in ARITHMETIC_DTYPES or rhs.dtype not in ARITHMETIC_DTYPES:
        raise ValueError(
            f"Pow node '{node_name}' requires same-dtype float32 inputs in the "
            f"M4/M5 static subset; got {lhs.dtype} and {rhs.dtype}"
        )
    if lhs.dtype != rhs.dtype:
        raise ValueError(
            f"Pow node '{node_name}' requires matching input dtypes; "
            f"got {lhs.dtype} and {rhs.dtype}"
        )
    result = TensorSpec(
        name=node.output[0],
        shape=_broadcast_shapes("Pow", node_name, lhs.shape, rhs.shape),
        dtype="float32",
    )
    _validate_declared_output("Pow", node_name, result, output_declarations, default_batch)
    return result


def _int64_constant_vector(
    op_type: str, node_name: str, name: str, params: dict[str, ParamTensor]
) -> list[int]:
    """Read a rank-1 int64 initializer/Constant payload as a list of ints.

    Shared by the Expand target shape and the Unsqueeze axes control inputs:
    the tensor must come from the static constant table, never from a
    dynamically produced value (the M3 shape-value slice owns that).
    """
    shape_param = params.get(name)
    if shape_param is None:
        raise ValueError(
            f"{op_type} node '{node_name}' control input '{name}' must be a static "
            "initializer or Constant node output; dynamic shape inputs are not "
            "supported in the M4/M5 static subset"
        )
    if shape_param.dtype != "int64":
        raise ValueError(
            f"{op_type} node '{node_name}' control input '{name}' must be int64; "
            f"got {shape_param.dtype}"
        )
    if len(shape_param.shape) != 1:
        raise ValueError(
            f"{op_type} node '{node_name}' control input '{name}' must be rank-1; "
            f"got rank {len(shape_param.shape)}"
        )
    if len(shape_param.data) != 8 * shape_param.shape[0]:
        raise ValueError(
            f"{op_type} node '{node_name}' control initializer '{name}' byte size "
            "is invalid"
        )
    return [int(value) for value in np.frombuffer(shape_param.data, dtype="<i8")]


def _expand_attrs(node: onnx.NodeProto, params: dict[str, ParamTensor]) -> dict[str, Any]:
    """Resolve the Expand constant shape control input into canonical attrs."""
    node_name = node.name or "<unnamed>"
    if _attrs_by_name(node):
        raise ValueError(f"Expand node '{node_name}' does not support attributes")
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(
            f"Expand node '{node_name}' requires exactly two non-empty inputs "
            "(data and target shape)"
        )
    values = _int64_constant_vector("Expand", node_name, node.input[1], params)
    for dim in values:
        if dim < 0:
            raise ValueError(
                f"Expand node '{node_name}' target shape dimensions must be "
                f"non-negative; got {dim}"
            )
    return {"target_shape": values}


def _infer_expand_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
    opset_version: int,
) -> TensorSpec:
    """Infer the static output of an Expand node under the M4/M5 static subset.

    The target shape must come from an initializer/Constant (unidirectional
    numpy broadcast_to rules: output rank equals the target rank and each
    aligned data dimension is 1 or equal to the target dimension). Dynamic
    shape inputs are rejected here and handed to the M3 shape-value slice.
    """
    node_name = node.name or "<unnamed>"
    if opset_version < M4M5_MATH_MIN_OPSET:
        raise UnsupportedONNXOpError(
            f"Unsupported ONNX Expand opset {opset_version} in node "
            f"'{node_name}': the opset >= {M4M5_MATH_MIN_OPSET} form is required"
        )
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(
            f"Expand node '{node_name}' requires exactly two non-empty inputs"
        )
    data = _resolve_static_input("Expand", node_name, node.input[0], input_specs, params,
                                 inferred_specs, value_info_by_name, default_batch)
    if any(dim < 0 for dim in data.shape):
        raise ValueError(
            f"Expand node '{node_name}' requires non-negative static data dimensions"
        )
    target = _expand_attrs(node, params)["target_shape"]
    if len(data.shape) > len(target):
        raise ValueError(
            f"Expand node '{node_name}' data rank {len(data.shape)} must not exceed "
            f"the target rank {len(target)}"
        )
    offset = len(target) - len(data.shape)
    for index, dim in enumerate(data.shape):
        if dim != target[offset + index] and dim != 1:
            raise ValueError(
                f"Expand node '{node_name}' data dimension {dim} at axis {index} must "
                f"be 1 or equal to the target dimension {target[offset + index]}"
            )
    result = TensorSpec(name=node.output[0], shape=list(target), dtype=data.dtype)
    _validate_declared_output("Expand", node_name, result, output_declarations, default_batch)
    return result


def _unsqueeze_axes(
    node: onnx.NodeProto, params: dict[str, ParamTensor], data_rank: int
) -> list[int]:
    """Validate static axes using fixed rank, independently of input extents."""
    node_name = node.name or "<unnamed>"
    if _attrs_by_name(node):
        raise ValueError(f"Unsqueeze node '{node_name}' does not support attributes")
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(
            f"Unsqueeze node '{node_name}' requires exactly two non-empty inputs "
            "(data and axes)"
        )
    axes = _int64_constant_vector("Unsqueeze", node_name, node.input[1], params)
    rank_out = data_rank + len(axes)
    normalized: list[int] = []
    for axis in axes:
        resolved = axis + rank_out if axis < 0 else axis
        if resolved < 0 or resolved >= rank_out:
            raise ValueError(
                f"Unsqueeze node '{node_name}' axis {axis} is out of range for "
                f"output rank {rank_out}"
            )
        if resolved in normalized:
            raise ValueError(
                f"Unsqueeze node '{node_name}' axes must be unique after "
                f"normalization; duplicate axis {resolved}"
            )
        normalized.append(resolved)
    return normalized


def _unsqueeze_attrs(
    node: onnx.NodeProto, params: dict[str, ParamTensor], data_shape: list[int]
) -> dict[str, Any]:
    """Normalize a static Unsqueeze-13 into reshape canonical attrs."""
    if any(dim < 0 for dim in data_shape):
        raise ValueError(f"Unsqueeze node '{node.name or '<unnamed>'}' requires non-negative static data dimensions")
    normalized = _unsqueeze_axes(node, params, len(data_shape))
    newshape = list(data_shape)
    for axis in sorted(normalized):
        newshape.insert(axis, 1)
    return {"newshape": newshape, "allowzero": 0}


def _infer_unsqueeze_spec(
    node: onnx.NodeProto,
    input_specs: dict[str, TensorSpec],
    params: dict[str, ParamTensor],
    inferred_specs: dict[str, TensorSpec],
    value_info_by_name: dict[str, onnx.ValueInfoProto],
    output_declarations: dict[str, list[onnx.ValueInfoProto]],
    default_batch: int | None,
    opset_version: int,
) -> TensorSpec:
    """Infer the static output of an Unsqueeze-13 node normalized to reshape."""
    node_name = node.name or "<unnamed>"
    if opset_version < M4M5_MATH_MIN_OPSET:
        raise UnsupportedONNXOpError(
            f"Unsupported ONNX Unsqueeze opset {opset_version} in node "
            f"'{node_name}': the axes-input opset >= {M4M5_MATH_MIN_OPSET} form "
            "is required"
        )
    if len(node.input) != 2 or not all(node.input):
        raise ValueError(
            f"Unsqueeze node '{node_name}' requires exactly two non-empty inputs"
        )
    data = _resolve_static_input("Unsqueeze", node_name, node.input[0], input_specs,
                                 params, inferred_specs, value_info_by_name, default_batch)
    attrs = _unsqueeze_attrs(node, params, data.shape)
    result = TensorSpec(name=node.output[0], shape=attrs["newshape"], dtype=data.dtype)
    _validate_declared_output("Unsqueeze", node_name, result, output_declarations,
                              default_batch)
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
    if node.op_type == "Trilu":
        return _trilu_attrs(node, params, opset_version)
    if node.op_type == "Shape":
        if attrs:
            raise ValueError("Shape import only admits full-rank shape without attributes")
        return {}
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
    if node.op_type == "Split":
        axis, sections = _split_sections(node, params, opset_version)
        return {"axis": axis, "sections": sections}
    if node.op_type == "Conv":
        auto_pad = attrs.get("auto_pad", "NOTSET")
        if not isinstance(auto_pad, str) or auto_pad not in {"NOTSET", "VALID"}:
            raise UnsupportedONNXOpError(
                f"Conv node '{node.name or '<unnamed>'}' requires explicit pads or auto_pad=VALID; got {auto_pad!r}")
        if auto_pad == "VALID" and "pads" in attrs:
            raise ValueError("Conv auto_pad=VALID cannot be combined with explicit pads")
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
        if set(attrs) - {"axis"}:
            raise ValueError("Softmax supports only axis attribute")
        if opset_version < 13:
            raise UnsupportedONNXOpError(
                f"Unsupported ONNX Softmax opset {opset_version} in node "
                f"'{node.name or '<unnamed>'}': pre-opset-13 flattened-axis semantics "
                "cannot be represented by Relay single-axis softmax"
            )
        return {"axis": _int_attr(attrs, "axis", -1)}
    if node.op_type == "Transpose":
        if set(attrs) - {"perm"}:
            raise ValueError("Transpose supports only perm attribute")
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
    if node.op_type in {"Add", "Mul", "Sub", "Div", "Sqrt"}:
        if attrs:
            raise ValueError(
                f"{node.op_type} node '{node.name or '<unnamed>'}' does not support attributes"
            )
        return {}
    if node.op_type in {"Neg", "Sigmoid", "Tanh", "Erf", "Pow"}:
        if attrs:
            raise ValueError(
                f"{node.op_type} node '{node.name or '<unnamed>'}' does not support attributes"
            )
        return {}
    if node.op_type == "Cast":
        target = _cast_target_dtype(node)
        return {"to": RELAY_CAST_DTYPE_CODES[target]}
    if node.op_type in {"ReduceMean", "ReduceMax", "ReduceMin"}:
        return _reduce_attrs(node.op_type, node, opset_version)
    if node.op_type == "ArgMax":
        return _argmax_attrs(node, opset_version)
    if node.op_type == "Reshape":
        data = _resolve_static_input("Reshape", node.name or "<unnamed>", node.input[0],
                                     input_specs, params, inferred_specs,
                                     value_info_by_name, None)
        return _reshape_attrs(node, params, data.shape)
    if node.op_type == "Expand":
        return _expand_attrs(node, params)
    if node.op_type == "Unsqueeze":
        data = _resolve_static_input("Unsqueeze", node.name or "<unnamed>", node.input[0],
                                     input_specs, params, inferred_specs,
                                     value_info_by_name, None)
        return _unsqueeze_attrs(node, params, data.shape)
    if node.op_type in {"Relu", "GlobalAveragePool", "MatMul"}:
        if node.op_type == "MatMul" and attrs:
            raise ValueError("MatMul does not accept attributes")
        return {}
    raise UnsupportedONNXOpError(
        f"Unsupported ONNX op '{node.op_type}' in node '{node.name or '<unnamed>'}'"
    )
