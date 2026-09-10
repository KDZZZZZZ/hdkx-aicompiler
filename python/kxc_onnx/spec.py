from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str


@dataclass(frozen=True)
class ParamTensor:
    name: str
    shape: list[int]
    dtype: str
    data: bytes


@dataclass(frozen=True)
class RelayNodeSpec:
    name: str
    op_name: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelayFunctionSpec:
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    nodes: list[RelayNodeSpec]


@dataclass(frozen=True)
class ImportedONNXModel:
    function: RelayFunctionSpec
    params: dict[str, ParamTensor]
    param_order: list[str]
    preserve_shape_values: bool = False


def _tensor_spec_to_dict(spec: TensorSpec) -> dict[str, Any]:
    return {"name": spec.name, "shape": list(spec.shape), "dtype": spec.dtype}


def _node_spec_to_dict(spec: RelayNodeSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "op_name": spec.op_name,
        "inputs": list(spec.inputs),
        "outputs": list(spec.outputs),
        "attrs": spec.attrs,
    }


def to_json_dict(imported: ImportedONNXModel) -> dict[str, Any]:
    offset = 0
    param_records: list[dict[str, Any]] = []
    for name in imported.param_order:
        param = imported.params[name]
        nbytes = len(param.data)
        param_records.append(
            {
                "name": param.name,
                "shape": list(param.shape),
                "dtype": param.dtype,
                "offset": offset,
                "nbytes": nbytes,
            }
        )
        offset += nbytes

    return {
        "format": ("kxc.onnx_shape_source.v1" if imported.preserve_shape_values
                   else "kxc.onnx_import.v1"),
        "function": {
            "inputs": [_tensor_spec_to_dict(x) for x in imported.function.inputs],
            "outputs": [_tensor_spec_to_dict(x) for x in imported.function.outputs],
            "nodes": [_node_spec_to_dict(x) for x in imported.function.nodes],
        },
        "params": param_records,
        "param_order": list(imported.param_order),
    }


def save_imported_model(imported: ImportedONNXModel, json_path: Path, params_path: Path) -> None:
    json_path = Path(json_path)
    params_path = Path(params_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.parent.mkdir(parents=True, exist_ok=True)

    with params_path.open("wb") as f:
        for name in imported.param_order:
            f.write(imported.params[name].data)

    json_path.write_text(
        json.dumps(to_json_dict(imported), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
