"""导入前的 ONNX 常量折叠：把只依赖 initializer/Constant 的子图求值成常量。

存在的理由是具体的：`torch.onnx.export` 会把因果 mask 这类**完全静态**的构造
留在图里（MiniMind 导出中是 `ConstantOfShape` → `Mul`/`Equal`/`Where` → `Trilu`
这条链），并用 `Shape` 链去喂 `Expand` 的目标形状。这些节点不需要新的运行时
算子——它们在导入前就可以求值掉。

求值用 ONNX 自己的 `ReferenceEvaluator`，因此折叠语义就是 ONNX 语义，不在这里
另写一套算子解释器。

折叠边界（fail-closed）：
- 只折叠**每个输入都静态**的节点；图输入的任何下游都不静态。
- 产出图输出的节点不折叠——图输出必须由节点产出，不能变成 initializer。
- 只物化**折叠前沿**：被保留节点消费的那些静态值；纯中间值不进 initializer。
- 物化总字节数超预算即拒绝，不静默膨胀权重。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import ModelProto, helper, numpy_helper

DEFAULT_FOLD_BYTE_BUDGET = 256 * 1024 * 1024


class ConstantFoldingError(RuntimeError):
    """折叠越过声明边界（预算、求值失败）时抛出。"""


@dataclass
class FoldReport:
    folded_nodes: int = 0
    materialized: list[str] = field(default_factory=list)
    materialized_bytes: int = 0
    remaining_nodes: int = 0
    folded_op_types: dict[str, int] = field(default_factory=dict)


def _static_values(graph: onnx.GraphProto) -> tuple[set[str], list[onnx.NodeProto]]:
    """返回 (静态值名集合, 可折叠节点列表)。"""
    graph_outputs = {value.name for value in graph.output}
    static = {initializer.name for initializer in graph.initializer}
    foldable: list[onnx.NodeProto] = []
    for node in graph.node:
        # 图输入不静态，因此任何以它为输入的节点都不静态；节点按拓扑序遍历。
        if any(name and name not in static for name in node.input):
            continue
        if any(output in graph_outputs for output in node.output):
            # 图输出必须由节点产出，保留该节点；它的静态输入仍可折叠。
            continue
        foldable.append(node)
        static.update(output for output in node.output if output)
    return static, foldable


def fold_static_subgraph(
    model: ModelProto, *, byte_budget: int = DEFAULT_FOLD_BYTE_BUDGET
) -> tuple[ModelProto, FoldReport]:
    """求值并消除模型里的纯静态子图，返回新模型与折叠报告。"""
    graph = model.graph
    _, foldable = _static_values(graph)
    report = FoldReport(remaining_nodes=len(graph.node))
    if not foldable:
        return model, report

    folded_ids = {id(node) for node in foldable}
    kept = [node for node in graph.node if id(node) not in folded_ids]
    produced_by_fold = {output for node in foldable for output in node.output if output}

    # 折叠前沿：被保留节点消费的静态产出。纯中间值不物化。
    frontier = sorted(
        {name for node in kept for name in node.input if name in produced_by_fold}
    )
    if not frontier:
        return model, report

    static_model = helper.make_model(
        helper.make_graph(
            foldable, f"{graph.name or 'graph'}_static_fold", [],
            [helper.make_empty_tensor_value_info(name) for name in frontier],
            initializer=list(graph.initializer),
        ),
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
    )
    try:
        from onnx.reference import ReferenceEvaluator

        values = ReferenceEvaluator(static_model).run(None, {})
    except Exception as error:  # noqa: BLE001 - 折叠失败必须显式拒绝
        raise ConstantFoldingError(
            f"constant folding failed to evaluate the static subgraph: {error}"
        ) from error

    total = sum(int(np.asarray(value).nbytes) for value in values)
    if total > byte_budget:
        raise ConstantFoldingError(
            f"constant folding would materialize {total} bytes, over the "
            f"{byte_budget}-byte budget; refusing to inline it"
        )

    folded_graph = onnx.GraphProto()
    folded_graph.CopyFrom(graph)
    del folded_graph.node[:]
    folded_graph.node.extend(kept)
    for name, value in zip(frontier, values):
        array = np.ascontiguousarray(np.asarray(value))
        folded_graph.initializer.append(numpy_helper.from_array(array, name=name))
    # 折叠掉的中间值留下的 value_info 会指向不存在的产出，一并剪掉。
    stale = produced_by_fold - set(frontier)
    surviving = [info for info in folded_graph.value_info if info.name not in stale]
    del folded_graph.value_info[:]
    folded_graph.value_info.extend(surviving)

    folded_model = helper.make_model(
        folded_graph,
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
    )
    folded_model.CopyFrom(_with_metadata(folded_model, model))

    report.folded_nodes = len(foldable)
    report.materialized = frontier
    report.materialized_bytes = total
    report.remaining_nodes = len(kept)
    counts: dict[str, int] = {}
    for node in foldable:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    report.folded_op_types = dict(sorted(counts.items(), key=lambda item: -item[1]))
    return folded_model, report


def _with_metadata(folded: ModelProto, source: ModelProto) -> ModelProto:
    """保留原模型的 producer/domain/版本元数据。"""
    folded.producer_name = source.producer_name
    folded.producer_version = source.producer_version
    folded.domain = source.domain
    folded.model_version = source.model_version
    return folded
