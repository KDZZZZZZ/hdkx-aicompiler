"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

DIAGNOSIS_CATEGORIES = (
    "compile_hotspot",
    "pass_regression",
    "kernel_launch_overhead",
    "cache_miss_pattern",
    "background_compile_stall",
    "copy_dominance",
    "sync_overhead",
    "shape_fragmentation",
    "execution_plan_imbalance",
    "lowering_failure_context",
)

HOTSPOT_RATIO = 0.35
COPY_DOMINANCE_RATIO = 0.40
SYNC_DOMINANCE_RATIO = 0.25
SMALL_KERNEL_RUN_NS = 200_000
SHAPE_FRAGMENTATION_THRESHOLD = 4
