"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from kxc_agent.services.diagnosis_engine import (
    analyze_bundle,
    compare_bundles,
    explain_logs,
    inspect_pass_trace,
)

__all__ = [
    "analyze_bundle",
    "compare_bundles",
    "explain_logs",
    "inspect_pass_trace",
]
