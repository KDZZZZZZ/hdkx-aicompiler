"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from kxc_agent.tools import (
    analyze_bundle,
    compare_bundles,
    explain_logs,
    inspect_pass_trace,
    profile_run,
)

__all__ = [
    "profile_run",
    "analyze_bundle",
    "compare_bundles",
    "explain_logs",
    "inspect_pass_trace",
]
