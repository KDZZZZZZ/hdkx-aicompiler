"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from kxc_agent.tools.analyze_bundle import analyze_bundle
from kxc_agent.tools.compare_bundles import compare_bundles
from kxc_agent.tools.explain_logs import explain_logs
from kxc_agent.tools.inspect_pass_trace import inspect_pass_trace
from kxc_agent.tools.profile_run import profile_run

__all__ = [
    "analyze_bundle",
    "compare_bundles",
    "explain_logs",
    "inspect_pass_trace",
    "profile_run",
]
