"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProfileTask:
    command: list[str]
    bundle_path: Path


@dataclass
class AnalysisTask:
    bundle_path: Path


@dataclass
class ComparisonTask:
    base_bundle_path: Path
    new_bundle_path: Path
