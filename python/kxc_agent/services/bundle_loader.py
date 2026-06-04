"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_BUNDLE_FILES = (
    "manifest.json",
    "events.jsonl",
    "trace.json",
    "summary.json",
    "diagnosis.json",
    "diagnosis.md",
)


@dataclass
class Bundle:
    """内存中的 profiling bundle 视图。"""

    path: Path
    manifest: dict[str, Any]
    events: list[dict[str, Any]]
    trace: dict[str, Any]
    summary: dict[str, Any]
    diagnosis: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    """按 UTF-8 读取单个 JSON 文件。"""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """按行读取 JSONL 事件文件，忽略空行。"""

    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def ensure_bundle_path(bundle_path: str | Path) -> Path:
    """校验 bundle_path 存在且为目录，并返回绝对路径。"""

    path = Path(bundle_path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Bundle directory does not exist: {path}")
    return path


def missing_required_files(bundle_path: str | Path) -> list[str]:
    """返回 profiling bundle 中缺失的标准文件列表。"""

    path = ensure_bundle_path(bundle_path)
    missing = []
    for name in REQUIRED_BUNDLE_FILES:
        if not (path / name).exists():
            missing.append(name)
    if not (path / "artifacts").exists():
        missing.append("artifacts/")
    return missing


def load_bundle(bundle_path: str | Path) -> Bundle:
    """加载完整 profiling bundle，并在缺文件时抛出 FileNotFoundError。"""

    path = ensure_bundle_path(bundle_path)
    missing = missing_required_files(path)
    if missing:
        raise FileNotFoundError(f"Bundle is incomplete: {', '.join(missing)}")
    return Bundle(
        path=path,
        manifest=_load_json(path / "manifest.json"),
        events=_load_jsonl(path / "events.jsonl"),
        trace=_load_json(path / "trace.json"),
        summary=_load_json(path / "summary.json"),
        diagnosis=_load_json(path / "diagnosis.json"),
    )
