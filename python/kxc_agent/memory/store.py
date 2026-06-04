"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """返回仓库根目录路径。"""

    return Path(__file__).resolve().parents[3]


def memory_dir() -> Path:
    """返回 agent 本地记忆目录，并确保目录存在。"""

    path = Path(
        os.environ.get(
            "KXC_AGENT_MEMORY_DIR", str(_repo_root() / ".kxc_agent_memory")
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_bundle(bundle_path: str, trace_id: str | None,
                  diagnostics: list[dict[str, Any]]) -> None:
    """把 bundle 诊断摘要写入索引和历史 JSONL。"""

    directory = memory_dir()
    index_path = directory / "bundle_index.json"
    history_path = directory / "diagnosis_history.jsonl"

    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"bundles": []}

    record = {
        "bundle_path": bundle_path,
        "trace_id": trace_id,
        "diagnostic_categories": [item["category"] for item in diagnostics],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    index["bundles"] = [
        item for item in index.get("bundles", []) if item.get("bundle_path") != bundle_path
    ]
    index["bundles"].append(record)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
