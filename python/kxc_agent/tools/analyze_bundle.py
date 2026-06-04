"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kxc_agent.memory.store import record_bundle
from kxc_agent.services.diagnosis_engine import analyze_bundle as _analyze_bundle


def analyze_bundle(bundle_path: str | Path) -> dict:
    """分析 bundle 并把诊断摘要登记到本地 memory。"""

    result = _analyze_bundle(bundle_path)
    record_bundle(result["bundle_path"], result.get("trace_id"), result["diagnostics"])
    return result


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Analyze a KXC profiling bundle.")
    parser.add_argument("--bundle", required=True, help="Path to the bundle directory.")
    args = parser.parse_args()

    result = analyze_bundle(args.bundle)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
