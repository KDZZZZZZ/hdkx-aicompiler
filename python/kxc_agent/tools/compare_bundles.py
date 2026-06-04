"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kxc_agent.services.diagnosis_engine import compare_bundles as _compare_bundles


def compare_bundles(base_bundle: str | Path, new_bundle: str | Path) -> dict:
    """比较基线和新 bundle 的性能退化。"""

    return _compare_bundles(base_bundle, new_bundle)


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Compare two KXC profiling bundles.")
    parser.add_argument("--base", required=True, help="Baseline bundle directory.")
    parser.add_argument("--new", required=True, help="New bundle directory.")
    args = parser.parse_args()

    report = compare_bundles(args.base, args.new)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
