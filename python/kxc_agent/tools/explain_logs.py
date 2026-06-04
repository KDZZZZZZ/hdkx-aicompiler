"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kxc_agent.services.diagnosis_engine import explain_logs as _explain_logs


def explain_logs(bundle_path: str | Path) -> dict:
    """提取并结构化 bundle 中的日志事件。"""

    return _explain_logs(bundle_path)


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Extract structured log explanations.")
    parser.add_argument("--bundle", required=True, help="Bundle directory.")
    args = parser.parse_args()

    print(json.dumps(explain_logs(args.bundle), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
