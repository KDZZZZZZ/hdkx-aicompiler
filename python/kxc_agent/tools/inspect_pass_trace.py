"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kxc_agent.services.diagnosis_engine import inspect_pass_trace as _inspect_pass_trace


def inspect_pass_trace(bundle_path: str | Path, stage: str | None = None) -> dict:
    """读取 pass trace，并可限制到 Relay 或 TIR 阶段。"""

    return _inspect_pass_trace(bundle_path, stage)


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Inspect relay/tir pass traces.")
    parser.add_argument("--bundle", required=True, help="Bundle directory.")
    parser.add_argument("--stage", choices=("relay", "tir"), help="Limit to one pass stage.")
    args = parser.parse_args()

    print(json.dumps(inspect_pass_trace(args.bundle, args.stage), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
