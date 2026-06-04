"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

from kxc_agent.services.diagnosis_engine import analyze_bundle


def profile_run(command: list[str], bundle_path: str | Path | None = None,
                cwd: str | Path | None = None, analyze: bool = False) -> Path:
    """带 profiling 环境变量执行命令，并返回生成的 bundle 目录。"""

    if not command:
        raise ValueError("profile_run requires a command to execute")

    resolved_bundle = Path(bundle_path or (Path.cwd() / "profile_bundles" / f"bundle-{uuid4().hex}"))
    resolved_bundle = resolved_bundle.expanduser().resolve()
    resolved_bundle.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["KXC_PROFILE_ENABLE"] = "1"
    env["KXC_PROFILE_BUNDLE_DIR"] = str(resolved_bundle)

    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)

    if analyze:
        analyze_bundle(resolved_bundle)
    return resolved_bundle


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Run a command with KXC profiling enabled.")
    parser.add_argument("--bundle", help="Bundle directory. Defaults to profile_bundles/<uuid>.")
    parser.add_argument("--cwd", help="Working directory for the profiled command.")
    parser.add_argument("--analyze", action="store_true", help="Analyze the bundle after the command exits.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute after '--'.")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    bundle = profile_run(command, args.bundle, args.cwd, args.analyze)
    print(json.dumps({"bundle_path": str(bundle)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
