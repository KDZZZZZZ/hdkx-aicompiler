"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

import argparse
import json

from kxc_agent.tools import (
    analyze_bundle,
    compare_bundles,
    explain_logs,
    inspect_pass_trace,
    profile_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="KXC agent tooling for profiling bundles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze_bundle")
    analyze_parser.add_argument("--bundle", required=True)

    compare_parser = subparsers.add_parser("compare_bundles")
    compare_parser.add_argument("--base", required=True)
    compare_parser.add_argument("--new", required=True)

    explain_parser = subparsers.add_parser("explain_logs")
    explain_parser.add_argument("--bundle", required=True)

    inspect_parser = subparsers.add_parser("inspect_pass_trace")
    inspect_parser.add_argument("--bundle", required=True)
    inspect_parser.add_argument("--stage", choices=("relay", "tir"))

    profile_parser = subparsers.add_parser("profile_run")
    profile_parser.add_argument("--bundle")
    profile_parser.add_argument("--cwd")
    profile_parser.add_argument("--analyze", action="store_true")
    profile_parser.add_argument("profile_command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.command == "analyze_bundle":
        result = analyze_bundle(args.bundle)
    elif args.command == "compare_bundles":
        result = compare_bundles(args.base, args.new)
    elif args.command == "explain_logs":
        result = explain_logs(args.bundle)
    elif args.command == "inspect_pass_trace":
        result = inspect_pass_trace(args.bundle, args.stage)
    else:
        command = args.profile_command
        if command and command[0] == "--":
            command = command[1:]
        result = {"bundle_path": str(profile_run(command, args.bundle, args.cwd, args.analyze))}

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
