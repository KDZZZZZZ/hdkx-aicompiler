#!/usr/bin/env python3
"""Validate project include directions after the 0721 architecture migration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')

ALLOWED = {
    "shape": {"shape"},
    "support": {"support"},
    "ffi": {"ffi", "support"},
    "runtime": {"runtime", "ffi", "support"},
    "profiling": {"profiling", "runtime", "support"},
    "target": {"target", "runtime", "support", "ffi"},
    "pass": {"pass", "target", "runtime", "support"},
    "ir": {"ir", "support"},
    "tir": {"tir", "ir", "ffi", "pass", "profiling", "runtime", "support", "target"},
    "te": {"te", "tir", "ir", "support"},
    "relay": {
        "relay", "te", "tir", "ir", "distributed", "pass", "profiling",
        "runtime", "support", "ffi", "target",
    },
    "distributed": {
        "distributed", "tir", "runtime", "target", "pass", "profiling", "support", "ffi",
    },
    "compiler": {
        "compiler", "shape", "relay", "te", "tir", "ir", "distributed", "pass",
        "profiling", "runtime", "target", "support", "ffi",
    },
    "relay_distributed": {
        "relay", "distributed", "runtime", "target", "pass", "support", "ffi",
    },
    "runtime_executable": {
        "runtime", "profiling", "target", "tir", "support", "ffi",
    },
    "frontend": {"frontend", "relay", "runtime", "support"},
}


def owner(path: Path, root: Path) -> str | None:
    relative = path.relative_to(root).parts
    if relative[:2] == ("include", "kxc") and len(relative) >= 3:
        return relative[2]
    if relative and relative[0] == "src" and len(relative) >= 2:
        if relative[:3] == ("src", "relay", "distributed"):
            return "relay_distributed"
        if relative[1] == "runtime" and (
            "compiled_module" in path.name or "compiled_module_node" in path.name
        ):
            return "runtime_executable"
        return relative[1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    failures: list[str] = []

    files = list((root / "include" / "kxc").rglob("*.h"))
    files += list((root / "src").rglob("*.h"))
    files += list((root / "src").rglob("*.cc"))
    for path in sorted(files):
        source = owner(path, root)
        if source not in ALLOWED:
            continue
        is_public = path.is_relative_to(root / "include" / "kxc")
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = INCLUDE_RE.match(line)
            if not match:
                continue
            include = match.group(1)
            if is_public and (include.startswith("src/") or "/internal/" in include):
                failures.append(f"{path.relative_to(root)}:{line_no}: public header includes private path {include}")
            if not include.startswith("kxc/"):
                continue
            parts = include.split("/")
            if len(parts) < 3:
                continue
            dependency = parts[1]
            if dependency not in ALLOWED[source]:
                failures.append(
                    f"{path.relative_to(root)}:{line_no}: {source} may not include {dependency}: {include}"
                )

    if failures:
        print("Include-layer violations:")
        print("\n".join(failures))
        return 1
    print(f"Include-layer check passed ({len(files)} files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
