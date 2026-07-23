#!/usr/bin/env python3
"""Check the public-header manifest, forbidden dependencies, and self-containment."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


FORBIDDEN_INCLUDE = re.compile(
    r'#\s*include\s*[<"](?:src/|.*?/internal/|llvm/|cuda|nvrtc|cupti|pybind11)'
)
OLD_INCLUDE = re.compile(
    r'#\s*include\s*[<"](?:base/|api/|codegen/|pass/|relay/|runtime/|te/|tir/|frontend/)'
)
DEFINE_CALL = re.compile(r'^\s*KXC_OBJECT_DEFINE(?:_WITH_KEY)?\s*\(')
REGISTER_CALL = re.compile(r'^\s*KXC_REGISTER_(?:GLOBAL|OP)\s*\(')


def manifest(cmake: str) -> set[str]:
    match = re.search(r'set\(KXC_PUBLIC_HEADERS\s+(.*?)\n\)', cmake, re.S)
    if not match:
        return set()
    return {token for token in re.findall(r'include/kxc/[A-Za-z0-9_./-]+\.h', match.group(1))}


def compile_headers(
    root: Path, compiler: str, headers: list[Path], defines: list[str]
) -> list[str]:
    failures: list[str] = []
    include_dir = root / "include"
    dlpack_dir = root / "third_party" / "dlpack" / "include"
    with tempfile.TemporaryDirectory(prefix="kxc-headers-") as tmp:
        source = Path(tmp) / "header_test.cc"
        for header in headers:
            # detail/*_inl.h is an implementation fragment included by its
            # owning declaration header, not a supported direct-include API.
            if "detail" in header.relative_to(root / "include" / "kxc").parts:
                continue
            include = header.relative_to(include_dir).as_posix()
            source.write_text(f'#include "{include}"\nint main() {{ return 0; }}\n', encoding="utf-8")
            result = subprocess.run(
                [compiler, "-std=c++17", "-fsyntax-only",
                 *[f"-D{define}" for define in defines], f"-I{include_dir}",
                 f"-I{dlpack_dir}", str(source)],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                failures.append(f"{include}:\n{result.stderr.strip()}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--define", action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    headers = sorted((root / "include" / "kxc").rglob("*.h"))
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    declared = manifest(cmake)
    actual = {path.relative_to(root).as_posix() for path in headers}
    failures: list[str] = []

    missing = sorted(actual - declared)
    stale = sorted(declared - actual)
    if missing:
        failures.append("Headers missing from KXC_PUBLIC_HEADERS: " + ", ".join(missing))
    if stale:
        failures.append("Stale KXC_PUBLIC_HEADERS entries: " + ", ".join(stale))

    for path in headers:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN_INCLUDE.search(line):
                failures.append(f"{path.relative_to(root)}:{line_no}: forbidden dependency: {line.strip()}")
            if OLD_INCLUDE.search(line):
                failures.append(f"{path.relative_to(root)}:{line_no}: old include path: {line.strip()}")
            if path.name != "object_registration.h" and DEFINE_CALL.match(line):
                failures.append(f"{path.relative_to(root)}:{line_no}: object definition in public header")
            if REGISTER_CALL.match(line):
                failures.append(f"{path.relative_to(root)}:{line_no}: static registration in public header")

    if args.compile:
        failures.extend(
            compile_headers(root, args.compiler, headers, args.define)
        )

    if failures:
        print("Public-header check failed:")
        print("\n\n".join(failures))
        return 1
    suffix = " and compiled" if args.compile else ""
    print(f"Public-header check passed ({len(headers)} headers{suffix}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
