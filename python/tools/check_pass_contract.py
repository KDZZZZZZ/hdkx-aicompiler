#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BINDING_BLOCK_RE = re.compile(
    r"static const std::vector<(?P<type>RelayPassBinding|TIRPassBinding)> bindings = "
    r"\{(?P<body>.*?)\n\s*\};",
    re.DOTALL,
)
ENTRY_RE = re.compile(r"\{(?P<body>.*?)\}", re.DOTALL)
STRING_RE = re.compile(r'"([^"]+)"')
BOOL_RE = re.compile(r"\b(true|false)\b")
DEFAULT_ORDER_RE = re.compile(
    r"Array<String> GetDefaultPassOrder\(\)\s*\{(?P<body>.*?)\n\}",
    re.DOTALL,
)
FFI_RE = re.compile(r'KXC_REGISTER_GLOBAL\s*\(\s*"([^"]+)"\s*\)')


@dataclass(frozen=True)
class PassBinding:
    dialect: str
    name: str
    implementation_key: str
    in_default_pipeline: bool
    idempotent: bool
    target_dependent: bool
    phase: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check PassSpec registrations against test/pass_contract.json."
    )
    parser.add_argument("--root", default=".", help="Repository root. Defaults to cwd.")
    parser.add_argument(
        "--matrix",
        default=None,
        help="Contract JSON path. Defaults to test/pass_contract.json under --root.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Report format.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print findings but exit 0.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def relpath(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def load_contract(root: Path, matrix_arg: str | None) -> tuple[Path, dict[str, Any]]:
    matrix_path = Path(matrix_arg) if matrix_arg else root / "test" / "pass_contract.json"
    if not matrix_path.is_absolute():
        matrix_path = root / matrix_path
    with matrix_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "passes" not in data or not isinstance(data["passes"], dict):
        raise ValueError(f"{matrix_path} must contain a 'passes' object")
    if "pipelines" not in data or not isinstance(data["pipelines"], dict):
        raise ValueError(f"{matrix_path} must contain a 'pipelines' object")
    return matrix_path, data


def parse_bindings_from_file(path: Path) -> list[PassBinding]:
    text = read_text(path)
    match = BINDING_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"{path} does not contain a pass binding table")
    dialect = "relay" if match.group("type") == "RelayPassBinding" else "tir"
    bindings: list[PassBinding] = []
    for entry in ENTRY_RE.finditer(match.group("body")):
        body = entry.group("body")
        strings = STRING_RE.findall(body)
        bools = [value == "true" for value in BOOL_RE.findall(body)]
        if dialect == "relay":
            if len(strings) < 2 or len(bools) < 2:
                continue
            name, implementation_key = strings[:2]
            in_default, idempotent = bools[:2]
            bindings.append(
                PassBinding(
                    dialect=dialect,
                    name=name,
                    implementation_key=implementation_key,
                    in_default_pipeline=in_default,
                    idempotent=idempotent,
                    target_dependent=False,
                    phase="relay_optimize",
                )
            )
        else:
            if len(strings) < 3 or len(bools) < 3:
                continue
            name, implementation_key = strings[:2]
            phase = strings[-1]
            in_default, idempotent, target_dependent = bools[:3]
            bindings.append(
                PassBinding(
                    dialect=dialect,
                    name=name,
                    implementation_key=implementation_key,
                    in_default_pipeline=in_default,
                    idempotent=idempotent,
                    target_dependent=target_dependent,
                    phase=phase,
                )
            )
    return bindings


def parse_default_order(path: Path) -> list[str]:
    text = read_text(path)
    match = DEFAULT_ORDER_RE.search(text)
    if not match:
        raise ValueError(f"{path} does not contain GetDefaultPassOrder")
    return STRING_RE.findall(match.group("body"))


def parse_ffi_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    for path in [
        root / "src" / "relay" / "transforms" / "pipeline.cc",
        root / "src" / "tir" / "transforms" / "pipeline.cc",
    ]:
        if path.exists():
            keys.update(FFI_RE.findall(read_text(path)))
    return keys


def analyze(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    relay_path = root / "src" / "relay" / "transforms" / "pipeline.cc"
    tir_path = root / "src" / "tir" / "transforms" / "pipeline.cc"
    bindings = parse_bindings_from_file(relay_path) + parse_bindings_from_file(tir_path)
    ffi_keys = parse_ffi_keys(root)

    cxx_by_key = {f"{binding.dialect}.{binding.name}": binding for binding in bindings}
    contract_by_key = contract["passes"]
    defaults = contract.get("pass_defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("pass_defaults must be an object")
    rows: list[dict[str, Any]] = []
    issues: list[str] = []

    if len(cxx_by_key) != len(bindings):
        seen: set[str] = set()
        for binding in bindings:
            key = f"{binding.dialect}.{binding.name}"
            if key in seen:
                issues.append(f"duplicate C++ PassSpec identity: {key}")
            seen.add(key)

    for key in sorted(set(cxx_by_key) | set(contract_by_key)):
        binding = cxx_by_key.get(key)
        expected = contract_by_key.get(key)
        row_issues: list[str] = []
        if binding is None:
            row_issues.append("missing C++ binding")
        if expected is None:
            row_issues.append("missing contract entry")
        if binding and expected:
            effective = {**defaults, **expected}
            for field in contract.get("rules", {}).get("required_fields", []):
                if field not in effective:
                    row_issues.append(f"missing required field: {field}")
            comparisons = {
                "name": binding.name,
                "schema_version": 1,
                "dialect": binding.dialect,
                "implementation_key": binding.implementation_key,
                "opt_level": 1 if binding.in_default_pipeline else 3,
                "deterministic": True,
                "idempotent": binding.idempotent,
                "thread_safe": False,
                "target_dependent": binding.target_dependent,
                "may_change_ir": True,
                "phase": binding.phase,
            }
            for field, actual in comparisons.items():
                if effective.get(field) != actual:
                    row_issues.append(f"{field} is {actual!r}, expected {effective.get(field)!r}")
            expected_scope = "graph" if binding.dialect == "relay" else "prim_func"
            if effective.get("scope") != expected_scope:
                row_issues.append(
                    f"scope is {effective.get('scope')!r}, expected {expected_scope!r}"
                )
            if binding.implementation_key not in ffi_keys:
                row_issues.append(f"implementation key lacks FFI registration: {binding.implementation_key}")
        rows.append(
            {
                "pass": key,
                "present_in_cxx": binding is not None,
                "present_in_contract": expected is not None,
                "issues": row_issues,
            }
        )

    pipeline_expectations = {
        "relay.optimize_default": parse_default_order(relay_path),
        "tir.optimize_default": parse_default_order(tir_path),
    }
    for pipeline_name, actual in pipeline_expectations.items():
        expected = contract["pipelines"].get(pipeline_name)
        if expected != actual:
            issues.append(f"{pipeline_name} order is {actual}, expected {expected}")

    for binding in bindings:
        pipeline_name = f"{binding.dialect}.optimize_default"
        in_contract_default = binding.name in contract["pipelines"].get(pipeline_name, [])
        if binding.in_default_pipeline != in_contract_default:
            issues.append(
                f"{binding.dialect}.{binding.name} default flag is {binding.in_default_pipeline}, "
                f"contract membership is {in_contract_default}"
            )

    return {
        "rows": rows,
        "global_issues": issues,
        "summary": {
            "checked": len(rows),
            "passed": sum(1 for row in rows if not row["issues"]),
            "failed": sum(1 for row in rows if row["issues"]),
            "global_issues": len(issues),
        },
    }


def print_text_report(root: Path, matrix_path: Path, report: dict[str, Any]) -> None:
    print("Pass contract report")
    print(f"Root: {root}")
    print(f"Matrix: {matrix_path}")
    summary = report["summary"]
    print(
        f"Passes: {summary['checked']} checked, {summary['passed']} passed, "
        f"{summary['failed']} failed"
    )
    print(f"Global issues: {summary['global_issues']}")
    failed = [row for row in report["rows"] if row["issues"]]
    if failed:
        print()
        print("Pass issues:")
        for row in failed:
            print(f"- {row['pass']}:")
            for issue in row["issues"]:
                print(f"  - {issue}")
    if report["global_issues"]:
        print()
        print("Global issues:")
        for issue in report["global_issues"]:
            print(f"- {issue}")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    try:
        matrix_path, contract = load_contract(root, args.matrix)
        report = analyze(root, contract)
    except Exception as exc:
        print(f"pass contract check failed to run: {exc}", file=sys.stderr)
        return 2

    has_issues = bool(report["global_issues"] or any(row["issues"] for row in report["rows"]))
    if args.format == "json":
        print(json.dumps({"root": str(root), "matrix": str(matrix_path), **report}, indent=2))
    else:
        print_text_report(root, matrix_path, report)
    return 0 if args.report_only or not has_issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
