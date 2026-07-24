#!/usr/bin/env python3
"""Validate the JSON-authoritative pass contract and generated C++ binding."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from generate_pass_contract import render


BINDING_BLOCK_RE = re.compile(
    r"static const std::vector<(?:RelayPassBinding|TIRPassBinding)> bindings = "
    r"\{(?P<body>.*?)\n\s*\};",
    re.DOTALL,
)
IMPLEMENTATION_KEY_RE = re.compile(r'"(kxc\.(?:relay|tir)\.transform\.[^"]+)"')
FFI_RE = re.compile(r'KXC_REGISTER_GLOBAL\s*\(\s*"([^"]+)"\s*\)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check generated PassSpec metadata and implementation bindings."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--matrix", default=None)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def load_contract(root: Path, matrix_arg: str | None) -> tuple[Path, dict[str, Any]]:
    path = Path(matrix_arg) if matrix_arg else root / "contracts" / "pass_contract.json"
    if not path.is_absolute():
        path = root / path
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("passes"), dict) or not isinstance(data.get("pipelines"), dict):
        raise ValueError(f"{path} requires passes and pipelines objects")
    return path, data


def parse_binding_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    match = BINDING_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"{path} has no implementation-only binding table")
    return set(IMPLEMENTATION_KEY_RE.findall(match.group("body")))


def analyze(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    defaults = contract.get("pass_defaults", {})
    required_fields = contract.get("rules", {}).get("required_fields", [])

    generated_path = root / "src" / "pass" / "generated" / "pass_contract.inc"
    expected_generated = render(contract)
    if not generated_path.exists() or generated_path.read_text(encoding="utf-8") != expected_generated:
        issues.append(
            "generated PassSpec source is stale; run python/tools/generate_pass_contract.py"
        )

    pipeline_paths = [
        root / "src" / "relay" / "transforms" / "pipeline.cc",
        root / "src" / "tir" / "transforms" / "pipeline.cc",
    ]
    binding_keys: set[str] = set()
    ffi_keys: set[str] = set()
    for path in pipeline_paths:
        text = path.read_text(encoding="utf-8")
        binding_keys.update(parse_binding_keys(path))
        ffi_keys.update(FFI_RE.findall(text))
        if "pass_contract_generated::Specs" not in text or "pass_contract_generated::Pipeline" not in text:
            issues.append(f"{path.relative_to(root)} bypasses generated pass metadata/pipeline")
        if "MakeRelayPassSpec" in text or "MakeTIRPassSpec" in text:
            issues.append(f"{path.relative_to(root)} still writes duplicate PassSpec metadata")

    expected_binding_keys: set[str] = set()
    seen_identity: set[tuple[str, str]] = set()
    for key, raw in contract["passes"].items():
        row_issues: list[str] = []
        effective = {**defaults, **raw}
        for field in required_fields:
            if field not in effective:
                row_issues.append(f"missing required field: {field}")
        dialect = effective.get("dialect")
        name = effective.get("name")
        identity = (str(dialect), str(name))
        if identity in seen_identity:
            row_issues.append("duplicate dialect/name identity")
        seen_identity.add(identity)
        if key != f"{dialect}.{name}":
            row_issues.append("JSON object key differs from dialect/name")
        implementation_key = effective.get("implementation_key")
        if not isinstance(implementation_key, str) or not implementation_key:
            row_issues.append("implementation_key is empty")
        else:
            expected_binding_keys.add(implementation_key)
            if implementation_key not in binding_keys:
                row_issues.append("missing implementation-only C++ binding")
            if implementation_key not in ffi_keys:
                row_issues.append("missing FFI implementation binding")
        rows.append({"pass": key, "issues": row_issues})

    extra_bindings = sorted(binding_keys - expected_binding_keys)
    if extra_bindings:
        issues.append(f"implementation bindings absent from contract: {extra_bindings}")

    declared_passes = set(contract["passes"])
    raw_repeatable = contract.get("rules", {}).get("repeatable_passes", [])
    if (not isinstance(raw_repeatable, list) or
            not all(isinstance(key, str) and key in declared_passes
                    for key in raw_repeatable)):
        issues.append("rules.repeatable_passes must name declared dialect.pass entries")
    repeatable_passes = set(raw_repeatable)

    pass_names_by_dialect = {
        dialect: {
            raw["name"]
            for raw in contract["passes"].values()
            if raw.get("dialect") == dialect
        }
        for dialect in ("relay", "tir")
    }
    for pipeline_name, pass_names in contract["pipelines"].items():
        if not isinstance(pass_names, list):
            issues.append(f"{pipeline_name} must be a pass list")
            continue
        dialect = pipeline_name.split(".", 1)[0]
        duplicates = {name for name in pass_names if pass_names.count(name) > 1}
        accidental = sorted(
            name for name in duplicates if f"{dialect}.{name}" not in repeatable_passes
        )
        if accidental:
            issues.append(f"{pipeline_name} has unapproved repeated passes: {accidental}")
        unknown = sorted(set(pass_names) - pass_names_by_dialect.get(dialect, set()))
        if unknown:
            issues.append(f"{pipeline_name} references unknown passes: {unknown}")

    documentation = contract.get("documentation")
    if not isinstance(documentation, str) or not (root / documentation).is_file():
        issues.append("pass contract documentation anchor is missing or invalid")

    return {
        "rows": rows,
        "global_issues": issues,
        "summary": {
            "checked": len(rows),
            "passed": sum(not row["issues"] for row in rows),
            "failed": sum(bool(row["issues"]) for row in rows),
            "global_issues": len(issues),
        },
    }


def print_text(matrix: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Pass contract report")
    print(f"Matrix: {matrix}")
    print(
        f"Passes: {summary['checked']} checked, {summary['passed']} passed, "
        f"{summary['failed']} failed"
    )
    print(f"Global issues: {summary['global_issues']}")
    for row in report["rows"]:
        for issue in row["issues"]:
            print(f"- {row['pass']}: {issue}")
    for issue in report["global_issues"]:
        print(f"- {issue}")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    try:
        matrix, contract = load_contract(root, args.matrix)
        report = analyze(root, contract)
    except Exception as exc:
        print(f"pass contract check failed to run: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print_text(matrix, report)
    failed = bool(report["global_issues"] or any(row["issues"] for row in report["rows"]))
    return 0 if args.report_only or not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
