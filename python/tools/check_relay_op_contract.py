#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from generate_relay_op_contract import (
    render as render_generated_contract,
    render_registration as render_generated_registration,
)


REGISTER_RE = re.compile(
    r"KXC_REGISTER_OP\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
    r"|OpRegEntry\s*\(\s*Op::Get\s*\(\s*\"([^\"]+)\"\s*\)\s*\)"
)
SET_ATTR_RE = re.compile(
    r"\.set_attr\s*<\s*([^>]+?)\s*>\s*"
    r"\(\s*\"([^\"]+)\"\s*,\s*([^)]+?)\s*\)",
    re.DOTALL,
)
GLOBAL_MAKE_RE = re.compile(
    r"KXC_REGISTER_GLOBAL\s*\(\s*\"kxc\.relay\.op\._make\.([^\"]+)\"\s*\)"
    r"\s*\.set_body\s*\(\s*ToPackedFunc\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\)",
    re.DOTALL,
)
CALL_FUNCTION_RE = re.compile(
    r"\bCall\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*\{(.*?)\n\}",
    re.DOTALL,
)
EMPTY_TENSOR_RETURN_RE = re.compile(r"return\s+(?:kxc::)?te::Tensor\s*\(\s*\)\s*;")
CPP_FUNCTION_START_RE = re.compile(
    r"(?m)^[ \t]*(?:[A-Za-z_][\w:<>,: \t*&]+)\s+"
    r"[A-Za-z_][A-Za-z0-9_]*\s*\([^;{}]*\)\s*(?:const\s*)?\{"
)

OPERATOR_FIELD_ORDER = [
    "schema_version",
    "category",
    "num_inputs",
    "min_inputs",
    "max_inputs",
    "attrs",
    "output_arity",
    "type_relation_key",
    "effect",
    "deterministic",
    "alias",
    "lowering",
    "lowering_key",
    "ffi",
    "tests",
    "onnx_ops",
    "registration",
]
REQUIRED_OPERATOR_FIELDS = {
    "schema_version",
    "category",
    "num_inputs",
    "attrs",
    "output_arity",
    "type_relation_key",
    "effect",
    "deterministic",
    "alias",
    "lowering",
    "lowering_key",
    "ffi",
    "tests",
    "onnx_ops",
}
ALLOWED_EFFECTS = {"pure", "stateful", "device_communication"}
ALLOWED_LOWERINGS = {"single", "multi", "exec_plan", "none"}
CPP_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class Registration:
    op: str
    file: str
    line: int
    num_inputs: int | None
    min_inputs: int | None
    max_inputs: int | None
    arg_count: int
    has_description: bool
    tattrs: str | None
    infer_type: str | None
    single_lowering: str | None
    multi_lowering: str | None
    placeholder_terms: list[str] = field(default_factory=list)


@dataclass
class FfiHelper:
    helper_name: str
    function: str
    file: str
    line: int
    op_names: list[str]
    call_input_count: int | None
    placeholder_terms: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Relay operators against the machine-readable implementation contract."
    )
    parser.add_argument("--root", default=".", help="Repository root. Defaults to cwd.")
    parser.add_argument(
        "--matrix",
        default=None,
        help="Contract JSON path. Defaults to contracts/relay_op_contract.json under --root.",
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
        help="Print findings but exit 0. Without this flag, any issue fails the command.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def relpath(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def mask_comments(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    return re.sub(r"/\*.*?\*/|//[^\n]*", repl, text, flags=re.DOTALL)


def normalize_attr_value(value: str) -> str:
    value = value.strip().rstrip(";").strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def operator_contract_issues(op: str, spec: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    missing = sorted(REQUIRED_OPERATOR_FIELDS - set(spec))
    if missing:
        issues.append(f"{op}: missing OperatorSpec field(s): {', '.join(missing)}")

    known_order = [key for key in OPERATOR_FIELD_ORDER if key in spec]
    actual_known = [key for key in spec if key in OPERATOR_FIELD_ORDER]
    if actual_known != known_order:
        issues.append(
            f"{op}: OperatorSpec fields are not in stable order; expected "
            + ", ".join(known_order)
        )

    schema_version = spec.get("schema_version")
    if not isinstance(schema_version, int) or schema_version <= 0:
        issues.append(f"{op}: schema_version must be a positive integer")

    num_inputs = spec.get("num_inputs")
    if not isinstance(num_inputs, int) or num_inputs < -1:
        issues.append(f"{op}: num_inputs must be an integer >= -1")
    if num_inputs == -1:
        min_inputs = spec.get("min_inputs")
        max_inputs = spec.get("max_inputs")
        if not isinstance(min_inputs, int) or min_inputs < 0:
            issues.append(f"{op}: variable arity requires non-negative min_inputs")
        if not isinstance(max_inputs, int) or max_inputs < 0:
            issues.append(f"{op}: variable arity requires non-negative max_inputs")
        if isinstance(min_inputs, int) and isinstance(max_inputs, int) and min_inputs > max_inputs:
            issues.append(f"{op}: min_inputs cannot exceed max_inputs")

    if not isinstance(spec.get("category"), str) or not spec.get("category"):
        issues.append(f"{op}: category must be a non-empty string")
    attrs = spec.get("attrs")
    if attrs is not None and not isinstance(attrs, str):
        issues.append(f"{op}: attrs must be a string or null")
    if not isinstance(spec.get("output_arity"), int) or spec.get("output_arity") < 0:
        issues.append(f"{op}: output_arity must be a non-negative integer")
    if not isinstance(spec.get("type_relation_key"), str) or not spec.get("type_relation_key"):
        issues.append(f"{op}: type_relation_key must be a non-empty string")
    if spec.get("effect") not in ALLOWED_EFFECTS:
        issues.append(f"{op}: effect must be one of {sorted(ALLOWED_EFFECTS)}")
    if not isinstance(spec.get("deterministic"), bool):
        issues.append(f"{op}: deterministic must be boolean")
    if spec.get("effect") == "pure" and spec.get("deterministic") is not True:
        issues.append(f"{op}: pure operators must be deterministic")
    if not isinstance(spec.get("alias"), str) or not spec.get("alias"):
        issues.append(f"{op}: alias must be a non-empty string")

    lowering = spec.get("lowering")
    lowering_key = spec.get("lowering_key")
    if lowering not in ALLOWED_LOWERINGS:
        issues.append(f"{op}: lowering must be one of {sorted(ALLOWED_LOWERINGS)}")
    if not isinstance(lowering_key, str):
        issues.append(f"{op}: lowering_key must be a string")
    elif lowering == "single" and lowering_key != "FRelayToTE":
        issues.append(f"{op}: single lowering requires lowering_key FRelayToTE")
    elif lowering == "multi" and lowering_key != "FRelayToTEMulti":
        issues.append(f"{op}: multi lowering requires lowering_key FRelayToTEMulti")
    elif lowering == "exec_plan" and not lowering_key:
        issues.append(f"{op}: exec_plan lowering requires a lowering_key")
    elif lowering == "none" and lowering_key:
        issues.append(f"{op}: none lowering must have an empty lowering_key")

    if not isinstance(spec.get("ffi"), bool):
        issues.append(f"{op}: ffi must be boolean")
    if not isinstance(spec.get("tests"), bool):
        issues.append(f"{op}: tests must be boolean")
    onnx_ops = spec.get("onnx_ops")
    if not isinstance(onnx_ops, list) or not all(isinstance(item, str) for item in onnx_ops):
        issues.append(f"{op}: onnx_ops must be a string array")

    binding = spec.get("registration")
    if binding is not None:
        if not isinstance(binding, dict):
            issues.append(f"{op}: registration must be an object")
        else:
            binding_fields = {
                "description", "arguments", "type_infer_symbol", "relay_to_te_symbol"
            }
            missing_binding_fields = sorted(binding_fields - set(binding))
            unknown_binding_fields = sorted(set(binding) - binding_fields)
            if missing_binding_fields:
                issues.append(
                    f"{op}: registration missing field(s): " +
                    ", ".join(missing_binding_fields)
                )
            if unknown_binding_fields:
                issues.append(
                    f"{op}: registration has unsupported field(s): " +
                    ", ".join(unknown_binding_fields)
                )
            description = binding.get("description")
            if not isinstance(description, str) or not description.strip():
                issues.append(f"{op}: registration description must be non-empty")

            arguments = binding.get("arguments")
            if (not isinstance(arguments, list) or
                    len(arguments) != spec.get("num_inputs")):
                issues.append(f"{op}: registration arguments must match fixed input arity")
            elif not all(
                isinstance(arg, dict) and set(arg) == {"name", "type", "description"} and
                all(
                    isinstance(arg.get(field), str) and bool(arg[field].strip())
                    for field in ("name", "type", "description")
                )
                for arg in arguments
            ):
                issues.append(
                    f"{op}: registration arguments need only non-empty name, type, and description"
                )
            elif len({arg["name"] for arg in arguments}) != len(arguments):
                issues.append(f"{op}: registration argument names must be unique")

            if not CPP_IDENTIFIER_RE.fullmatch(op):
                issues.append(f"{op}: generated registration requires a C++ identifier op name")
            for field in ("type_infer_symbol", "relay_to_te_symbol"):
                symbol = binding.get(field)
                if not isinstance(symbol, str) or not CPP_IDENTIFIER_RE.fullmatch(symbol):
                    issues.append(f"{op}: registration {field} must be a C++ identifier")
            binding_num_inputs = spec.get("num_inputs")
            if not isinstance(binding_num_inputs, int) or binding_num_inputs < 0:
                issues.append(f"{op}: generated registration currently requires fixed arity")
            if spec.get("output_arity") != 1:
                issues.append(f"{op}: generated registration currently requires one output")
            if spec.get("type_relation_key") != "FInferType":
                issues.append(f"{op}: generated registration requires FInferType")
            if spec.get("lowering") != "single":
                issues.append(f"{op}: generated registration currently supports single lowering only")
    return issues


def validate_contract_shape(data: dict[str, Any]) -> None:
    operators = data.get("operators")
    if not isinstance(operators, dict):
        raise ValueError("contract must contain an 'operators' object")
    ordered_names = list(operators)
    if ordered_names != sorted(ordered_names):
        raise ValueError("operators must be serialized in canonical sorted order")

    issues: list[str] = []
    for op, spec in operators.items():
        if not isinstance(spec, dict):
            issues.append(f"{op}: operator entry must be an object")
            continue
        issues.extend(operator_contract_issues(op, spec))
    if issues:
        raise ValueError("invalid relay op contract:\n" + "\n".join(f"- {issue}" for issue in issues))


def load_contract(root: Path, matrix_arg: str | None) -> tuple[Path, dict[str, Any]]:
    matrix_path = Path(matrix_arg) if matrix_arg else root / "contracts" / "relay_op_contract.json"
    if not matrix_path.is_absolute():
        matrix_path = root / matrix_path
    with matrix_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "operators" not in data or not isinstance(data["operators"], dict):
        raise ValueError(f"{matrix_path} must contain an 'operators' object")
    validate_contract_shape(data)
    return matrix_path, data


def find_cpp_files(root: Path) -> list[Path]:
    source_root = root / "src"
    if not source_root.exists():
        return []
    return sorted(
        p
        for p in source_root.rglob("*")
        if p.suffix in {".cc", ".cpp", ".h", ".hpp"}
        and "build" not in p.parts
    )


def find_op_source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    op_root = root / "src" / "relay" / "op"
    if op_root.exists():
        files.extend(
            p
            for p in op_root.rglob("*")
            if p.suffix in {".cc", ".cpp", ".h", ".hpp"}
        )
    importer = root / "python" / "kxc_onnx" / "importer.py"
    if importer.exists():
        files.append(importer)
    topi_root = root / "include" / "te" / "topi"
    if topi_root.exists():
        files.extend(
            p
            for p in topi_root.rglob("*")
            if p.suffix in {".h", ".hpp"}
        )
    return sorted(set(files))


def extract_placeholder_terms(text: str, terms: list[str]) -> list[str]:
    found: list[str] = []
    lower_text = text.lower()
    for term in terms:
        if term.lower() in lower_text:
            found.append(term)
    return found


def parse_registrations(root: Path, terms: list[str]) -> dict[str, list[Registration]]:
    registrations: dict[str, list[Registration]] = {}
    for path in find_cpp_files(root):
        raw = read_text(path)
        masked = mask_comments(raw)
        matches = list(REGISTER_RE.finditer(masked))
        for index, match in enumerate(matches):
            op = match.group(1) or match.group(2)
            block_start = match.end()
            block_end = matches[index + 1].start() if index + 1 < len(matches) else len(masked)
            block = masked[block_start:block_end]
            raw_block = raw[match.start():block_end]

            num_inputs_match = re.search(r"\.set_num_inputs\s*\(\s*(-?\d+)\s*\)", block)
            arity_range_match = re.search(
                r"\.set_input_arity_range\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", block
            )
            num_inputs = int(num_inputs_match.group(1)) if num_inputs_match else None
            min_inputs = None
            max_inputs = None
            if arity_range_match:
                num_inputs = -1
                min_inputs = int(arity_range_match.group(1))
                max_inputs = int(arity_range_match.group(2))

            attrs_by_key: dict[str, str] = {}
            for _value_type, key, value in SET_ATTR_RE.findall(block):
                attrs_by_key[key] = normalize_attr_value(value)

            registration = Registration(
                op=op,
                file=relpath(root, path),
                line=line_number(raw, match.start()),
                num_inputs=num_inputs,
                min_inputs=min_inputs,
                max_inputs=max_inputs,
                arg_count=len(re.findall(r"\.add_argument\s*\(", block)),
                has_description=".describe" in block,
                tattrs=attrs_by_key.get("TAttrs"),
                infer_type=attrs_by_key.get("FInferType"),
                single_lowering=attrs_by_key.get("FRelayToTE"),
                multi_lowering=attrs_by_key.get("FRelayToTEMulti"),
                placeholder_terms=extract_placeholder_terms(raw_block, terms),
            )
            registrations.setdefault(op, []).append(registration)
    return registrations


def has_callback_definition(root: Path, symbol: str, return_type: str,
                            arguments: str) -> bool:
    pattern = re.compile(
        r"(?m)^\s*(?!static\b)" + return_type + re.escape(symbol) +
        r"\s*\(" + arguments + r"\)\s*\{"
    )
    for path in find_cpp_files(root):
        if (path.suffix not in {".cc", ".cpp"} or
                path.as_posix().endswith("/generated/relay_op_registration.cc")):
            continue
        if pattern.search(mask_comments(read_text(path))):
            return True
    return False


def generated_binding_issues(root: Path, regs: list[Registration],
                             binding: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if len(regs) != 1 or not regs[0].file.endswith("generated/relay_op_registration.cc"):
        issues.append("generated registration must be the sole registration authority")
    if not has_callback_definition(
        root, binding["type_infer_symbol"], r"Type\s+",
        r"\s*const\s+Attrs\s*&\s*\w+\s*,\s*const\s+Array\s*<\s*Type\s*>\s*&\s*\w+\s*",
    ):
        issues.append("generated type_infer_symbol lacks the non-static FInferType signature")
    if not has_callback_definition(
        root, binding["relay_to_te_symbol"], r"te::Tensor\s+",
        r"\s*const\s+Attrs\s*&\s*\w+\s*,\s*const\s+Array\s*<\s*te::Tensor\s*>\s*&\s*\w+\s*,\s*const\s+kxc::Type\s*&\s*\w+\s*",
    ):
        issues.append("generated relay_to_te_symbol lacks the non-static FRelayToTE signature")
    return issues


def count_top_level_items(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0

    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closing = {v: k for k, v in pairs.items()}
    stack: list[str] = []
    in_string: str | None = None
    escaped = False
    count = 1

    for ch in stripped:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = None
            continue

        if ch in {'"', "'"}:
            in_string = ch
        elif ch in pairs:
            stack.append(ch)
        elif ch in closing and stack and stack[-1] == closing[ch]:
            stack.pop()
        elif ch == "," and not stack:
            count += 1

    return count


def parse_ffi_helpers(root: Path, terms: list[str]) -> dict[str, list[FfiHelper]]:
    path = root / "src" / "relay" / "op" / "op_ffi.cc"
    if not path.exists():
        return {}

    raw = read_text(path)
    functions: dict[str, tuple[str, int]] = {}
    for match in CALL_FUNCTION_RE.finditer(raw):
        functions[match.group(1)] = (match.group(3), line_number(raw, match.start()))

    helpers_by_op: dict[str, list[FfiHelper]] = {}
    for match in GLOBAL_MAKE_RE.finditer(raw):
        helper_name = match.group(1)
        function_name = match.group(2)
        body, function_line = functions.get(function_name, ("", line_number(raw, match.start())))
        op_names = re.findall(r"GetOp\s*\(\s*\"([^\"]+)\"\s*\)", body)
        call_input_count: int | None = None
        if len(op_names) == 1:
            call_re = re.search(
                r"Call\s*\(\s*GetOp\s*\(\s*\""
                + re.escape(op_names[0])
                + r"\"\s*\)\s*,\s*\{(?P<inputs>.*?)\}",
                body,
                re.DOTALL,
            )
            if call_re:
                call_input_count = count_top_level_items(call_re.group("inputs"))

        helper = FfiHelper(
            helper_name=helper_name,
            function=function_name,
            file=relpath(root, path),
            line=function_line,
            op_names=op_names,
            call_input_count=call_input_count,
            placeholder_terms=extract_placeholder_terms(body, terms),
        )
        if op_names:
            for op in op_names:
                helpers_by_op.setdefault(op, []).append(helper)
        else:
            helpers_by_op.setdefault("<unknown>", []).append(helper)
    return helpers_by_op


def parse_onnx_mapping(root: Path) -> dict[str, list[str]]:
    path = root / "python" / "kxc_onnx" / "importer.py"
    if not path.exists():
        return {}

    tree = ast.parse(read_text(path), filename=str(path))
    relay_to_onnx: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "ONNX_TO_RELAY" for target in node.targets):
            continue
        mapping = ast.literal_eval(node.value)
        for onnx_op, relay_op in mapping.items():
            relay_to_onnx.setdefault(str(relay_op), []).append(str(onnx_op))
    return relay_to_onnx


def extract_cpp_function_blocks(text: str) -> list[str]:
    masked = mask_comments(text)
    blocks: list[str] = []
    for match in CPP_FUNCTION_START_RE.finditer(masked):
        brace = masked.find("{", match.start(), match.end())
        if brace < 0:
            continue
        depth = 0
        end = -1
        for index in range(brace, len(masked)):
            ch = masked[index]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end > brace:
            blocks.append(text[match.start():end])
    return blocks


def count_test_refs(root: Path, op_names: set[str]) -> dict[str, dict[str, int]]:
    counts = {
        op: {
            "refs": 0,
            "tir_refs": 0,
            "backend_refs": 0,
            "exec_plan_refs": 0,
        }
        for op in op_names
    }
    test_root = root / "test"
    if not test_root.exists():
        return counts

    test_files = [
        p
        for p in test_root.rglob("*")
        if p.suffix in {".cc", ".cpp", ".h", ".hpp", ".py"}
        and "profile_bundle_test_output" not in p.parts
    ]
    for path in test_files:
        text = read_text(path)
        for op in op_names:
            counts[op]["refs"] += text.count(f'"{op}"')

        regions = (
            extract_cpp_function_blocks(text)
            if path.suffix in {".cc", ".cpp", ".h", ".hpp"}
            else [text]
        )
        for region in regions:
            has_tir_signal = any(
                signal in region
                for signal in (
                    "LowerPrimitiveUnits",
                    "LowerFirstPrimitive",
                )
            )
            has_backend_signal = (
                "Compiler::Compile" in region
                or "CompileConfig" in region
                or "CodeGenLLVM" in region
                or "LLVMJIT" in region
                or "CompileAndRun" in region
            )
            has_exec_plan_signal = (
                "SerializeExecutionPlan" in region
                or "CommExec" in region
                or "ExecutionPlanExecutor" in region
            )
            if not (has_tir_signal or has_backend_signal or has_exec_plan_signal):
                continue
            for op in op_names:
                refs = region.count(f'"{op}"')
                if refs and has_tir_signal:
                    counts[op]["tir_refs"] += refs
                if refs and has_backend_signal:
                    counts[op]["backend_refs"] += refs
                if refs and has_exec_plan_signal:
                    counts[op]["exec_plan_refs"] += refs
    return counts


def expected_input_issue(
    expected: dict[str, Any] | None,
    actual: int | None,
    label: str,
    *,
    allow_variable_marker: bool = False,
) -> str | None:
    if expected is None:
        return None
    if actual is None:
        return f"{label} does not expose a parseable input count"

    expected_num = expected.get("num_inputs")
    if isinstance(expected_num, int) and expected_num >= 0 and actual != expected_num:
        return f"{label} input count is {actual}, expected {expected_num}"
    if isinstance(expected_num, int) and expected_num == -1:
        if allow_variable_marker and actual == -1:
            return None
        min_inputs = expected.get("min_inputs")
        max_inputs = expected.get("max_inputs")
        if isinstance(min_inputs, int) and actual < min_inputs:
            return f"{label} input count is {actual}, expected at least {min_inputs}"
        if isinstance(max_inputs, int) and actual > max_inputs:
            return f"{label} input count is {actual}, expected at most {max_inputs}"
    return None


def registration_schema_issues(reg: Registration, expected: dict[str, Any] | None) -> list[str]:
    issues: list[str] = []
    if not reg.has_description:
        issues.append(f"{reg.file}:{reg.line} missing describe()")
    if reg.num_inputs is None:
        issues.append(
            f"{reg.file}:{reg.line} missing set_num_inputs() or set_input_arity_range()"
        )
    else:
        input_issue = expected_input_issue(
            expected,
            reg.num_inputs,
            "schema",
            allow_variable_marker=True,
        )
        if input_issue:
            issues.append(f"{reg.file}:{reg.line} {input_issue}")
        if expected and expected.get("num_inputs") == -1:
            expected_min = expected.get("min_inputs")
            expected_max = expected.get("max_inputs")
            if reg.min_inputs != expected_min or reg.max_inputs != expected_max:
                issues.append(
                    f"{reg.file}:{reg.line} schema input range is "
                    f"[{reg.min_inputs}, {reg.max_inputs}], expected "
                    f"[{expected_min}, {expected_max}]"
                )

    required_arg_count = 0
    if expected:
        expected_num = expected.get("num_inputs")
        if isinstance(expected_num, int) and expected_num >= 0:
            required_arg_count = expected_num
        elif isinstance(expected.get("min_inputs"), int):
            required_arg_count = int(expected["min_inputs"])
    elif reg.num_inputs is not None and reg.num_inputs > 0:
        required_arg_count = reg.num_inputs

    if reg.arg_count < required_arg_count:
        issues.append(
            f"{reg.file}:{reg.line} has {reg.arg_count} add_argument() entries, "
            f"expected at least {required_arg_count}"
        )

    if expected and "attrs" in expected:
        expected_attrs = expected.get("attrs")
        if expected_attrs and reg.tattrs != expected_attrs:
            issues.append(
                f"{reg.file}:{reg.line} TAttrs is {reg.tattrs or '<missing>'}, "
                f"expected {expected_attrs}"
            )
        if expected_attrs is None and reg.tattrs is not None:
            issues.append(f"{reg.file}:{reg.line} has unexpected TAttrs {reg.tattrs}")

    if reg.placeholder_terms:
        issues.append(
            f"{reg.file}:{reg.line} contains placeholder marker(s): "
            + ", ".join(sorted(set(reg.placeholder_terms)))
        )
    return issues


def has_complete_schema(reg: Registration, expected: dict[str, Any] | None) -> bool:
    return not registration_schema_issues(reg, expected)


def cmake_set_contains(text: str, variable: str, value: str) -> bool:
    match = re.search(
        r"\bset\s*\(\s*" + re.escape(variable) + r"\b(?P<body>.*?)\)",
        text,
        re.DOTALL,
    )
    if not match:
        return False
    body = re.sub(r"#[^\n]*", "", match.group("body"))
    return bool(re.search(r"(?m)^\s*" + re.escape(value) + r"\s*$", body))


def generated_anchor_is_reachable(text: str) -> bool:
    masked = mask_comments(text)
    declaration = re.search(
        r"\bvoid\s+RelayGeneratedOpBindings\s*\(\s*\)\s*;", masked
    )
    register_builtins = re.search(
        r"\bvoid\s+RegisterBuiltins\s*\(\s*\)\s*\{(?P<body>[^}]*)\}",
        masked,
        re.DOTALL,
    )
    return bool(
        declaration and register_builtins and re.search(
            r"\bbuiltin_anchor::RelayGeneratedOpBindings\s*\(\s*\)\s*;",
            register_builtins.group("body"),
        )
    )


def analyze(
    root: Path,
    contract: dict[str, Any],
    registrations: dict[str, list[Registration]],
    helpers_by_op: dict[str, list[FfiHelper]],
    relay_to_onnx: dict[str, list[str]],
) -> dict[str, Any]:
    expected_ops: dict[str, dict[str, Any]] = contract["operators"]
    rules = contract.get("rules", {})
    forbidden_ops = set(rules.get("forbidden_op_names", []))
    forbidden_helpers = set(rules.get("forbidden_helper_names", []))
    required_stages = set(rules.get("required_stages", []))

    op_names = set(expected_ops) | set(registrations) | set(relay_to_onnx)
    for helpers in helpers_by_op.values():
        for helper in helpers:
            op_names.update(helper.op_names)

    test_counts = count_test_refs(root, op_names)

    rows: list[dict[str, Any]] = []
    global_issues: list[str] = []

    generated_path = root / "src" / "relay" / "generated" / "relay_op_contract.inc"
    expected_generated = render_generated_contract(contract)
    if (not generated_path.exists() or
            generated_path.read_text(encoding="utf-8") != expected_generated):
        global_issues.append(
            "generated OperatorSpec source is stale; run "
            "python/tools/generate_relay_op_contract.py"
        )
    generated_registration_path = (
        root / "src" / "relay" / "generated" / "relay_op_registration.cc"
    )
    expected_generated_registration = render_generated_registration(contract)
    if (not generated_registration_path.exists() or
            generated_registration_path.read_text(encoding="utf-8") !=
            expected_generated_registration):
        global_issues.append("generated Relay registration source is stale")
    cmake_text = read_text(root / "CMakeLists.txt")
    builtin_text = read_text(root / "src" / "ffi" / "builtin_registry.cc")
    if not cmake_set_contains(
        cmake_text,
        "KXC_RELAY_IR_SOURCES",
        "src/relay/generated/relay_op_registration.cc",
    ):
        global_issues.append(
            "generated Relay registration source is not in KXC_RELAY_IR_SOURCES"
        )
    if not generated_anchor_is_reachable(builtin_text):
        global_issues.append(
            "generated Relay registration anchor is not retained by RegisterBuiltins"
        )
    registry_text = read_text(root / "src" / "relay" / "op_registry.cc")
    if "op_contract_generated::Spec" not in registry_text:
        global_issues.append("operator registry bypasses generated OperatorSpec metadata")
    if "InferCategoryFromName" in registry_text or "FillLegacyDefaults" in registry_text:
        global_issues.append("operator registry still infers or defaults critical contract fields")
    documentation = contract.get("documentation")
    if not isinstance(documentation, str) or not (root / documentation).is_file():
        global_issues.append("operator contract documentation anchor is missing or invalid")

    for op in sorted(op_names):
        expected = expected_ops.get(op)
        regs = registrations.get(op, [])
        helpers = helpers_by_op.get(op, [])
        onnx_ops = sorted(relay_to_onnx.get(op, []))
        issues: list[str] = []

        if expected is None:
            issues.append("op is not declared in contracts/relay_op_contract.json")
        elif "registration" in expected:
            issues.extend(generated_binding_issues(root, regs, expected["registration"]))
        if op in forbidden_ops:
            issues.append("op name is forbidden alias")
        if expected and not regs:
            issues.append("op is declared in contract but has no registration")
        if regs and len(regs) != 1:
            locations = ", ".join(f"{reg.file}:{reg.line}" for reg in regs)
            issues.append(f"op has {len(regs)} registrations; expected exactly 1 ({locations})")

        for reg in regs:
            issues.extend(registration_schema_issues(reg, expected))
            if not reg.infer_type:
                issues.append(f"{reg.file}:{reg.line} missing FInferType")
            if expected and expected.get("type_relation_key") == "FInferType" and not reg.infer_type:
                issues.append(f"{reg.file}:{reg.line} missing type relation key FInferType")

            expected_lowering = expected.get("lowering") if expected else None
            expected_lowering_key = expected.get("lowering_key") if expected else None
            if expected_lowering == "single":
                if expected_lowering_key != "FRelayToTE":
                    issues.append(f"{reg.file}:{reg.line} single lowering key must be FRelayToTE")
                if not reg.single_lowering:
                    issues.append(f"{reg.file}:{reg.line} missing FRelayToTE")
                if reg.multi_lowering:
                    issues.append(f"{reg.file}:{reg.line} should not register FRelayToTEMulti")
            elif expected_lowering == "multi":
                if expected_lowering_key != "FRelayToTEMulti":
                    issues.append(f"{reg.file}:{reg.line} multi lowering key must be FRelayToTEMulti")
                if not reg.multi_lowering:
                    issues.append(f"{reg.file}:{reg.line} missing FRelayToTEMulti")
                if reg.single_lowering:
                    issues.append(f"{reg.file}:{reg.line} should not register FRelayToTE")
            elif expected_lowering == "exec_plan":
                if not expected_lowering_key:
                    issues.append(f"{reg.file}:{reg.line} exec_plan lowering key is missing")
                if reg.single_lowering or reg.multi_lowering:
                    issues.append(
                        f"{reg.file}:{reg.line} exec_plan op should not register TE lowering hooks"
                    )

        strict_helpers = [
            helper for helper in helpers if len(helper.op_names) == 1 and helper.helper_name == op
        ]
        ffi_required = bool(expected.get("ffi", True)) if expected else True
        ffi_present = bool(strict_helpers)
        ffi_satisfied = (not ffi_required) or ffi_present
        if expected and ffi_required and not strict_helpers:
            issues.append(f"missing canonical FFI helper kxc.relay.op._make.{op}")

        for helper in helpers:
            if helper.helper_name in forbidden_helpers:
                issues.append(
                    f"{helper.file}:{helper.line} helper _make.{helper.helper_name} is forbidden alias"
                )
            if len(helper.op_names) != 1:
                issues.append(
                    f"{helper.file}:{helper.line} helper _make.{helper.helper_name} "
                    f"must reference exactly one GetOp(), found {len(helper.op_names)}"
                )
            elif helper.helper_name != helper.op_names[0]:
                issues.append(
                    f"{helper.file}:{helper.line} helper _make.{helper.helper_name} aliases "
                    f"{helper.op_names[0]}; helper name must be canonical"
                )
            if helper.call_input_count is None:
                issues.append(
                    f"{helper.file}:{helper.line} helper _make.{helper.helper_name} "
                    "does not expose a parseable Call input list"
                )
            else:
                input_issue = expected_input_issue(expected, helper.call_input_count, "FFI Call")
                if input_issue:
                    issues.append(
                        f"{helper.file}:{helper.line} helper _make.{helper.helper_name}: {input_issue}"
                    )
            if helper.placeholder_terms:
                issues.append(
                    f"{helper.file}:{helper.line} helper _make.{helper.helper_name} contains "
                    "placeholder marker(s): "
                    + ", ".join(sorted(set(helper.placeholder_terms)))
                )

        coverage = test_counts.get(
            op,
            {
                "refs": 0,
                "tir_refs": 0,
                "backend_refs": 0,
                "exec_plan_refs": 0,
            },
        )
        test_ref_count = coverage["refs"]
        tir_test_ref_count = coverage["tir_refs"]
        backend_test_ref_count = coverage["backend_refs"]
        exec_plan_test_ref_count = coverage["exec_plan_refs"]

        if expected:
            expected_onnx = set(expected.get("onnx_ops", []))
            missing_onnx = sorted(expected_onnx - set(onnx_ops))
            if missing_onnx:
                issues.append("missing ONNX mapping(s): " + ", ".join(missing_onnx))

            if expected.get("tests", True) and test_ref_count == 0:
                issues.append("missing test reference")

        schema_ok = bool(regs) and any(has_complete_schema(reg, expected) for reg in regs)
        has_type = any(reg.infer_type for reg in regs)
        expected_lowering = expected.get("lowering") if expected else None
        if expected_lowering == "multi":
            has_lowering = any(reg.multi_lowering for reg in regs)
        elif expected_lowering == "single":
            has_lowering = any(reg.single_lowering for reg in regs)
        elif expected_lowering == "exec_plan":
            has_lowering = bool(regs)
        else:
            has_lowering = any(reg.single_lowering or reg.multi_lowering for reg in regs)
        has_ffi = ffi_present
        requires_tir_test = expected_lowering in {"single", "multi"}
        if expected and requires_tir_test and tir_test_ref_count == 0:
            issues.append(
                "missing production primitive lowering contract test reference"
            )

        requires_exec_plan_test = expected_lowering == "exec_plan"
        if expected and requires_exec_plan_test and exec_plan_test_ref_count == 0:
            issues.append("missing execution-plan contract test reference")

        requires_backend_test = bool(
            expected
            and (
                "backend" in required_stages
                or expected.get("llvm_required")
                or expected.get("backend") == "llvm"
                or expected.get("executable")
                or expected.get("tir_executable")
            )
        )
        if requires_backend_test and backend_test_ref_count == 0:
            issues.append("missing backend compile/runtime test reference")

        has_tests = (
            test_ref_count > 0
            and (not requires_tir_test or tir_test_ref_count > 0)
            and (not requires_exec_plan_test or exec_plan_test_ref_count > 0)
        )
        if requires_backend_test:
            has_tests = has_tests and backend_test_ref_count > 0

        if not regs:
            stage = "missing"
        elif not schema_ok:
            stage = "registered"
        elif not has_type:
            stage = "schema"
        elif not has_lowering:
            stage = "typed"
        elif not ffi_satisfied:
            stage = "lowered"
        elif not has_tests:
            stage = "ffi"
        else:
            stage = "tested"

        rows.append(
            {
                "op": op,
                "category": expected.get("category") if expected else "<unknown>",
                "stage": stage,
                "registered_count": len(regs),
                "registration_locations": [f"{reg.file}:{reg.line}" for reg in regs],
                "schema": schema_ok,
                "type_inference": has_type,
                "lowering": has_lowering,
                "ffi": has_ffi,
                "ffi_required": ffi_required,
                "ffi_satisfied": ffi_satisfied,
                "ffi_helpers": [
                    {
                        "helper_name": helper.helper_name,
                        "function": helper.function,
                        "file": helper.file,
                        "line": helper.line,
                        "op_names": helper.op_names,
                        "call_input_count": helper.call_input_count,
                    }
                    for helper in helpers
                ],
                "onnx_ops": onnx_ops,
                "test_refs": test_ref_count,
                "tir_test_refs": tir_test_ref_count,
                "backend_test_refs": backend_test_ref_count,
                "exec_plan_test_refs": exec_plan_test_ref_count,
                "issues": issues,
            }
        )

    for op, helpers in sorted(helpers_by_op.items()):
        if op != "<unknown>":
            continue
        for helper in helpers:
            global_issues.append(
                f"{helper.file}:{helper.line} helper _make.{helper.helper_name} "
                "does not reference GetOp()"
            )

    return {
        "operators": rows,
        "global_issues": global_issues,
    }


def scan_source_placeholders(root: Path, terms: list[str]) -> list[str]:
    issues: list[str] = []
    lower_terms = [(term, term.lower()) for term in terms]
    for path in find_op_source_files(root):
        text = read_text(path)
        for index, line in enumerate(text.splitlines(), start=1):
            lower_line = line.lower()
            for term, lower_term in lower_terms:
                if lower_term in lower_line:
                    issues.append(f"{relpath(root, path)}:{index} contains placeholder marker: {term}")
                    break
            if EMPTY_TENSOR_RETURN_RE.search(line):
                issues.append(f"{relpath(root, path)}:{index} returns an empty te::Tensor()")
    return issues


def print_text_report(root: Path, matrix_path: Path, report: dict[str, Any]) -> None:
    rows = report["operators"]
    failed = [row for row in rows if row["issues"]]
    global_issues = report["global_issues"]

    print("Relay op contract report")
    print(f"Root: {root}")
    print(f"Matrix: {matrix_path}")
    print(f"Operators: {len(rows)} checked, {len(rows) - len(failed)} passed, {len(failed)} failed")
    print(f"Global issues: {len(global_issues)}")
    print()
    print(
        f"{'op':<34} {'stage':<11} {'reg':>3} {'schema':>6} {'type':>5} "
        f"{'lower':>6} {'ffi':>4} {'onnx':>4} {'tests':>5} {'tir':>4} "
        f"{'be':>3} {'exec':>4} {'issues':>6}"
    )
    print("-" * 106)
    for row in rows:
        print(
            f"{row['op']:<34} {row['stage']:<11} {row['registered_count']:>3} "
            f"{yn(row['schema']):>6} {yn(row['type_inference']):>5} "
            f"{yn(row['lowering']):>6} {ffi_text(row):>4} "
            f"{len(row['onnx_ops']):>4} {row['test_refs']:>5} "
            f"{row['tir_test_refs']:>4} {row['backend_test_refs']:>3} "
            f"{row['exec_plan_test_refs']:>4} "
            f"{len(row['issues']):>6}"
        )

    if failed:
        print()
        print("Operator issues:")
        for row in failed:
            print(f"- {row['op']} ({row['stage']}):")
            for issue in row["issues"]:
                print(f"  - {issue}")

    if global_issues:
        print()
        print("Global issues:")
        for issue in global_issues:
            print(f"- {issue}")


def yn(flag: bool) -> str:
    return "Y" if flag else "-"


def ffi_text(row: dict[str, Any]) -> str:
    if not row.get("ffi_required", True):
        return "n/a"
    return yn(row["ffi"])


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    try:
        matrix_path, contract = load_contract(root, args.matrix)
        terms = list(contract.get("rules", {}).get("placeholder_terms", []))
        registrations = parse_registrations(root, terms)
        helpers = parse_ffi_helpers(root, terms)
        relay_to_onnx = parse_onnx_mapping(root)
        report = analyze(root, contract, registrations, helpers, relay_to_onnx)
        report["global_issues"].extend(scan_source_placeholders(root, terms))
    except Exception as exc:
        print(f"relay op contract check failed to run: {exc}", file=sys.stderr)
        return 2

    failed_rows = [row for row in report["operators"] if row["issues"]]
    has_issues = bool(failed_rows or report["global_issues"])

    if args.format == "json":
        json_report = {
            "root": str(root),
            "matrix": str(matrix_path),
            "operators": report["operators"],
            "global_issues": report["global_issues"],
            "summary": {
                "checked": len(report["operators"]),
                "passed": len(report["operators"]) - len(failed_rows),
                "failed": len(failed_rows),
                "global_issues": len(report["global_issues"]),
            },
        }
        print(json.dumps(json_report, ensure_ascii=False, indent=2))
    else:
        print_text_report(root, matrix_path, report)

    if has_issues and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
