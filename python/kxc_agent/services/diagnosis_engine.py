"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .bundle_loader import Bundle, load_bundle
from .rule_library import (
    COPY_DOMINANCE_RATIO,
    HOTSPOT_RATIO,
    SHAPE_FRAGMENTATION_THRESHOLD,
    SMALL_KERNEL_RUN_NS,
    SYNC_DOMINANCE_RATIO,
)
from .schema import validate_bundle


def _duration(events: list[dict[str, Any]], *, component: str | None = None,
              event_type: str | None = None) -> int:
    """汇总满足 component/event_type 过滤条件的 duration_ns。"""

    total = 0
    for event in events:
        if component and event.get("component") != component:
            continue
        if event_type and event.get("event_type") != event_type:
            continue
        total += int(event.get("duration_ns", 0) or 0)
    return total


def _events_by_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    """按 event_type 过滤事件列表。"""

    return [event for event in events if event.get("event_type") == event_type]


def _host_device_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only measured successful host work; completion-observation delay is not execution time."""

    return [event for event in events
            if event.get("component") == "device_api"
            and event.get("phase") == "complete" and event.get("status") == "ok"
            and event.get("fields", {}).get("timing") == "host_execute"]


def _make_diagnostic(category: str, severity: str, component: str, summary: str,
                     evidence: dict[str, Any], next_steps: list[str]) -> dict[str, Any]:
    """构造统一结构的诊断项。"""

    return {
        "category": category,
        "severity": severity,
        "component": component,
        "summary": summary,
        "evidence": evidence,
        "next_steps": next_steps,
    }


def analyze_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """分析单个 profiling bundle，并把诊断结果写回 bundle 目录。"""

    bundle = load_bundle(bundle_path)
    validation_errors = validate_bundle(bundle)

    diagnostics: list[dict[str, Any]] = []
    if validation_errors:
        diagnostics.append(
            _make_diagnostic(
                "bundle_schema_error",
                "error",
                "schema",
                "Bundle schema validation failed.",
                {"errors": validation_errors},
                ["Regenerate the bundle with profiling enabled and schema_version=1."],
            )
        )
    else:
        diagnostics.extend(_analyze_valid_bundle(bundle))

    result = {
        "trace_id": bundle.manifest.get("trace_id"),
        "bundle_path": str(bundle.path),
        "schema_version": bundle.manifest.get("schema_version"),
        "diagnostics": diagnostics,
    }

    _write_diagnosis_files(bundle.path, result)
    return result


def compare_bundles(base_bundle_path: str | Path, new_bundle_path: str | Path) -> dict[str, Any]:
    """比较两个 bundle 的 pass、cache 和设备开销退化情况。"""

    base = load_bundle(base_bundle_path)
    new = load_bundle(new_bundle_path)

    regressions: list[dict[str, Any]] = []
    base_pass = _aggregate_pass_durations(base.events)
    new_pass = _aggregate_pass_durations(new.events)
    for pass_name, new_duration in sorted(new_pass.items()):
        base_duration = base_pass.get(pass_name, 0)
        if base_duration <= 0 or new_duration <= base_duration:
            continue
        ratio = new_duration / base_duration
        if ratio >= 1.2:
            regressions.append(
                {
                    "category": "pass_regression",
                    "component": "pass_pipeline",
                    "summary": f"{pass_name} regressed by {ratio:.2f}x",
                    "evidence": {
                        "pass_name": pass_name,
                        "base_duration_ns": base_duration,
                        "new_duration_ns": new_duration,
                    },
                }
            )

    regressions.extend(_compare_cache_health(base.events, new.events))
    regressions.extend(_compare_device_overheads(base.events, new.events))

    return {
        "base_bundle": str(base.path),
        "new_bundle": str(new.path),
        "regressions": regressions,
    }


def explain_logs(bundle_path: str | Path) -> dict[str, Any]:
    """提取 bundle 中的日志事件并整理为可读结构。"""

    bundle = load_bundle(bundle_path)
    log_events = [event for event in bundle.events if event.get("event_type") == "log"]
    explanations = []
    for event in log_events:
        explanations.append(
            {
                "severity": event.get("severity"),
                "component": event.get("component"),
                "message": event.get("message"),
                "run_id": event.get("run_id"),
                "fields": event.get("fields", {}),
            }
        )
    return {
        "bundle_path": str(bundle.path),
        "log_count": len(log_events),
        "logs": explanations,
    }


def inspect_pass_trace(bundle_path: str | Path, stage: str | None = None) -> dict[str, Any]:
    """提取 Relay/TIR pass trace 的耗时、状态和 IR hash。"""

    bundle = load_bundle(bundle_path)
    components = {"relay_pass", "tir_pass"} if not stage else {f"{stage}_pass"}
    passes = []
    for event in bundle.events:
        if event.get("component") not in components:
            continue
        passes.append(
            {
                "component": event.get("component"),
                "pass_name": event.get("pass_name"),
                "duration_ns": event.get("duration_ns", 0),
                "status": event.get("status"),
                "ir_before_hash": event.get("fields", {}).get("ir_before_hash"),
                "ir_after_hash": event.get("fields", {}).get("ir_after_hash"),
            }
        )
    return {
        "bundle_path": str(bundle.path),
        "passes": passes,
    }


def _analyze_valid_bundle(bundle: Bundle) -> list[dict[str, Any]]:
    """在 schema 合法的 bundle 上执行规则库诊断。"""

    diagnostics: list[dict[str, Any]] = []
    events = bundle.events

    total_duration = sum(int(event.get("duration_ns", 0) or 0) for event in events)
    compile_events = [
        event for event in events if event.get("component") in {"compiler", "relay_pass", "tir_pass", "lowering", "codegen"}
    ]
    if total_duration > 0 and compile_events:
        hottest = max(compile_events, key=lambda event: int(event.get("duration_ns", 0) or 0))
        hottest_duration = int(hottest.get("duration_ns", 0) or 0)
        if hottest_duration / total_duration >= HOTSPOT_RATIO:
            diagnostics.append(
                _make_diagnostic(
                    "compile_hotspot",
                    "warn",
                    str(hottest.get("component")),
                    f"{hottest.get('event_type')} dominates compile time.",
                    {
                        "duration_ns": hottest_duration,
                        "pass_name": hottest.get("pass_name"),
                        "component": hottest.get("component"),
                    },
                    ["Inspect pass trace for this stage.", "Review IR artifacts for this pass."],
                )
            )

    cache_hits = len(_events_by_type(events, "cache_exact_hit"))
    cache_misses = len(_events_by_type(events, "cache_miss_sync_compile"))
    if cache_misses > cache_hits:
        diagnostics.append(
            _make_diagnostic(
                "cache_miss_pattern",
                "warn",
                "runtime_session",
                "Synchronous cache misses outnumber exact cache hits.",
                {"cache_exact_hit": cache_hits, "cache_miss_sync_compile": cache_misses},
                ["Inspect shape signatures.", "Warm up frequent shapes before serving traffic."],
            )
        )

    submitted = len(_events_by_type(events, "background_compile_submitted"))
    completed = len(_events_by_type(events, "background_compile_completed"))
    if submitted > completed:
        diagnostics.append(
            _make_diagnostic(
                "background_compile_stall",
                "warn",
                "background_compiler",
                "Background compile submissions are not fully completing.",
                {"submitted": submitted, "completed": completed},
                ["Check worker pool health.", "Inspect background compiler error logs."],
            )
        )

    host_device_events = _host_device_events(events)
    host_copies = _events_by_type(host_device_events, "copy")
    copy_duration = _duration(host_copies)
    device_duration = _duration(host_device_events)
    if device_duration > 0 and copy_duration / device_duration >= COPY_DOMINANCE_RATIO:
        diagnostics.append(
            _make_diagnostic(
                "copy_dominance",
                "warn",
                "device_api",
                "Copy work dominates measured host device-API activity.",
                {"copy_duration_ns": copy_duration, "device_duration_ns": device_duration,
                 "measurement_domain": "host_execute", "copy_count": len(host_copies),
                 "copy_bytes": sum(event.get("metrics", {}).get("bytes", 0) for event in host_copies)},
                ["Inspect redundant copies and copied values.", "Investigate execution plan placement."],
            )
        )

    sync_events = [
        event
        for event in events
        if event.get("event_type") in {"stream_sync", "execute_barrier_node"}
    ]
    sync_duration = sum(int(event.get("duration_ns", 0) or 0) for event in sync_events)
    if total_duration > 0 and sync_duration / total_duration >= SYNC_DOMINANCE_RATIO:
        diagnostics.append(
            _make_diagnostic(
                "sync_overhead",
                "warn",
                "runtime",
                "Synchronization overhead is a large portion of the run.",
                {"sync_duration_ns": sync_duration, "total_duration_ns": total_duration},
                ["Inspect barrier nodes and stream sync usage.", "Reduce unnecessary waits."],
            )
        )

    shape_counter = Counter(
        event.get("shape_signature")
        for event in events
        if event.get("shape_signature")
    )
    if len(shape_counter) >= SHAPE_FRAGMENTATION_THRESHOLD:
        diagnostics.append(
            _make_diagnostic(
                "shape_fragmentation",
                "info",
                "runtime_session",
                "Many unique shape signatures were observed.",
                {"unique_shape_signatures": len(shape_counter)},
                ["Bucket dynamic shapes.", "Inspect cache warmup coverage."],
            )
        )

    tiny_kernel_runs = [
        event
        for event in events
        if event.get("event_type") == "kernel_exec"
        and event.get("phase") == "complete" and event.get("status") == "ok"
        and event.get("fields", {}).get("timing") == "host_execute"
        and 0 < int(event.get("duration_ns", 0) or 0) < SMALL_KERNEL_RUN_NS
    ]
    if len(tiny_kernel_runs) >= 5:
        diagnostics.append(
            _make_diagnostic(
                "kernel_launch_overhead",
                "info",
                "runtime",
                "Many short host kernel executions warrant checking invocation overhead.",
                {"tiny_kernel_count": len(tiny_kernel_runs), "threshold_ns": SMALL_KERNEL_RUN_NS,
                 "measurement_domain": "host_execute", "event_type": "kernel_exec"},
                ["Measure invocation overhead before changing the plan.", "Inspect batching and kernel fusion opportunities."],
            )
        )

    comm_count = len(_events_by_type(events, "execute_comm_node"))
    barrier_count = len(_events_by_type(events, "execute_barrier_node"))
    kernel_count = len(_events_by_type(events, "execute_kernel_node"))
    if comm_count + barrier_count > kernel_count and (comm_count + barrier_count) > 0:
        diagnostics.append(
            _make_diagnostic(
                "execution_plan_imbalance",
                "warn",
                "execution_plan",
                "Communication and barrier nodes outnumber kernel nodes.",
                {
                    "kernel_nodes": kernel_count,
                    "comm_nodes": comm_count,
                    "barrier_nodes": barrier_count,
                },
                ["Review disco placement.", "Reduce communication boundaries between kernels."],
            )
        )

    lowering_errors = [
        event
        for event in events
        if event.get("status") == "error"
        and event.get("component") in {"lowering", "relay_pass", "tir_pass"}
    ]
    if lowering_errors:
        first = lowering_errors[0]
        diagnostics.append(
            _make_diagnostic(
                "lowering_failure_context",
                "error",
                str(first.get("component")),
                "Lowering or pass execution failed.",
                {
                    "message": first.get("message"),
                    "pass_name": first.get("pass_name"),
                    "run_id": first.get("run_id"),
                },
                ["Inspect the failure artifact in artifacts/.", "Replay the failing pass trace."],
            )
        )

    if not diagnostics:
        diagnostics.append(
            _make_diagnostic(
                "no_issues_detected",
                "info",
                "analysis",
                "No obvious bottleneck pattern matched the current rules.",
                {},
                ["Inspect pass trace for fine-grained detail.", "Compare this bundle against a baseline."],
            )
        )

    return diagnostics


def _aggregate_pass_durations(events: list[dict[str, Any]]) -> dict[str, int]:
    """按 pass_name 汇总 pass duration。"""

    durations: dict[str, int] = defaultdict(int)
    for event in events:
        pass_name = event.get("pass_name")
        if not pass_name:
            continue
        durations[str(pass_name)] += int(event.get("duration_ns", 0) or 0)
    return dict(durations)


def _compare_cache_health(base_events: list[dict[str, Any]],
                          new_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """比较 cache 命中/同步 miss 的健康度变化。"""

    results: list[dict[str, Any]] = []
    base_hits = len(_events_by_type(base_events, "cache_exact_hit"))
    new_hits = len(_events_by_type(new_events, "cache_exact_hit"))
    base_misses = len(_events_by_type(base_events, "cache_miss_sync_compile"))
    new_misses = len(_events_by_type(new_events, "cache_miss_sync_compile"))
    if new_hits < base_hits:
        results.append(
            {
                "category": "cache_miss_pattern",
                "component": "runtime_session",
                "summary": "Exact cache hit count regressed.",
                "evidence": {
                    "base_cache_exact_hit": base_hits,
                    "new_cache_exact_hit": new_hits,
                    "base_cache_miss": base_misses,
                    "new_cache_miss": new_misses,
                },
            }
        )
    return results


def _compare_device_overheads(base_events: list[dict[str, Any]],
                              new_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """比较设备拷贝和同步开销是否退化。"""

    results: list[dict[str, Any]] = []
    base_copies = _events_by_type(_host_device_events(base_events), "copy")
    new_copies = _events_by_type(_host_device_events(new_events), "copy")
    base_copy = _duration(base_copies)
    new_copy = _duration(new_copies)
    if base_copy > 0 and new_copy > base_copy * 1.2:
        results.append(
            {
                "category": "copy_dominance",
                "component": "device_api",
                "summary": "Measured host copy time increased.",
                "evidence": {
                    "base_copy_duration_ns": base_copy,
                    "new_copy_duration_ns": new_copy,
                    "measurement_domain": "host_execute",
                    "base_copy_bytes": sum(event.get("metrics", {}).get("bytes", 0) for event in base_copies),
                    "new_copy_bytes": sum(event.get("metrics", {}).get("bytes", 0) for event in new_copies),
                },
            }
        )

    base_sync = sum(
        int(event.get("duration_ns", 0) or 0)
        for event in base_events
        if event.get("event_type") in {"stream_sync", "execute_barrier_node"}
    )
    new_sync = sum(
        int(event.get("duration_ns", 0) or 0)
        for event in new_events
        if event.get("event_type") in {"stream_sync", "execute_barrier_node"}
    )
    if base_sync > 0 and new_sync > base_sync * 1.2:
        results.append(
            {
                "category": "sync_overhead",
                "component": "runtime",
                "summary": "Synchronization overhead regressed.",
                "evidence": {
                    "base_sync_duration_ns": base_sync,
                    "new_sync_duration_ns": new_sync,
                },
            }
        )
    return results


def _write_diagnosis_files(bundle_path: Path, result: dict[str, Any]) -> None:
    """将诊断结果写为 JSON 和 Markdown 两种格式。"""

    diagnosis_json = bundle_path / "diagnosis.json"
    diagnosis_md = bundle_path / "diagnosis.md"
    diagnosis_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = ["# Diagnostics", ""]
    for item in result["diagnostics"]:
        lines.append(
            f"- [{item['severity']}] {item['category']} ({item['component']}): {item['summary']}"
        )
    diagnosis_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
