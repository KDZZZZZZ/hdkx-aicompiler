"""
职责简介：
- 提供 profiling bundle 离线分析 CLI、规则引擎和工具。
"""

from __future__ import annotations

from .bundle_loader import Bundle


REQUIRED_EVENT_FIELDS = (
    "trace_id",
    "session_id",
    "run_id",
    "span_id",
    "parent_span_id",
    "component",
    "event_type",
    "phase",
    "ts_ns",
    "duration_ns",
    "status",
    "severity",
    "device",
    "worker_id",
    "op_name",
    "pass_name",
    "kernel_symbol",
    "shape_signature",
    "message",
    "metrics",
)


def validate_bundle(bundle: Bundle) -> list[str]:
    """校验 bundle schema 的关键字段和事件数量一致性。"""

    errors: list[str] = []

    schema_version = bundle.manifest.get("schema_version")
    if schema_version != 1:
        errors.append(f"Unsupported schema_version: {schema_version}")

    if not isinstance(bundle.events, list) or not bundle.events:
        errors.append("events.jsonl did not contain any events")
        return errors

    for index, event in enumerate(bundle.events):
        missing = [field for field in REQUIRED_EVENT_FIELDS if field not in event]
        if missing:
            errors.append(f"event[{index}] missing fields: {', '.join(missing)}")
            if len(errors) >= 10:
                break

    if bundle.summary.get("event_count") != len(bundle.events):
        errors.append(
            "summary.event_count does not match the number of events in events.jsonl"
        )

    return errors
