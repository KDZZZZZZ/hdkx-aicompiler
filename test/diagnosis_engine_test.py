"""Public bundle consumers must respect runtime execution/observation semantics."""
import json

from kxc_agent.services.diagnosis_engine import analyze_bundle, compare_bundles
from kxc_agent.services.schema import REQUIRED_EVENT_FIELDS


def event(kind="copy", duration=100, timing="host_execute", **changes):
    record = {field: "" for field in REQUIRED_EVENT_FIELDS}
    record.update(trace_id="trace", session_id="session", run_id="run", span_id="span",
                  component="device_api", event_type=kind, phase="complete", ts_ns=10,
                  duration_ns=duration, status="ok", severity="info", device="cpu:0",
                  worker_id=-1, metrics={"bytes": 32}, fields={"timing": timing})
    record.update(changes)
    return record


def bundle(path, events):
    path.mkdir()
    (path / "artifacts").mkdir()
    for name, payload in {"manifest": {"schema_version": 1, "trace_id": "trace"},
                          "summary": {"event_count": len(events)},
                          "trace": {"traceEvents": []}, "diagnosis": {}}.items():
        (path / f"{name}.json").write_text(json.dumps(payload))
    (path / "diagnosis.md").write_text("")
    (path / "events.jsonl").write_text("".join(json.dumps(item) + "\n" for item in events))
    return path


def diagnostics(path):
    result = analyze_bundle(path)
    records = {item["category"]: item for item in result["diagnostics"]}
    assert "bundle_schema_error" not in records
    return records


def test_copy_counts_only_measured_successful_completions(tmp_path):
    path = bundle(tmp_path / "copies", [
        event(), event(phase="submit", duration=999, timing="host_submit"),
        event(duration=10**9, timing="host_observed_complete"),
        event(duration=10**9, status="error"), event(duration=10**9, fields={}),
    ])
    evidence = diagnostics(path)["copy_dominance"]["evidence"]
    assert evidence == {"copy_duration_ns": 100, "device_duration_ns": 100,
                        "measurement_domain": "host_execute", "copy_count": 1, "copy_bytes": 32}


def test_late_observation_is_not_a_copy_performance_regression(tmp_path):
    base = bundle(tmp_path / "base", [event(), event(timing="host_observed_complete")])
    later = bundle(tmp_path / "later", [event(), event(duration=10**9, timing="host_observed_complete")])
    assert compare_bundles(base, later)["regressions"] == []
    changed = bundle(tmp_path / "changed", [event(duration=130)])
    regression = compare_bundles(base, changed)["regressions"][0]
    assert regression["category"] == "copy_dominance"
    assert regression["evidence"] == {"base_copy_duration_ns": 100, "new_copy_duration_ns": 130,
                                     "measurement_domain": "host_execute", "base_copy_bytes": 32,
                                     "new_copy_bytes": 32}
    pending_only = bundle(tmp_path / "observed", [event(duration=10**9, timing="host_observed_complete")])
    assert "copy_dominance" not in diagnostics(pending_only)


def test_short_kernels_use_current_runtime_events(tmp_path):
    events = [event("kernel_exec", component="execution_plan", kernel_symbol="relu") for _ in range(6)]
    events += [event("compiled_module_run"), event("runtime_session_run")]
    events += [event("kernel_exec", status="error"), event("kernel_exec", duration=0),
               event("kernel_exec", timing="host_observed_complete"), event("kernel_exec", phase="submit")]
    evidence = diagnostics(bundle(tmp_path / "kernels", events))["kernel_launch_overhead"]["evidence"]
    assert evidence["tiny_kernel_count"] == 6
    assert evidence["measurement_domain"] == "host_execute"
    assert evidence["event_type"] == "kernel_exec"


def test_module_or_observation_spans_do_not_establish_short_kernels(tmp_path):
    events = [event("adaptive_module_run") for _ in range(6)]
    events += [event("kernel_exec", timing="host_observed_complete") for _ in range(6)]
    assert "kernel_launch_overhead" not in diagnostics(bundle(tmp_path / "not_kernel_time", events))
