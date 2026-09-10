"""Run a CUDA consumer and verify actual CUPTI-to-runtime joins (stdlib only)."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate(events, model, run_count, call_count, stream_count):
    spans = {e["span_id"]: e for e in events if e["span_id"]}
    require(len(spans) == sum(bool(e["span_id"]) for e in events), "duplicate span identity")
    runs = [e for e in events if e["event_type"] == "runtime_session_run" and e["status"] == "ok"]
    require(len(runs) == run_count, "missing or extra completed model runs")
    require(all(e["fields"].get("model") == model for e in runs), "model context contamination")
    run_ids = {e["run_id"] for e in runs}
    require(len(run_ids) == run_count, "reused run identity")
    launches = [e for e in events if e["event_type"] == "kernel_launch"]
    kernels = [e for e in events if e["event_type"] == "cuda_kernel"]
    require(len(launches) == len(kernels) == run_count * call_count, "missing or extra CUDA launches/activities")
    completed = {(e["run_id"], e["fields"]["call_index"]): e for e in events
                 if e["event_type"] == "kernel_exec"}
    require(len(completed) == len(kernels), "missing runtime completion events")
    joined = Counter()
    leads = []
    streams = set()
    for kernel in kernels:
        require(kernel["run_id"] in run_ids, "device kernel lost its run")
        launch = spans.get(kernel["parent_span_id"])
        require(launch and launch["event_type"] == "kernel_launch", "device kernel lost its launch span")
        require(launch["run_id"] == kernel["run_id"], "device/launch run mismatch")
        run = spans.get(launch["parent_span_id"])
        require(run in runs and run["run_id"] == launch["run_id"], "launch lost its model run span")
        require(launch["fields"].get("model") == model, "launch lost model metadata")
        for name in ("repeat", "stage", "worker", "export_receipt", "sequence_length",
                     "batch", "sequence", "shape_case", "plan_abi", "layers",
                     "past", "step", "state_extent_after", "case",
                     "request_ids", "request_batch_size", "request_past_extent"):
            require(launch["fields"].get(name) == run["fields"].get(name), "launch metadata mismatch: " + name)
        index = launch["fields"]["call_index"]
        end = completed[(kernel["run_id"], index)]
        require(end["parent_span_id"] == run["span_id"], "completion lost its run parent")
        require(launch["fields"]["timing"] == "host_submit" and
                end["fields"]["timing"] == "host_observed_complete", "host timing domain changed")
        require(kernel["status"] == launch["status"] == "ok" and kernel["duration_ns"] > 0,
                "invalid device activity")
        require(kernel["fields"].get("timing") == "device_execute", "device timing domain missing")
        require(kernel["device"] == launch["device"] == run["device"] == "cuda:0", "device mismatch")
        # Host/CUPTI clocks are sampled separately; tolerate 1 ms, never the
        # previous ~15 ms initialization-origin offset. RunAsync's run span
        # ends at submission, so completion supplies the valid upper bound.
        lead = kernel["ts_ns"] - launch["ts_ns"]
        require(lead >= -1_000_000, "device activity precedes its host launch")
        require(kernel["ts_ns"] + kernel["duration_ns"] <=
                end["ts_ns"] + end["duration_ns"] + 1_000_000, "device activity follows observed completion")
        leads.append(lead)
        streams.add(kernel["fields"]["backend.cuda.stream_id"])
        joined[(kernel["run_id"], int(index))] += 1
    require(joined == Counter({(run_id, i): 1 for run_id in run_ids for i in range(call_count)}),
            "CUDA association is not one-to-one with plan calls")
    require(len(streams) == stream_count, "expected independent nondefault streams")
    require(all(int(s) > 0 for s in streams), "default stream used")
    require(not any(int(e["fields"].get("backend.cuda.dropped_records", "0")) for e in events),
            "CUPTI dropped activity records")
    if model.startswith("model_"):
        copies = [e for e in events if e["event_type"] == "cuda_memcpy" and e["run_id"] == "caller"]
        require(len(copies) == 10 and all(e["parent_span_id"] == "caller_parent" for e in copies),
                "moved/disabled scope leaked CUPTI IDs into caller copies")
    return {"model": model, "runs": run_count, "correlated_kernels": len(kernels),
            "streams": sorted(streams), "minimum_launch_to_device_ns": min(leads),
            "maximum_launch_to_device_ns": max(leads)}


def load_bundle(path, cupti=True):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    require(bool(manifest["enable_cupti"]) == cupti, "unexpected CUPTI configuration")
    if cupti:
        require(manifest["cupti_available"], "CUPTI collector unavailable")
    events = [json.loads(line) for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    require(all(e["session_id"] == manifest["session_id"] and
                e["trace_id"] == manifest["trace_id"] for e in events), "bundle identity mismatch")
    return events


def check_bundle(path, model, runs, calls, streams):
    events = load_bundle(path)
    summary = validate(events, model, runs, calls, streams)
    # Exercise the checker on real evidence so an empty/stale bundle cannot
    # make this hardware gate green after a native process exits silently.
    for corruption in ("empty", "run", "parent", "clock", "duplicate", "model"):
        damaged = copy.deepcopy(events)
        device = next(e for e in damaged if e["event_type"] == "cuda_kernel")
        if corruption == "empty":
            damaged = []
        elif corruption == "run":
            device["run_id"] = "foreign"
        elif corruption == "parent":
            device["parent_span_id"] = "missing"
        elif corruption == "clock":
            device["ts_ns"] = -100_000_000
        elif corruption == "duplicate":
            damaged.append(copy.deepcopy(device))
        else:
            next(e for e in damaged if e["event_type"] == "runtime_session_run" and
                 e["status"] == "ok")["fields"]["model"] = "foreign"
        try:
            validate(damaged, model, runs, calls, streams)
        except (AssertionError, KeyError):
            continue
        raise AssertionError("checker accepted corrupted " + corruption)
    summary["negative_checks"] = 6
    (path / "correlation-audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


def validate_copies(events, mode):
    spans = {e["span_id"]: e for e in events if e["span_id"]}
    require(len(spans) == sum(bool(e["span_id"]) for e in events), "duplicate span identity")
    launches = [e for e in events if e["event_type"] == "copy_launch"]
    copies = [e for e in events if e["event_type"] == "copy" and e["fields"].get("copy_id")]
    devices = [e for e in events if e["event_type"] == "cuda_memcpy"]
    expected_count = 4 if mode == "chain" else 1
    require(len(launches) == len(devices) == expected_count, "missing or extra copy launches/DMA")
    require(len(copies) == (7 if mode == "chain" else 2), "missing or extra copy points")
    require(all(e["status"] in ("ok", "info") for e in events), "unexpected error activity")
    require(not any(int(e["fields"].get("backend.cuda.dropped_records", "0")) for e in events),
            "CUPTI dropped records")
    joined, endpoints = Counter(), Counter()
    streams, completed_bytes, device_ns = set(), 0, 0
    for device in devices:
        launch = spans.get(device["parent_span_id"])
        require(launch and launch["event_type"] == "copy_launch", "DMA lost its copy launch")
        copy_id = launch["span_id"]
        points = [e for e in copies if e["fields"].get("copy_id") == copy_id]
        asynchronous = launch["fields"]["submitted_async"] == "true"
        require(Counter(e["phase"] for e in points) ==
                Counter({"submit": 1, "complete": 1} if asynchronous else {"complete": 1}),
                "copy submit/completion pairing is incorrect")
        end = next(e for e in points if e["phase"] == "complete")
        caller = spans.get(launch["parent_span_id"])
        require(caller and caller["event_type"] == "copy_test", "copy lost its caller span")
        require(device["run_id"] == launch["run_id"] == caller["run_id"] == "copy_" + mode,
                "copy run contamination")
        source, target = launch["fields"]["from_device"], launch["fields"]["to_device"]
        size = launch["metrics"]["bytes"]
        require(device["metrics"]["bytes"] == size, "DMA byte count differs from logical copy")
        kind = {("cpu:0", "cuda:0"): "1", ("cuda:0", "cpu:0"): "2",
                ("cuda:0", "cuda:0"): "8"}[(source, target)]
        require(device["fields"]["backend.cuda.copy_kind"] == kind, "DMA direction mismatch")
        require(device["device"] == "cuda:0" and launch["device"] == target, "copy device mismatch")
        for point in points:
            require(point["parent_span_id"] == caller["span_id"] and point["run_id"] == caller["run_id"],
                    "copy completion/submit inherited another parent/run")
            require(point["fields"]["from_device"] == source and point["fields"]["to_device"] == target and
                    point["metrics"]["bytes"] == size, "copy endpoint/byte facts changed")
            if point["phase"] == "submit":
                require(point["fields"]["timing"] == "host_submit", "submit timing changed")
        require(launch["fields"]["timing"] == "host_submit", "launch timing changed")
        require(end["fields"]["timing"] == ("host_observed_complete" if asynchronous else "host_execute"),
                "host completion timing changed")
        require(device["fields"]["timing"] == "device_execute" and device["duration_ns"] > 0,
                "invalid device timing")
        require(device["ts_ns"] >= launch["ts_ns"] - 1_000_000 and
                device["ts_ns"] + device["duration_ns"] <= end["ts_ns"] + end["duration_ns"] + 1_000_000,
                "DMA clock outside launch/completion interval")
        stream = device["fields"]["backend.cuda.stream_id"]
        if asynchronous:
            require(int(stream) > 0, "async copy used default stream")
            streams.add(stream)
        endpoints[(source, target, int(size), asynchronous)] += 1
        joined[copy_id] += 1
        completed_bytes += int(size)
        device_ns += device["duration_ns"]
    require(joined == Counter({e["span_id"]: 1 for e in launches}), "DMA join is not one-to-one")
    expected = Counter({("cuda:0", "cuda:0", 1028, True): 1})
    if mode == "chain":
        expected.update({("cpu:0", "cuda:0", 1028, True): 1,
                         ("cuda:0", "cpu:0", 1028, True): 1,
                         ("cuda:0", "cpu:0", 1036, False): 1})
        validate(events, "copy_chain", 1, 1, 1)
    else:
        require(not any(e["event_type"] == "cuda_kernel" for e in events), "lifetime test launched an extra kernel")
    require(endpoints == expected, "copy direction/size/async coverage changed")
    require(len(streams) == (2 if mode == "chain" else 1), "copy stream coverage changed")
    return {"mode": mode, "correlated_dma": len(devices), "copy_points": len(copies),
            "completed_bytes": completed_bytes, "device_duration_ns": device_ns, "streams": sorted(streams)}


def check_copy_bundle(path, mode):
    events = load_bundle(path)
    summary = validate_copies(events, mode)
    for corruption in ("empty", "run", "parent", "clock", "duplicate", "bytes", "id", "missing", "timing"):
        damaged = copy.deepcopy(events)
        device = next(e for e in damaged if e["event_type"] == "cuda_memcpy")
        completion = next(e for e in damaged if e["event_type"] == "copy" and e["phase"] == "complete"
                          and e["fields"].get("copy_id"))
        if corruption == "empty":
            damaged = []
        elif corruption == "run":
            completion["run_id"] = "foreign"
        elif corruption == "parent":
            device["parent_span_id"] = "missing"
        elif corruption == "clock":
            device["ts_ns"] = -100_000_000
        elif corruption == "duplicate":
            damaged.append(copy.deepcopy(device))
        elif corruption == "bytes":
            device["metrics"]["bytes"] += 1
        elif corruption == "id":
            completion["fields"]["copy_id"] = "missing"
        elif corruption == "missing":
            damaged.remove(completion)
        else:
            completion["fields"]["timing"] = "device_execute"
        try:
            validate_copies(damaged, mode)
        except (AssertionError, KeyError):
            continue
        raise AssertionError("copy checker accepted corrupted " + corruption)
    summary["negative_checks"] = 9
    (path / "copy-correlation-audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


def validate_state(events, mode):
    small = mode == "state"
    calls, run_count = (2, 3) if small else (682, 6)
    summary = validate(events, "cuda_state_contract" if small else "minimind", run_count, calls, 2)
    runs = {e["run_id"]: e for e in events if e["event_type"] == "runtime_session_run" and e["status"] == "ok"}
    errors = [e for e in events if e["event_type"] == "runtime_session_run" and e["status"] == "error"]
    require(len(errors) == (3 if small else 0) and
            all(e["metrics"]["submit_count"] == 0 for e in errors), "preflight errors submitted work")
    require(all(e["status"] in ("ok", "info") or e in errors for e in events), "unexpected failed activity")
    require(all(e["metrics"]["submit_count"] == calls for e in runs.values()), "run receipt lost kernel submissions")
    expected_steps = Counter({("append", str(i)): 1 for i in (1, 2, 3)}) if small else Counter(
        {**{("decode", str(i)): 1 for i in (16, 17, 18, 19)}, ("decode_replay", "19"): 2})
    require(Counter((e["fields"].get("stage"), e["fields"].get("state_extent"))
                    for e in runs.values()) == expected_steps, "state step/extent coverage changed")
    if not small:
        main = [e for e in runs.values() if e["fields"]["stage"] == "decode"]
        replay = [e for e in runs.values() if e["fields"]["stage"] == "decode_replay"]
        require(len({e["fields"].get("plan_abi") for e in main}) == 1 and
                len({e["fields"].get("plan_abi") for e in main + replay}) == 3,
                "fixed decode or replay fill identity changed")
        for e in runs.values():
            f = e["fields"]
            index = int(f["state_extent"]) - 16
            require(f.get("token_index") == str(index) and f.get("token_id") == str((5756, 5756, 1576, 1576)[index])
                    and f.get("generation") == f.get("state_version") == "1"
                    and f.get("export_receipt", "").startswith("sha256:minimind_decode_capacity.onnx:"),
                    "model state/greedy receipt changed")
    fields = ("model", "stage", "state_extent", "generation", "state_version", "plan_abi",
              "export_receipt", "token_index", "token_id")
    for e in events:
        if e["event_type"] in ("kernel_submit", "kernel_launch", "kernel_exec", "copy_launch", "copy") and e["run_id"] in runs:
            require(all(e["fields"].get(k) == runs[e["run_id"]]["fields"].get(k) for k in fields),
                    "kernel/copy lost state metadata")

    launches = {e["span_id"]: e for e in events if e["event_type"] == "copy_launch"}
    points = [e for e in events if e["event_type"] == "copy" and e["fields"].get("copy_id")]
    devices = [e for e in events if e["event_type"] == "cuda_memcpy" and e["parent_span_id"] in launches]
    require(len(launches) == len(points) == len(devices) == (30 if small else 144), "state copy coverage changed")
    require(Counter(e["fields"]["copy_id"] for e in points) == Counter({k: 1 for k in launches}),
            "state copy completion is not one-to-one")
    require(Counter(e["parent_span_id"] for e in devices) == Counter({k: 1 for k in launches}),
            "state DMA is not one-to-one")
    # Constant snapshots belong to compilation. Nonzero state fills use the
    # existing raw DeviceAPI H2D path, before any run and outside Storage copy_id.
    fills = [e for e in events if e["event_type"] == "cuda_memcpy" and
             e["parent_span_id"] not in launches and not e["run_id"].startswith("compile-")]
    require(len(fills) == (2 if small else 32) and all(not e["run_id"] and not e["parent_span_id"] and
            e["fields"]["backend.cuda.copy_kind"] == "1" and e["metrics"]["bytes"] == (480 if small else 49152)
            for e in fills), "unexpected raw state fill DMA")
    ends = {e["fields"]["copy_id"]: e for e in points}
    kernel_end = {r: max(e["ts_ns"] + e["duration_ns"] for e in events
                        if e["event_type"] == "cuda_kernel" and e["run_id"] == r) for r in runs}
    coverage, device_ns = Counter(), 0
    for dma in devices:
        launch = launches[dma["parent_span_id"]]
        end = ends[launch["span_id"]]
        run_id, size = launch["run_id"], launch["metrics"]["bytes"]
        require(dma["run_id"] == end["run_id"] == run_id and
                end["parent_span_id"] == launch["parent_span_id"], "state copy correlation changed")
        require(end["phase"] == "complete" and end["fields"]["timing"] == "host_execute" and
                launch["fields"]["timing"] == "host_submit" and dma["fields"]["timing"] == "device_execute",
                "state copy timing domain changed")
        require(all(e["device"] == "cuda:0" and e["status"] == "ok" for e in (launch, end, dma)) and
                all(e["fields"]["from_device"] == e["fields"]["to_device"] == "cuda:0" and
                    e["fields"]["submitted_async"] == "false" for e in (launch, end)), "state copy device/contract changed")
        require(dma["fields"]["backend.cuda.copy_kind"] == "8" and
                dma["metrics"]["bytes"] == end["metrics"]["bytes"] == size, "state D2D byte count changed")
        require(dma["duration_ns"] > 0 and dma["ts_ns"] >= launch["ts_ns"] - 1_000_000 and
                dma["ts_ns"] + dma["duration_ns"] <= end["ts_ns"] + end["duration_ns"] + 1_000_000,
                "state DMA lies outside observed copy")
        if run_id:
            require(run_id in runs and launch["parent_span_id"] == runs[run_id]["span_id"], "state append lost its run")
            require(dma["ts_ns"] >= kernel_end[run_id], "state commit began before all kernels finished")
            run_end = runs[run_id]["ts_ns"] + runs[run_id]["duration_ns"]
            require(end["ts_ns"] + end["duration_ns"] <= run_end, "RunAsync returned before state copy completed")
        else:
            require(not launch["parent_span_id"], "initialization borrowed another run")
        coverage[(run_id, size)] += 1
        device_ns += dma["duration_ns"]
    expected = Counter({(r, 20 if small else 1536): 6 if small else 16 for r in runs})
    expected.update({("", 20): 12} if small else {("", 24576): 16, ("", 29184): 32})
    require(coverage == expected, "initialization/append byte ranges changed")
    summary.update(state_copies=len(devices), state_bytes=sum(e["metrics"]["bytes"] for e in devices),
                   state_dma_ns=device_ns, zero_submit_rejections=len(errors))
    return summary


def check_state_bundle(path, mode):
    events = load_bundle(path)
    summary = validate_state(events, mode)
    for corruption in ("empty", "extent", "metadata", "kernel_parent", "duplicate", "copy_parent",
                       "bytes", "missing_completion", "commit_order", "run_receipt"):
        damaged = copy.deepcopy(events)
        run = next(e for e in damaged if e["event_type"] == "runtime_session_run" and e["status"] == "ok")
        kernel = next(e for e in damaged if e["event_type"] == "cuda_kernel")
        dma = next(e for e in damaged if e["event_type"] == "cuda_memcpy" and e["run_id"] == run["run_id"])
        if corruption == "empty":
            damaged = []
        elif corruption == "extent":
            run["fields"]["state_extent"] = "99"
        elif corruption == "metadata":
            next(e for e in damaged if e["event_type"] == "kernel_launch")["fields"]["state_extent"] = "99"
        elif corruption == "kernel_parent":
            kernel["parent_span_id"] = "missing"
        elif corruption == "duplicate":
            damaged.append(copy.deepcopy(kernel))
        elif corruption == "copy_parent":
            dma["parent_span_id"] = "missing"
        elif corruption == "bytes":
            dma["metrics"]["bytes"] += 1
        elif corruption == "missing_completion":
            damaged.remove(next(e for e in damaged if e["event_type"] == "copy" and e["fields"].get("copy_id")))
        elif corruption == "commit_order":
            dma["ts_ns"] = kernel["ts_ns"]
        else:
            rejected = next((e for e in damaged if e["event_type"] == "runtime_session_run" and e["status"] == "error"), run)
            rejected["metrics"]["submit_count"] += 1
        try:
            validate_state(damaged, mode)
        except (AssertionError, KeyError):
            continue
        raise AssertionError("state checker accepted corrupted " + corruption)
    summary["negative_checks"] = 10
    (path / "state-audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


def validate_bounded(events, kind):
    # Expected shapes and launches come from each consumer's declared bounds.
    cases = {
        "elementwise": (16, 4, 9, [2, 4, 18, 16, 6, 18, 0, 0], [10, 10, 10, 1], [2] * 4),
        "matmul": (18, 1, 5, [5, 17, 16, 17, 1, 17, 0, 5, 0], [2], [3]),
        "reductions": (16, 9, 4, [1, 5, 18, 16, 7, 18, 1, 5], [10, 1, 2, 1, 10, 2, 2, 2, 10], [2] * 9),
        "attention": (12, 5, 4, [1, 5, 23, 21, 7, 23], [6, 16, 16, 1, 4], [2, 3, 3, 3, 3]),
    }
    model = "bounded_" + kind
    runs, calls, rejects, expected_shapes, expected_grids, scalars_per_call = cases[kind]
    summary = validate(events, model, runs, calls, 2)
    successful = [e for e in events if e["event_type"] == "runtime_session_run" and e["status"] == "ok"]
    expected = Counter((str(i), str(r), str(length)) for i, length in enumerate(expected_shapes) for r in range(2))
    actual = Counter((e["fields"].get("shape_case"), e["fields"].get("repeat"),
                      e["fields"].get("sequence_length")) for e in successful)
    require(actual == expected, "bounded run shape/repeat coverage changed")
    if kind == "reductions":
        require(all(e["fields"].get("masked_nonfinite") == ("1" if e["fields"]["shape_case"] == "7" else "0")
                    for e in successful), "masked non-finite coverage changed")
    failed = [e for e in events if e["event_type"] == "runtime_session_run" and e["status"] == "error"]
    require(len(failed) == rejects and all(e["metrics"]["submit_count"] == 0 for e in failed),
            "invalid bounded input reached a kernel")
    failed_ids = {e["run_id"] for e in failed}
    require(not any(e["run_id"] in failed_ids and e["event_type"] in ("kernel_launch", "cuda_kernel", "cuda_memcpy")
                    for e in events), "rejected input allocated/copied/launched device work")
    kernels = [e for e in events if e["event_type"] == "cuda_kernel"]
    spans = {e["span_id"]: e for e in events if e["span_id"]}
    grids = {}
    for e in kernels:
        launch = spans[e["parent_span_id"]]
        index = int(launch["fields"]["call_index"])
        configuration = tuple(e["metrics"][key] for key in ("grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z"))
        require(configuration == (expected_grids[index], 1, 1, 256, 1, 1),
                "bounded kernel did not use one fixed launch derived from its upper bounds")
        require(grids.setdefault(index, configuration) == configuration, "runtime shape changed kernel launch artifact")
    run_ids = {r["run_id"] for r in successful}
    copies = [e for e in events if e["event_type"] == "cuda_memcpy" and e["run_id"] in run_ids]
    # Extent buffers use the existing target-resident uint64[1] ABI. Input
    # tensors were uploaded before RunAsync; only generated scalars copy here.
    scalars_per_run = sum(scalars_per_call)
    require(len(copies) == runs * scalars_per_run and all(e["metrics"]["bytes"] == 8 and
            e["fields"]["backend.cuda.copy_kind"] == "1" for e in copies),
            "runtime extent ABI copy count, bytes or direction changed")
    require(Counter(e["run_id"] for e in copies) == Counter({r["run_id"]: scalars_per_run for r in successful}),
            "runtime extent copies crossed invocation ownership")
    for kernel in kernels:
        extents = [e for e in copies if e["parent_span_id"] == kernel["parent_span_id"]]
        index = int(spans[kernel["parent_span_id"]]["fields"]["call_index"])
        require(len(extents) == scalars_per_call[index] and
                all(e["run_id"] == kernel["run_id"] and e["status"] == "ok" and e["duration_ns"] > 0 and
                    e["ts_ns"] + e["duration_ns"] <= kernel["ts_ns"] for e in extents),
                "extent DMA was not complete before its consuming kernel")
    summary.update(zero_submit_rejections=rejects, extent_copies=len(copies),
                   extent_bytes=sum(e["metrics"]["bytes"] for e in copies), fixed_launches=grids)
    return summary


def check_bounded_bundle(path, kind):
    events = load_bundle(path)
    summary = validate_bounded(events, kind)
    for corruption in ("empty", "parent", "shape", "grid", "scalar", "scalar_order", "duplicate", "rejection"):
        damaged = copy.deepcopy(events)
        kernel = next(e for e in damaged if e["event_type"] == "cuda_kernel")
        run = next(e for e in damaged if e["event_type"] == "runtime_session_run" and e["status"] == "ok")
        if corruption == "empty": damaged = []
        elif corruption == "parent": kernel["parent_span_id"] = "missing"
        elif corruption == "shape": run["fields"]["shape_case"] = "99"
        elif corruption == "grid": kernel["metrics"]["grid_x"] += 1
        elif corruption == "scalar":
            next(e for e in damaged if e["event_type"] == "cuda_memcpy" and e["run_id"] == run["run_id"])["metrics"]["bytes"] = 4
        elif corruption == "scalar_order":
            next(e for e in damaged if e["event_type"] == "cuda_memcpy" and
                 e["parent_span_id"] == kernel["parent_span_id"])["ts_ns"] = kernel["ts_ns"]
        elif corruption == "duplicate": damaged.append(copy.deepcopy(kernel))
        else:
            next(e for e in damaged if e["event_type"] == "runtime_session_run" and e["status"] == "error")["metrics"]["submit_count"] = 1
        try:
            validate_bounded(damaged, kind)
        except (AssertionError, KeyError):
            continue
        raise AssertionError("bounded checker accepted corrupted " + corruption)
    summary["negative_checks"] = 8
    (path / "bounded-audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


def validate_bounded_prefill(events):
    summary = validate(events, "minimind_bounded", 10, 742, 2)
    runs = [e for e in events if e["event_type"] == "runtime_session_run" and e["status"] == "ok"]
    expected = Counter((str(i), str(b), str(s), stage, str(repeat))
                       for i, b, s in ((0, 1, 1), (1, 1, 4), (2, 2, 3), (3, 3, 8))
                       for stage in (["minimind_bounded_prefill", "minimind_bounded_prefill_future"]
                                     if i == 3 else ["minimind_bounded_prefill"])
                       for repeat in range(2))
    fields = ("shape_case", "batch", "sequence_length", "stage", "repeat")
    require(Counter(tuple(e["fields"].get(f) for f in fields) for e in runs) == expected,
            "full bounded MiniMind shape/stage/repeat coverage changed")
    require(all(e["fields"].get("layers") == "8" and e["fields"].get("export_receipt") ==
                "c52ef7d36dc5d37611341adf7d3e4ba0cc9255e665199acc1ec4712dd473a1d6" for e in runs),
            "full model lost its actual eight-layer export receipt")
    require(len({e["fields"].get("plan_abi") for e in runs}) == 1 and runs[0]["fields"].get("plan_abi"),
            "runtime shapes did not share one compiled plan ABI")
    rejected = [e for e in events if e["event_type"] == "runtime_session_run" and e["status"] == "error"]
    require(len(rejected) == 9 and all(e["metrics"]["submit_count"] == 0 for e in rejected),
            "full model lost its nine preflight rejections")
    failed_ids = {e["run_id"] for e in rejected}
    require(not any(e["run_id"] in failed_ids and e["event_type"] in
                    ("alloc", "kernel_launch", "cuda_kernel", "cuda_memcpy") for e in events),
            "invalid full model input allocated or launched device work")
    spans = {e["span_id"]: e for e in events if e["span_id"]}
    run_ids = {e["run_id"] for e in runs}
    copies = [e for e in events if e["event_type"] == "cuda_memcpy" and e["run_id"] in run_ids]
    # This actual ONNX export has ten S-only calls; the other 732 use B and S.
    single_extent_calls = {21, 28, 62, 153, 244, 335, 426, 517, 608, 699}
    require(len(copies) == 14740 and all(e["metrics"]["bytes"] == 8 and
            e["fields"]["backend.cuda.copy_kind"] == "1" for e in copies),
            "full model extent count, uint64 bytes or H2D direction changed")
    require(Counter(e["run_id"] for e in copies) == Counter({r: 1474 for r in run_ids}),
            "full model extent copies crossed invocation ownership")
    by_parent = {}
    for e in copies:
        by_parent.setdefault(e["parent_span_id"], []).append(e)
    launches, grids = set(), {}
    for kernel in (e for e in events if e["event_type"] == "cuda_kernel"):
        launch = spans[kernel["parent_span_id"]]
        index = int(launch["fields"]["call_index"])
        launches.add(launch["span_id"])
        extents = by_parent.get(launch["span_id"], [])
        require(len(extents) == (1 if index in single_extent_calls else 2) and
                all(e["run_id"] == kernel["run_id"] and e["status"] == "ok" and e["duration_ns"] > 0 and
                    e["ts_ns"] + e["duration_ns"] <= kernel["ts_ns"] for e in extents),
                "full model kernel did not consume its own completed extent DMA")
        configuration = tuple(kernel["metrics"][key] for key in
                              ("grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z"))
        require(configuration[0] > 0 and configuration[1:] == (1, 1, 256, 1, 1) and
                grids.setdefault(index, configuration) == configuration,
                "full model runtime shape changed a fixed CUDA launch artifact")
    require(set(by_parent) == launches, "full model extent copy has a foreign consumer")
    summary.update(zero_submit_rejections=9, extent_copies=len(copies), extent_bytes=117920,
                   fixed_launches=grids)
    return summary


def check_bounded_prefill_bundle(path):
    events = load_bundle(path)
    summary = validate_bounded_prefill(events)
    for corruption in ("empty", "shape", "abi", "receipt", "grid", "scalar", "scalar_order", "rejection"):
        damaged = copy.deepcopy(events)
        run = next(e for e in damaged if e["event_type"] == "runtime_session_run" and e["status"] == "ok")
        kernel = next(e for e in damaged if e["event_type"] == "cuda_kernel")
        if corruption == "empty": damaged = []
        elif corruption == "shape": run["fields"]["batch"] = "99"
        elif corruption == "abi": run["fields"]["plan_abi"] = "foreign"
        elif corruption == "receipt": run["fields"]["export_receipt"] = "foreign"
        elif corruption == "grid": kernel["metrics"]["grid_x"] += 1
        elif corruption == "scalar":
            next(e for e in damaged if e["event_type"] == "cuda_memcpy" and
                 e["run_id"] == run["run_id"])["metrics"]["bytes"] = 4
        elif corruption == "scalar_order":
            next(e for e in damaged if e["event_type"] == "cuda_memcpy" and
                 e["parent_span_id"] == kernel["parent_span_id"])["ts_ns"] = kernel["ts_ns"]
        else:
            next(e for e in damaged if e["event_type"] == "runtime_session_run" and
                 e["status"] == "error")["metrics"]["submit_count"] = 1
        try:
            validate_bounded_prefill(damaged)
        except (AssertionError, KeyError):
            continue
        raise AssertionError("full model checker accepted corrupted " + corruption)
    summary["negative_checks"] = 8
    (path / "bounded-prefill-audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


def validate_request_schedule(runs):
    # Locked full-model scenario: A/C start at P=4, B/D at P=0; D reuses A's
    # released physical slot, but its request id must never reuse A's id.
    cases = (("admit_ac_p4_skip_b_p0", "1,3", 2, 4), ("b_p0", "2", 1, 0),
             ("replacement_d_p0", "4", 1, 0), ("merge_bd_p1_skip_c_p5", "2,4", 2, 1),
             ("c_p5", "3", 1, 5))
    batches = sorted((e for e in runs if e["fields"].get("stage") == "minimind_request_batching"),
                     key=lambda e: e["ts_ns"])
    keys = ("case", "request_ids", "batch", "past", "request_batch_size", "request_past_extent", "state_extent_after")
    require([tuple(e["fields"].get(k) for k in keys) for e in batches] ==
            [(case, ids, str(b), str(p), str(b), str(p), str(p + 1)) for case, ids, b, p in cases],
            "request FIFO, merge, departure/reuse or extent receipt changed")
    references = [e for e in runs if e["fields"].get("stage") == "request_batch_reference"]
    require(len(runs) == 12 and len(batches) == 5 and len(references) == 7 and
            Counter((e["fields"].get("case"), e["fields"].get("batch"), e["fields"].get("past"))
                    for e in references) == Counter({(case, "1", str(p)): b for case, _, b, p in cases}),
            "independent request reference coverage changed")


def validate_request_management_copies(events, device="cuda:0"):
    # These Storage copies belong to the caller context, outside graph runs:
    # 32 initial KV rows, 8 queued snapshots, 7 merged token rows and 112
    # diagnostic KV rows. Raw host byte uploads/downloads have no Storage id.
    launches = {e["span_id"]: e for e in events if e["event_type"] == "copy_launch" and
                e["run_id"] == "minimind_bounded_decode"}
    points = [e for e in events if e["event_type"] == "copy" and e["run_id"] == "minimind_bounded_decode"]
    require(len(launches) == len(points) == 159 and
            Counter(e["fields"].get("copy_id") for e in points) == Counter({k: 1 for k in launches}),
            "request management copies lost their submission/completion pairing")
    ends = {e["fields"]["copy_id"]: e for e in points}
    expected = Counter({6144: 32, 8: 15, 7680: 32, 1536: 32, 3072: 32, 9216: 16})
    require(Counter(int(e["metrics"]["bytes"]) for e in launches.values()) == expected,
            "request snapshot, initialization, merge or diagnostic copy ranges changed")
    for copy_id, launch in launches.items():
        end = ends[copy_id]
        require(all(e["status"] == "ok" and e["device"] == device and not e["parent_span_id"] and
                    e["fields"]["from_device"] == e["fields"]["to_device"] == device and
                    e["fields"]["submitted_async"] == "false" for e in (launch, end)) and
                end["phase"] == "complete" and end["fields"]["timing"] == "host_execute" and
                launch["fields"]["timing"] == "host_submit" and
                launch["metrics"]["bytes"] == end["metrics"]["bytes"] and
                launch["ts_ns"] <= end["ts_ns"] + end["duration_ns"],
                "request management copy lost its synchronous same-device contract")
    runs = [e for e in events if e["event_type"] == "runtime_session_run" and e["status"] == "ok"]
    for run in (e for e in runs if e["fields"].get("stage") == "minimind_request_batching"):
        preceding = [e for e in runs if e["fields"].get("stage") == "request_batch_reference" and
                     e["fields"].get("case") == run["fields"]["case"]]
        start = max(e["ts_ns"] + e["duration_ns"] for e in preceding)
        require(sum(e["metrics"]["bytes"] == 8 and start <= e["ts_ns"] and
                    e["ts_ns"] + e["duration_ns"] <= run["ts_ns"] for e in points) ==
                int(run["fields"]["request_batch_size"]), "merged token rows were not ready before the graph run")
    if device == "cuda:0":
        dmas = [e for e in events if e["event_type"] == "cuda_memcpy" and e["parent_span_id"] in launches]
        require(Counter(e["parent_span_id"] for e in dmas) == Counter({k: 1 for k in launches}),
                "request management DMA lost a copy id")
        for dma in dmas:
            launch, end = launches[dma["parent_span_id"]], ends[dma["parent_span_id"]]
            require(dma["status"] == "ok" and dma["device"] == device and dma["run_id"] == launch["run_id"] and
                    dma["fields"]["backend.cuda.copy_kind"] == "8" and dma["duration_ns"] > 0 and
                    dma["metrics"]["bytes"] == launch["metrics"]["bytes"] and
                    dma["ts_ns"] >= launch["ts_ns"] - 1_000_000 and dma["ts_ns"] + dma["duration_ns"] <=
                    end["ts_ns"] + end["duration_ns"] + 1_000_000,
                    "request management copy completed before its DMA or changed bytes/device")
        by_id = {e["parent_span_id"]: e for e in dmas}
        for run in (e for e in runs if e["fields"].get("stage") == "minimind_request_batching"):
            start = max(e["ts_ns"] + e["duration_ns"] for e in runs if
                        e["fields"].get("stage") == "request_batch_reference" and
                        e["fields"].get("case") == run["fields"]["case"])
            first_kernel = min(e["ts_ns"] for e in events if e["event_type"] == "cuda_kernel" and e["run_id"] == run["run_id"])
            for copy_id, end in ends.items():
                if end["metrics"]["bytes"] == 8 and start <= end["ts_ns"] and end["ts_ns"] + end["duration_ns"] <= run["ts_ns"]:
                    dma = by_id[copy_id]
                    require(dma["ts_ns"] + dma["duration_ns"] <= first_kernel,
                            "request kernel read unfinished merged token DMA")
    else:
        require(device == "cpu:0", "unsupported request management audit device")
    return {"management_copies": len(points), "management_bytes": sum(e["metrics"]["bytes"] for e in points),
            "management_dma_verified": device == "cuda:0"}


def validate_bounded_decode(events, requests=False):
    summary = validate(events, "minimind_bounded", 12 if requests else 16, 774, 2)
    runs = {e["run_id"]: e for e in events if e["event_type"] == "runtime_session_run" and e["status"] == "ok"}
    expected = Counter(("minimind_bounded_" + stage, str(b), str(p))
                       for stage in ("decode", "owned_decode") for b, p in ((1, 0), (1, 1), (2, 4), (3, 8)))
    expected.update(("minimind_bounded_" + stage, "1", str(p))
                    for stage in ("decode_loop", "state_greedy") for p in range(4, 8))
    if requests:
        validate_request_schedule(list(runs.values()))
    else:
        require(Counter((e["fields"].get("stage"), e["fields"].get("batch"), e["fields"].get("past"))
                        for e in runs.values()) == expected, "bounded decode shape/stage coverage changed")
    require(all(e["metrics"]["submit_count"] == 774 and e["fields"].get("layers") == "8" and
                e["fields"].get("export_receipt") == "158ea389d1a332ee51df9901459c387034fd6b1eed4436fcbcd4740f9fba5fb9"
                and e["fields"].get("plan_abi") for e in runs.values()), "decode lost model, ABI or submit receipt")
    fresh_abis, greedy_abis, state_abis = set(), set(), set()
    state_stages = ("minimind_request_batching",) if requests else (
        "minimind_bounded_owned_decode", "minimind_bounded_state_greedy")
    for run in runs.values():
        f = run["fields"]
        stateful = f["stage"] in state_stages
        if stateful:
            require(f.get("state_extent_after") == str(int(f["past"]) + 1), "state did not commit one token")
            state_abis.add(f["plan_abi"])
        else:
            fresh_abis.add(f["plan_abi"])
        if f["stage"].endswith(("decode_loop", "state_greedy")):
            require(f.get("step") == str(int(f["past"]) - 4), "greedy step differs from its past extent")
        if f["stage"] == "minimind_bounded_state_greedy":
            greedy_abis.add(f["plan_abi"])
    require(len(fresh_abis) == 1 and (requests or len(greedy_abis) == 1) and
            len(state_abis) == (1 if requests else 4) and
            not fresh_abis.intersection(state_abis), "state capacity/shape changed the wrong plan identity")
    errors = [e for e in events if e["event_type"] == "runtime_session_run" and e["status"] == "error"]
    require(len(errors) == (0 if requests else 13) and all(e["metrics"]["submit_count"] == 0 for e in errors),
            "bounded decode lost preflight rejection coverage")
    require(all(e["status"] in ("ok", "info") or e in errors for e in events),
            "unexpected failed decode activity")
    failed_ids = {e["run_id"] for e in errors}
    require(not any(e["run_id"] in failed_ids and e["event_type"] in
                    ("alloc", "copy", "copy_launch", "kernel_launch", "cuda_kernel", "cuda_memcpy") for e in events),
            "rejected decode allocated, copied or launched")
    spans = {e["span_id"]: e for e in events if e["span_id"]}
    kernels = [e for e in events if e["event_type"] == "cuda_kernel"]
    copies = [e for e in events if e["event_type"] == "cuda_memcpy" and e["run_id"] in runs]
    scalars = [e for e in copies if e["fields"]["backend.cuda.copy_kind"] == "1"]
    state_copies = [e for e in copies if e["fields"]["backend.cuda.copy_kind"] == "8"]
    require(len(copies) == len(scalars) + len(state_copies) and len(scalars) == 934 * len(runs),
            "unexpected runtime DMA or missing uint64 extents")
    # Locked actual export: 18 static calls, 178 B/P calls, 578 single-extent calls.
    static_calls = {21, 28} | {i + 95 * layer for layer in range(8) for i in (63, 64)}
    double_calls = {20, 27} | {i + 95 * layer for layer in range(8)
                             for i in (*range(52, 63), *range(66, 69), *range(72, 80))}
    by_parent = {}
    for dma in scalars:
        by_parent.setdefault(dma["parent_span_id"], []).append(dma)
    require(Counter(e["run_id"] for e in scalars) == Counter({r: 934 for r in runs}),
            "extent DMA crossed a decode invocation")
    consumed, grids, kernel_ranges, first_submits = set(), {}, {}, {}
    for kernel in kernels:
        launch = spans[kernel["parent_span_id"]]
        index = int(launch["fields"]["call_index"])
        r = kernel["run_id"]
        first_submits[r] = min(first_submits.get(r, launch["ts_ns"]), launch["ts_ns"])
        extent_count = 0 if index in static_calls else 2 if index in double_calls else 1
        extents = by_parent.get(launch["span_id"], [])
        require(len(extents) == extent_count and all(e["run_id"] == kernel["run_id"] and
                e["status"] == "ok" and e["metrics"]["bytes"] == 8 and e["duration_ns"] > 0 and
                e["ts_ns"] + e["duration_ns"] <= kernel["ts_ns"] for e in extents),
                "decode kernel consumed missing, foreign or unfinished extents")
        if extent_count:
            consumed.add(launch["span_id"])
        grid = tuple(kernel["metrics"][k] for k in ("grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z"))
        require(grid[0] > 0 and grid[1:] == (1, 1, 256, 1, 1) and grids.setdefault(index, grid) == grid,
                "decode shape changed its fixed launch artifact")
        bounds = kernel_ranges.setdefault(kernel["run_id"], [kernel["ts_ns"], kernel["ts_ns"] + kernel["duration_ns"]])
        bounds[0] = min(bounds[0], kernel["ts_ns"])
        bounds[1] = max(bounds[1], kernel["ts_ns"] + kernel["duration_ns"])
    require(set(by_parent) == consumed, "extent DMA has an unknown consumer")
    launches = {e["span_id"]: e for e in events if e["event_type"] == "copy_launch" and e["run_id"] in runs}
    points = [e for e in events if e["event_type"] == "copy" and e["run_id"] in runs]
    require(len(launches) == len(points) == len(state_copies) == (192 if requests else 336) and
            Counter(e["fields"].get("copy_id") for e in points) == Counter({k: 1 for k in launches}) and
            Counter(e["parent_span_id"] for e in state_copies) == Counter({k: 1 for k in launches}),
            "bounded state DMA/launch/completion is not one-to-one")
    ends = {e["fields"]["copy_id"]: e for e in points}
    coverage = Counter()
    for dma in state_copies:
        launch = launches[dma["parent_span_id"]]
        end = ends[launch["span_id"]]
        run = runs[dma["run_id"]]
        f = run["fields"]
        require(f["stage"] in state_stages,
                "fresh decode mutated session state")
        require(launch["run_id"] == end["run_id"] == dma["run_id"] and
                launch["parent_span_id"] == end["parent_span_id"] == run["span_id"], "state copy lost its run")
        require(all(e["fields"].get(k) == f.get(k) for e in (launch, end)
                    for k in ("model", "stage", "batch", "past", "step", "state_extent_after", "plan_abi", "export_receipt",
                              "case", "request_ids", "request_batch_size", "request_past_extent")),
                "state copy lost model/extent metadata")
        require(all(e["status"] == "ok" and e["device"] == "cuda:0" for e in (launch, end, dma)) and
                all(e["fields"]["from_device"] == e["fields"]["to_device"] == "cuda:0" and
                    e["fields"]["submitted_async"] == "false" for e in (launch, end)) and
                end["phase"] == "complete" and end["fields"]["timing"] == "host_execute" and
                launch["fields"]["timing"] == "host_submit" and dma["fields"]["timing"] == "device_execute",
                "bounded state copy lost its synchronous device contract")
        size = dma["metrics"]["bytes"]
        require(size == launch["metrics"]["bytes"] == end["metrics"]["bytes"] and dma["duration_ns"] > 0 and
                dma["ts_ns"] >= launch["ts_ns"] - 1_000_000 and dma["ts_ns"] + dma["duration_ns"] <=
                end["ts_ns"] + end["duration_ns"] + 1_000_000, "bounded state copy bytes/timing changed")
        first, last = kernel_ranges[dma["run_id"]]
        phase = "prefix" if dma["ts_ns"] + dma["duration_ns"] <= first else "append"
        if phase == "prefix":
            require(end["ts_ns"] + end["duration_ns"] <= first_submits[dma["run_id"]],
                    "state prefix copy did not finish before submission")
        else:
            require(dma["ts_ns"] >= last, "state append overlapped a kernel reading the old cache")
        require(end["ts_ns"] + end["duration_ns"] <= run["ts_ns"] + run["duration_ns"],
                "RunAsync returned before the state copy finished")
        coverage[(dma["run_id"], phase, int(size))] += 1
    expected_copies = Counter()
    for r, run in runs.items():
        if run["fields"]["stage"] in state_stages:
            batch, past = int(run["fields"]["batch"]), int(run["fields"]["past"])
            expected_copies[(r, "append", 1536)] = 16 * batch
            if past:
                expected_copies[(r, "prefix", past * 1536)] = 16 * batch
    require(coverage == expected_copies, "state prefix/append row ranges changed")
    summary.update(zero_submit_rejections=len(errors), extent_copies=len(scalars), extent_bytes=8*len(scalars),
                   state_copies=len(state_copies), state_bytes=sum(e["metrics"]["bytes"] for e in state_copies))
    if requests:
        summary.update(validate_request_management_copies(events))
    return summary


def check_bounded_decode_bundle(path, requests=False):
    events = load_bundle(path)
    summary = validate_bounded_decode(events, requests)
    corruptions = ("empty", "past", "extent", "abi", "receipt", "grid", "scalar", "scalar_order",
                   "state_bytes", "copy_parent", "copy_completion", "commit_order") + (
                   ("request_ids", "request_batch_size", "request_past_extent", "case", "queue_copy") if requests else ("rejection",))
    for corruption in corruptions:
        damaged = copy.deepcopy(events)
        run = next(e for e in damaged if e["event_type"] == "runtime_session_run" and
                   e["fields"].get("stage") == ("minimind_request_batching" if requests else "minimind_bounded_state_greedy")
                   and e["status"] == "ok")
        kernel = next(e for e in damaged if e["event_type"] == "cuda_kernel")
        scalar = next(e for e in damaged if e["event_type"] == "cuda_memcpy" and e["parent_span_id"] == kernel["parent_span_id"] and
                      e["fields"]["backend.cuda.copy_kind"] == "1")
        state = next(e for e in damaged if e["event_type"] == "cuda_memcpy" and e["run_id"] == run["run_id"] and
                     e["fields"]["backend.cuda.copy_kind"] == "8")
        if corruption == "empty": damaged = []
        elif corruption == "past": run["fields"]["past"] = "99"
        elif corruption == "extent": run["fields"]["state_extent_after"] = "99"
        elif corruption == "abi": run["fields"]["plan_abi"] = "foreign"
        elif corruption == "receipt": run["fields"]["export_receipt"] = "foreign"
        elif corruption == "grid": kernel["metrics"]["grid_x"] += 1
        elif corruption == "scalar": scalar["metrics"]["bytes"] = 4
        elif corruption == "scalar_order": scalar["ts_ns"] = kernel["ts_ns"]
        elif corruption == "state_bytes": state["metrics"]["bytes"] += 4
        elif corruption == "copy_parent": state["parent_span_id"] = "missing"
        elif corruption == "copy_completion":
            damaged.remove(next(e for e in damaged if e["event_type"] == "copy" and e["fields"].get("copy_id") == state["parent_span_id"]))
        elif corruption == "commit_order":
            state["ts_ns"] = next(e["ts_ns"] for e in damaged if e["event_type"] == "cuda_kernel" and e["run_id"] == state["run_id"])
        elif corruption == "queue_copy":
            damaged.remove(next(e for e in damaged if e["event_type"] == "copy_launch" and
                                e["run_id"] == "minimind_bounded_decode"))
        elif requests:
            run["fields"][corruption] = "foreign"
        else:
            next(e for e in damaged if e["event_type"] == "runtime_session_run" and e["status"] == "error")["metrics"]["submit_count"] = 1
        try:
            validate_bounded_decode(damaged, requests)
        except (AssertionError, KeyError):
            continue
        raise AssertionError("bounded decode checker accepted corrupted " + corruption)
    summary["negative_checks"] = len(corruptions)
    (path / ("bounded-requests-audit.json" if requests else "bounded-decode-audit.json")).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--mode", choices=("concurrent", "prefill", "copy", "state", "decode", "bounded", "bounded-prefill", "bounded-decode", "bounded-requests"), required=True)
    parser.add_argument("--bundle-root", required=True)
    args = parser.parse_args()
    root = Path(args.bundle_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundle = Path(tempfile.mkdtemp(prefix=args.mode + "-", dir=root))
    print("[INFO] CUPTI evidence: " + str(bundle), flush=True)
    env = os.environ.copy()
    env.pop("KXC_PROFILE_BUNDLE_DIR", None)
    if args.mode == "prefill":
        env["KXC_PROFILE_BUNDLE_DIR"] = str(bundle / "prefill")
    if args.mode in ("bounded-prefill", "bounded-decode", "bounded-requests"):
        env["KXC_PROFILE_BUNDLE_DIR"] = str(bundle)
    if args.mode == "decode":
        env["KXC_MINIMIND_CUDA_STATE_BUNDLE_DIR"] = str(bundle)
    arguments = [] if args.mode == "prefill" else ["--cuda"] if args.mode in ("decode", "bounded-prefill", "bounded-decode") else [str(bundle)]
    if args.mode == "bounded-requests": arguments = ["--cuda-batching"]
    command = [args.executable] + arguments
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="", flush=True)
    if result.returncode == 77:
        return 77
    require(result.returncode == 0 and "[PASS]" in result.stdout, "CUDA consumer failed or produced no final success marker")
    if args.mode == "concurrent":
        for model in ("model_0", "model_1"):
            check_bundle(bundle / model, model, 6, 3, 2)
    elif args.mode == "prefill":
        check_bundle(bundle / "prefill", "minimind", 2, 650, 1)
    elif args.mode == "copy":
        for mode in ("chain", "wait", "query", "destroy"):
            check_copy_bundle(bundle / mode, mode)
            if mode != "chain":
                foreign = load_bundle(bundle / (mode + "_foreign"), cupti=False)
                require(not any(e["event_type"] in ("copy", "copy_launch", "cuda_memcpy") for e in foreign),
                        "completion leaked into worker profile")
    elif args.mode == "bounded":
        require("[PASS] bounded_cuda_compact_shapes_and_runtime_extents" in result.stdout,
                "missing final bounded CUDA consumer marker")
        require("[PASS] bounded_reductions: runs=16 calls=9" in result.stdout,
                "missing bounded multistage numeric consumer marker")
        require("[PASS] bounded_attention: runs=12 calls=5" in result.stdout,
                "missing bounded attention numeric consumer marker")
        for kind in ("elementwise", "matmul", "reductions", "attention"):
            check_bounded_bundle(bundle / kind, kind)
        print("[PASS] bounded_cuda_profile_verified", flush=True)
    elif args.mode == "bounded-prefill":
        require("[PASS] full_minimind_bounded_cuda_prefill_and_all_kv_no_runtime_compile" in result.stdout and
                "[PASS] bounded_minimind_prefill: runs=10 calls=742 reference_values=878080 causal_values=602112 streams=2" in result.stdout,
                "missing full bounded CUDA model numerical completion markers")
        check_bounded_prefill_bundle(bundle / "prefill")
        print("[PASS] bounded_minimind_prefill_profile_verified", flush=True)
    elif args.mode == "bounded-decode":
        require("[PASS] full_minimind_bounded_cuda_decode_and_owned_state_no_runtime_compile" in result.stdout and
                "[PASS] bounded_minimind_decode: fresh_runs=8 state_runs=8 calls=774 streams=2 reference_values=692224" in result.stdout,
                "missing full bounded CUDA decode/state numerical completion markers")
        check_bounded_decode_bundle(bundle / "decode")
        check_bundle(bundle / "prefill", "minimind_bounded", 1, 742, 1)
        print("[PASS] bounded_minimind_decode_profile_verified", flush=True)
    elif args.mode == "bounded-requests":
        require("[PASS] full_minimind_bounded_cuda_request_batching_no_runtime_compile" in result.stdout and
                "[PASS] bounded_minimind_requests: batches=5 request_steps=7 calls=774 streams=2 reference_values=99328" in result.stdout,
                "missing full bounded CUDA request batching numerical completion markers")
        check_bounded_decode_bundle(bundle / "batching", requests=True)
        print("[PASS] bounded_minimind_requests_profile_verified", flush=True)
    else:
        marker = "cuda_capacity_decode_loop" if args.mode == "decode" else "cuda_state_external_axis2_capacity_and_ownership"
        require("[PASS] " + marker in result.stdout, "missing final state consumer marker")
        check_state_bundle(bundle, args.mode)
        if args.mode == "decode":
            check_bundle(bundle / "prefill", "minimind", 1, 650, 1)
        print("[PASS] cuda_state_profile_verified", flush=True)
    print("[PASS] cuda_profile_correlation_verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
