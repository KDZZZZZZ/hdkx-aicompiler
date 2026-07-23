#!/usr/bin/env python3
"""Deterministic, dependency-free NLP reference and capability gate."""

import argparse
import hashlib
import json
import math
from pathlib import Path


CAPABILITIES = (
    "stable_softmax", "masked_softmax_all_masked", "batched_matmul",
    "embedding_gather", "mask_select", "normalization", "slice_concat", "prefill_exact",
    "decode_external_kv", "kv_cache", "dynamic_batching", "copy_event",
)
LAYERS = ("frontend", "relay", "lowering", "llvm", "cuda", "runtime", "numeric", "profile")
STATUSES = {"unsupported", "contracted", "implemented", "validated"}
VALIDATED = {
    ("stable_softmax", "numeric"),
    ("mask_select", "cuda"),
    ("slice_concat", "cuda"),
    ("prefill_exact", "numeric"),
    ("decode_external_kv", "numeric"),
    ("kv_cache", "numeric"),
}
CUDA_INJECTIVE_IMPLEMENTED = {"mask_select", "slice_concat"}
DYNAMIC_BATCHING_GATES = {
    "frontend": "unresolved_onnx_rank_or_dims_rejected",
    "relay": "unknown_extents_representable_not_executable",
    "lowering": "negative_static_extents_rejected",
    "runtime": "no_dynamic_execution_plan",
}
IMPLEMENTED = {
    ("stable_softmax", "frontend"),
    ("stable_softmax", "relay"),
    ("stable_softmax", "lowering"),
    ("stable_softmax", "llvm"),
    ("stable_softmax", "runtime"),
    ("batched_matmul", "frontend"),
    ("batched_matmul", "relay"),
    ("batched_matmul", "lowering"),
    ("batched_matmul", "llvm"),
    ("batched_matmul", "runtime"),
    ("batched_matmul", "numeric"),
    ("embedding_gather", "frontend"),
    ("embedding_gather", "relay"),
    ("embedding_gather", "lowering"),
    ("embedding_gather", "llvm"),
    ("embedding_gather", "runtime"),
    ("embedding_gather", "numeric"),
    ("mask_select", "frontend"),
    ("mask_select", "relay"),
    ("mask_select", "lowering"),
    ("mask_select", "llvm"),
    ("mask_select", "runtime"),
    ("mask_select", "numeric"),
    ("normalization", "frontend"),
    ("normalization", "relay"),
    ("normalization", "lowering"),
    ("normalization", "llvm"),
    ("normalization", "runtime"),
    ("normalization", "numeric"),
    ("slice_concat", "frontend"),
    ("slice_concat", "relay"),
    ("slice_concat", "lowering"),
    ("slice_concat", "llvm"),
    ("slice_concat", "runtime"),
    ("slice_concat", "numeric"),
    ("prefill_exact", "relay"),
    ("prefill_exact", "lowering"),
    ("prefill_exact", "llvm"),
    ("prefill_exact", "runtime"),
    ("decode_external_kv", "relay"),
    ("decode_external_kv", "lowering"),
    ("decode_external_kv", "llvm"),
    ("decode_external_kv", "runtime"),
}
WORKLOAD_KEYS = {
    "id", "fixture", "kind", "logical_extent", "physical_extent", "valid_extent",
    "seed", "tolerance", "expected_gate", "model_revision", "target", "driver",
    "runtime_abi", "compiler_fingerprint", "pipeline_fingerprint", "backend_fingerprint",
    "shape_profile", "fingerprint",
}
SHAPE_PROFILE_KEYS = {
    "batch", "sequence", "hidden", "heads", "head_dim", "dtype", "layout",
    "past_kv_length", "kv_capacity", "kv_page_size", "causal", "padding_lengths",
}
REFERENCE_WORKLOAD_IDENTITY = {
    "model_revision": "reference-v1",
    "target": "cpu-reference",
    "driver": "none",
    "runtime_abi": "not-applicable",
    "compiler_fingerprint": "not-applicable",
    "pipeline_fingerprint": "reference-v1",
    "backend_fingerprint": "not-applicable",
}
COMMON_FIXTURE_KEYS = {
    "id", "kind", "logical_extent", "physical_extent", "valid_extent", "seed",
    "tolerance", "expected_gate", "fingerprint",
}
FIXTURE_EXTRA_KEYS = {
    "stable_softmax": {"logits", "mask", "expected"},
    "prefill_attention": {"causal", "scale", "query", "key", "value", "expected"},
    "decode_external_kv": {"scale", "query", "external_key", "external_value", "expected"},
    "kv_cache_trace": {"page_size", "append_tokens", "expected_trace", "expected_final", "capacity_probe"},
}


class ValidationError(ValueError):
    pass


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def load_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=reject_duplicates)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ValidationError("{}: {}".format(path, error)) from error


def require_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValidationError("{} keys must be {}; got {}".format(label, sorted(expected), actual))


def fingerprint(value):
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def check_fingerprint(value, label):
    actual = value.get("fingerprint")
    expected = fingerprint(value)
    if not isinstance(actual, str) or actual != expected:
        raise ValidationError("{} fingerprint mismatch".format(label))


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError("{} must be a non-negative concrete integer".format(label))
    return value


def validate_extents(value, label):
    logical = integer(value["logical_extent"], label + ".logical_extent")
    physical = integer(value["physical_extent"], label + ".physical_extent")
    valid = integer(value["valid_extent"], label + ".valid_extent")
    if not valid <= logical <= physical:
        raise ValidationError("{} requires valid <= logical <= physical".format(label))


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError("{} must be finite numeric data".format(label))


def numeric_tree(value, label):
    if isinstance(value, list):
        if not value:
            raise ValidationError("{} must not be empty".format(label))
        for index, item in enumerate(value):
            numeric_tree(item, "{}[{}]".format(label, index))
    else:
        finite_number(value, label)


def matrix_shape(value, label):
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        raise ValidationError("{} must be a non-empty matrix".format(label))
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValidationError("{} rows must have a common width".format(label))
    numeric_tree(value, label)
    return len(value), width


def check_evidence(root, evidence, label):
    if not isinstance(evidence, list) or not evidence:
        raise ValidationError("{} evidence must be a non-empty list".format(label))
    for item in evidence:
        if not isinstance(item, str) or not item:
            raise ValidationError("{} has invalid evidence".format(label))
        candidate = (root / item).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as error:
            raise ValidationError("{} evidence escapes root".format(label)) from error
        if not candidate.is_file():
            raise ValidationError("{} evidence is missing: {}".format(label, item))


def validate_matrix(root, matrix):
    require_keys(matrix, {"schema_version", "capabilities"}, "matrix")
    if matrix["schema_version"] != 1:
        raise ValidationError("matrix schema_version must be 1")
    require_keys(matrix["capabilities"], CAPABILITIES, "matrix.capabilities")
    for capability in CAPABILITIES:
        layers = matrix["capabilities"][capability]
        require_keys(layers, LAYERS, "matrix.{}".format(capability))
        for layer in LAYERS:
            record = layers[layer]
            require_keys(record, {"status", "gate", "reason", "evidence"}, "matrix.{}.{}".format(capability, layer))
            if record["status"] not in STATUSES:
                raise ValidationError("matrix.{}.{} has invalid status".format(capability, layer))
            if (capability, layer) in VALIDATED:
                if record["status"] != "validated":
                    raise ValidationError("matrix.{}.{} must retain its reference validation".format(capability, layer))
            elif (capability, layer) in IMPLEMENTED:
                if record["status"] != "implemented":
                    raise ValidationError("matrix.{}.{} must retain its implementation gate".format(capability, layer))
            elif record["status"] in {"implemented", "validated"}:
                raise ValidationError("matrix.{}.{} opens an unvalidated gate".format(capability, layer))
            if not isinstance(record["gate"], str) or not record["gate"]:
                raise ValidationError("matrix.{}.{} gate is invalid".format(capability, layer))
            if not isinstance(record["reason"], str) or not record["reason"]:
                raise ValidationError("matrix.{}.{} reason is invalid".format(capability, layer))
            check_evidence(root, record["evidence"], "matrix.{}.{}".format(capability, layer))
            if (layer == "cuda" and capability not in CUDA_INJECTIVE_IMPLEMENTED and
                    record["status"] != "unsupported"):
                raise ValidationError("matrix.{}.cuda must remain closed".format(capability))
            if layer == "llvm" and record["status"] == "validated":
                raise ValidationError("matrix.{}.llvm must not claim unrun local validation".format(capability))
    for capability in ("stable_softmax", "batched_matmul", "normalization",
                       "prefill_exact", "decode_external_kv"):
        cuda = matrix["capabilities"][capability]["cuda"]
        if cuda["gate"] != "cuda_reduction_unsupported":
            raise ValidationError("{} CUDA reduction gate is not closed".format(capability))
    gather_cuda = matrix["capabilities"]["embedding_gather"]["cuda"]
    if (gather_cuda["status"] != "unsupported" or
            gather_cuda["gate"] != "cuda_indirect_load_schedule_unsupported"):
        raise ValidationError("embedding_gather CUDA indirect-load gate is open or inaccurate")
    for layer, gate in DYNAMIC_BATCHING_GATES.items():
        dynamic = matrix["capabilities"]["dynamic_batching"][layer]
        if dynamic["status"] != "unsupported" or dynamic["gate"] != gate:
            raise ValidationError("dynamic batching gate is open or inaccurate at {}".format(layer))


def validate_fixture(fixture):
    kind = fixture.get("kind")
    if kind not in FIXTURE_EXTRA_KEYS:
        raise ValidationError("unknown fixture kind: {}".format(kind))
    required = COMMON_FIXTURE_KEYS | FIXTURE_EXTRA_KEYS[kind]
    if kind == "stable_softmax" and fixture.get("id") == "masked_softmax_all_masked":
        required = COMMON_FIXTURE_KEYS | {"logits", "mask"}
    require_keys(fixture, required, "fixture.{}".format(fixture.get("id", "<unknown>")))
    if not isinstance(fixture["id"], str) or not fixture["id"]:
        raise ValidationError("fixture id is invalid")
    validate_extents(fixture, "fixture.{}".format(fixture["id"]))
    integer(fixture["seed"], "fixture.{}.seed".format(fixture["id"]))
    finite_number(fixture["tolerance"], "fixture.{}.tolerance".format(fixture["id"]))
    if fixture["tolerance"] < 0 or not isinstance(fixture["expected_gate"], str):
        raise ValidationError("fixture {} gate/tolerance is invalid".format(fixture["id"]))
    check_fingerprint(fixture, "fixture {}".format(fixture["id"]))
    if kind == "stable_softmax":
        if len(fixture["logits"]) != fixture["logical_extent"] or len(fixture["mask"]) != len(fixture["logits"]):
            raise ValidationError("fixture {} softmax extent mismatch".format(fixture["id"]))
        numeric_tree(fixture["logits"], "fixture logits")
        if not all(isinstance(item, bool) for item in fixture["mask"]):
            raise ValidationError("fixture {} mask must contain booleans".format(fixture["id"]))
        if any(fixture["mask"]):
            numeric_tree(fixture["expected"], "fixture expected")
        elif fixture["expected_gate"] != "reject_unsupported":
            raise ValidationError("all-masked softmax must fail closed")
    elif kind == "prefill_attention":
        if not isinstance(fixture["causal"], bool):
            raise ValidationError("prefill causal must be boolean")
        finite_number(fixture["scale"], "prefill scale")
        query_shape = matrix_shape(fixture["query"], "prefill query")
        key_shape = matrix_shape(fixture["key"], "prefill key")
        value_shape = matrix_shape(fixture["value"], "prefill value")
        matrix_shape(fixture["expected"], "prefill expected")
        if query_shape[0] != fixture["logical_extent"] or key_shape[0] != fixture["valid_extent"] or key_shape[1] != query_shape[1] or value_shape[0] != key_shape[0]:
            raise ValidationError("prefill attention dimensions disagree with extents")
    elif kind == "decode_external_kv":
        finite_number(fixture["scale"], "decode scale")
        numeric_tree(fixture["query"], "decode query")
        key_shape = matrix_shape(fixture["external_key"], "decode external key")
        value_shape = matrix_shape(fixture["external_value"], "decode external value")
        numeric_tree(fixture["expected"], "decode expected")
        if len(fixture["query"]) != key_shape[1] or key_shape[0] < fixture["valid_extent"] or value_shape[0] < fixture["valid_extent"]:
            raise ValidationError("decode external KV dimensions disagree with extents")
    else:
        if integer(fixture["page_size"], "KV page_size") == 0 or not isinstance(fixture["append_tokens"], list):
            raise ValidationError("KV trace geometry is invalid")
        require_keys(fixture["expected_final"], {"logical_extent", "physical_extent", "valid_extent"}, "KV expected_final")
        validate_extents(fixture["expected_final"], "KV expected_final")
        require_keys(fixture["capacity_probe"], {"append_count", "expected_gate"}, "KV capacity_probe")
        integer(fixture["capacity_probe"]["append_count"], "KV capacity append_count")
        if fixture["capacity_probe"]["expected_gate"] != "reject_capacity":
            raise ValidationError("KV capacity probe must reject")


def expected_shape_profile(fixture):
    profile = {
        "batch": None, "sequence": None, "hidden": None, "heads": None, "head_dim": None,
        "dtype": None, "layout": None, "past_kv_length": 0, "kv_capacity": 0,
        "kv_page_size": 0, "causal": None, "padding_lengths": None,
    }
    if fixture["kind"] == "stable_softmax":
        profile.update(batch=1, sequence=fixture["logical_extent"], dtype="float64", layout="contiguous")
    elif fixture["kind"] == "prefill_attention":
        head_dim = len(fixture["query"][0])
        profile.update(batch=1, sequence=fixture["logical_extent"], hidden=head_dim,
                       heads=1, head_dim=head_dim, dtype="float64", layout="contiguous",
                       causal=fixture["causal"], padding_lengths=[fixture["valid_extent"]])
    elif fixture["kind"] == "decode_external_kv":
        head_dim = len(fixture["query"])
        profile.update(batch=1, sequence=1, hidden=head_dim, heads=1, head_dim=head_dim,
                       dtype="float64", layout="contiguous",
                       past_kv_length=fixture["valid_extent"],
                       kv_capacity=fixture["physical_extent"], causal=True,
                       padding_lengths=[1])
    else:
        profile.update(batch=1, past_kv_length=fixture["valid_extent"],
                       kv_capacity=fixture["physical_extent"],
                       kv_page_size=fixture["page_size"])
    return profile


def validate_workload_identity(workload, fixture):
    for key, expected in REFERENCE_WORKLOAD_IDENTITY.items():
        if workload[key] != expected:
            raise ValidationError("workload {} {} must be {!r}".format(workload["id"], key, expected))
    require_keys(workload["shape_profile"], SHAPE_PROFILE_KEYS, "workload {} shape_profile".format(workload["id"]))
    if workload["shape_profile"] != expected_shape_profile(fixture):
        raise ValidationError("workload {} shape_profile does not match its fixture".format(workload["id"]))


def validate_manifests(fixtures, manifests):
    require_keys(manifests, {"schema_version", "fixtures_file", "workloads"}, "manifests")
    if manifests["schema_version"] != 1 or manifests["fixtures_file"] != "test/nlp_validation/deterministic_fixtures.json":
        raise ValidationError("manifest header is invalid")
    if not isinstance(manifests["workloads"], list) or not manifests["workloads"]:
        raise ValidationError("workloads must be non-empty")
    fixture_by_id = {fixture["id"]: fixture for fixture in fixtures}
    if len(fixture_by_id) != len(fixtures):
        raise ValidationError("fixture IDs must be unique")
    workload_ids = set()
    for workload in manifests["workloads"]:
        require_keys(workload, WORKLOAD_KEYS, "workload")
        if workload["id"] in workload_ids or workload["fixture"] not in fixture_by_id:
            raise ValidationError("workload IDs and fixture references must be unique and known")
        workload_ids.add(workload["id"])
        validate_extents(workload, "workload.{}".format(workload["id"]))
        integer(workload["seed"], "workload.{}.seed".format(workload["id"]))
        finite_number(workload["tolerance"], "workload.{}.tolerance".format(workload["id"]))
        check_fingerprint(workload, "workload {}".format(workload["id"]))
        fixture = fixture_by_id[workload["fixture"]]
        for key in ("id", "kind", "logical_extent", "physical_extent", "valid_extent", "seed", "tolerance", "expected_gate"):
            fixture_key = "id" if key == "fixture" else key
            if workload[key] != fixture[fixture_key]:
                raise ValidationError("workload {} disagrees with its fixture on {}".format(workload["id"], key))
        validate_workload_identity(workload, fixture)
    if workload_ids != set(fixture_by_id):
        raise ValidationError("every fixture must have exactly one workload")


def stable_softmax(logits, mask):
    active = [value for value, enabled in zip(logits, mask) if enabled]
    if not active:
        raise ValidationError("all-masked softmax is unsupported")
    maximum = max(active)
    exponentials = [math.exp(value - maximum) if enabled else 0.0 for value, enabled in zip(logits, mask)]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def assert_close(actual, expected, tolerance, label):
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValidationError("{} shape mismatch".format(label))
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_close(left, right, tolerance, "{}[{}]".format(label, index))
    elif abs(actual - expected) > tolerance:
        raise ValidationError("{} differs: {} != {}".format(label, actual, expected))


def prefill_attention(fixture):
    query, key, value = fixture["query"], fixture["key"], fixture["value"]
    if len(key) != fixture["valid_extent"] or len(value) != len(key) or len(query) != fixture["logical_extent"]:
        raise ValidationError("prefill attention extent mismatch")
    result = []
    for query_index, query_row in enumerate(query):
        scores = [sum(a * b for a, b in zip(query_row, key_row)) * fixture["scale"] for key_row in key]
        mask = [not fixture["causal"] or key_index <= query_index for key_index in range(len(key))]
        result.append([sum(weight * row[column] for weight, row in zip(stable_softmax(scores, mask), value)) for column in range(len(value[0]))])
    return result


def decode_external_kv(fixture):
    key = fixture["external_key"][:fixture["valid_extent"]]
    value = fixture["external_value"][:fixture["valid_extent"]]
    if len(key) != fixture["valid_extent"] or len(value) != len(key):
        raise ValidationError("decode external KV extent mismatch")
    scores = [sum(a * b for a, b in zip(fixture["query"], row)) * fixture["scale"] for row in key]
    weights = stable_softmax(scores, [True] * len(scores))
    return [sum(weight * row[column] for weight, row in zip(weights, value)) for column in range(len(value[0]))]


def kv_trace(fixture):
    logical, physical, valid = fixture["logical_extent"], fixture["physical_extent"], fixture["valid_extent"]
    page_size = fixture["page_size"]
    if page_size <= 0 or physical % page_size:
        raise ValidationError("KV page geometry is invalid")
    trace = []
    for token in fixture["append_tokens"]:
        if logical >= physical:
            raise ValidationError("KV append exceeded capacity")
        trace.append({"token": token, "logical_index": logical, "page": logical // page_size, "offset": logical % page_size})
        logical += 1
        valid += 1
    return trace, {"logical_extent": logical, "physical_extent": physical, "valid_extent": valid}


def require_valid_context(fixture, requested_extent):
    validate_extents(fixture, "KV context")
    if requested_extent != fixture["valid_extent"]:
        raise ValidationError("physical capacity is not valid context")
    return requested_extent


def require_rejection(label, action):
    try:
        action()
    except ValidationError:
        return
    raise ValidationError("negative gate did not reject: {}".format(label))


def run_references(fixtures, matrix, manifests):
    by_id = {fixture["id"]: fixture for fixture in fixtures}
    stable = by_id["stable_softmax_extreme"]
    assert_close(stable_softmax(stable["logits"], stable["mask"]), stable["expected"], stable["tolerance"], stable["id"])
    print("PASS reference stable_softmax_extreme")
    masked = by_id["masked_softmax_all_masked"]
    require_rejection(masked["id"], lambda: stable_softmax(masked["logits"], masked["mask"]))
    print("PASS negative masked_softmax_all_masked")
    prefill = by_id["prefill_exact_attention"]
    assert_close(prefill_attention(prefill), prefill["expected"], prefill["tolerance"], prefill["id"])
    print("PASS reference prefill_exact_attention")
    decode = by_id["decode_external_kv"]
    assert_close(decode_external_kv(decode), decode["expected"], decode["tolerance"], decode["id"])
    print("PASS reference decode_external_kv")
    cache = by_id["kv_append_capacity_page_trace"]
    trace, final = kv_trace(cache)
    if trace != cache["expected_trace"] or final != cache["expected_final"]:
        raise ValidationError("KV append/page trace mismatch")
    print("PASS reference kv_append_capacity_page_trace")

    require_rejection("unknown/symbolic dim", lambda: validate_extents({"logical_extent": "N", "physical_extent": 4, "valid_extent": 1}, "negative"))
    require_rejection("all-masked softmax", lambda: stable_softmax([1.0], [False]))
    if require_valid_context(cache, cache["valid_extent"]) != cache["valid_extent"]:
        raise ValidationError("valid KV context was not preserved")
    require_rejection("capacity used as valid context", lambda: require_valid_context(cache, cache["physical_extent"]))
    probe = cache["capacity_probe"]
    require_rejection("capacity overflow", lambda: kv_trace({**cache, "append_tokens": cache["append_tokens"] + list(range(probe["append_count"]))}))
    require_rejection("fingerprint mismatch", lambda: check_fingerprint({**manifests["workloads"][0], "fingerprint": "0" * 64}, "negative manifest"))
    cuda = matrix["capabilities"]["stable_softmax"]["cuda"]
    if cuda["status"] != "unsupported" or cuda["gate"] != "cuda_reduction_unsupported":
        raise ValidationError("CUDA reduction unsupported gate is open")
    print("PASS negative gates unknown-symbolic/all-masked/capacity/gather-reduction-CUDA/fingerprint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--matrix", type=Path, default=Path("test/nlp_validation/transformer_capability_matrix.json"))
    parser.add_argument("--fixtures", type=Path, default=Path("test/nlp_validation/deterministic_fixtures.json"))
    parser.add_argument("--manifests", type=Path, default=Path("test/nlp_validation/workload_manifests.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    matrix = load_json(root / args.matrix)
    fixture_document = load_json(root / args.fixtures)
    manifests = load_json(root / args.manifests)
    require_keys(fixture_document, {"schema_version", "fixtures"}, "fixtures")
    if fixture_document["schema_version"] != 1 or not isinstance(fixture_document["fixtures"], list):
        raise ValidationError("fixture header is invalid")
    for fixture in fixture_document["fixtures"]:
        validate_fixture(fixture)
    validate_matrix(root, matrix)
    validate_manifests(fixture_document["fixtures"], manifests)
    print("PASS schemas, fingerprints, and evidence")
    run_references(fixture_document["fixtures"], matrix, manifests)
    print("PASS NLP reference/capability gate; injective CUDA validated, gather/reductions closed")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as error:
        print("FAIL NLP reference and capability gate: {}".format(error))
        raise SystemExit(1)
