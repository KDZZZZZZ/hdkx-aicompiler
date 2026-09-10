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
    "vision_encoder", "vlm_joint",
)
LAYERS = ("frontend", "relay", "lowering", "llvm", "cuda", "runtime", "numeric", "profile")
STATUSES = {"unsupported", "contracted", "implemented", "validated"}
VALIDATED = {
    ("copy_event", "numeric"),
    ("masked_softmax_all_masked", "numeric"),
    ("stable_softmax", "numeric"),
    ("batched_matmul", "numeric"),
    ("embedding_gather", "numeric"),
    ("normalization", "numeric"),
    ("slice_concat", "numeric"),
    ("mask_select", "numeric"),
    ("prefill_exact", "numeric"),
    ("decode_external_kv", "numeric"),
    ("kv_cache", "numeric"),
    ("dynamic_batching", "numeric"),
    ("vision_encoder", "numeric"),
    ("vlm_joint", "numeric"),
}
CUDA_LOCAL_EVIDENCE_GATES = {
    "stable_softmax": "cuda_softmax_static_and_bounded_multistage_local_evidence",
    "masked_softmax_all_masked": "cuda_masked_softmax_static_and_bounded_multistage_local_evidence",
    "normalization": "cuda_normalization_static_and_bounded_rms_local_evidence",
    "batched_matmul": "cuda_output_owned_matmul_static_and_bounded_attention_local_evidence",
    "mask_select": "cuda_where_static_nd_local_evidence",
    "slice_concat": "cuda_slice_concat_static_nd_local_evidence",
    "embedding_gather": "cuda_guarded_gather_static_local_evidence",
    "prefill_exact": "cuda_minimind_prefill_b1_s16_local_evidence",
    "copy_event": "cuda_copy_cupti_pending_lifetime_local_evidence",
    "decode_external_kv": "cuda_minimind_capacity_decode_b1_c32_local_evidence",
    "kv_cache": "cuda_static_external_state_commit_local_evidence",
    "dynamic_batching": "cuda_request_batching_hardware_evidence",
}
CUDA_PENDING_EVIDENCE_GATES = {}
COPY_EVENT_GATES = {
    "frontend": "runtime_api_not_frontend_operator",
    "relay": "runtime_api_not_relay_operator",
    "lowering": "runtime_copy_not_tir_lowering",
    "llvm": "cpu_copy_llvm_copy_executed",
    "runtime": "storage_async_completion_ownership",
    "numeric": "copy_offset_llvm_cuda_exact",
    "profile": "paired_copy_completion_and_timing_domains",
}
PREFILL_EVIDENCE_GATES = {
    "frontend": "minimind_static_and_bounded_prefill_import",
    "relay": "minimind_exact_and_bounded_prefill_contracts",
    "lowering": "ordinary_minimind_prefill_primitives",
    "llvm": "cpu_llvm_minimind_prefill_executed",
    "runtime": "minimind_prefill_and_state_handoff",
    "numeric": "minimind_prefill_logits_and_present_cpu",
    "profile": "minimind_prefill_receipt_shape_generation",
}
MASKED_SOFTMAX_GATES = {
    "frontend": "explicit_relay_ffi_masked_softmax",
    "relay": "bool_mask_zero_row_contract_v1",
    "lowering": "existing_te_masked_reductions",
    "llvm": "cpu_llvm_masked_softmax_executed",
    "runtime": "bounded_masked_attention_shared_axes",
    "numeric": "masked_attention_independent_cpu",
    "profile": "masked_attention_shape_bundle",
}
DYNAMIC_BATCHING_GATES = {
    "frontend": "minimind_bounded_decode_shape_source",
    "relay": "restricted_leading_batch_and_past_axes",
    "lowering": "bounded_independent_request_kernels",
    "llvm": "cpu_llvm_request_batches_executed",
    "runtime": "session_owned_request_slots_v1",
    "numeric": "minimind_request_batches_match_independent_llvm",
    "profile": "request_batch_ids_and_kernel_receipts",
}
KV_STATE_EVIDENCE_GATES = {
    "frontend": "minimind_static_and_bounded_onnx",
    "relay": "static_capacity_and_bounded_past_present",
    "lowering": "ordinary_static_and_bounded_kv_kernels",
    "llvm": "cpu_llvm_capacity_and_bounded_state_executed",
    "runtime": "session_owned_static_and_bounded_state_v1",
    "numeric": "minimind_compiled_state_handoff_cpu",
    "profile": "minimind_capacity_and_bounded_state_bundles"
}

BOUNDED_ATTENTION_EVIDENCE_GATES = {
    "relay": "restricted_attention_shapes",
    "lowering": "bounded_serial_attention_v2",
    "llvm": "cpu_llvm_bounded_attention_executed",
    "runtime": "bounded_fresh_output_attention",
    "numeric": "bounded_attention_independent_cpu",
    "profile": "bounded_attention_shape_bundle",
}
NORMALIZATION_EVIDENCE_GATES = {
    "frontend": "static_layer_norm_and_minimind_projection_import",
    "relay": "static_layer_norm_and_fixed_reduction_rmsnorm",
    "lowering": "existing_te_weighted_bounded_projection",
    "llvm": "cpu_llvm_normalization_executed",
    "runtime": "bounded_projection_frozen_weights",
    "numeric": "normalization_independent_cpu",
    "profile": "minimind_projection_shape_bundle",
}
EMBEDDING_EVIDENCE_GATES = {
    "frontend": "static_and_bounded_runtime_gather_source",
    "relay": "static_table_dynamic_index_shape",
    "lowering": "existing_te_guarded_bounded_gather",
    "llvm": "cpu_llvm_bounded_embedding_executed",
    "runtime": "bounded_embedding_fresh_outputs",
    "numeric": "embedding_independent_and_full_prefill_cpu",
    "profile": "minimind_bounded_prefill_shape_bundle",
}
DECODE_EVIDENCE_GATES = {
    "frontend": "minimind_decode_shape_source",
    "relay": "bounded_single_token_external_kv",
    "lowering": "existing_te_decode_window_and_shapes",
    "llvm": "cpu_llvm_full_bounded_decode_executed",
    "runtime": "bounded_decode_external_fresh_outputs",
    "numeric": "full_decode_and_greedy_onnx_cpu",
    "profile": "minimind_bounded_decode_shape_bundle",
}
SLICE_CONCAT_EVIDENCE_GATES = {
    "frontend": "static_slice_and_proved_rope_source",
    "relay": "static_operated_axes_symbolic_other_axes",
    "lowering": "existing_te_bounded_slice_concat",
    "llvm": "cpu_llvm_rope_slice_concat_executed",
    "runtime": "bounded_rope_fresh_outputs",
    "numeric": "rope_independent_and_actual_onnx_cpu",
    "profile": "minimind_rope_shape_bundle",
}
CAUSAL_MASK_EVIDENCE_GATES = {
    "frontend": "static_where_and_causal_mask_source",
    "relay": "where_and_float32_trilu_contract",
    "lowering": "existing_te_causal_mask_select",
    "llvm": "cpu_llvm_causal_attention_executed",
    "runtime": "bounded_causal_attention_repeated_axes",
    "numeric": "causal_attention_independent_and_actual_onnx",
    "profile": "minimind_causal_attention_shape_bundle"
}
# L2 rows: CPU/LLVM only. Their CUDA cells stay unsupported until GPU evidence exists.
VISION_ENCODER_GATES = {
    "frontend": "static_siglip2_fixed_shape_import",
    "relay": "static_conv_layernorm_gelu_contracts",
    "lowering": "ordinary_static_vision_primitives",
    "llvm": "cpu_llvm_vision_executed",
    "runtime": "static_vision_fresh_outputs",
    "numeric": "vision_independent_and_upstream_cpu",
    "profile": "vision_run_receipt_bundle",
}
VLM_JOINT_GATES = {
    "frontend": "minimind_v_fixed_and_slot_prefill_import",
    "relay": "fixed_layout_and_bounded_slot_prefill",
    "lowering": "ordinary_static_and_bounded_vlm_primitives",
    "llvm": "cpu_llvm_vlm_prefill_decode_executed",
    "runtime": "vlm_prefill_to_session_owned_state",
    "numeric": "vlm_upstream_references_cpu",
    "profile": "vlm_images_sequence_receipt_bundle",
}
IMPLEMENTED = {
    ("decode_external_kv", "cuda"),
    ("kv_cache", "cuda"),
    ("copy_event", "cuda"),
    ("embedding_gather", "cuda"),
    ("prefill_exact", "cuda"),
    ("stable_softmax", "cuda"),
    ("masked_softmax_all_masked", "cuda"),
    ("normalization", "cuda"),
    ("dynamic_batching", "cuda"),
    ("copy_event", "llvm"),
    ("copy_event", "runtime"),
    ("copy_event", "profile"),
    ("prefill_exact", "frontend"),
    ("prefill_exact", "profile"),
    ("masked_softmax_all_masked", "frontend"),
    ("masked_softmax_all_masked", "relay"),
    ("masked_softmax_all_masked", "lowering"),
    ("masked_softmax_all_masked", "llvm"),
    ("masked_softmax_all_masked", "runtime"),
    ("masked_softmax_all_masked", "profile"),
    ("dynamic_batching", "frontend"),
    ("dynamic_batching", "relay"),
    ("dynamic_batching", "lowering"),
    ("dynamic_batching", "llvm"),
    ("dynamic_batching", "runtime"),
    ("dynamic_batching", "profile"),
    ("stable_softmax", "frontend"),
    ("stable_softmax", "relay"),
    ("stable_softmax", "lowering"),
    ("stable_softmax", "llvm"),
    ("stable_softmax", "runtime"),
    ("stable_softmax", "profile"),
    ("batched_matmul", "frontend"),
    ("batched_matmul", "relay"),
    ("batched_matmul", "lowering"),
    ("batched_matmul", "llvm"),
    ("batched_matmul", "cuda"),
    ("batched_matmul", "runtime"),
    ("batched_matmul", "profile"),
    ("embedding_gather", "frontend"),
    ("embedding_gather", "relay"),
    ("embedding_gather", "lowering"),
    ("embedding_gather", "llvm"),
    ("embedding_gather", "runtime"),
    ("embedding_gather", "profile"),
    ("mask_select", "frontend"),
    ("mask_select", "relay"),
    ("mask_select", "lowering"),
    ("mask_select", "llvm"),
    ("mask_select", "runtime"),
    ("mask_select", "profile"),
    ("mask_select", "cuda"),
    ("normalization", "frontend"),
    ("normalization", "relay"),
    ("normalization", "lowering"),
    ("normalization", "llvm"),
    ("normalization", "runtime"),
    ("normalization", "profile"),
    ("slice_concat", "frontend"),
    ("slice_concat", "relay"),
    ("slice_concat", "lowering"),
    ("slice_concat", "llvm"),
    ("slice_concat", "runtime"),
    ("slice_concat", "profile"),
    ("slice_concat", "cuda"),
    ("prefill_exact", "relay"),
    ("prefill_exact", "lowering"),
    ("prefill_exact", "llvm"),
    ("prefill_exact", "runtime"),
    ("decode_external_kv", "frontend"),
    ("decode_external_kv", "relay"),
    ("decode_external_kv", "lowering"),
    ("decode_external_kv", "llvm"),
    ("decode_external_kv", "runtime"),
    ("decode_external_kv", "profile"),
    ("kv_cache", "frontend"),
    ("kv_cache", "relay"),
    ("kv_cache", "lowering"),
    ("kv_cache", "llvm"),
    ("kv_cache", "runtime"),
    ("kv_cache", "profile"),
    ("vision_encoder", "frontend"),
    ("vision_encoder", "relay"),
    ("vision_encoder", "lowering"),
    ("vision_encoder", "llvm"),
    ("vision_encoder", "runtime"),
    ("vision_encoder", "profile"),
    ("vlm_joint", "frontend"),
    ("vlm_joint", "relay"),
    ("vlm_joint", "lowering"),
    ("vlm_joint", "llvm"),
    ("vlm_joint", "runtime"),
    ("vlm_joint", "profile"),
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
                    raise ValidationError("matrix.{}.{} must retain its recorded validation".format(capability, layer))
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
            if layer == "cuda" and record["status"] == "validated":
                raise ValidationError(
                    "matrix.{}.cuda cannot be promoted to validated by text-only evidence".format(
                        capability
                    )
                )
            if layer == "cuda" and capability not in CUDA_LOCAL_EVIDENCE_GATES:
                expected = "contracted" if capability in CUDA_PENDING_EVIDENCE_GATES else "unsupported"
                if record["status"] != expected:
                    raise ValidationError("matrix.{}.cuda lacks local hardware evidence".format(capability))
            if layer == "llvm" and record["status"] == "validated":
                raise ValidationError("matrix.{}.llvm must not claim unrun local validation".format(capability))
    for capability, gate in CUDA_LOCAL_EVIDENCE_GATES.items():
        cuda = matrix["capabilities"][capability]["cuda"]
        required = {"test/codegen_cuda_test.cpp", "src/tir/transforms/bind_cuda_threads.cc"}
        if capability in ("decode_external_kv", "kv_cache"):
            required |= {"src/runtime/executable_plan.cc", "src/runtime/session.cc",
                         "test/cuda_state_runtime_test.cpp", "test/minimind_decode_loop_llvm_test.cpp",
                         "test/cuda_profile_bundle_test.py", "python/tools/make_minimind_decode_loop_fixture.py",
                         "docs/implementation/GPU_KV_STATE_REPORT.md"}
        elif capability == "copy_event":
            required = {"src/runtime/device_stream.cc", "include/kxc/profiling/runtime_observer.h",
                        "test/cuda_copy_profiling_test.cpp", "test/cuda_profile_bundle_test.py",
                        "docs/implementation/M1_CUDA_COPY_REPORT.md"}
        elif capability == "prefill_exact":
            required |= {"test/minimind_prefill_cuda_test.cpp",
                         "test/cuda_profile_bundle_test.py",
                         "docs/implementation/M1_CUDA_CORRELATION_REPORT.md",
                         "docs/implementation/GPU_MINIMIND_PREFILL_REPORT.md"}
        elif capability == "embedding_gather":
            required.add("docs/implementation/GPU_GATHER_POW_REPORT.md")
        elif capability == "batched_matmul":
            required |= {"docs/implementation/GPU_OWNED_REDUCTION_REPORT.md",
                         "docs/implementation/GPU_BOUNDED_CORE_REPORT.md"}
        elif capability == "dynamic_batching":
            required |= {"src/runtime/executable_plan.cc", "src/runtime/session.cc",
                         "test/minimind_bounded_decode_llvm_test.cpp",
                         "test/cuda_profile_bundle_test.py",
                         "docs/implementation/GPU_REQUEST_BATCHING_REPORT.md"}
        elif capability in ("stable_softmax", "masked_softmax_all_masked", "normalization"):
            required.add("docs/implementation/GPU_MULTISTAGE_REDUCTION_REPORT.md")
        else:
            required.add("docs/implementation/GPU_OWNED_REDUCTION_REPORT.md")
        if capability in ("stable_softmax", "masked_softmax_all_masked", "normalization", "batched_matmul"):
            required |= {"docs/implementation/GPU_BOUNDED_REDUCTION_REPORT.md",
                         "test/bounded_cuda_test.cpp", "test/cuda_profile_bundle_test.py",
                         "test/cuda_schedule_test.cpp", "contracts/pass_contract.json",
                         "src/compiler/lowering/te_to_tir.cc", "src/runtime/session.cc"}
        if (cuda["status"] != "implemented" or cuda["gate"] != gate or
                not required.issubset(cuda["evidence"])):
            raise ValidationError(
                "{} CUDA record must remain implemented/local-evidence only".format(
                    capability
                )
            )
    for capability, gate in CUDA_PENDING_EVIDENCE_GATES.items():
        record = matrix["capabilities"][capability]["cuda"]
        required = {"src/runtime/executable_plan.cc", "src/runtime/session.cc",
                    "test/request_batching_llvm_test.cpp", "test/minimind_bounded_decode_llvm_test.cpp",
                    "test/cuda_profile_bundle_test.py", "docs/implementation/GPU_REQUEST_BATCHING_REPORT.md"}
        if record["gate"] != gate or not required.issubset(record["evidence"]):
            raise ValidationError("pending CUDA request contract lacks its implementation/validation entry points")
    for layer, gate in COPY_EVENT_GATES.items():
        record = matrix["capabilities"]["copy_event"][layer]
        required = {"test/runtime_profiling_test.cpp", "test/diagnosis_engine_test.py",
                    "docs/implementation/M1_COPY_EVENT_REPORT.md"}
        if record["gate"] != gate or not required.issubset(record["evidence"]):
            raise ValidationError("copy/event record lacks its execution/observation evidence at {}".format(layer))
    for layer, gate in PREFILL_EVIDENCE_GATES.items():
        record = matrix["capabilities"]["prefill_exact"][layer]
        required = {"test/adaptive_runtime_test.cpp", "test/minimind_bounded_prefill_llvm_test.cpp",
                    "docs/implementation/M6_RUNTIME_REPORT.md", "docs/implementation/M3_FULL_PREFILL_REPORT.md"}
        if record["gate"] != gate or not required.issubset(record["evidence"]):
            raise ValidationError("prefill record lacks its actual model evidence at {}".format(layer))
    for layer, gate in MASKED_SOFTMAX_GATES.items():
        record = matrix["capabilities"]["masked_softmax_all_masked"][layer]
        required = {
            "test/masked_softmax_llvm_test.cpp",
            "docs/implementation/M5_MASKED_SOFTMAX_REPORT.md",
        }
        if record["gate"] != gate or not required.issubset(record["evidence"]):
            raise ValidationError("masked softmax record lacks its CPU evidence at {}".format(layer))
    for layer, gate in DYNAMIC_BATCHING_GATES.items():
        dynamic = matrix["capabilities"]["dynamic_batching"][layer]
        required = {
            "test/request_batching_llvm_test.cpp",
            "test/minimind_bounded_decode_llvm_test.cpp",
            "docs/implementation/M2_REQUEST_BATCHING_REPORT.md",
        }
        if dynamic["gate"] != gate or not required.issubset(dynamic["evidence"]):
            raise ValidationError("request batching record lacks its CPU evidence at {}".format(layer))
    for layer, gate in KV_STATE_EVIDENCE_GATES.items():
        state = matrix["capabilities"]["kv_cache"][layer]
        required = {
            "test/minimind_decode_loop_llvm_test.cpp",
            "docs/implementation/M2_MINIMIND_STATE_REPORT.md",
            "test/minimind_bounded_decode_llvm_test.cpp",
            "docs/implementation/M2_BOUNDED_STATE_REPORT.md",
        }
        if state["gate"] != gate or not required.issubset(state["evidence"]):
            raise ValidationError("KV state record must retain its CPU capacity evidence at {}".format(layer))
    for capability in ("stable_softmax", "batched_matmul"):
        for layer, gate in BOUNDED_ATTENTION_EVIDENCE_GATES.items():
            record = matrix["capabilities"][capability][layer]
            required = {
                "test/bounded_attention_llvm_test.cpp",
                "docs/implementation/M3_BOUNDED_ATTENTION_REPORT.md",
            }
            if record["gate"] != gate or not required.issubset(record["evidence"]):
                raise ValidationError("bounded attention record lacks its CPU evidence at {}.{}".format(capability, layer))
    for layer, gate in NORMALIZATION_EVIDENCE_GATES.items():
        record = matrix["capabilities"]["normalization"][layer]
        required = {
            "test/bounded_projection_llvm_test.cpp",
            "docs/implementation/M3_WEIGHTED_PROJECTION_REPORT.md",
        }
        if record["gate"] != gate or not required.issubset(record["evidence"]):
            raise ValidationError("normalization record lacks its CPU evidence at {}".format(layer))
    for layer, gate in SLICE_CONCAT_EVIDENCE_GATES.items():
        record = matrix["capabilities"]["slice_concat"][layer]
        required = {"test/bounded_rope_llvm_test.cpp", "docs/implementation/M3_ROPE_REPORT.md"}
        if record["gate"] != gate or not required.issubset(record["evidence"]):
            raise ValidationError("Slice/Concat record lacks its CPU evidence at {}".format(layer))

    for layer, gate in EMBEDDING_EVIDENCE_GATES.items():
        record = matrix["capabilities"]["embedding_gather"][layer]
        required = {"test/minimind_bounded_prefill_llvm_test.cpp", "docs/implementation/M3_FULL_PREFILL_REPORT.md"}
        if record["gate"] != gate or not required.issubset(record["evidence"]):
            raise ValidationError("embedding record lacks its CPU evidence at {}".format(layer))


    for layer, gate in DECODE_EVIDENCE_GATES.items():
        record = matrix["capabilities"]["decode_external_kv"][layer]
        required = {"test/minimind_bounded_decode_llvm_test.cpp", "docs/implementation/M3_FULL_DECODE_REPORT.md"}
        if record["gate"] != gate or not required.issubset(record["evidence"]):
            raise ValidationError("decode record lacks its CPU evidence at {}".format(layer))

    for layer, gate in VISION_ENCODER_GATES.items():
        record = matrix["capabilities"]["vision_encoder"][layer]
        required = {"test/minimind_vision_llvm_test.cpp", "test/minimind_vlm_bounded_llvm_test.cpp",
                    "docs/implementation/M9_MINIMIND_V_VISION_REPORT.md",
                    "docs/implementation/M9_MINIMIND_V_BOUNDED_REPORT.md"}
        if record["gate"] != gate or not required.issubset(record["evidence"]):
            raise ValidationError("vision encoder record lacks its CPU evidence at {}".format(layer))
    for layer, gate in VLM_JOINT_GATES.items():
        record = matrix["capabilities"]["vlm_joint"][layer]
        required = {"test/minimind_decode_loop_llvm_test.cpp", "test/minimind_vlm_bounded_llvm_test.cpp",
                    "docs/implementation/M9_MINIMIND_V_JOINT_REPORT.md",
                    "docs/implementation/M9_MINIMIND_V_BOUNDED_REPORT.md"}
        if record["gate"] != gate or not required.issubset(record["evidence"]):
            raise ValidationError("VLM joint record lacks its CPU evidence at {}".format(layer))

    for layer, gate in CAUSAL_MASK_EVIDENCE_GATES.items():
        record = matrix["capabilities"]["mask_select"][layer]
        required = {"test/bounded_causal_attention_llvm_test.cpp", "docs/implementation/M3_CAUSAL_ATTENTION_REPORT.md"}
        if record["gate"] != gate or not required.issubset(record["evidence"]):
            raise ValidationError("causal mask record lacks its CPU evidence at {}".format(layer))


def validate_fixture(fixture):
    kind = fixture.get("kind")
    if kind not in FIXTURE_EXTRA_KEYS:
        raise ValidationError("unknown fixture kind: {}".format(kind))
    required = COMMON_FIXTURE_KEYS | FIXTURE_EXTRA_KEYS[kind]
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
        numeric_tree(fixture["expected"], "fixture expected")
        if len(fixture["expected"]) != len(fixture["logits"]):
            raise ValidationError("fixture {} expected extent mismatch".format(fixture["id"]))
        if not any(fixture["mask"]) and (fixture["expected_gate"] != "cpu_reference" or
                any(value != 0 for value in fixture["expected"])):
            raise ValidationError("explicit all-masked softmax reference must return zeros")
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
    if not logits or len(logits) != len(mask) or not all(isinstance(item, bool) for item in mask):
        raise ValidationError("masked softmax requires a positive matching extent and bool mask")
    active = [value for value, enabled in zip(logits, mask) if enabled]
    if not active:
        return [0.0] * len(logits)
    if not all(math.isfinite(value) for value in active):
        raise ValidationError("masked softmax reference requires finite active logits")
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
    assert_close(stable_softmax(masked["logits"], masked["mask"]), masked["expected"], masked["tolerance"], masked["id"])
    assert_close(stable_softmax([float("nan"), float("inf")], [False, False]), [0.0, 0.0], 0, "masked nonfinite payload")
    print("PASS reference masked_softmax_all_masked: exact zeros including masked NaN/Inf")
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
    require_rejection("mask extent mismatch", lambda: stable_softmax([1.0], []))
    require_rejection("mask dtype", lambda: stable_softmax([1.0], [1]))
    require_rejection("empty reduction", lambda: stable_softmax([], []))
    if require_valid_context(cache, cache["valid_extent"]) != cache["valid_extent"]:
        raise ValidationError("valid KV context was not preserved")
    require_rejection("capacity used as valid context", lambda: require_valid_context(cache, cache["physical_extent"]))
    probe = cache["capacity_probe"]
    require_rejection("capacity overflow", lambda: kv_trace({**cache, "append_tokens": cache["append_tokens"] + list(range(probe["append_count"]))}))
    require_rejection("fingerprint mismatch", lambda: check_fingerprint({**manifests["workloads"][0], "fingerprint": "0" * 64}, "negative manifest"))
    cuda = matrix["capabilities"]["dynamic_batching"]["cuda"]
    if cuda["status"] != "implemented" or cuda["gate"] != CUDA_LOCAL_EVIDENCE_GATES["dynamic_batching"]:
        raise ValidationError("CUDA request batching hardware evidence is missing")
    print("PASS negative gates unknown-symbolic/mask-contract/capacity/request-batching-CUDA/fingerprint")


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
    print("PASS NLP reference/capability gate; static and bounded model/request-batching CUDA evidence recorded")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as error:
        print("FAIL NLP reference and capability gate: {}".format(error))
        raise SystemExit(1)
