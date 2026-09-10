"""Compile every emitted model kernel to PTX; this gate never runs a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("prefill", "decode", "requests-axis1", "requests-axis2"), default="prefill")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="bounded-" + args.model + "-", dir=root))
    print("[INFO] CUDA source/PTX evidence: " + str(out), flush=True)
    expected_calls = 3 if args.model.startswith("requests-") else 774 if args.model == "decode" else 742
    arguments = [args.executable, str(out)] + ([args.model] if args.model != "prefill" else [])
    result = subprocess.run(arguments, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    (out / "lowering.log").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    if result.returncode == 77:
        return 77
    if result.returncode or f"[PASS] full_minimind_bounded_cuda_codegen: {expected_calls}/{expected_calls} primitives" not in result.stdout:
        raise AssertionError("full model CUDA lowering did not complete")
    source, ptx = out / "full-minimind.cu", out / "full-minimind.ptx"
    command = [args.nvcc, "--ptx", "--std=c++17", "--gpu-architecture=sm_89", str(source), "-o", str(ptx)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out / "nvcc.log").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise AssertionError("NVCC rejected emitted full model CUDA source")
    symbols = re.findall(r'extern "C" __global__ void (kxc_unit_\d+_\w+)\(', source.read_text(encoding="utf-8"))
    entries = re.findall(r"\.entry\s+(kxc_unit_\d+_\w+)\s*\(", ptx.read_text(encoding="utf-8"))
    if len(symbols) != expected_calls or len(set(symbols)) != expected_calls or sorted(symbols) != sorted(entries):
        raise AssertionError("PTX did not preserve all distinct model entry points")
    version = subprocess.check_output([args.nvcc, "--version"], text=True)
    audit = {"scope": ("shared request-test graph" if args.model.startswith("requests-") else "actual model") +
                     " lowering, CUDA proof, source and PTX compilation",
             "model": args.model, "target": "synthetic sm_89", "gpu_execution": False, "entry_points": expected_calls,
             "nvcc_version": version, "command": command,
             "files": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in (source, ptx, out / "lowering.log", out / "nvcc.log")}}
    (out / "codegen-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] full_minimind_bounded_cuda_ptx_verified: {expected_calls} entries; {args.model}; no GPU execution", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
