#!/usr/bin/env python3
"""Verify a Pantheon common-workload accuracy/arrival contract.

The verifier compares a reference TensorRT DAG trace with a Pantheon trace.
It intentionally does not require byte-identical model outputs: the gate
requires identical inputs/labels and an explicitly bounded accuracy delta,
while preserving Pantheon's block/exit decisions for later latency analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_COMMIT = "1caa4321fe9f9902ffacb78978f11a32a7a62f64"
WORKLOAD = "p9-dependent-tensorrt-dag"
TRACE_KEYS = {
    "schema_version", "request_id", "arrival_sequence", "input_sha256",
    "expected_label", "prediction", "correct", "selected_exit",
    "block_sequence", "wall_latency_us", "deadline_us", "deadline_miss",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_pinned_checkout(source_path: Path) -> dict[str, str]:
    source_path = source_path.resolve()
    try:
        root = Path(subprocess.check_output(
            ["git", "-C", str(source_path.parent), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()).resolve()
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        relative = source_path.relative_to(root)
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", str(relative)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        ).returncode == 0
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError("upstream source is not inside a Git checkout") from error
    if head != UPSTREAM_COMMIT:
        raise ValueError("upstream source checkout HEAD is not the pinned Pantheon commit")
    if not tracked:
        raise ValueError("upstream source is not a tracked file in the pinned checkout")
    return {
        "upstream_git_root": str(root),
        "upstream_git_head": head,
        "upstream_source_relative_path": str(relative),
    }


def _hex(value: Any, label: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return value


def _sha(value: Any, label: str) -> str:
    return _hex(value, label, 64)


def _load_common_workload_contract(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    raw = resolved.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("common workload contract is not newline-complete")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("common workload contract is invalid JSON") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("common workload contract schema differs")
    required = (
        "workload_id", "topology", "placement", "input_tensor", "payload_bytes",
        "arrival_trace_path", "arrival_trace_sha256",
        "dataset_manifest_path", "dataset_manifest_sha256",
    )
    if any(key not in value for key in required):
        raise ValueError("common workload contract is incomplete")
    for path_key, digest_key in (
        ("arrival_trace_path", "arrival_trace_sha256"),
        ("dataset_manifest_path", "dataset_manifest_sha256"),
    ):
        evidence = Path(value[path_key]).resolve()
        digest = _sha(value[digest_key], digest_key)
        if not evidence.is_file() or sha256(evidence) != digest:
            raise ValueError(f"common workload evidence SHA mismatches: {path_key}")
        value[path_key] = str(evidence)
    value["contract_path"] = str(resolved)
    value["contract_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{label} has an invalid value")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return value


def _read_training_result(path: Path, expected_commit: str) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    raw = resolved.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} training result is not newline-complete")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} training result is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} training result must be an object")
    if (
        value.get("kind") != "pantheon-cifar10-resnet50-training"
        or value.get("system") != "Pantheon"
        or value.get("upstream_commit") != expected_commit
        or value.get("formal_training_contract") is not True
        or value.get("accuracy_gate_passed") is not True
        or value.get("full_output_max_abs_error") != 0.0
    ):
        raise ValueError("Pantheon training artifact has not passed the formal accuracy gate")
    source_hashes = value.get("source_sha256")
    dataset_hashes = value.get("dataset_sha256")
    artifacts = value.get("artifacts")
    if (
        not isinstance(source_hashes, dict) or not source_hashes
        or not isinstance(dataset_hashes, dict) or not dataset_hashes
        or not isinstance(artifacts, dict) or not artifacts
    ):
        raise ValueError("Pantheon training artifact lacks bound source/data/artifact hashes")
    for name, digest in source_hashes.items():
        _sha(digest, f"training source {name}")
    for name, digest in dataset_hashes.items():
        _sha(digest, f"training dataset {name}")
    return value, hashlib.sha256(raw).hexdigest()


def _read_trace(path: Path) -> list[dict[str, Any]]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    rows: list[dict[str, Any]] = []
    arrivals: set[int] = set()
    request_ids: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        if not isinstance(value, dict) or set(value) != TRACE_KEYS:
            raise ValueError(f"{path}:{line_number} canonical schema differs")
        if value.get("schema_version") != 1:
            raise ValueError(f"{path}:{line_number} schema version differs")
        request_id = value.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"{path}:{line_number} request_id is invalid")
        if request_id in request_ids:
            raise ValueError(f"{path}:{line_number} duplicate request_id")
        request_ids.add(request_id)
        arrival = _integer(value.get("arrival_sequence"), "arrival_sequence")
        if arrival in arrivals:
            raise ValueError(f"{path}:{line_number} duplicate arrival_sequence")
        arrivals.add(arrival)
        _sha(value.get("input_sha256"), "input_sha256")
        _integer(value.get("expected_label"), "expected_label")
        _integer(value.get("prediction"), "prediction")
        if not isinstance(value.get("correct"), bool):
            raise ValueError(f"{path}:{line_number} correct must be boolean")
        if value["correct"] != (value["prediction"] == value["expected_label"]):
            raise ValueError(f"{path}:{line_number} correct does not match prediction")
        _integer(value.get("selected_exit"), "selected_exit")
        sequence = value.get("block_sequence")
        if (
            not isinstance(sequence, list)
            or not sequence
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                   for item in sequence)
        ):
            raise ValueError(f"{path}:{line_number} block_sequence is invalid")
        if value["selected_exit"] != sequence[-1]:
            raise ValueError(f"{path}:{line_number} exit does not match block sequence")
        _finite(value.get("wall_latency_us"), "wall_latency_us")
        _finite(value.get("deadline_us"), "deadline_us")
        if not isinstance(value.get("deadline_miss"), bool):
            raise ValueError(f"{path}:{line_number} deadline_miss must be boolean")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    if sorted(arrivals) != list(range(len(rows))):
        raise ValueError(f"{path} arrival sequence is not dense and zero-based")
    return sorted(rows, key=lambda row: row["arrival_sequence"])


def verify(
    reference_path: Path,
    pantheon_path: Path,
    *,
    upstream_commit: str = UPSTREAM_COMMIT,
    upstream_source_path: Path,
    training_result_path: Path,
    upstream_source_sha256: str | None = None,
    runtime_binary_path: Path | None = None,
    runtime_binary_sha256: str | None = None,
    expected_cases: int | None = None,
    accuracy_tolerance: float = 0.01,
    deadline_us: float,
    require_pinned_checkout: bool = True,
    common_workload_contract: Path | None = None,
    require_common_workload: bool = False,
) -> dict[str, Any]:
    _hex(upstream_commit, "upstream_commit", 40)
    if upstream_commit != UPSTREAM_COMMIT:
        raise ValueError("upstream commit is not the pinned Pantheon artifact")
    if require_common_workload and common_workload_contract is None:
        raise ValueError("Pantheon accuracy gate requires a common workload contract")
    common_workload = (
        _load_common_workload_contract(common_workload_contract)
        if common_workload_contract is not None else None
    )
    source_path = upstream_source_path.resolve()
    if not source_path.is_file():
        raise ValueError("upstream_source_path must name a regular file")
    checkout = (
        _verify_pinned_checkout(source_path)
        if require_pinned_checkout
        else {
            "upstream_git_root": None,
            "upstream_git_head": None,
            "upstream_source_relative_path": None,
        }
    )
    computed_source_sha256 = sha256(source_path)
    if upstream_source_sha256 is not None:
        _sha(upstream_source_sha256, "upstream_source_sha256")
        if upstream_source_sha256 != computed_source_sha256:
            raise ValueError("Pantheon source SHA256 does not match source bytes")
    runtime_binary = None
    computed_runtime_binary_sha256 = None
    if runtime_binary_path is not None:
        runtime_binary = runtime_binary_path.resolve()
        if not runtime_binary.is_file():
            raise ValueError("runtime_binary_path must name a regular file")
        computed_runtime_binary_sha256 = sha256(runtime_binary)
        if runtime_binary_sha256 is not None:
            _sha(runtime_binary_sha256, "runtime_binary_sha256")
            if runtime_binary_sha256 != computed_runtime_binary_sha256:
                raise ValueError("Pantheon runtime binary SHA256 does not match binary bytes")
    if require_pinned_checkout and common_workload is not None and runtime_binary is None:
        raise ValueError(
            "Pantheon numeric fidelity gate requires the pinned runtime binary"
        )
    training_result, training_result_sha256 = _read_training_result(
        training_result_path, UPSTREAM_COMMIT
    )
    if not math.isfinite(accuracy_tolerance) or accuracy_tolerance < 0.0:
        raise ValueError("accuracy_tolerance must be finite and nonnegative")
    if not math.isfinite(deadline_us) or deadline_us <= 0.0:
        raise ValueError("deadline_us must be finite and positive")
    reference = _read_trace(reference_path)
    pantheon = _read_trace(pantheon_path)
    if expected_cases is not None and len(reference) != expected_cases:
        raise ValueError("reference trace case count differs")
    if len(reference) != len(pantheon):
        raise ValueError("reference and Pantheon trace lengths differ")
    for index, (left, right) in enumerate(zip(reference, pantheon, strict=True)):
        for key in ("request_id", "arrival_sequence", "input_sha256", "expected_label"):
            if left[key] != right[key]:
                raise ValueError(f"shared workload mismatch at case {index}: {key}")
        for trace_name, trace_row in (("reference", left), ("Pantheon", right)):
            if not math.isclose(float(trace_row["deadline_us"]), deadline_us,
                                rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{trace_name} deadline differs from frozen contract")
            expected_miss = float(trace_row["wall_latency_us"]) > deadline_us
            if trace_row["deadline_miss"] != expected_miss:
                raise ValueError(f"{trace_name} deadline classification is inconsistent")
    reference_accuracy = sum(row["correct"] for row in reference) / len(reference)
    pantheon_accuracy = sum(row["correct"] for row in pantheon) / len(pantheon)
    accuracy_delta = abs(reference_accuracy - pantheon_accuracy)
    if accuracy_delta > accuracy_tolerance:
        raise ValueError("Pantheon accuracy differs beyond the declared tolerance")
    return {
        "schema_version": 1,
        "kind": "pantheon-common-workload-accuracy-gate",
        "system": "Pantheon",
        "status": "passed",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_path": str(source_path),
        "upstream_source_sha256": computed_source_sha256,
        "upstream_source_verified": True,
        "runtime_binary_path": str(runtime_binary) if runtime_binary else None,
        "runtime_binary_sha256": computed_runtime_binary_sha256,
        "runtime_binary_verified": runtime_binary is not None,
        "upstream_checkout_verified": require_pinned_checkout,
        **checkout,
        "training_result_path": str(training_result_path.resolve()),
        "training_result_sha256": training_result_sha256,
        "training_artifact_verified": True,
        "training_accuracy_gate": {
            "formal_training_contract": training_result["formal_training_contract"],
            "accuracy_gate_passed": training_result["accuracy_gate_passed"],
            "full_output_max_abs_error": training_result["full_output_max_abs_error"],
        },
        "workload": WORKLOAD,
        "deadline_us": deadline_us,
        "decision_cases": len(reference),
        "reference_trace_path": str(reference_path.resolve()),
        "reference_trace_sha256": sha256(reference_path.resolve()),
        "port_trace_path": str(pantheon_path.resolve()),
        "port_trace_sha256": sha256(pantheon_path.resolve()),
        "shared_arrival_trace": True,
        "common_workload": common_workload,
        "accuracy_equivalent": True,
        "reference_accuracy": reference_accuracy,
        "pantheon_accuracy": pantheon_accuracy,
        "accuracy_delta": accuracy_delta,
        "accuracy_tolerance": accuracy_tolerance,
        "numeric_comparison_allowed": (
            require_pinned_checkout
            and common_workload is not None
            and runtime_binary is not None
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--pantheon-trace", type=Path, required=True)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--upstream-source-sha256")
    parser.add_argument("--runtime-binary", type=Path)
    parser.add_argument("--runtime-binary-sha256")
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.01)
    parser.add_argument("--common-workload-contract", type=Path)
    parser.add_argument("--require-common-workload", action="store_true")
    parser.add_argument(
        "--allow-unpinned-source", action="store_true",
        help="emit non-promoting local fixture evidence; numeric comparison remains disabled",
    )
    parser.add_argument("--deadline-us", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(
        args.reference_trace,
        args.pantheon_trace,
        upstream_source_path=args.upstream_source,
        training_result_path=args.training_result,
        upstream_source_sha256=args.upstream_source_sha256,
        runtime_binary_path=args.runtime_binary,
        runtime_binary_sha256=args.runtime_binary_sha256,
        expected_cases=args.expected_cases,
        accuracy_tolerance=args.accuracy_tolerance,
        deadline_us=args.deadline_us,
        require_pinned_checkout=not args.allow_unpinned_source,
        common_workload_contract=args.common_workload_contract,
        require_common_workload=args.require_common_workload,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
