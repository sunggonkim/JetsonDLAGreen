#!/usr/bin/env python3
"""Verify Pantheon's pinned runtime on the current labelled ImageNette gate.

The adapter is deliberately separate from the CIFAR model-recovery verifier.
It binds the pinned online runtime, generated TorchScript modules, current
input/arrival traces, post-completion logits, labels, and the shared deadline
lock.  Pantheon's protobuf interface has integer-microsecond release/deadline
fields; that quantization is recorded explicitly instead of being hidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "1caa4321fe9f9902ffacb78978f11a32a7a62f64"
EXIT = re.compile(
    r"\[EXEC:EXIT\] HIGH_PRIORITY (\d+) (\d+) (\d+) (\d+) "
    r"(\d+) (\d+) ([0-9]+(?:\.[0-9]+)?)$"
)
JDGOUT_MAGIC = b"JDGOUT1\0"
JDGINT_MAGIC = b"JDGINT1\0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return value


def _file(path: Path, expected: str | None, label: str) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    actual = sha256(path)
    if expected is not None and actual != _sha(expected, f"{label} SHA"):
        raise ValueError(f"{label} SHA differs")
    return {"path": str(path), "sha256": actual}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{label} is not newline-complete")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _pinned_source(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError("Pantheon source path is missing")
    try:
        root = Path(subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()).resolve()
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        relative = path.relative_to(root)
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", str(relative)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        ).returncode == 0
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError("Pantheon source is not inside a Git checkout") from error
    if head != UPSTREAM_COMMIT or not tracked:
        raise ValueError("Pantheon source checkout is not the pinned tracked source")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "git_root": str(root),
        "git_head": head,
        "relative_path": str(relative),
    }


def _adapter(path: Path) -> dict[str, Any]:
    value = _load_json(path, "Pantheon adapter")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "pantheon-resnet50-imagenette-torchscript-adapter"
        or value.get("model") != "resnet50-imagenette"
        or value.get("numeric_comparison_allowed") is not False
    ):
        raise ValueError("Pantheon adapter contract differs")
    sources = value.get("source_models")
    if not isinstance(sources, dict) or set(sources) != {"backbone", "head"}:
        raise ValueError("Pantheon adapter source models differ")
    for label, record in sources.items():
        if not isinstance(record, dict):
            raise ValueError(f"Pantheon adapter source is missing: {label}")
        _file(Path(record["path"]), record.get("sha256"), f"Pantheon {label} ONNX")
    generated = value.get("generated_modules")
    if not isinstance(generated, dict) or set(generated) != {"block_00.pth", "branch_00.pth"}:
        raise ValueError("Pantheon adapter modules differ")
    for label, record in generated.items():
        if not isinstance(record, dict):
            raise ValueError(f"Pantheon adapter module is missing: {label}")
        _file(Path(record["path"]), record.get("sha256"), f"Pantheon {label}")
    config = Path(value.get("config_path", ""))
    _file(config, None, "Pantheon model config")
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "source_models": sources, "generated_modules": generated,
            "config_path": str(config.resolve())}


def _common(path: Path, operational_arrival: Path) -> dict[str, Any]:
    value = _load_json(path, "common workload")
    if (
        value.get("schema_version") != 1
        or value.get("workload_id") != "resnet50-classification"
        or value.get("topology") != "fixed-2g+1g"
        or value.get("placement") != "fixed-1g-producer-2g-consumer"
        or value.get("input_tensor") != "gpu_0/res4_5_branch2c_bn_2"
        or value.get("payload_bytes") != 802816
        or value.get("request_count") != 90
    ):
        raise ValueError("Pantheon common workload contract differs")
    for key in ("arrival_trace_path", "dataset_manifest_path"):
        evidence = Path(value[key]).resolve()
        expected = value["arrival_trace_sha256" if key == "arrival_trace_path" else "dataset_manifest_sha256"]
        if not evidence.is_file() or sha256(evidence) != _sha(expected, key):
            raise ValueError(f"Pantheon common evidence differs: {key}")
        value[key] = str(evidence)
    operational_arrival = operational_arrival.resolve()
    operational_sha = sha256(operational_arrival)
    if (
        value.get("operational_arrival_trace_path") != str(operational_arrival)
        or value.get("operational_arrival_trace_sha256") != operational_sha
    ):
        raise ValueError("Pantheon operational arrival trace differs")
    producer = Path(value["producer_input_trace_path"]).resolve()
    if not producer.is_file() or sha256(producer) != _sha(
        value["producer_input_trace_sha256"], "producer input trace"
    ):
        raise ValueError("Pantheon producer input trace differs")
    value["contract_path"] = str(path.resolve())
    value["contract_sha256"] = sha256(path)
    return value


def _input_trace(path: Path, expected_count: int, expected_offset: int) -> dict[str, Any]:
    raw = path.resolve().read_bytes()
    if len(raw) < 24 or raw[:8] != JDGINT_MAGIC:
        raise ValueError("Pantheon input trace header differs")
    schema, count = struct.unpack_from("<II", raw, 8)
    sample_bytes = struct.unpack_from("<Q", raw, 16)[0]
    record_bytes = 4 + 64 + sample_bytes
    if schema != 1 or count != expected_count or sample_bytes != 602112:
        raise ValueError("Pantheon input trace schema differs")
    if len(raw) != 24 + count * record_bytes:
        raise ValueError("Pantheon input trace length differs")
    for index in range(count):
        iteration = struct.unpack_from("<I", raw, 24 + index * record_bytes)[0]
        if iteration != expected_offset + index:
            raise ValueError("Pantheon input trace iteration offset differs")
    return {"path": str(path.resolve()), "sha256": sha256(path), "count": count,
            "sample_bytes": sample_bytes, "iteration_offset": expected_offset}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.resolve().read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"prediction row {line_number} is not an object")
        rows.append(value)
    return rows


def _output(path: Path, count: int) -> tuple[dict[str, Any], list[int]]:
    raw = path.resolve().read_bytes()
    if raw[:8] != JDGOUT_MAGIC or len(raw) < 20:
        raise ValueError("Pantheon output trace header differs")
    output_count = struct.unpack_from("<I", raw, 8)[0]
    output_bytes = struct.unpack_from("<Q", raw, 12)[0]
    if output_count != 1 or output_bytes != 40:
        raise ValueError("Pantheon output tensor contract differs")
    record_bytes = 4 + output_bytes
    if len(raw) != 20 + count * record_bytes:
        raise ValueError("Pantheon output trace record count differs")
    predictions: list[int] = []
    seen: set[int] = set()
    for index in range(count):
        offset = 20 + index * record_bytes
        iteration = struct.unpack_from("<I", raw, offset)[0]
        if iteration != index or iteration in seen:
            raise ValueError("Pantheon output iterations are not dense")
        seen.add(iteration)
        logits = struct.unpack_from("<10f", raw, offset + 4)
        if any(not math.isfinite(value) for value in logits):
            raise ValueError("Pantheon output contains a non-finite logit")
        predictions.append(max(range(10), key=logits.__getitem__))
    return ({"path": str(path.resolve()), "sha256": sha256(path),
             "capture_boundary": "post-completion", "record_count": count,
             "output_count": 1, "output_sizes": [40]}, predictions)


def _timings(path: Path, count: int, deadline_us: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_jobs: set[int] = set()
    for line in path.resolve().read_text(encoding="utf-8").splitlines():
        match = EXIT.search(line)
        if match is None:
            continue
        release, end, duration, service, job, exit_id, accuracy = match.groups()
        release_i, end_i, duration_i, service_i, job_i, exit_i = map(
            int, (release, end, duration, service, job, exit_id)
        )
        if (
            end_i - release_i != duration_i
            or job_i < 0
            or job_i in seen_jobs
            or exit_i != 0
        ):
            raise ValueError("Pantheon runtime exit trace is not unique or consistent")
        seen_jobs.add(job_i)
        rows.append({
            "arrival_sequence": job_i,
            "release_us_epoch": release_i,
            "completion_us_epoch": end_i,
            "wall_latency_us": float(duration_i),
            "service_us": float(service_i),
            "selected_exit": exit_i,
            "profile_accuracy": float(accuracy),
            "deadline_us": float(deadline_us),
            "deadline_miss": duration_i > deadline_us,
        })
    if len(rows) != count:
        raise ValueError("Pantheon runtime did not complete every request")
    return rows


def verify(
    *, common_path: Path, operational_arrival: Path, input_trace: Path,
    reference_predictions: Path, reference_output: Path, reference_pipeline: Path,
    output_trace: Path, runtime_log: Path, adapter: Path, runtime_binary: Path,
    sources: list[Path], workload_json: Path, deadline_lock: Path, background_json: Path,
    background_engine: Path, class_map: Path, input_iteration_offset: int = 10,
    minimum_accuracy: float = 0.8, accuracy_tolerance: float = 0.0,
) -> dict[str, Any]:
    if minimum_accuracy < 0.8 or minimum_accuracy > 1.0 or accuracy_tolerance < 0.0:
        raise ValueError("Pantheon accuracy thresholds are invalid")
    common = _common(common_path, operational_arrival)
    lock = _load_json(deadline_lock, "deadline lock")
    deadline = float(lock.get("deadline_us"))
    if not math.isfinite(deadline) or deadline <= 0.0:
        raise ValueError("Pantheon deadline lock is invalid")
    workload = _load_json(workload_json, "Pantheon workload")
    effective_deadline = workload.get("deadline_us")
    if (
        not isinstance(effective_deadline, int)
        or effective_deadline <= 0
        or effective_deadline != math.floor(deadline)
        or workload.get("arrival_trace_sha256") != sha256(operational_arrival)
        or workload.get("request_count") != common["request_count"]
    ):
        raise ValueError("Pantheon protobuf workload is not bound to the current lock")
    quantization = deadline - effective_deadline
    if quantization < 0.0 or quantization >= 1.0:
        raise ValueError("Pantheon deadline quantization exceeds one microsecond")
    input_info = _input_trace(input_trace, common["request_count"] + input_iteration_offset, 0)
    reference_rows = _jsonl(reference_predictions)
    if len(reference_rows) != common["request_count"]:
        raise ValueError("Pantheon reference prediction count differs")
    class_values = _load_json(class_map, "Pantheon class map")
    labels: list[str] = []
    input_hashes: list[str] = []
    request_ids: list[str] = []
    for index, row in enumerate(reference_rows):
        if row.get("arrival_sequence") != index or row.get("schema_version") != 1:
            raise ValueError("Pantheon reference prediction sequence differs")
        expected = row.get("expected_label")
        input_hash = row.get("input_sha256")
        request_id = row.get("request_id")
        if not isinstance(expected, str) or not isinstance(request_id, str):
            raise ValueError("Pantheon reference labels are invalid")
        _sha(input_hash, "Pantheon input hash")
        labels.append(expected)
        input_hashes.append(input_hash)
        request_ids.append(request_id)
    requests = _jsonl(Path(common["arrival_trace_path"]))
    if [row.get("input_sha256") for row in requests] != input_hashes:
        raise ValueError("Pantheon labels are not bound to the common input order")
    output_info, prediction_indices = _output(output_trace, common["request_count"])
    predictions = []
    for index in prediction_indices:
        value = class_values.get(str(index))
        if not isinstance(value, str):
            raise ValueError("Pantheon class map is incomplete")
        predictions.append(value)
    reference_correct = [row.get("prediction") == row.get("expected_label") for row in reference_rows]
    candidate_correct = [prediction == label for prediction, label in zip(predictions, labels)]
    reference_accuracy = sum(reference_correct) / len(reference_correct)
    candidate_accuracy = sum(candidate_correct) / len(candidate_correct)
    accuracy_delta = abs(candidate_accuracy - reference_accuracy)
    if (
        reference_accuracy < minimum_accuracy
        or candidate_accuracy < minimum_accuracy
        or accuracy_delta > accuracy_tolerance
    ):
        raise ValueError("Pantheon labelled application gate failed")
    timing = _timings(runtime_log, common["request_count"], effective_deadline)
    for row, label, prediction, input_hash, request_id in zip(
        timing, labels, predictions, input_hashes, request_ids
    ):
        row.update({
            "request_id": request_id, "input_sha256": input_hash,
            "expected_label": label, "prediction": prediction,
            "correct": prediction == label,
        })
    background = _load_json(background_json, "Pantheon background result")
    if (
        background.get("model") != "distilbert-sst2"
        or background.get("config", {}).get("period_ms") != 4
        or background.get("gpu", {}).get("multiprocessors") != 8
        or background.get("completed_requests", 0) <= 0
        or background.get("throughput_per_second", 0.0) <= 0.0
    ):
        raise ValueError("Pantheon background workload differs")
    return {
        "schema_version": 1,
        "kind": "pantheon-resnet50-imagenette-common-workload-fidelity-gate",
        "system": "Pantheon (Thor port)",
        "status": "passed",
        "numeric_comparison_allowed": True,
        "upstream_commit": UPSTREAM_COMMIT,
        "workload": "resnet50-classification",
        "common_workload": common,
        "deadline_us": deadline,
        "effective_pantheon_deadline_us": effective_deadline,
        "deadline_quantization": {
            "interface": "Pantheon protobuf integer microseconds",
            "method": "floor",
            "difference_us": quantization,
        },
        "requests": len(timing),
        "warmup_requests": input_iteration_offset,
        "reference_accuracy": reference_accuracy,
        "pantheon_accuracy": candidate_accuracy,
        "accuracy_delta": accuracy_delta,
        "accuracy_tolerance": accuracy_tolerance,
        "minimum_accuracy": minimum_accuracy,
        "decision_cases": len(timing),
        "deadline_misses": sum(row["deadline_miss"] for row in timing),
        "dmr": sum(row["deadline_miss"] for row in timing) / len(timing),
        "p99_us": sorted(row["wall_latency_us"] for row in timing)[max(0, math.ceil(0.99 * len(timing)) - 1)],
        "background_goodput_rps": background["throughput_per_second"],
        "production_wall_definition": "Pantheon actual-release-to-exit-completion",
        "correctness_validation_placement": "post-completion",
        "reference_predictions": _file(reference_predictions, None, "reference predictions"),
        "reference_output_trace": _file(reference_output, None, "reference output trace"),
        "reference_pipeline": _file(reference_pipeline, None, "reference pipeline trace"),
        "pantheon_output_trace": output_info,
        "runtime_log": _file(runtime_log, None, "Pantheon runtime log"),
        "background": _file(background_json, None, "Pantheon background result"),
        "background_engine": _file(background_engine, None, "Pantheon background engine"),
        "runtime_binary": _file(runtime_binary, None, "Pantheon runtime binary"),
        "upstream_sources": [_pinned_source(source) for source in sources],
        "adapter": _adapter(adapter),
        "input_trace": input_info,
        "class_map": _file(class_map, None, "Pantheon class map"),
        "workload_protobuf": _file(workload_json.with_suffix(".pb"), None, "Pantheon workload protobuf"),
        "timing_trace": timing,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "common", "operational-arrival", "input-trace", "reference-predictions",
        "reference-output", "reference-pipeline", "output-trace", "runtime-log",
        "adapter", "runtime-binary", "source", "workload-json", "deadline-lock",
        "background-json", "background-engine", "class-map",
    ):
        if name == "source":
            parser.add_argument(f"--{name}", type=Path, action="append", required=True)
        else:
            parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--input-iteration-offset", type=int, default=10)
    parser.add_argument("--minimum-accuracy", type=float, default=0.8)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = verify(
        common_path=args.common.resolve(), operational_arrival=args.operational_arrival.resolve(),
        input_trace=args.input_trace.resolve(), reference_predictions=args.reference_predictions.resolve(),
        reference_output=args.reference_output.resolve(), reference_pipeline=args.reference_pipeline.resolve(),
        output_trace=args.output_trace.resolve(), runtime_log=args.runtime_log.resolve(),
        adapter=args.adapter.resolve(), runtime_binary=args.runtime_binary.resolve(),
        sources=[path.resolve() for path in args.source], workload_json=args.workload_json.resolve(),
        deadline_lock=args.deadline_lock.resolve(), background_json=args.background_json.resolve(),
        background_engine=args.background_engine.resolve(), class_map=args.class_map.resolve(),
        input_iteration_offset=args.input_iteration_offset,
        minimum_accuracy=args.minimum_accuracy, accuracy_tolerance=args.accuracy_tolerance,
    )
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
