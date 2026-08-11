#!/usr/bin/env python3
"""Verify an end-to-end application accuracy contract for a dependent DAG.

Timing checksums prove that a tensor crossed the edge unchanged.  They do not
prove that the downstream stage performs the intended application task.  This
gate binds a reference and candidate request trace to the same dataset/input
trace, engine bytes, output bytes, and deadline classification before a formal
numeric comparison may use the candidate.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


TRACE_KEYS = {
    "schema_version",
    "request_id",
    "arrival_sequence",
    "input_sha256",
    "expected_label",
    "prediction",
    "correct",
    "output_sha256",
    "deadline_us",
    "wall_latency_us",
    "deadline_miss",
}

DATASET_KEYS = {"schema_version", "sample_id", "input_sha256", "expected_label"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _hex(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase hexadecimal SHA-256")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _detection_request_correct(
    prediction: str, expected_label: str, *, iou_threshold: float,
) -> bool:
    try:
        predicted = json.loads(prediction)
        expected = json.loads(expected_label)
    except json.JSONDecodeError as error:
        raise ValueError("detection labels must be canonical JSON") from error
    if not isinstance(predicted, dict) or not isinstance(expected, dict):
        raise ValueError("detection labels must be JSON objects")
    predicted_boxes = predicted.get("detections")
    expected_boxes = expected.get("detections")
    if not isinstance(predicted_boxes, list) or not isinstance(expected_boxes, list):
        raise ValueError("detection labels must contain detections arrays")
    if len(predicted_boxes) != len(expected_boxes):
        return False
    if not predicted_boxes:
        return True

    def box(value: Any) -> tuple[str, float, float, float, float] | None:
        if not isinstance(value, dict) or not isinstance(value.get("class"), str):
            return None
        coords = value.get("box")
        if not isinstance(coords, list) or len(coords) != 4:
            return None
        try:
            x, y, width, height = (float(item) for item in coords)
        except (TypeError, ValueError):
            return None
        if any(not math.isfinite(item) for item in (x, y, width, height)) or width <= 0 or height <= 0:
            return None
        return value["class"], x, y, x + width, y + height

    parsed_predicted = [box(item) for item in predicted_boxes]
    parsed_expected = [box(item) for item in expected_boxes]
    if any(item is None for item in parsed_predicted + parsed_expected):
        raise ValueError("detection boxes are invalid")
    unmatched = set(range(len(parsed_expected)))
    for candidate in parsed_predicted:
        assert candidate is not None
        best: tuple[float, int] | None = None
        for index in unmatched:
            reference = parsed_expected[index]
            assert reference is not None
            if candidate[0] != reference[0]:
                continue
            left = max(candidate[1], reference[1])
            top = max(candidate[2], reference[2])
            right = min(candidate[3], reference[3])
            bottom = min(candidate[4], reference[4])
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            candidate_area = (candidate[3] - candidate[1]) * (candidate[4] - candidate[2])
            reference_area = (reference[3] - reference[1]) * (reference[4] - reference[2])
            iou = intersection / (candidate_area + reference_area - intersection)
            if iou >= iou_threshold and (best is None or iou > best[0]):
                best = (iou, index)
        if best is None:
            return False
        unmatched.remove(best[1])
    return not unmatched


def _transcript_words(value: str) -> list[str]:
    """Normalize an externally decoded transcript for word-error scoring."""
    return re.findall(r"\w+", value.casefold(), flags=re.UNICODE)


def _word_error_rate(prediction: str, expected_label: str) -> float:
    """Return normalized Levenshtein word error rate for an ASR request."""
    reference = _transcript_words(expected_label)
    hypothesis = _transcript_words(prediction)
    if not reference:
        raise ValueError("ASR reference transcript has no words")
    previous = list(range(len(reference) + 1))
    for hypothesis_index, hypothesis_word in enumerate(hypothesis, 1):
        current = [hypothesis_index]
        for reference_index, reference_word in enumerate(reference, 1):
            current.append(min(
                current[-1] + 1,
                previous[reference_index] + 1,
                previous[reference_index - 1] + (hypothesis_word != reference_word),
            ))
        previous = current
    return previous[-1] / len(reference)


def _asr_request_correct(
    prediction: str, expected_label: str, *, max_wer: float,
) -> bool:
    return _word_error_rate(prediction, expected_label) <= max_wer


def _read_trace(
    path: Path, deadline_us: float, *, structured_predictions: bool = False,
    detection_iou_threshold: float = 0.5, task: str = "classification",
    asr_max_wer: float = 0.20,
) -> list[dict[str, Any]]:
    resolved = path.resolve()
    raw = resolved.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    rows: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    arrivals: set[int] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is invalid JSON") from error
        if not isinstance(value, dict) or set(value) != TRACE_KEYS:
            raise ValueError(f"{path}:{line_number} trace schema differs")
        if value.get("schema_version") != 1:
            raise ValueError(f"{path}:{line_number} trace schema version differs")
        request_id = value.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in request_ids:
            raise ValueError(f"{path}:{line_number} request_id is invalid or duplicated")
        request_ids.add(request_id)
        arrival = value.get("arrival_sequence")
        if isinstance(arrival, bool) or not isinstance(arrival, int) or arrival < 0 or arrival in arrivals:
            raise ValueError(f"{path}:{line_number} arrival_sequence is invalid")
        arrivals.add(arrival)
        _hex(value.get("input_sha256"), f"{path}:{line_number} input_sha256")
        for field in ("expected_label", "prediction"):
            if not isinstance(value.get(field), str) or not value[field]:
                raise ValueError(f"{path}:{line_number} {field} is invalid")
        if not isinstance(value.get("correct"), bool):
            raise ValueError(f"{path}:{line_number} correct must be boolean")
        if structured_predictions:
            expected_correct = _detection_request_correct(
                value["prediction"], value["expected_label"],
                iou_threshold=detection_iou_threshold,
            )
        elif task == "asr":
            expected_correct = _asr_request_correct(
                value["prediction"], value["expected_label"], max_wer=asr_max_wer,
            )
        else:
            expected_correct = value["prediction"] == value["expected_label"]
        if value["correct"] != expected_correct:
            raise ValueError(f"{path}:{line_number} correct disagrees with labels")
        _hex(value.get("output_sha256"), f"{path}:{line_number} output_sha256")
        observed_deadline = _finite(value.get("deadline_us"), f"{path}:{line_number} deadline_us", positive=True)
        if not math.isclose(observed_deadline, deadline_us, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{path}:{line_number} deadline differs from frozen contract")
        latency = _finite(value.get("wall_latency_us"), f"{path}:{line_number} wall_latency_us")
        if not isinstance(value.get("deadline_miss"), bool):
            raise ValueError(f"{path}:{line_number} deadline_miss must be boolean")
        if value["deadline_miss"] != (latency > deadline_us):
            raise ValueError(f"{path}:{line_number} deadline classification is inconsistent")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    if sorted(arrivals) != list(range(len(rows))):
        raise ValueError(f"{path} arrival sequence is not dense and zero-based")
    return sorted(rows, key=lambda row: row["arrival_sequence"])


def _read_dataset_manifest(path: Path) -> dict[str, str]:
    """Read the immutable sample-to-label manifest used by both traces.

    A dataset path/hash alone is not an accuracy oracle: a forged trace could
    otherwise invent labels while still agreeing with its reference trace.
    The manifest binds every input digest consumed by the latency run to its
    expected task label.
    """
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("dataset manifest is not newline-complete")
    samples: dict[str, str] = {}
    sample_ids: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"dataset manifest:{line_number} is invalid JSON") from error
        if not isinstance(value, dict) or set(value) != DATASET_KEYS:
            raise ValueError(f"dataset manifest:{line_number} schema differs")
        if value.get("schema_version") != 1:
            raise ValueError(f"dataset manifest:{line_number} schema version differs")
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError(f"dataset manifest:{line_number} sample_id is invalid or duplicated")
        sample_ids.add(sample_id)
        digest = _hex(value.get("input_sha256"), f"dataset manifest:{line_number} input_sha256")
        label = value.get("expected_label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"dataset manifest:{line_number} expected_label is invalid")
        if digest in samples and samples[digest] != label:
            raise ValueError(f"dataset manifest:{line_number} input has conflicting labels")
        samples[digest] = label
    if not samples:
        raise ValueError("dataset manifest is empty")
    return samples


def _read_output_trace(
    path: Path, expected_requests: int, label: str, *, warmup: int, output_index: int,
) -> dict[str, Any]:
    """Validate the raw post-completion output container at the gate boundary.

    Prediction JSONL is derived metadata.  The binary trace is the producer's
    independent completion evidence, so a formal gate records both its digest
    and its request cardinality instead of trusting a copied prediction file.
    """
    module_path = Path(__file__).with_name("read_application_output_trace.py")
    spec = importlib.util.spec_from_file_location("p9_output_trace_reader", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("application output trace parser is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        parsed = module.parse(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} output trace is invalid: {error}") from error
    records = parsed.get("records")
    if (
        not isinstance(records, list)
        or parsed.get("record_count") != warmup + expected_requests
        or output_index < 0
        or output_index >= parsed.get("output_count", 0)
    ):
        raise ValueError(f"{label} output trace record count differs")
    iterations = [record.get("iteration") for record in records]
    if iterations != list(range(warmup + expected_requests)):
        raise ValueError(f"{label} output trace iterations are not dense and ordered")
    measured = records[warmup:]
    return {
        "path": str(path.resolve()),
        "sha256": parsed["sha256"],
        "record_count": parsed["record_count"],
        "output_count": parsed["output_count"],
        "output_sizes": parsed["output_sizes"],
        "measured_output_sha256": [
            record["outputs"][output_index]["sha256"] for record in measured
        ],
        "capture_boundary": "post-completion",
    }


def _accuracy_breakdown(rows: list[dict[str, Any]]) -> str:
    counts: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        bucket = counts[row["expected_label"]]
        bucket[0] += 1
        bucket[1] += int(row["correct"])
    return ", ".join(
        f"{label}={correct}/{total}"
        for label, (total, correct) in sorted(counts.items())
    )


def _accuracy_breakdown_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        bucket = counts[row["expected_label"]]
        bucket[0] += 1
        bucket[1] += int(row["correct"])
    return {
        label: {"requests": total, "correct": correct}
        for label, (total, correct) in sorted(counts.items())
    }


def _read_pipeline_binding(
    path: Path,
    trace: list[dict[str, Any]],
    deadline_us: float,
    *,
    warmup: int,
    label: str,
) -> dict[str, Any]:
    """Bind a production wall CSV to the already validated prediction rows.

    The CSV is the timing source used by the benchmark; prediction JSONL is
    derived metadata.  Requiring their measured request/hash sequence to
    agree prevents a forged prediction trace from silently changing the
    workload represented by a numeric comparison.
    """
    if not path.resolve().is_file():
        raise ValueError(f"{label} pipeline CSV is missing")
    rows: list[dict[str, Any]] = []
    with path.resolve().open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"request", "input_sha256", "wall_end_to_end_us", "deadline_miss"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{label} pipeline CSV lacks input binding columns")
        for line_number, value in enumerate(reader, 2):
            try:
                request = int(value["request"])
                latency = float(value["wall_end_to_end_us"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label} pipeline CSV:{line_number} has invalid timing") from error
            if request < 0 or not math.isfinite(latency) or latency < 0.0:
                raise ValueError(f"{label} pipeline CSV:{line_number} request/timing is invalid")
            digest = _hex(value.get("input_sha256"), f"{label} pipeline CSV:{line_number} input_sha256")
            observed = value.get("deadline_miss") in {"1", "true", "True"}
            expected = latency > deadline_us
            if observed != expected:
                raise ValueError(f"{label} pipeline CSV:{line_number} deadline classification differs")
            rows.append({"request": request, "input_sha256": digest,
                         "latency": latency, "deadline_miss": observed})
    measured_rows = [row for row in rows if row["request"] >= warmup]
    if len(measured_rows) != len(trace):
        raise ValueError(f"{label} pipeline CSV request count differs")
    requests = [row["request"] for row in measured_rows]
    if requests != list(range(warmup, warmup + len(trace))):
        raise ValueError(f"{label} pipeline CSV requests are not dense after warmup")
    for index, (csv_row, trace_row) in enumerate(zip(measured_rows, trace, strict=True)):
        if csv_row["input_sha256"] != trace_row["input_sha256"]:
            raise ValueError(f"{label} pipeline input_sha256 differs at request {index}")
        if csv_row["deadline_miss"] != trace_row["deadline_miss"]:
            raise ValueError(f"{label} pipeline deadline classification differs at request {index}")
        if not math.isclose(csv_row["latency"], trace_row["wall_latency_us"], rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{label} pipeline wall latency differs at request {index}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "warmup": warmup,
        "measured_requests": len(measured_rows),
        "contract": "production-wall-input-bound",
    }


def verify(
    reference_trace: Path,
    candidate_trace: Path,
    *,
    dataset: Path,
    reference_engine: Path,
    candidate_engine: Path,
    workload: str,
    task: str,
    deadline_us: float,
    accuracy_tolerance: float = 0.01,
    minimum_accuracy: float = 0.90,
    detection_iou_threshold: float = 0.5,
    asr_max_wer: float = 0.20,
    asr_wer_tolerance: float = 0.02,
    reference_output_trace: Path | None = None,
    candidate_output_trace: Path | None = None,
    require_output_traces: bool = False,
    output_trace_warmup: int = 0,
    output_trace_index: int = 0,
    reference_pipeline_csv: Path | None = None,
    candidate_pipeline_csv: Path | None = None,
    require_input_binding: bool = False,
    pipeline_warmup: int = 0,
) -> dict[str, Any]:
    if not workload or not task:
        raise ValueError("workload and task are required")
    if task not in {"classification", "object-detection", "asr"}:
        raise ValueError("task must be classification, object-detection, or asr")
    deadline = _finite(deadline_us, "deadline_us", positive=True)
    tolerance = _finite(accuracy_tolerance, "accuracy_tolerance")
    if tolerance > 1.0:
        raise ValueError("accuracy_tolerance must not exceed one")
    minimum = _finite(minimum_accuracy, "minimum_accuracy")
    if minimum > 1.0:
        raise ValueError("minimum_accuracy must not exceed one")
    iou_threshold = _finite(detection_iou_threshold, "detection_iou_threshold")
    if iou_threshold > 1.0:
        raise ValueError("detection_iou_threshold must not exceed one")
    max_wer = _finite(asr_max_wer, "asr_max_wer")
    wer_tolerance = _finite(asr_wer_tolerance, "asr_wer_tolerance")
    if (
        isinstance(output_trace_warmup, bool)
        or not isinstance(output_trace_warmup, int)
        or output_trace_warmup < 0
        or isinstance(output_trace_index, bool)
        or not isinstance(output_trace_index, int)
        or output_trace_index < 0
    ):
        raise ValueError("output trace warmup/index is invalid")
    if (
        isinstance(pipeline_warmup, bool)
        or not isinstance(pipeline_warmup, int)
        or pipeline_warmup < 0
    ):
        raise ValueError("pipeline warmup is invalid")
    if (reference_pipeline_csv is None) != (candidate_pipeline_csv is None):
        raise ValueError("reference and candidate pipeline CSVs must be supplied together")
    if require_input_binding and reference_pipeline_csv is None:
        raise ValueError("formal accuracy gate requires reference and candidate pipeline CSVs")
    paths = {
        "dataset": dataset,
        "reference_engine": reference_engine,
        "candidate_engine": candidate_engine,
    }
    digests: dict[str, str] = {}
    for name, path in paths.items():
        if not path.resolve().is_file():
            raise ValueError(f"{name} is missing")
        digests[name + "_sha256"] = sha256(path)
    structured_predictions = task == "object-detection"
    reference = _read_trace(
        reference_trace, deadline,
        structured_predictions=structured_predictions,
        detection_iou_threshold=iou_threshold,
        task=task, asr_max_wer=max_wer,
    )
    candidate = _read_trace(
        candidate_trace, deadline,
        structured_predictions=structured_predictions,
        detection_iou_threshold=iou_threshold,
        task=task, asr_max_wer=max_wer,
    )
    input_evidence: dict[str, Any] = {
        "application_input_binding_required": require_input_binding,
        "application_input_binding_contract": "not-required",
    }
    if reference_pipeline_csv is not None and candidate_pipeline_csv is not None:
        reference_pipeline = _read_pipeline_binding(
            reference_pipeline_csv, reference, deadline,
            warmup=pipeline_warmup, label="reference",
        )
        candidate_pipeline = _read_pipeline_binding(
            candidate_pipeline_csv, candidate, deadline,
            warmup=pipeline_warmup, label="candidate",
        )
        input_evidence = {
            "application_input_binding_required": require_input_binding,
            "application_input_binding_contract": "passed",
            "reference_pipeline_csv": reference_pipeline,
            "candidate_pipeline_csv": candidate_pipeline,
        }
    if (reference_output_trace is None) != (candidate_output_trace is None):
        raise ValueError("reference and candidate output traces must be supplied together")
    if require_output_traces and reference_output_trace is None:
        raise ValueError("formal accuracy gate requires reference and candidate output traces")
    output_evidence: dict[str, Any] = {
        "application_output_trace_required": require_output_traces,
        "application_output_trace_contract": "not-required",
    }
    if reference_output_trace is not None and candidate_output_trace is not None:
        reference_output = _read_output_trace(
            reference_output_trace, len(reference), "reference",
            warmup=output_trace_warmup, output_index=output_trace_index,
        )
        candidate_output = _read_output_trace(
            candidate_output_trace, len(candidate), "candidate",
            warmup=output_trace_warmup, output_index=output_trace_index,
        )
        for index, (row, output_sha) in enumerate(
            zip(reference, reference_output["measured_output_sha256"], strict=True)
        ):
            if row["output_sha256"] != output_sha:
                raise ValueError(f"reference output trace differs at request {index}")
        for index, (row, output_sha) in enumerate(
            zip(candidate, candidate_output["measured_output_sha256"], strict=True)
        ):
            if row["output_sha256"] != output_sha:
                raise ValueError(f"candidate output trace differs at request {index}")
        output_evidence = {
            "application_output_trace_required": require_output_traces,
            "application_output_trace_contract": "passed",
            "reference_output_trace": reference_output,
            "candidate_output_trace": candidate_output,
        }
    dataset_samples = _read_dataset_manifest(dataset)
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate request counts differ")
    for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        for key in ("request_id", "arrival_sequence", "input_sha256", "expected_label"):
            if left[key] != right[key]:
                raise ValueError(f"shared application workload differs at request {index}: {key}")
        expected_label = dataset_samples.get(left["input_sha256"])
        if expected_label is None:
            raise ValueError(f"dataset manifest lacks input {left['input_sha256']}")
        if expected_label != left["expected_label"]:
            raise ValueError(f"dataset manifest label differs at request {index}")
    reference_accuracy = sum(row["correct"] for row in reference) / len(reference)
    candidate_accuracy = sum(row["correct"] for row in candidate) / len(candidate)
    accuracy_delta = abs(reference_accuracy - candidate_accuracy)
    if accuracy_delta > tolerance:
        raise ValueError("candidate application accuracy differs beyond tolerance")
    if reference_accuracy < minimum or candidate_accuracy < minimum:
        raise ValueError(
            "application accuracy is below minimum: "
            f"reference={reference_accuracy:.6f}, candidate={candidate_accuracy:.6f}, "
            f"minimum={minimum:.6f}; "
            f"reference_by_label=[{_accuracy_breakdown(reference)}]; "
            f"candidate_by_label=[{_accuracy_breakdown(candidate)}]"
        )
    asr_metrics: dict[str, Any] = {
        "asr_max_wer": max_wer,
        "asr_wer_tolerance": wer_tolerance,
        "reference_wer": None,
        "candidate_wer": None,
        "wer_delta": None,
    }
    if task == "asr":
        reference_wers = [
            _word_error_rate(row["prediction"], row["expected_label"])
            for row in reference
        ]
        candidate_wers = [
            _word_error_rate(row["prediction"], row["expected_label"])
            for row in candidate
        ]
        reference_wer = sum(reference_wers) / len(reference_wers)
        candidate_wer = sum(candidate_wers) / len(candidate_wers)
        wer_delta = abs(reference_wer - candidate_wer)
        if reference_wer > max_wer or candidate_wer > max_wer:
            raise ValueError(
                "ASR word error rate is above maximum: "
                f"reference={reference_wer:.6f}, candidate={candidate_wer:.6f}, "
                f"maximum={max_wer:.6f}"
            )
        if wer_delta > wer_tolerance:
            raise ValueError("candidate ASR word error rate differs beyond tolerance")
        asr_metrics.update({
            "reference_wer": reference_wer,
            "candidate_wer": candidate_wer,
            "wer_delta": wer_delta,
        })
    return {
        "schema_version": 1,
        "kind": "p9-application-accuracy-gate",
        "status": "passed",
        "numeric_comparison_allowed": True,
        "workload": workload,
        "task": task,
        "requests": len(reference),
        "deadline_us": deadline,
        "accuracy_tolerance": tolerance,
        "minimum_accuracy": minimum,
        "detection_iou_threshold": iou_threshold,
        "reference_accuracy": reference_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "reference_accuracy_by_label": _accuracy_breakdown_map(reference),
        "candidate_accuracy_by_label": _accuracy_breakdown_map(candidate),
        "accuracy_delta": accuracy_delta,
        **asr_metrics,
        "dataset_samples": len(dataset_samples),
        "dataset_coverage": len(reference) / len(dataset_samples),
        "dataset_manifest_schema": 1,
        "dataset_manifest_path": str(dataset.resolve()),
        "dataset_manifest_sha256": sha256(dataset),
        "reference_trace_path": str(reference_trace.resolve()),
        "reference_trace_sha256": sha256(reference_trace),
        "candidate_trace_path": str(candidate_trace.resolve()),
        "candidate_trace_sha256": sha256(candidate_trace),
        **output_evidence,
        **input_evidence,
        **digests,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-engine", type=Path, required=True)
    parser.add_argument("--candidate-engine", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--deadline-us", type=float, required=True)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.01)
    parser.add_argument("--minimum-accuracy", type=float, default=0.90)
    parser.add_argument("--detection-iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--asr-max-wer", type=float, default=0.20,
        help="maximum mean/request ASR word error rate accepted by the gate",
    )
    parser.add_argument(
        "--asr-wer-tolerance", type=float, default=0.02,
        help="maximum reference/candidate mean WER difference for ASR",
    )
    parser.add_argument("--reference-output-trace", type=Path)
    parser.add_argument("--candidate-output-trace", type=Path)
    parser.add_argument("--output-trace-warmup", type=int, default=0)
    parser.add_argument("--output-trace-index", type=int, default=0)
    parser.add_argument("--reference-pipeline-csv", type=Path)
    parser.add_argument("--candidate-pipeline-csv", type=Path)
    parser.add_argument("--pipeline-warmup", type=int, default=0)
    parser.add_argument(
        "--require-input-binding",
        action="store_true",
        help="require production CSV input hashes and wall timing to bind prediction traces",
    )
    parser.add_argument(
        "--require-output-traces",
        action="store_true",
        help="require independent post-completion binary traces for a formal gate",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(
        args.reference_trace,
        args.candidate_trace,
        dataset=args.dataset,
        reference_engine=args.reference_engine,
        candidate_engine=args.candidate_engine,
        workload=args.workload,
        task=args.task,
        deadline_us=args.deadline_us,
        accuracy_tolerance=args.accuracy_tolerance,
        minimum_accuracy=args.minimum_accuracy,
        detection_iou_threshold=args.detection_iou_threshold,
        asr_max_wer=args.asr_max_wer,
        asr_wer_tolerance=args.asr_wer_tolerance,
        reference_output_trace=args.reference_output_trace,
        candidate_output_trace=args.candidate_output_trace,
        require_output_traces=args.require_output_traces,
        output_trace_warmup=args.output_trace_warmup,
        output_trace_index=args.output_trace_index,
        reference_pipeline_csv=args.reference_pipeline_csv,
        candidate_pipeline_csv=args.candidate_pipeline_csv,
        require_input_binding=args.require_input_binding,
        pipeline_warmup=args.pipeline_warmup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
