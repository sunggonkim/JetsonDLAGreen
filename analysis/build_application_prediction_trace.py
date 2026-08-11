#!/usr/bin/env python3
"""Convert a production output trace into an accuracy-gate prediction trace.

The request manifest and class map are external trust inputs.  This converter
only joins them with the benchmark's post-completion output bytes and measured
wall CSV; it never invents labels or timing.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    from .read_application_output_trace import parse
except ImportError:  # direct CLI execution
    _reader_path = Path(__file__).with_name("read_application_output_trace.py")
    _reader_spec = importlib.util.spec_from_file_location(
        "p9_read_application_output_trace", _reader_path
    )
    if _reader_spec is None or _reader_spec.loader is None:
        raise ImportError(f"cannot load {_reader_path}")
    _reader_module = importlib.util.module_from_spec(_reader_spec)
    sys.modules[_reader_spec.name] = _reader_module
    _reader_spec.loader.exec_module(_reader_module)
    parse = _reader_module.parse


REQUEST_KEYS = {
    "schema_version", "iteration", "request_id", "arrival_sequence",
    "input_sha256", "expected_label",
}
RESNET10_OUTPUT_LABELS = ("Car", "RoadSign", "TwoWheeler", "Person")
RESNET10_DECODED_LABEL_COUNT = 3


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _read_request_manifest(path: Path) -> dict[int, dict[str, Any]]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("request manifest is not newline-complete")
    rows: dict[int, dict[str, Any]] = {}
    request_ids: set[str] = set()
    arrivals: set[int] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != REQUEST_KEYS or value.get("schema_version") != 1:
            raise ValueError(f"request manifest:{line_number} schema differs")
        iteration = value.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0 or iteration in rows:
            raise ValueError(f"request manifest:{line_number} iteration is invalid")
        request_id = value.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in request_ids:
            raise ValueError(f"request manifest:{line_number} request_id is invalid")
        arrival = value.get("arrival_sequence")
        if isinstance(arrival, bool) or not isinstance(arrival, int) or arrival < 0 or arrival in arrivals:
            raise ValueError(f"request manifest:{line_number} arrival is invalid")
        if not isinstance(value.get("expected_label"), str) or not value["expected_label"]:
            raise ValueError(f"request manifest:{line_number} label is invalid")
        _hex(value.get("input_sha256"), f"request manifest:{line_number} input_sha256")
        rows[iteration] = value
        request_ids.add(request_id)
        arrivals.add(arrival)
    if not rows or sorted(arrivals) != list(range(len(rows))):
        raise ValueError("request manifest must have dense zero-based arrivals")
    return rows


def _read_class_map(path: Path) -> dict[int, str]:
    value = json.loads(path.resolve().read_bytes())
    if not isinstance(value, dict) or not value:
        raise ValueError("class map must be a non-empty JSON object")
    result: dict[int, str] = {}
    for key, label in value.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError("class map keys must be integer indices") from error
        if index < 0 or not isinstance(label, str) or not label:
            raise ValueError("class map entries are invalid")
        if index in result:
            raise ValueError("class map repeats an index")
        result[index] = label
    return result


def _read_transcript_map(path: Path) -> dict[str, str]:
    """Read transcripts emitted by an external ASR decoder, keyed by output hash."""
    value = json.loads(path.resolve().read_bytes())
    if not isinstance(value, dict) or not value:
        raise ValueError("transcript map must be a non-empty JSON object")
    result: dict[str, str] = {}
    for output_sha, transcript in value.items():
        _hex(output_sha, "transcript map output_sha256")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("transcript map values must be non-empty strings")
        result[output_sha] = transcript
    return result


def _word_error_rate(prediction: str, expected_label: str) -> float:
    reference = re.findall(r"\w+", expected_label.casefold(), flags=re.UNICODE)
    hypothesis = re.findall(r"\w+", prediction.casefold(), flags=re.UNICODE)
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


def _validate_resnet10_class_map(class_map: dict[int, str]) -> None:
    """Require the vendor output slots, including its parser-excluded slot."""
    expected = {index: label for index, label in enumerate(RESNET10_OUTPUT_LABELS)}
    if class_map != expected:
        raise ValueError(
            "ResNet10 class map must match vendor labels.txt output slots "
            "including parser-excluded final Person slot"
        )


def _read_wall_csv(
    path: Path, deadline_us: float, *, require_input_binding: bool = False,
    warmup: int = 0,
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.resolve().open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"request", "wall_end_to_end_us", "deadline_miss"}
        if require_input_binding:
            required.add("input_sha256")
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("pipeline CSV lacks application wall columns")
        for line_number, value in enumerate(reader, 2):
            try:
                iteration = int(value["request"])
                latency = float(value["wall_end_to_end_us"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"pipeline CSV:{line_number} has invalid timing") from error
            if iteration < 0 or iteration in rows or not math.isfinite(latency) or latency < 0:
                raise ValueError(f"pipeline CSV:{line_number} timing is invalid")
            expected_miss = latency > deadline_us
            observed = value["deadline_miss"] in {"1", "true", "True"}
            if observed != expected_miss:
                raise ValueError(f"pipeline CSV:{line_number} deadline classification differs")
            input_sha256 = value.get("input_sha256", "")
            if require_input_binding and not input_sha256:
                raise ValueError(f"pipeline CSV:{line_number} lacks input_sha256")
            if input_sha256:
                _hex(input_sha256, f"pipeline CSV:{line_number} input_sha256")
            rows[iteration] = {
                "latency": latency,
                "deadline_miss": observed,
                "input_sha256": input_sha256,
            }
    if not rows:
        raise ValueError("pipeline CSV is empty")
    return {
        iteration: row for iteration, row in rows.items() if iteration >= warmup
    }


def _decode_resnet10_detection(
    outputs: list[dict[str, Any]], class_map: dict[int, str], *,
    threshold: float = 0.1, grid_width: int = 40, grid_height: int = 23,
) -> str:
    """Decode the vendor ResNet10 ``Layer7_cov``/``Layer7_bbox`` contract.

    The Jetson Multimedia API uses stride 16, a 35-pixel normalization, and
    ``groupRectangles(..., 1, 0.1)`` per class.  Predictions are serialized as
    a canonical JSON string so the external dataset manifest can bind true
    detector annotations without pretending that an argmax is detection
    accuracy.
    """
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("ResNet10 detection threshold is invalid")
    if grid_width <= 0 or grid_height <= 0 or not class_map:
        raise ValueError("ResNet10 detection dimensions/classes are invalid")
    cov = outputs[0].get("values")
    bbox = outputs[1].get("values")
    cells = grid_width * grid_height
    if (
        not isinstance(cov, list) or not isinstance(bbox, list)
        or len(cov) != len(class_map) * cells
        or len(bbox) != len(class_map) * 4 * cells
    ):
        raise ValueError("ResNet10 output tensor dimensions differ from the vendor contract")
    try:
        import cv2
    except ImportError as error:
        raise ValueError("ResNet10 detection mode requires OpenCV groupRectangles") from error

    detections: list[dict[str, Any]] = []
    # The vendor parser clips to the network input bounds.  The trace contract
    # does not carry an image shape, so retain its fixed 640x368 model bounds.
    for class_index in range(RESNET10_DECODED_LABEL_COUNT):
        rectangles: list[list[int]] = []
        cov_offset = class_index * cells
        bbox_offset = class_index * 4 * cells
        for h in range(grid_height):
            for w in range(grid_width):
                cell = w + h * grid_width
                score = float(cov[cov_offset + cell])
                if score < threshold:
                    continue
                cx = (w * 16.0 + 0.5) / 35.0
                cy = (h * 16.0 + 0.5) / 35.0
                x1 = (float(bbox[bbox_offset + cell]) - cx) * -35.0
                y1 = (float(bbox[bbox_offset + cells + cell]) - cy) * -35.0
                x2 = (float(bbox[bbox_offset + 2 * cells + cell]) + cx) * 35.0
                y2 = (float(bbox[bbox_offset + 3 * cells + cell]) + cy) * 35.0
                left = max(0, min(639, int(x1)))
                top = max(0, min(367, int(y1)))
                right = max(0, min(639, int(x2)))
                bottom = max(0, min(367, int(y2)))
                rectangles.append([left, top, right - left, bottom - top])
        if not rectangles:
            continue
        grouped, _weights = cv2.groupRectangles(rectangles, 1, 0.1)
        for left, top, width, height in grouped.tolist() if len(grouped) else []:
            detections.append({
                "class": class_map[class_index],
                "box": [int(left), int(top), int(width), int(height)],
            })
    detections.sort(key=lambda item: (item["class"], item["box"]))
    return json.dumps({"detections": detections}, separators=(",", ":"), sort_keys=True)


def _detection_request_correct(
    prediction: str, expected_label: str, *, iou_threshold: float = 0.5,
) -> bool:
    """Apply a request-level, label-bound detector correctness criterion.

    A request is correct only when every predicted and expected box can be
    matched one-to-one with the same class and IoU >= threshold.  This is
    intentionally stricter than reporting mAP: it makes the boolean trace
    suitable for a conservative end-to-end application gate.
    """
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("detection IoU threshold is invalid")
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
    if not predicted_boxes and not expected_boxes:
        return True
    if len(predicted_boxes) != len(expected_boxes):
        return False

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


def build(output_trace: Path, pipeline_csv: Path, request_manifest: Path,
          class_map: Path | None, *, warmup: int, deadline_us: float,
          output_index: int = 0, prediction_mode: str = "argmax",
          detection_threshold: float = 0.1, detection_grid_width: int = 40,
          detection_grid_height: int = 23,
          detection_iou_threshold: float = 0.5,
          asr_transcript_map: Path | None = None,
          asr_max_wer: float = 0.20,
          require_input_binding: bool = False) -> list[dict[str, Any]]:
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if isinstance(deadline_us, bool) or not isinstance(deadline_us, (int, float)) or not math.isfinite(float(deadline_us)) or deadline_us <= 0:
        raise ValueError("deadline_us must be positive and finite")
    if isinstance(output_index, bool) or not isinstance(output_index, int) or output_index < 0:
        raise ValueError("output_index must be nonnegative")
    if prediction_mode not in {"argmax", "resnet10-detection", "asr"}:
        raise ValueError("prediction_mode is unsupported")
    if not math.isfinite(asr_max_wer) or asr_max_wer < 0.0:
        raise ValueError("asr_max_wer is invalid")
    if not math.isfinite(detection_iou_threshold) or not 0.0 <= detection_iou_threshold <= 1.0:
        raise ValueError("detection_iou_threshold is invalid")
    records = parse(
        output_trace,
        float32_output=prediction_mode == "argmax",
        float32_values=prediction_mode == "resnet10-detection",
    )["records"]
    requests = _read_request_manifest(request_manifest)
    timing = _read_wall_csv(
        pipeline_csv, float(deadline_us), require_input_binding=require_input_binding,
        warmup=warmup,
    )
    labels = _read_class_map(class_map) if class_map is not None else None
    if prediction_mode == "resnet10-detection":
        if labels is None:
            raise ValueError("ResNet10 detection mode requires --class-map")
        _validate_resnet10_class_map(labels)
    if prediction_mode == "argmax" and labels is None:
        raise ValueError("argmax mode requires --class-map")
    transcripts = (
        _read_transcript_map(asr_transcript_map)
        if prediction_mode == "asr" and asr_transcript_map is not None else None
    )
    if prediction_mode == "asr" and transcripts is None:
        raise ValueError("ASR mode requires --transcript-map")
    measured = [record for record in records if record["iteration"] >= warmup]
    if len(measured) != len(timing):
        raise ValueError("output trace and pipeline CSV measured counts differ")
    rows: list[dict[str, Any]] = []
    for record in measured:
        iteration = record["iteration"]
        request = requests.get(iteration)
        if request is None:
            raise ValueError(f"request manifest lacks iteration {iteration}")
        if iteration not in timing:
            raise ValueError(f"pipeline CSV lacks iteration {iteration}")
        measured_input_sha = timing[iteration].get("input_sha256", "")
        if measured_input_sha and measured_input_sha != request["input_sha256"]:
            raise ValueError(f"pipeline CSV input hash differs at iteration {iteration}")
        outputs = record["outputs"]
        if prediction_mode == "resnet10-detection":
            if len(outputs) != 2:
                raise ValueError("ResNet10 detection mode requires cov and bbox outputs")
            prediction = _decode_resnet10_detection(
                outputs, labels, threshold=detection_threshold,
                grid_width=detection_grid_width, grid_height=detection_grid_height,
            )
            prediction_output = outputs[0]
        elif prediction_mode == "asr":
            if output_index >= len(outputs):
                raise ValueError("output index exceeds captured outputs")
            prediction_output = outputs[output_index]
            prediction = transcripts.get(prediction_output["sha256"])  # type: ignore[union-attr]
            if prediction is None:
                raise ValueError(
                    "transcript map lacks post-completion output hash "
                    f"{prediction_output['sha256']}"
                )
        else:
            if output_index >= len(outputs):
                raise ValueError("output index exceeds captured outputs")
            prediction_index = outputs[output_index].get("argmax")
            assert labels is not None
            prediction = labels.get(prediction_index)
            if prediction is None:
                raise ValueError(f"class map lacks output index {prediction_index}")
            prediction_output = outputs[output_index]
        if prediction is None:
            raise ValueError("output index exceeds captured outputs")
        measured_timing = timing[iteration]
        rows.append({
            "schema_version": 1,
            "request_id": request["request_id"],
            "arrival_sequence": request["arrival_sequence"],
            "input_sha256": request["input_sha256"],
            "expected_label": request["expected_label"],
            "prediction": prediction,
            "correct": (
                _detection_request_correct(
                    prediction, request["expected_label"],
                    iou_threshold=detection_iou_threshold,
                )
                if prediction_mode == "resnet10-detection"
                else _asr_request_correct(
                    prediction, request["expected_label"], max_wer=asr_max_wer,
                ) if prediction_mode == "asr"
                else prediction == request["expected_label"]
            ),
            "output_sha256": prediction_output["sha256"],
            "deadline_us": float(deadline_us),
            "wall_latency_us": measured_timing["latency"],
            "deadline_miss": measured_timing["deadline_miss"],
        })
    if sorted(row["arrival_sequence"] for row in rows) != list(range(len(rows))):
        raise ValueError("application prediction arrivals are not dense")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-trace", type=Path, required=True)
    parser.add_argument("--pipeline-csv", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument(
        "--transcript-map", type=Path,
        help="ASR JSON object mapping post-completion output SHA-256 to transcript",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--deadline-us", type=float, required=True)
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument(
        "--prediction-mode", choices=("argmax", "resnet10-detection", "asr"), default="argmax",
        help="decode a classifier argmax, vendor detector tensors, or Whisper tokens",
    )
    parser.add_argument("--detection-threshold", type=float, default=0.1)
    parser.add_argument("--detection-grid-width", type=int, default=40)
    parser.add_argument("--detection-grid-height", type=int, default=23)
    parser.add_argument("--detection-iou-threshold", type=float, default=0.5)
    parser.add_argument("--asr-max-wer", type=float, default=0.20)
    parser.add_argument(
        "--require-input-binding", action="store_true",
        help="require the benchmark CSV to bind every measured tensor input hash",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.output_trace, args.pipeline_csv, args.request_manifest,
                 args.class_map, warmup=args.warmup, deadline_us=args.deadline_us,
                 output_index=args.output_index, prediction_mode=args.prediction_mode,
                 detection_threshold=args.detection_threshold,
                 detection_grid_width=args.detection_grid_width,
                 detection_grid_height=args.detection_grid_height,
                 detection_iou_threshold=args.detection_iou_threshold,
                 asr_transcript_map=args.transcript_map,
                 asr_max_wer=args.asr_max_wer,
                 require_input_binding=args.require_input_binding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"kind": "p9-application-prediction-trace", "rows": len(rows), "path": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
