#!/usr/bin/env python3
"""Build an explicit repeated current ImageNette formal-session contract.

The labelled ImageNette gate contains 90 measured requests.  Formal sessions
need a longer request-level sample without changing the application, model,
arrival cadence, or deadline.  This tool makes that reuse explicit: every
generated request receives a fresh request ID and arrival sequence, while its
input bytes, external label, reference logits, and source provenance are
copied from the frozen current gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable


INPUT_MAGIC = b"JDGINT1\x00"
OUTPUT_MAGIC = b"JDGOUT1\x00"
ARRIVAL_MAGIC = b"JDGARR1\x00"
INPUT_HEADER = struct.Struct("<IIQ")
INPUT_PREFIX = struct.Struct("<I64s")
OUTPUT_HEADER = struct.Struct("<IQ")
OUTPUT_RECORD = struct.Struct("<I10f")
ARRIVAL_HEADER = struct.Struct("<IIQ")
ARRIVAL_RECORD = struct.Struct("<IIQ64s64s")
REQUEST_KEYS = {
    "schema_version", "iteration", "request_id", "arrival_sequence",
    "input_sha256", "expected_label",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path} is not newline-complete")
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} is not a nonempty object JSONL")
    return rows


def _input_records(path: Path) -> tuple[int, list[tuple[str, bytes]]]:
    raw = path.resolve().read_bytes()
    if len(raw) < len(INPUT_MAGIC) + INPUT_HEADER.size or not raw.startswith(INPUT_MAGIC):
        raise ValueError("current input trace header differs")
    schema, count, sample_bytes = INPUT_HEADER.unpack_from(raw, len(INPUT_MAGIC))
    if schema != 1 or count != 100 or sample_bytes != 602112:
        raise ValueError("current ImageNette input trace dimensions differ")
    offset = len(INPUT_MAGIC) + INPUT_HEADER.size
    records: list[tuple[str, bytes]] = []
    for expected in range(count):
        end = offset + INPUT_PREFIX.size + sample_bytes
        if end > len(raw):
            raise ValueError("current input trace is truncated")
        iteration, digest = INPUT_PREFIX.unpack_from(raw, offset)
        if iteration != expected:
            raise ValueError("current input trace iterations are not dense")
        offset += INPUT_PREFIX.size
        input_sha = digest.decode("ascii")
        if len(input_sha) != 64 or any(char not in "0123456789abcdef" for char in input_sha):
            raise ValueError("current input trace hash is invalid")
        records.append((input_sha, raw[offset:end]))
        offset = end
    if offset != len(raw):
        raise ValueError("current input trace has trailing bytes")
    return sample_bytes, records


def _output_records(path: Path) -> list[bytes]:
    raw = path.resolve().read_bytes()
    if len(raw) < len(OUTPUT_MAGIC) + OUTPUT_HEADER.size or not raw.startswith(OUTPUT_MAGIC):
        raise ValueError("current reference output header differs")
    output_count, output_bytes = OUTPUT_HEADER.unpack_from(raw, len(OUTPUT_MAGIC))
    if output_count != 1 or output_bytes != 40:
        raise ValueError("current reference output dimensions differ")
    record_bytes = 4 + output_bytes
    count = (len(raw) - len(OUTPUT_MAGIC) - OUTPUT_HEADER.size) // record_bytes
    if count != 100 or len(raw) != len(OUTPUT_MAGIC) + OUTPUT_HEADER.size + count * record_bytes:
        raise ValueError("current reference output record count differs")
    records: list[bytes] = []
    offset = len(OUTPUT_MAGIC) + OUTPUT_HEADER.size
    for expected in range(count):
        record = raw[offset:offset + record_bytes]
        iteration = struct.unpack_from("<I", record)[0]
        if iteration != expected:
            raise ValueError("current reference output iterations are not dense")
        logits = struct.unpack_from("<10f", record, 4)
        if any(not math.isfinite(value) for value in logits):
            raise ValueError("current reference output contains a non-finite logit")
        records.append(record)
        offset += record_bytes
    return records


def _reference_csv(path: Path) -> tuple[list[str], dict[int, dict[str, str]]]:
    with path.resolve().open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = ["request", "input_sha256", "wall_end_to_end_us", "deadline_miss"]
        if reader.fieldnames != expected:
            raise ValueError("current reference CSV schema differs")
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            request = int(row["request"])
            if request in rows:
                raise ValueError("current reference CSV repeats a request")
            rows[request] = row
    if sorted(rows) != list(range(100)):
        raise ValueError("current reference CSV must contain requests 0..99")
    return expected, rows


def build(
    *, output: Path, source_dir: Path, requests: int, warmup: int,
    period_us: int, deadline_us: float,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing existing formal workload directory: {output}")
    if requests <= 0 or warmup < 0 or period_us <= 0 or not math.isfinite(deadline_us):
        raise ValueError("formal workload parameters are invalid")
    source_dir = source_dir.resolve()
    source_paths = {
        "input_trace": source_dir / "inputs.bin",
        "reference_output": source_dir / "reference-output.bin",
        "reference_predictions": source_dir / "reference-predictions-current-deadline.jsonl",
        "reference_pipeline": source_dir / "reference-current-deadline.csv",
        "request_manifest": source_dir / "requests.jsonl",
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise ValueError(f"missing current source artifact {label}: {path}")
    sample_bytes, input_records = _input_records(source_paths["input_trace"])
    output_records = _output_records(source_paths["reference_output"])
    requests_source = _jsonl(source_paths["request_manifest"])
    predictions_source = _jsonl(source_paths["reference_predictions"])
    if len(requests_source) != 90 or len(predictions_source) != 90:
        raise ValueError("current ImageNette measured source count differs")
    for index, row in enumerate(requests_source):
        if set(row) != REQUEST_KEYS or row.get("iteration") != warmup + index:
            raise ValueError("current request manifest schema or warmup differs")
    for index, row in enumerate(predictions_source):
        if row.get("arrival_sequence") != index or row.get("input_sha256") != requests_source[index]["input_sha256"]:
            raise ValueError("current reference prediction binding differs")
    csv_fields, csv_rows = _reference_csv(source_paths["reference_pipeline"])
    output.mkdir(parents=True)

    total = warmup + requests
    formal_input = output / "inputs.bin"
    with formal_input.open("wb") as stream:
        stream.write(INPUT_MAGIC)
        stream.write(INPUT_HEADER.pack(1, total, sample_bytes))
        for index in range(total):
            source_index = index if index < warmup else warmup + ((index - warmup) % 90)
            input_sha, sample = input_records[source_index]
            stream.write(INPUT_PREFIX.pack(index, input_sha.encode("ascii")))
            stream.write(sample)

    formal_output = output / "reference-output.bin"
    with formal_output.open("wb") as stream:
        stream.write(OUTPUT_MAGIC)
        stream.write(OUTPUT_HEADER.pack(1, 40))
        for index in range(total):
            source_index = index if index < warmup else warmup + ((index - warmup) % 90)
            stream.write(struct.pack("<I", index))
            stream.write(output_records[source_index][4:])

    formal_requests: list[dict[str, Any]] = []
    formal_predictions: list[dict[str, Any]] = []
    for index in range(requests):
        source_index = index % 90
        source_request = requests_source[source_index]
        source_prediction = predictions_source[source_index]
        request_id = f"imagenette-formal-{index:06d}"
        formal_requests.append({
            "schema_version": 1,
            "iteration": warmup + index,
            "request_id": request_id,
            "arrival_sequence": index,
            "input_sha256": source_request["input_sha256"],
            "expected_label": source_request["expected_label"],
        })
        prediction = dict(source_prediction)
        prediction["request_id"] = request_id
        prediction["arrival_sequence"] = index
        prediction["deadline_us"] = deadline_us
        prediction["deadline_miss"] = float(prediction["wall_latency_us"]) > deadline_us
        formal_predictions.append(prediction)

    request_path = output / "requests.jsonl"
    request_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in formal_requests),
        encoding="utf-8",
    )
    prediction_path = output / "reference-predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in formal_predictions),
        encoding="utf-8",
    )

    arrival_path = output / "arrivals.bin"
    with arrival_path.open("wb") as stream:
        stream.write(ARRIVAL_MAGIC)
        stream.write(ARRIVAL_HEADER.pack(1, requests, ARRIVAL_RECORD.size))
        for row in formal_requests:
            request_id = row["request_id"].encode("ascii")
            stream.write(ARRIVAL_RECORD.pack(
                row["iteration"], row["arrival_sequence"],
                period_us * 1000 * row["arrival_sequence"],
                row["input_sha256"].encode("ascii"), request_id.ljust(64, b"\x00"),
            ))

    reference_csv_path = output / "reference.csv"
    with reference_csv_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for index in range(total):
            if index < warmup:
                source = csv_rows[index]
            else:
                source = csv_rows[warmup + ((index - warmup) % 90)]
            writer.writerow({
                "request": index,
                "input_sha256": source["input_sha256"],
                "wall_end_to_end_us": source["wall_end_to_end_us"],
                "deadline_miss": int(float(source["wall_end_to_end_us"]) > deadline_us),
            })

    dataset = source_dir.parent / "p9-resnet50-imagenette-val10-20260811" / "dataset-manifest.jsonl"
    if not dataset.is_file():
        raise ValueError(f"cannot locate current ImageNette dataset manifest: {dataset}")
    contract = {
        "schema_version": 1,
        "workload_id": "resnet50-classification",
        "topology": "fixed-2g+1g",
        "placement": "fixed-1g-producer-2g-consumer",
        "input_tensor": "gpu_0/res4_5_branch2c_bn_2",
        "payload_bytes": 802816,
        "request_count": requests,
        "arrival_trace_path": str(request_path.resolve()),
        "arrival_trace_sha256": sha256(request_path),
        "dataset_manifest_path": str(dataset.resolve()),
        "dataset_manifest_sha256": sha256(dataset),
        "binding": "explicit-cyclic-replay-of-labelled-current-gate",
        "producer_input_trace_path": str(formal_input.resolve()),
        "producer_input_trace_sha256": sha256(formal_input),
        "producer_input_binding": "JDGINT1-bytes-bound-to-arrival-contract",
        "operational_arrival_trace_path": str(arrival_path.resolve()),
        "operational_arrival_trace_sha256": sha256(arrival_path),
        "operational_arrival_binding": "JDGARR1-release-offsets-consumed-by-pipeline",
    }
    contract_path = output / "common-workload.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "kind": "p9-current-imagenette-formal-session-workload",
        "requests": requests,
        "warmup": warmup,
        "period_us": period_us,
        "deadline_us": deadline_us,
        "sample_reuse": {
            "enabled": True,
            "source_measured_requests": 90,
            "source_labelled_samples": 100,
            "method": "cyclic-replay-with-fresh-request-and-arrival-identities",
        },
        "current_source_artifacts": {
            label: {"path": str(path.resolve()), "sha256": sha256(path)}
            for label, path in source_paths.items()
        },
        "generated_artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in {
                "input_trace": formal_input,
                "arrival_trace": arrival_path,
                "request_manifest": request_path,
                "reference_predictions": prediction_path,
                "reference_output": formal_output,
                "reference_pipeline": reference_csv_path,
                "common_workload": contract_path,
            }.items()
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=1100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--period-us", type=int, default=5000)
    parser.add_argument("--deadline-us", type=float, required=True)
    args = parser.parse_args(argv)
    result = build(
        output=args.output.resolve(), source_dir=args.source_dir.resolve(),
        requests=args.requests, warmup=args.warmup, period_us=args.period_us,
        deadline_us=args.deadline_us,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
