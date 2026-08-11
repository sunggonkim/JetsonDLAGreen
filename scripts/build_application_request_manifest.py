#!/usr/bin/env python3
"""Expand an externally labelled dataset into a fixed request manifest.

The production binary numbers warmup and measured iterations in one sequence,
while the accuracy gate compares only measured requests.  This tool binds the
measured iteration numbers to dataset rows without inferring labels or
repeating samples silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_KEYS = {"schema_version", "sample_id", "input_sha256", "expected_label"}
REQUEST_KEYS = {
    "schema_version", "iteration", "request_id", "arrival_sequence",
    "input_sha256", "expected_label",
}
SAMPLE_KEYS = {"iteration", "sample_id", "path", "input_sha256"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def read_dataset(path: Path) -> list[dict[str, Any]]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("dataset manifest is not newline-complete")
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    input_labels: dict[str, str] = {}
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
        digest = value.get("input_sha256")
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"dataset manifest:{line_number} input_sha256 is invalid")
        label = value.get("expected_label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"dataset manifest:{line_number} expected_label is invalid")
        previous_label = input_labels.get(digest)
        if previous_label is not None and previous_label != label:
            raise ValueError(f"dataset manifest:{line_number} input_sha256 has conflicting labels")
        sample_ids.add(sample_id)
        input_labels[digest] = label
        rows.append(value)
    if not rows:
        raise ValueError("dataset manifest is empty")
    return rows


def read_sample_list(path: Path) -> list[dict[str, Any]]:
    """Read the dense producer trace index used for warmup/measurement binding."""
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("sample list is not newline-complete")
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"sample list:{line_number} is invalid JSON") from error
        if not isinstance(value, dict) or set(value) != SAMPLE_KEYS:
            raise ValueError(f"sample list:{line_number} schema differs")
        iteration = value.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError(f"sample list:{line_number} iteration is invalid")
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError(f"sample list:{line_number} sample_id is invalid")
        digest = value.get("input_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"sample list:{line_number} input_sha256 is invalid")
        if not isinstance(value.get("path"), str) or not value["path"]:
            raise ValueError(f"sample list:{line_number} path is invalid")
        rows.append(value)
        sample_ids.add(sample_id)
    if not rows:
        raise ValueError("sample list is empty")
    rows.sort(key=lambda row: row["iteration"])
    if [row["iteration"] for row in rows] != list(range(len(rows))):
        raise ValueError("sample list iterations must be dense and zero-based")
    return rows


def build(
    dataset_manifest: Path,
    *,
    warmup: int,
    requests: int | None = None,
    request_id_prefix: str = "app",
    sample_list: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if requests is not None and (
        isinstance(requests, bool) or not isinstance(requests, int) or requests <= 0
    ):
        raise ValueError("requests must be a positive integer")
    if not isinstance(request_id_prefix, str) or not request_id_prefix:
        raise ValueError("request_id_prefix must be nonempty")
    dataset_rows = read_dataset(dataset_manifest)
    count = len(dataset_rows) if requests is None else requests
    if count > len(dataset_rows):
        raise ValueError("requests exceed labelled dataset samples; refusing implicit reuse")
    sample_rows = None if sample_list is None else read_sample_list(sample_list)
    if sample_rows is not None and warmup + count > len(sample_rows):
        raise ValueError("producer sample list is shorter than warmup plus requests")
    labels_by_digest = {row["input_sha256"]: row["expected_label"] for row in dataset_rows}
    selected_rows = (
        dataset_rows
        if sample_rows is None
        else sample_rows[warmup : warmup + count]
    )
    rows: list[dict[str, Any]] = []
    for arrival, sample in enumerate(selected_rows):
        digest = sample["input_sha256"]
        label = labels_by_digest.get(digest)
        if label is None:
            raise ValueError(
                "producer sample list contains an input absent from the dataset manifest"
            )
        rows.append({
            "schema_version": 1,
            "iteration": warmup + arrival,
            "request_id": f"{request_id_prefix}-{arrival:06d}",
            "arrival_sequence": arrival,
            "input_sha256": digest,
            "expected_label": label,
        })
    provenance = {
        "schema_version": 1,
        "kind": "p9-application-request-manifest-provenance",
        "dataset_manifest": {
            "path": str(dataset_manifest.resolve()),
            "sha256": sha256(dataset_manifest),
        },
        "warmup": warmup,
        "requests": count,
        "request_id_prefix": request_id_prefix,
        "sample_reuse": False,
        "label_source": "external-dataset-owner-map",
    }
    if sample_list is not None:
        provenance["producer_sample_list"] = {
            "path": str(sample_list.resolve()),
            "sha256": sha256(sample_list),
            "warmup_records_skipped": warmup,
            "measured_records_bound": count,
        }
    return rows, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--request-id-prefix", default="app")
    parser.add_argument(
        "--sample-list",
        type=Path,
        help="dense producer sample list; measured rows are selected after warmup",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, provenance = build(
        args.dataset_manifest,
        warmup=args.warmup,
        requests=args.requests,
        request_id_prefix=args.request_id_prefix,
        sample_list=args.sample_list,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "request_manifest": str(args.output),
        "provenance": str(provenance_path),
        "warmup": args.warmup,
        "requests": len(rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
