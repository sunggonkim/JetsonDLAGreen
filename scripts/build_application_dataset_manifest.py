#!/usr/bin/env python3
"""Build the immutable input/label manifest for a QUIET accuracy gate.

Labels are supplied by the dataset owner as a JSON object mapping a path
relative to ``--root`` to a task label.  The tool never infers labels from a
filename or from a model prediction.  That separation is required for the
reference/candidate accuracy gate to be meaningful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_labels(path: Path) -> dict[str, str]:
    raw = path.resolve().read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("label map is not newline-complete")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("label map is not valid JSON") from error
    if not isinstance(value, dict) or not value:
        raise ValueError("label map must be a non-empty JSON object")
    labels: dict[str, str] = {}
    for key, label in value.items():
        if not isinstance(key, str) or not key or Path(key).is_absolute():
            raise ValueError("label map keys must be relative paths")
        normalized = Path(key).as_posix()
        if normalized != key or normalized in labels:
            raise ValueError("label map contains duplicate/non-canonical paths")
        if not isinstance(label, str) or not label:
            raise ValueError(f"label for {key!r} must be a non-empty string")
        labels[normalized] = label
    return labels


def build(root: Path, labels_path: Path, *, pattern: str = "*",
          limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve()
    labels_path = labels_path.resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    if not labels_path.is_file():
        raise ValueError(f"label map is not a regular file: {labels_path}")
    labels = _read_labels(labels_path)
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        files = files[:limit]
    if not files:
        raise ValueError("dataset selection is empty")
    selected: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        selected.add(relative)
        if relative not in labels:
            raise ValueError(f"label map lacks selected sample {relative}")
        rows.append({
            "schema_version": 1,
            "sample_id": relative,
            "input_sha256": sha256(path),
            "expected_label": labels[relative],
        })
    unused = sorted(set(labels) - selected)
    if unused:
        raise ValueError(f"label map contains samples outside selection: {unused[0]}")
    provenance = {
        "schema_version": 1,
        "kind": "p9-application-dataset-manifest-provenance",
        "root": str(root),
        "root_pattern": pattern,
        "label_map": {"path": str(labels_path), "sha256": sha256(labels_path)},
        "samples": len(rows),
        "label_source": "external-dataset-owner-map",
        "automatic_filename_labels": False,
    }
    return rows, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True,
                        help="JSON object: relative sample path -> expected label")
    parser.add_argument("--pattern", default="*",
                        help="glob relative to root (default: *)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, provenance = build(args.root, args.labels, pattern=args.pattern,
                             limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps({"manifest": str(args.output),
                      "provenance": str(provenance_path),
                      "samples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
