#!/usr/bin/env python3
"""Convert a strict KITTI 2-D detection subset into COCO annotations.

This is an input-preparation tool for the vendor ResNet10 detector.  KITTI
classes are never inferred from filenames or model predictions: every class
must be explicitly mapped to an emitted ResNet10 label or explicitly ignored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SUPPORTED_OUTPUT_LABELS = frozenset({"Car", "RoadSign", "TwoWheeler"})
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _finite(value: str, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _parse_mapping(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        source, separator, target = value.partition("=")
        if not separator or not source or not target:
            raise ValueError(f"class mapping must be SOURCE=TARGET: {value!r}")
        if source in mapping:
            raise ValueError(f"class mapping repeats source class: {source}")
        if target not in SUPPORTED_OUTPUT_LABELS:
            raise ValueError(
                f"target label is not emitted by ResNet10: {target}; "
                f"supported={sorted(SUPPORTED_OUTPUT_LABELS)}"
            )
        mapping[source] = target
    if not mapping:
        raise ValueError("at least one explicit class mapping is required")
    return mapping


def _parse_label_file(path: Path, mapping: dict[str, str], ignored: set[str],
                      width: int, height: int, annotation_start: int) -> tuple[list[dict[str, Any]], int]:
    annotations: list[dict[str, Any]] = []
    next_id = annotation_start
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        source = fields[0]
        if source == "DontCare" or source in ignored:
            continue
        if source not in mapping:
            raise ValueError(
                f"{path}:{line_number}: class {source!r} is neither mapped nor ignored"
            )
        if len(fields) < 8:
            raise ValueError(f"{path}:{line_number}: KITTI row has fewer than 8 fields")
        left = _finite(fields[4], f"{path}:{line_number} left")
        top = _finite(fields[5], f"{path}:{line_number} top")
        right = _finite(fields[6], f"{path}:{line_number} right")
        bottom = _finite(fields[7], f"{path}:{line_number} bottom")
        left = max(0.0, min(float(width), left))
        top = max(0.0, min(float(height), top))
        right = max(0.0, min(float(width), right))
        bottom = max(0.0, min(float(height), bottom))
        if right <= left or bottom <= top:
            raise ValueError(f"{path}:{line_number}: bbox is empty after clipping")
        annotations.append({
            "id": next_id,
            "category_name": mapping[source],
            "bbox": [left, top, right - left, bottom - top],
            "area": (right - left) * (bottom - top),
            "iscrowd": 0,
        })
        next_id += 1
    return annotations, next_id


def convert(image_dir: Path, label_dir: Path, output: Path,
            *, mappings: list[str], ignored: set[str] | None = None,
            category_map_output: Path | None = None,
            limit: int | None = None) -> dict[str, Any]:
    image_dir = image_dir.resolve()
    label_dir = label_dir.resolve()
    output = output.resolve()
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError("image and label directories must exist")
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise ValueError("limit must be positive")
    mapping = _parse_mapping(mappings)
    ignored = set() if ignored is None else set(ignored)
    if any(not isinstance(item, str) or not item for item in ignored):
        raise ValueError("ignored KITTI classes must be non-empty strings")
    if set(mapping) & ignored:
        raise ValueError("a KITTI class cannot be both mapped and ignored")

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError("image directory contains no supported images")

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("KITTI conversion requires Pillow") from error

    target_labels = sorted(set(mapping.values()))
    category_ids = {label: index for index, label in enumerate(target_labels, 1)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_annotation_id = 1
    for image_id, image_path in enumerate(image_paths, 1):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"missing KITTI label file for {image_path.name}")
        with Image.open(image_path) as image:
            width, height = image.size
        image_record = {
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        }
        rows, next_annotation_id = _parse_label_file(
            label_path, mapping, ignored, width, height, next_annotation_id
        )
        for row in rows:
            row["image_id"] = image_id
        images.append(image_record)
        annotations.extend(rows)

    # Convert target names to deterministic integer category IDs only after
    # all input files have been validated and the target vocabulary is known.
    categories = [{"id": category_ids[label], "name": label} for label in target_labels]
    for row in annotations:
        row["category_id"] = category_ids[row.pop("category_name")]
    result = {
        "info": {"description": "KITTI subset converted for ResNet10 accuracy gating", "version": "1"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "source": {
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "class_mapping": mapping,
            "ignored_classes": sorted(ignored),
            "resnet10_supported_labels": sorted(SUPPORTED_OUTPUT_LABELS),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    map_path = category_map_output.resolve() if category_map_output else output.with_name("category-map.json")
    map_path.write_text(json.dumps({str(category_ids[label]): label for label in target_labels}, indent=2) + "\n", encoding="utf-8")
    result["category_map_path"] = str(map_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map", dest="mappings", action="append", required=True,
                        metavar="KITTI_CLASS=RESNET10_LABEL")
    parser.add_argument("--ignore", dest="ignored", action="append", default=[],
                        metavar="KITTI_CLASS")
    parser.add_argument("--category-map-output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = convert(args.image_dir, args.label_dir, args.output,
                     mappings=args.mappings, ignored=set(args.ignored),
                     category_map_output=args.category_map_output, limit=args.limit)
    print(json.dumps({"images": len(result["images"]), "annotations": len(result["annotations"]),
                      "output": str(args.output.resolve()), "category_map": result["category_map_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
