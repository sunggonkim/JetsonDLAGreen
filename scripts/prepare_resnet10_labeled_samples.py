#!/usr/bin/env python3
"""Prepare externally labelled ResNet10 detector inputs.

The Jetson Multimedia API detector consumes a resized ``640x368`` RGB tensor
in NCHW order, scaled by ``1/255``.  This tool turns an image directory and a
COCO-style annotation file into two immutable manifests:

* ``samples.jsonl`` for :mod:`build_producer_input_trace` (tensor paths/SHA),
* ``dataset-manifest.jsonl`` for the application accuracy gate (expected
  detection JSON).

Category labels are supplied by an explicit category map.  The tool never
derives labels from filenames or model output, and refuses annotations whose
category is absent from that map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


WIDTH = 640
HEIGHT = 368
# ParseFunc_ID=1 in the vendor TensorRT sample iterates over classCnt - 1;
# the final labels.txt entry is the background slot, not a decoded detection.
SUPPORTED_DETECTION_LABELS = frozenset({"Car", "RoadSign", "TwoWheeler"})
SAMPLE_KEYS = {"iteration", "sample_id", "path", "input_sha256"}
DATASET_KEYS = {"schema_version", "sample_id", "input_sha256", "expected_label"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.resolve().read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _read_category_map(path: Path) -> dict[int, str]:
    value = _read_json(path, "category map")
    if not isinstance(value, dict) or not value:
        raise ValueError("category map must be a non-empty JSON object")
    result: dict[int, str] = {}
    for raw_id, label in value.items():
        try:
            category_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise ValueError("category map keys must be integer category IDs") from error
        if category_id <= 0 or not isinstance(label, str) or not label:
            raise ValueError("category map entries are invalid")
        if category_id in result:
            raise ValueError("category map repeats a category ID")
        result[category_id] = label
    return result


def _clip_box(box: list[Any], image_width: int, image_height: int) -> list[int]:
    if len(box) != 4 or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in box):
        raise ValueError("COCO bbox must contain four numeric values")
    x, y, width, height = (float(item) for item in box)
    if not all(math.isfinite(item) for item in (x, y, width, height)) or width <= 0 or height <= 0:
        raise ValueError("COCO bbox must be finite with positive width and height")
    # Match the vendor parser: truncate coordinates before clipping to the
    # fixed network bounds, then serialize xywh.
    left = max(0, min(WIDTH - 1, int(x * WIDTH / image_width)))
    top = max(0, min(HEIGHT - 1, int(y * HEIGHT / image_height)))
    right = max(0, min(WIDTH - 1, int((x + width) * WIDTH / image_width)))
    bottom = max(0, min(HEIGHT - 1, int((y + height) * HEIGHT / image_height)))
    if right <= left or bottom <= top:
        raise ValueError("COCO bbox becomes empty after resize/clipping")
    return [left, top, right - left, bottom - top]


def _expected_detections(
    annotations: list[dict[str, Any]],
    category_map: dict[int, str],
    ignored_categories: set[int],
    image_width: int,
    image_height: int,
) -> str:
    detections: list[dict[str, Any]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("annotation must be an object")
        category_id = annotation.get("category_id")
        if isinstance(category_id, bool) or not isinstance(category_id, int):
            raise ValueError("annotation category_id is invalid")
        if category_id in ignored_categories:
            continue
        if category_id not in category_map:
            raise ValueError(
                f"annotation category {category_id} is absent from the explicit category map"
            )
        detections.append({
            "class": category_map[category_id],
            "box": _clip_box(annotation.get("bbox"), image_width, image_height),
        })
    detections.sort(key=lambda item: (item["class"], item["box"]))
    return json.dumps({"detections": detections}, separators=(",", ":"), sort_keys=True)


def _preprocess(
    image_path: Path,
    *,
    expected_width: int,
    expected_height: int,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {image_path}")
    actual_height, actual_width = image.shape[:2]
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise ValueError(
            f"COCO dimensions differ from decoded image for {image_path}: "
            f"metadata={expected_width}x{expected_height}, "
            f"decoded={actual_width}x{actual_height}"
        )
    resized = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Explicitly preserve the vendor's input_scale={1/255,...} and zero offsets.
    tensor = resized.astype(np.float32) * (1.0 / 255.0)
    return np.ascontiguousarray(np.transpose(tensor, (2, 0, 1)))


def prepare(
    image_root: Path,
    annotations_path: Path,
    category_map_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    ignored_categories: set[int] | None = None,
) -> dict[str, Any]:
    image_root = image_root.resolve()
    annotations_path = annotations_path.resolve()
    category_map_path = category_map_path.resolve()
    output_dir = output_dir.resolve()
    if not image_root.is_dir():
        raise ValueError(f"image root is not a directory: {image_root}")
    if not annotations_path.is_file() or not category_map_path.is_file():
        raise ValueError("annotation and category-map files must exist")
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise ValueError("limit must be positive")
    categories = _read_category_map(category_map_path)
    unsupported_labels = sorted(set(categories.values()) - SUPPORTED_DETECTION_LABELS)
    if unsupported_labels:
        raise ValueError(
            "category map contains labels not emitted by the vendor ResNet10 parser: "
            + ", ".join(unsupported_labels)
        )
    ignored_categories = set() if ignored_categories is None else set(ignored_categories)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in ignored_categories):
        raise ValueError("ignored category IDs must be positive integers")
    if categories.keys() & ignored_categories:
        raise ValueError("a category cannot be both mapped and ignored")
    coco = _read_json(annotations_path, "COCO annotations")
    images = coco.get("images") if isinstance(coco, dict) else None
    annotations = coco.get("annotations") if isinstance(coco, dict) else None
    if not isinstance(images, list) or not isinstance(annotations, list) or not images:
        raise ValueError("COCO annotations require non-empty images and annotations arrays")
    ann_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict) or not isinstance(annotation.get("image_id"), int):
            raise ValueError("annotation image_id is invalid")
        ann_by_image.setdefault(annotation["image_id"], []).append(annotation)
    selected = images if limit is None else images[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(exist_ok=True)
    samples: list[dict[str, Any]] = []
    dataset: list[dict[str, Any]] = []
    image_hashes: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_inodes: set[tuple[int, int]] = set()
    for iteration, image in enumerate(selected):
        if not isinstance(image, dict) or not isinstance(image.get("id"), int):
            raise ValueError("COCO image entry is invalid")
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name or Path(file_name).is_absolute():
            raise ValueError("COCO image file_name must be a relative path")
        image_path = (image_root / file_name).resolve()
        try:
            image_path.relative_to(image_root)
        except ValueError as error:
            raise ValueError("COCO image escapes image root") from error
        if not image_path.is_file():
            raise ValueError(f"COCO image is missing: {file_name}")
        stat = image_path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen_inodes:
            raise ValueError("COCO selection reuses an image inode")
        width = image.get("width")
        height = image.get("height")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("COCO image width is invalid")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError("COCO image height is invalid")
        tensor = _preprocess(
            image_path,
            expected_width=width,
            expected_height=height,
        )
        tensor_path = tensor_dir / f"{iteration:06d}.f32"
        tensor.tofile(tensor_path)
        digest = sha256(tensor_path)
        sample_id = Path(file_name).as_posix()
        if sample_id in seen_ids:
            raise ValueError("COCO image file_name is duplicated")
        expected_label = _expected_detections(
            ann_by_image.get(image["id"], []), categories, ignored_categories, width, height
        )
        samples.append({"iteration": iteration, "sample_id": sample_id, "path": str(tensor_path), "input_sha256": digest})
        dataset.append({"schema_version": 1, "sample_id": sample_id, "input_sha256": digest, "expected_label": expected_label})
        image_hashes.append({"sample_id": sample_id, "image_sha256": sha256(image_path), "tensor_sha256": digest})
        seen_ids.add(sample_id)
        seen_inodes.add(inode)
    sample_list = output_dir / "samples.jsonl"
    dataset_manifest = output_dir / "dataset-manifest.jsonl"
    sample_list.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in samples), encoding="utf-8")
    dataset_manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in dataset), encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "kind": "p9-resnet10-labelled-input-provenance",
        "preprocessing": {
            "source_color": "bgr",
            "model_color": "rgb",
            "resize": [WIDTH, HEIGHT],
            "layout": "NCHW",
            "dtype": "float32",
            "scale": 1.0 / 255.0,
            "offsets": [0, 0, 0],
            "contract_source": "/usr/src/jetson_multimedia_api/samples/common/algorithm/trt/trt_inference.h",
        },
        "annotations": {"path": str(annotations_path), "sha256": sha256(annotations_path)},
        "category_map": {"path": str(category_map_path), "sha256": sha256(category_map_path)},
        "ignored_category_ids": sorted(ignored_categories),
        "image_root": str(image_root),
        "samples": image_hashes,
        "sample_list": {"path": str(sample_list), "sha256": sha256(sample_list)},
        "dataset_manifest": {"path": str(dataset_manifest), "sha256": sha256(dataset_manifest)},
        "labels_external": True,
        "filename_labels_inferred": False,
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="COCO JSON")
    parser.add_argument("--category-map", type=Path, required=True, help="JSON: COCO category ID -> ResNet10 class label")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ignore-category", type=int, action="append", default=[],
                        help="COCO category ID outside the ResNet10 label set; repeatable")
    args = parser.parse_args()
    result = prepare(args.image_root, args.annotations, args.category_map, args.output_dir,
                     limit=args.limit, ignored_categories=set(args.ignore_category))
    print(json.dumps({"samples": len(result["samples"]), "output_dir": str(args.output_dir.resolve()), "provenance": str((args.output_dir / "provenance.json").resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
