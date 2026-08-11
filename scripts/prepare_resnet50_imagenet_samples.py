#!/usr/bin/env python3
"""Prepare externally labelled ImageNet samples for the split ResNet-50 DAG.

The image filename may carry an ImageNet WNID (for example
``n03792782_123.JPEG-224.jpg``), but the label is never inferred from that
string.  ``--synset-map`` is an explicit external annotation contract whose
entries bind each WNID to the model output index, label, and annotation
source.  The tool emits the fixed NCHW float32 tensors consumed by
``build_producer_input_trace.py`` plus the dataset/request manifests consumed
by the application accuracy gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


INPUT_SIZE = 224
RESIZE_SHORT_SIDE = 256
SYNSET_RE = re.compile(r"^(n\d{8})(?:[_./-]|$)")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SAMPLE_KEYS = {"iteration", "sample_id", "path", "input_sha256"}
DATASET_KEYS = {"schema_version", "sample_id", "input_sha256", "expected_label"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.resolve().read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _read_synset_map(path: Path) -> dict[str, dict[str, Any]]:
    value = _read_json(path, "synset map")
    if not isinstance(value, dict) or not value:
        raise ValueError("synset map must be a non-empty JSON object")
    result: dict[str, dict[str, Any]] = {}
    indices: set[int] = set()
    for synset, entry in value.items():
        if not isinstance(synset, str) or not re.fullmatch(r"n\d{8}", synset):
            raise ValueError("synset map keys must be ImageNet WNIDs")
        if not isinstance(entry, dict) or set(entry) != {"index", "label", "source"}:
            raise ValueError("synset map entries require index, label, and source")
        index = entry["index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 1000:
            raise ValueError(f"synset map index is invalid for {synset}")
        label = entry["label"]
        source = entry["source"]
        if not isinstance(label, str) or not label.strip() or not isinstance(source, str) or not source.strip():
            raise ValueError(f"synset map annotation is invalid for {synset}")
        if index in indices:
            raise ValueError("synset map reuses a model output index")
        indices.add(index)
        result[synset] = {"index": index, "label": label, "source": source}
    return result


def _synset(path: Path, image_root: Path | None = None, *, from_parent: bool = False) -> str:
    match = SYNSET_RE.match(path.name)
    if match is not None:
        return match.group(1)
    if from_parent and image_root is not None:
        try:
            relative = path.resolve().relative_to(image_root.resolve())
        except ValueError as error:
            raise ValueError(f"image is outside image root: {path}") from error
        for component in reversed(relative.parts[:-1]):
            if re.fullmatch(r"n\d{8}", component):
                return component
    raise ValueError(f"image filename lacks an ImageNet WNID: {path.name}")


def _preprocess(path: Path) -> np.ndarray:
    try:
        image = Image.open(path).convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot decode image: {path}") from error
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions are invalid: {path}")
    scale = RESIZE_SHORT_SIDE / min(width, height)
    resized = image.resize(
        (max(INPUT_SIZE, round(width * scale)), max(INPUT_SIZE, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    left = (resized.width - INPUT_SIZE) // 2
    top = (resized.height - INPUT_SIZE) // 2
    cropped = resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    pixels = np.asarray(cropped, dtype=np.float32)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = ((pixels / 255.0) - mean) / std
    tensor = np.transpose(tensor, (2, 0, 1))
    if tensor.shape != (3, INPUT_SIZE, INPUT_SIZE) or not np.isfinite(tensor).all():
        raise ValueError(f"preprocessed tensor is invalid: {path}")
    return np.ascontiguousarray(tensor, dtype=np.float32)


def prepare(
    image_root: Path,
    synset_map: Path,
    output_dir: Path,
    *,
    limit_per_synset: int | None = None,
    source_archive: Path | None = None,
    wnid_from_parent: bool = False,
) -> dict[str, Any]:
    image_root = image_root.resolve()
    synset_map = synset_map.resolve()
    output_dir = output_dir.resolve()
    if not image_root.is_dir():
        raise ValueError(f"image root is not a directory: {image_root}")
    if source_archive is not None:
        source_archive = source_archive.resolve()
        if not source_archive.is_file():
            raise ValueError(f"source archive is not a file: {source_archive}")
    if limit_per_synset is not None and (
        isinstance(limit_per_synset, bool) or not isinstance(limit_per_synset, int) or limit_per_synset <= 0
    ):
        raise ValueError("limit_per_synset must be positive")
    annotations = _read_synset_map(synset_map)
    images = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError("image root contains no supported images")
    selected: list[tuple[Path, str]] = []
    counts: dict[str, int] = {}
    seen_inodes: set[tuple[int, int]] = set()
    for path in images:
        synset = _synset(path, image_root, from_parent=wnid_from_parent)
        if synset not in annotations:
            raise ValueError(f"image synset is absent from explicit map: {synset}")
        if limit_per_synset is not None and counts.get(synset, 0) >= limit_per_synset:
            continue
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen_inodes:
            raise ValueError(f"image inode is reused: {path}")
        seen_inodes.add(inode)
        selected.append((path, synset))
        counts[synset] = counts.get(synset, 0) + 1
    if not selected:
        raise ValueError("no images selected")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(exist_ok=True)
    samples: list[dict[str, Any]] = []
    dataset: list[dict[str, Any]] = []
    image_hashes: list[dict[str, Any]] = []
    for iteration, (path, synset) in enumerate(selected):
        tensor_path = tensor_dir / f"{iteration:06d}.f32"
        _preprocess(path).tofile(tensor_path)
        digest = sha256(tensor_path)
        sample_id = path.relative_to(image_root).as_posix()
        annotation = annotations[synset]
        samples.append({"iteration": iteration, "sample_id": sample_id, "path": str(tensor_path), "input_sha256": digest})
        dataset.append({"schema_version": 1, "sample_id": sample_id, "input_sha256": digest, "expected_label": annotation["label"]})
        image_hashes.append({
            "sample_id": sample_id,
            "synset": synset,
            "image_sha256": sha256(path),
            "tensor_sha256": digest,
            "label": annotation["label"],
            "label_index": annotation["index"],
            "label_source": annotation["source"],
        })
    sample_list = output_dir / "samples.jsonl"
    dataset_manifest = output_dir / "dataset-manifest.jsonl"
    class_map = output_dir / "class-map.json"
    sample_list.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in samples), encoding="utf-8")
    dataset_manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in dataset), encoding="utf-8")
    class_map.write_text(json.dumps({str(v["index"]): v["label"] for v in annotations.values()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "kind": "p9-resnet50-imagenet-labelled-input-provenance",
        "preprocessing": {
            "input_shape": [1, 3, 224, 224],
            "resize_short_side": 256,
            "crop": "center",
            "color": "RGB",
            "layout": "NCHW",
            "dtype": "float32",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "contract_source": "ONNX Model Zoo ResNet-50 v2 preprocessing",
            "wnid_source": "explicit synset map; filename or dataset parent directory",
        },
        "image_root": str(image_root),
        "synset_map": {"path": str(synset_map), "sha256": sha256(synset_map)},
        "samples": image_hashes,
        "sample_list": {"path": str(sample_list), "sha256": sha256(sample_list)},
        "dataset_manifest": {"path": str(dataset_manifest), "sha256": sha256(dataset_manifest)},
        "class_map": {"path": str(class_map), "sha256": sha256(class_map)},
        "labels_external": True,
        "filename_labels_inferred": False,
        "wnid_from_parent_directory": wnid_from_parent,
    }
    if source_archive is not None:
        provenance["source_archive"] = {
            "path": str(source_archive),
            "sha256": sha256(source_archive),
        }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"samples": samples, "dataset": dataset, "provenance": provenance}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--synset-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-synset", type=int)
    parser.add_argument(
        "--wnid-from-parent",
        action="store_true",
        help="read the WNID from an ImageNet-style class directory when filenames lack it",
    )
    parser.add_argument(
        "--source-archive", type=Path,
        help="optional archive whose digest explains how image-root was materialized",
    )
    args = parser.parse_args()
    result = prepare(
        args.image_root,
        args.synset_map,
        args.output_dir,
        limit_per_synset=args.limit_per_synset,
        source_archive=args.source_archive,
        wnid_from_parent=args.wnid_from_parent,
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "samples": len(result["samples"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
