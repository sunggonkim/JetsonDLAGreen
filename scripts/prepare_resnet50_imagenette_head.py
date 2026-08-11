#!/usr/bin/env python3
"""Train and export a reproducible learned ResNet-50 Imagenette head.

The ResNet-50 backbone remains the pinned ONNX Model Zoo model.  This tool
trains only a linear classifier over its global-average-pooled split tensor
using a deterministic, label-bound training slice from Imagenette.  It emits
the learned head, a composed unsplit reference graph, a compact class map,
and provenance sufficient to replay the CPU reference before a TensorRT run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, compose, helper, numpy_helper

import prepare_resnet50_imagenet_samples as sample_preparer


SPLIT_TENSOR = "gpu_0/res4_5_branch2c_bn_2"
HEAD_INPUT = SPLIT_TENSOR
HEAD_OUTPUT = "imagenette_logits"
CLASS_MAP_NAME = "class-map.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_map(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.resolve().read_bytes())
    if not isinstance(value, dict) or not value:
        raise ValueError("synset map must be a non-empty JSON object")
    result: dict[str, dict[str, Any]] = {}
    for synset, entry in value.items():
        if not isinstance(synset, str) or not isinstance(entry, dict):
            raise ValueError("synset map entry is invalid")
        if set(entry) != {"index", "label", "source"}:
            raise ValueError("synset map entries require index, label, and source")
        if not isinstance(entry["index"], int) or isinstance(entry["index"], bool):
            raise ValueError("synset map ImageNet index is invalid")
        if not isinstance(entry["label"], str) or not entry["label"]:
            raise ValueError("synset map label is invalid")
        result[synset] = entry
    return result


def image_files(root: Path, classes: list[str], limit: int) -> list[tuple[Path, int]]:
    if limit <= 0:
        raise ValueError("training limit must be positive")
    result: list[tuple[Path, int]] = []
    for class_index, synset in enumerate(classes):
        directory = root.resolve() / synset
        if not directory.is_dir():
            raise ValueError(f"dataset class directory is missing: {directory}")
        files = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in sample_preparer.IMAGE_SUFFIXES
        )
        if len(files) < limit:
            raise ValueError(f"dataset class {synset} has fewer than {limit} images")
        result.extend((path, class_index) for path in files[:limit])
    return result


def extract_features(
    session: ort.InferenceSession,
    files: list[tuple[Path, int]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    input_name = session.get_inputs()[0].name
    features: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    for path, label in files:
        tensor = sample_preparer._preprocess(path)
        output = session.run(None, {input_name: tensor[None, ...]})[0]
        if output.shape != (1, 1024, 14, 14):
            raise ValueError(f"backbone split shape differs for {path}: {output.shape}")
        pooled = np.asarray(output.mean(axis=(2, 3))[0], dtype=np.float64)
        if not np.isfinite(pooled).all():
            raise ValueError(f"backbone feature contains non-finite values: {path}")
        tensor_bytes = np.ascontiguousarray(tensor, dtype=np.float32).tobytes()
        features.append(pooled)
        labels.append(label)
        records.append({
            "sample_id": path.relative_to(path.parents[1]).as_posix(),
            "path": str(path.resolve()),
            "image_sha256": sha256(path),
            "input_tensor_sha256": hashlib.sha256(tensor_bytes).hexdigest(),
            "head_label_index": label,
        })
    return np.asarray(features), np.asarray(labels, dtype=np.int64), records


def train_ridge(
    features: np.ndarray, labels: np.ndarray, class_count: int, regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("training feature dimensions differ")
    if regularization <= 0.0 or not np.isfinite(regularization):
        raise ValueError("regularization must be positive and finite")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    normalized = (features - mean) / scale
    design = np.concatenate([normalized, np.ones((len(normalized), 1))], axis=1)
    targets = np.eye(class_count, dtype=np.float64)[labels]
    system = design.T @ design + regularization * np.eye(design.shape[1])
    weights = np.linalg.solve(system, design.T @ targets)
    # Fold training normalization into the ONNX head so the runtime input is
    # the original pooled activation, not a hidden CPU-side transformation.
    folded_weights = weights[:-1] / scale[:, None]
    folded_bias = weights[-1] - (mean / scale) @ weights[:-1]
    return folded_weights, folded_bias, mean, scale


def make_head(weights: np.ndarray, bias: np.ndarray) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        HEAD_INPUT, TensorProto.FLOAT, [1, 1024, 14, 14]
    )
    output_info = helper.make_tensor_value_info(
        HEAD_OUTPUT, TensorProto.FLOAT, [1, int(bias.size)]
    )
    weight = numpy_helper.from_array(np.asarray(weights, dtype=np.float32), "head_weight")
    bias_tensor = numpy_helper.from_array(np.asarray(bias, dtype=np.float32), "head_bias")
    nodes = [
        helper.make_node("ReduceMean", [HEAD_INPUT], ["pooled"], axes=[2, 3], keepdims=0),
        helper.make_node("MatMul", ["pooled", "head_weight"], ["projected"]),
        helper.make_node("Add", ["projected", "head_bias"], [HEAD_OUTPUT]),
    ]
    graph = helper.make_graph(
        nodes, "imagenette-learned-head", [input_info], [output_info],
        initializer=[weight, bias_tensor],
    )
    model = helper.make_model(
        graph, producer_name="QUIET Imagenette head preparation", opset_imports=[helper.make_opsetid("", 13)]
    )
    onnx.checker.check_model(model)
    return model


def compose_reference(backbone_path: Path, head_path: Path) -> onnx.ModelProto:
    backbone = onnx.load(str(backbone_path))
    head = onnx.load(str(head_path))
    # The extracted Model Zoo backbone is an IR-v3 graph that declares its
    # weights as graph inputs.  Mirror that legacy declaration while invoking
    # the composer; remove only the head constants again after the merge and
    # upgrade the composed container to IR-v10.
    head_initializer_names = {initializer.name for initializer in head.graph.initializer}
    declared = {value.name for value in head.graph.input}
    for initializer in head.graph.initializer:
        if initializer.name not in declared:
            head.graph.input.append(helper.make_tensor_value_info(
                initializer.name, initializer.data_type, list(initializer.dims)
            ))
    # ONNX's compose helper requires a common container IR even though the
    # head graph uses only standard ops.  The backbone's IR is authoritative
    # for the composed reference artifact.
    head.ir_version = backbone.ir_version
    del head.opset_import[:]
    head.opset_import.extend(backbone.opset_import)
    reference = compose.merge_models(
        backbone, head, io_map=[(SPLIT_TENSOR, HEAD_INPUT)],
        prefix1="backbone/", prefix2="head/",
    )
    merged_initializer_names = {initializer.name for initializer in reference.graph.initializer}
    retained_inputs = [
        value for value in reference.graph.input
        if value.name not in merged_initializer_names
    ]
    del reference.graph.input[:]
    reference.graph.input.extend(retained_inputs)
    reference.ir_version = 10
    onnx.checker.check_model(reference)
    return reference


def build_engine(onnx_path: Path, engine_path: Path, trtexec: str, device: str | None) -> None:
    environment = os.environ.copy()
    if device:
        environment["CUDA_VISIBLE_DEVICES"] = device
    subprocess.run([
        trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}",
        "--noTF32", "--skipInference",
    ], check=True, env=environment)


def prepare(
    backbone_path: Path, train_root: Path, synset_map_path: Path, output_dir: Path,
    *, train_limit: int, regularization: float, trtexec: str | None,
    producer_device: str | None, consumer_device: str | None,
) -> dict[str, Any]:
    backbone_path = backbone_path.resolve()
    train_root = train_root.resolve()
    synset_map_path = synset_map_path.resolve()
    output_dir = output_dir.resolve()
    if not backbone_path.is_file() or not train_root.is_dir():
        raise ValueError("backbone or training root is missing")
    mapping = load_map(synset_map_path)
    classes = sorted(mapping)
    files = image_files(train_root, classes, train_limit)
    session = ort.InferenceSession(str(backbone_path), providers=["CPUExecutionProvider"])
    features, labels, records = extract_features(session, files)
    weights, bias, mean, scale = train_ridge(
        features, labels, len(classes), regularization,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / "resnet50-imagenette-head.onnx"
    reference_path = output_dir / "resnet50-imagenette-unsplit.onnx"
    onnx.save(make_head(weights, bias), str(head_path))
    onnx.save(compose_reference(backbone_path, head_path), str(reference_path))
    class_map = {str(index): mapping[synset]["label"] for index, synset in enumerate(classes)}
    class_map_path = output_dir / CLASS_MAP_NAME
    class_map_path.write_text(json.dumps(class_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    training_path = output_dir / "training-samples.jsonl"
    training_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    model_manifest = {
        "schema_version": 1,
        "kind": "p9-resnet50-imagenette-learned-head",
        "proposed_system": "QUIET",
        "backbone": {"path": str(backbone_path), "sha256": sha256(backbone_path)},
        "synset_map": {"path": str(synset_map_path), "sha256": sha256(synset_map_path)},
        "classes": classes,
        "class_map": {"path": str(class_map_path), "sha256": sha256(class_map_path)},
        "training": {
            "dataset_root": str(train_root),
            "samples_per_class": train_limit,
            "sample_order": "lexicographic path prefix per WNID",
            "feature_tensor": SPLIT_TENSOR,
            "feature_shape": [1, 1024, 14, 14],
            "pooling": "global average over axes 2,3",
            "normalization": "training feature mean/std folded into head",
            "solver": "multi-output ridge regression",
            "regularization": regularization,
            "samples": len(records),
            "training_samples": {"path": str(training_path), "sha256": sha256(training_path)},
        },
        "head": {"path": str(head_path), "sha256": sha256(head_path), "output": HEAD_OUTPUT},
        "unsplit_reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if trtexec is not None:
        build_engine(
            backbone_path, output_dir / "resnet50-imagenette-backbone.engine",
            trtexec, producer_device,
        )
        build_engine(
            head_path, output_dir / "resnet50-imagenette-head.engine",
            trtexec, consumer_device,
        )
    return model_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--synset-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=20)
    parser.add_argument("--regularization", type=float, default=100.0)
    parser.add_argument("--trtexec")
    parser.add_argument("--producer-device")
    parser.add_argument("--consumer-device")
    args = parser.parse_args()
    result = prepare(
        args.backbone, args.train_root, args.synset_map, args.output_dir,
        train_limit=args.train_limit, regularization=args.regularization,
        trtexec=args.trtexec, producer_device=args.producer_device,
        consumer_device=args.consumer_device,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "classes": len(result["classes"]),
        "training_samples": result["training"]["samples"],
        "head_sha256": result["head"]["sha256"],
        "unsplit_reference_sha256": result["unsplit_reference"]["sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
