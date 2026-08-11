#!/usr/bin/env python3
"""Extract the learned ResNet10 backbone/detection-head dependent DAG.

The vendor ResNet10 detector already contains a learned detection head.  This
tool preserves its weights while splitting the graph at ``Layer6_relu_Y`` so
the backbone and head can execute in different MIG instances.  It deliberately
does not invent a downstream MLP or relabel a checksum-only control as an
application workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SPLIT_TENSOR = "Layer6_relu_Y"
PRODUCER_INPUT = "data"
HEAD_OUTPUTS = ("Layer7_cov", "Layer7_bbox")
BACKBONE_NAME = "resnet10-backbone"
HEAD_NAME = "resnet10-detection-head"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(source: Path, output_dir: Path) -> tuple[Path, Path]:
    try:
        import onnx
        from onnx.utils import extract_model
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the real DAG extractor requires the 'onnx' Python package"
        ) from error

    model = onnx.load(str(source))
    output_names = {output.name for output in model.graph.output}
    if not set(HEAD_OUTPUTS).issubset(output_names):
        raise ValueError("source model lacks the ResNet10 detection outputs")
    value_names = {
        value.name for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
    }
    if SPLIT_TENSOR not in value_names:
        raise ValueError("source model lacks the learned split tensor")

    output_dir.mkdir(parents=True, exist_ok=True)
    backbone = output_dir / f"{BACKBONE_NAME}.onnx"
    head = output_dir / f"{HEAD_NAME}.onnx"
    extract_model(str(source), str(backbone), [PRODUCER_INPUT], [SPLIT_TENSOR])
    extract_model(str(source), str(head), [SPLIT_TENSOR], list(HEAD_OUTPUTS))
    onnx.checker.check_model(onnx.load(str(backbone)))
    onnx.checker.check_model(onnx.load(str(head)))
    return backbone, head


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    device: str | None,
    trtexec: str,
    shapes: str,
) -> None:
    environment = os.environ.copy()
    if device:
        environment["CUDA_VISIBLE_DEVICES"] = device
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--skipInference",
    ]
    if shapes:
        command.extend(
            [f"--minShapes={shapes}", f"--optShapes={shapes}", f"--maxShapes={shapes}"]
        )
    subprocess.run(command, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("models/cache/resnet10-detection.onnx"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-engines", action="store_true")
    parser.add_argument("--producer-device", help="CUDA_VISIBLE_DEVICES for the backbone build")
    parser.add_argument("--consumer-device", help="CUDA_VISIBLE_DEVICES for the head build")
    parser.add_argument("--trtexec", default="trtexec")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file():
        raise ValueError(f"source model is not a regular file: {source}")
    backbone, head = extract(source, args.output_dir.resolve())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "p9-real-resnet10-dependent-dag-artifacts",
        "source_model": {"path": str(source), "sha256": sha256(source)},
        "split_tensor": SPLIT_TENSOR,
        "producer": {
            "model": BACKBONE_NAME,
            "input": PRODUCER_INPUT,
            "output": SPLIT_TENSOR,
            "shape": [1, 512, 23, 40],
            "payload_bytes": 512 * 23 * 40 * 4,
            "onnx": {"path": str(backbone), "sha256": sha256(backbone)},
        },
        "consumer": {
            "model": HEAD_NAME,
            "input": SPLIT_TENSOR,
            "outputs": list(HEAD_OUTPUTS),
            "onnx": {"path": str(head), "sha256": sha256(head)},
        },
        "application": "ResNet10 object detection",
        "accuracy_gate_required": True,
    }
    if args.build_engines:
        producer_engine = args.output_dir / f"{BACKBONE_NAME}.engine"
        consumer_engine = args.output_dir / f"{HEAD_NAME}.engine"
        build_engine(
            backbone,
            producer_engine,
            args.producer_device,
            args.trtexec,
            "data:1x3x368x640",
        )
        build_engine(head, consumer_engine, args.consumer_device, args.trtexec, "")
        manifest["producer"]["engine"] = {
            "path": str(producer_engine),
            "sha256": sha256(producer_engine),
        }
        manifest["consumer"]["engine"] = {
            "path": str(consumer_engine),
            "sha256": sha256(consumer_engine),
        }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
