#!/usr/bin/env python3
"""Prepare a learned ResNet-50 classification DAG for QUIET.

The ONNX model shipped by the ONNX Model Zoo stores weights both as graph
inputs and as initializers.  ``onnx.utils.extract_model`` drops unused graph
inputs while retaining the corresponding initializers, which makes the
extracted graph fail validation (and, consequently, fail a TensorRT build).
This tool extracts a producer/head split and restores the initializer input
declarations before checking or building either graph.

This is an artifact-preparation tool, not an accuracy oracle.  A dataset label
manifest and reference/candidate traces are still required by
``analysis/verify_application_accuracy.py`` before a numeric result is
promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SOURCE_DEFAULT = Path("models/cache/resnet50-v2.onnx")
PRODUCER_INPUT = "gpu_0/data_0"
FINAL_OUTPUT = "gpu_0/softmax_1"
SPLIT_TENSOR = "gpu_0/res4_5_branch2c_bn_2"
PRODUCER_NAME = "resnet50-backbone"
HEAD_NAME = "resnet50-classification-head"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_model(extract_model: Any, source: Path, output: Path,
                   inputs: list[str], outputs: list[str]) -> None:
    """Extract one graph, tolerating ONNX's post-write checker failure.

    Some ONNX releases raise after writing the output because an initializer
    is not also declared as a graph input.  We normalize and check that output
    immediately afterwards, so swallowing an exception is safe only when the
    file was actually produced.
    """
    try:
        extract_model(str(source), str(output), inputs, outputs)
    except Exception:
        if not output.is_file() or output.stat().st_size == 0:
            raise


def _shape(value_info: Any) -> list[int | str]:
    tensor = value_info.type.tensor_type
    if not tensor.HasField("shape"):
        return []
    result: list[int | str] = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append("dynamic")
    return result


def normalize_graph(onnx: Any, helper: Any, path: Path) -> dict[str, Any]:
    """Declare every initializer as an input and validate the graph."""
    model = onnx.load(str(path))
    declared = {value.name for value in model.graph.input}
    added = 0
    for initializer in model.graph.initializer:
        if initializer.name in declared:
            continue
        model.graph.input.append(
            helper.make_tensor_value_info(
                initializer.name,
                initializer.data_type,
                list(initializer.dims),
            )
        )
        declared.add(initializer.name)
        added += 1
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    inputs = {
        value.name: _shape(value)
        for value in model.graph.input
    }
    outputs = {
        value.name: _shape(value)
        for value in model.graph.output
    }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "graph_inputs": len(model.graph.input),
        "graph_nodes": len(model.graph.node),
        "initializer_inputs_added": added,
        "inputs": inputs,
        "outputs": outputs,
    }


def extract(source: Path, output_dir: Path, *, split_tensor: str = SPLIT_TENSOR,
            producer_input: str = PRODUCER_INPUT,
            final_output: str = FINAL_OUTPUT) -> tuple[Path, Path, dict[str, Any]]:
    try:
        import onnx
        from onnx import helper
        from onnx.utils import extract_model
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("the ResNet50 DAG tool requires the 'onnx' package") from error

    source_model = onnx.load(str(source))
    names = {
        value.name
        for value in (*source_model.graph.input, *source_model.graph.output,
                      *source_model.graph.value_info)
    }
    # Older ONNX exports omit intermediate tensors from value_info and expose
    # them only as node inputs/outputs.  They are still valid extraction
    # boundaries, so include the complete graph name set here.
    for node in source_model.graph.node:
        names.update(node.input)
        names.update(node.output)
    if producer_input not in names:
        raise ValueError(f"source model lacks producer input {producer_input!r}")
    if final_output not in names:
        raise ValueError(f"source model lacks final output {final_output!r}")
    if split_tensor not in names:
        raise ValueError(f"source model lacks split tensor {split_tensor!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    producer = output_dir / f"{PRODUCER_NAME}.onnx"
    head = output_dir / f"{HEAD_NAME}.onnx"
    _extract_model(extract_model, source, producer, [producer_input], [split_tensor])
    _extract_model(extract_model, source, head, [split_tensor], [final_output])
    producer_info = normalize_graph(onnx, helper, producer)
    head_info = normalize_graph(onnx, helper, head)
    return producer, head, {"producer": producer_info, "consumer": head_info}


def build_engine(onnx_path: Path, engine_path: Path, device: str | None,
                 trtexec: str, shape: str) -> None:
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
    if shape:
        command.extend([f"--minShapes={shape}", f"--optShapes={shape}",
                        f"--maxShapes={shape}"])
    subprocess.run(command, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-tensor", default=SPLIT_TENSOR)
    parser.add_argument("--build-engines", action="store_true")
    parser.add_argument("--producer-device")
    parser.add_argument("--consumer-device")
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--dataset-manifest", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise ValueError(f"source model is not a regular file: {source}")
    if args.dataset_manifest is not None and not args.dataset_manifest.resolve().is_file():
        raise ValueError(f"dataset manifest is not a regular file: {args.dataset_manifest}")

    producer, head, graph_info = extract(
        source,
        output_dir,
        split_tensor=args.split_tensor,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "p9-real-resnet50-dependent-dag-artifacts",
        "proposed_system": "QUIET",
        "source_model": {"path": str(source), "sha256": sha256(source)},
        "split_tensor": args.split_tensor,
        "producer": {
            "model": PRODUCER_NAME,
            "input": PRODUCER_INPUT,
            "output": args.split_tensor,
            "shape": graph_info["producer"]["outputs"].get(args.split_tensor, []),
            "onnx": graph_info["producer"],
        },
        "consumer": {
            "model": HEAD_NAME,
            "input": args.split_tensor,
            "output": FINAL_OUTPUT,
            "shape": graph_info["consumer"]["inputs"].get(args.split_tensor, []),
            "output_shape": graph_info["consumer"]["outputs"].get(FINAL_OUTPUT, []),
            "onnx": graph_info["consumer"],
        },
        "application": "ImageNet image classification",
        "accuracy_gate_required": True,
        "accuracy_gate": {
            "status": "pending",
            "verifier": "analysis/verify_application_accuracy.py",
        },
    }
    if args.dataset_manifest is not None:
        manifest["dataset_manifest"] = {
            "path": str(args.dataset_manifest.resolve()),
            "sha256": sha256(args.dataset_manifest.resolve()),
        }
    if args.build_engines:
        producer_engine = output_dir / f"{PRODUCER_NAME}.engine"
        head_engine = output_dir / f"{HEAD_NAME}.engine"
        build_engine(producer, producer_engine, args.producer_device,
                     args.trtexec, "gpu_0/data_0:1x3x224x224")
        shape = graph_info["consumer"]["inputs"].get(args.split_tensor, [])
        # The head input shape is the producer output shape.  TensorRT accepts
        # the same shape string when the split tensor is the only data input.
        head_input_shape = ""
        if shape and all(isinstance(item, int) and item > 0 for item in shape):
            head_input_shape = f"{args.split_tensor}:{'x'.join(map(str, shape))}"
        build_engine(head, head_engine, args.consumer_device, args.trtexec,
                     head_input_shape)
        manifest["producer"]["engine"] = {
            "path": str(producer_engine), "sha256": sha256(producer_engine),
        }
        manifest["consumer"]["engine"] = {
            "path": str(head_engine), "sha256": sha256(head_engine),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
