#!/usr/bin/env python3
"""Build Pantheon TorchScript modules from the current ImageNette split.

This is a mechanical adapter around the pinned Pantheon online runtime.  It
does not replace Pantheon's scheduler or block executor.  The source ONNX
artifacts are explicitly hashed so the generated block/exit repository cannot
silently drift from the current QUIET application model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    output: Path,
    backbone: Path,
    head: Path,
    *,
    block_latency_us: int,
    branch_latency_us: int,
    accuracy: float,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing existing Pantheon asset directory: {output}")
    if block_latency_us <= 0 or branch_latency_us <= 0:
        raise ValueError("Pantheon profile latencies must be positive")
    if not 0.0 < accuracy <= 1.0:
        raise ValueError("Pantheon branch accuracy must be in (0, 1]")
    for path in (backbone, head):
        if not path.is_file():
            raise ValueError(f"missing ImageNette ONNX input: {path}")

    # Imports are intentionally lazy: the repository's Python unit tests do
    # not require the external Pantheon CUDA wheel.
    import torch
    from onnx2torch import convert

    output.mkdir(parents=True)
    model_root = output / "model-repository" / "resnet50-imagenette"
    model_files = model_root / "model_files"
    model_files.mkdir(parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise ValueError("ImageNette Pantheon assets require one CUDA device")

    specs = (
        ("block_00.pth", backbone, (1, 3, 224, 224)),
        ("branch_00.pth", head, (1, 1024, 14, 14)),
    )
    generated: dict[str, dict[str, Any]] = {}
    for filename, source, shape in specs:
        module = convert(str(source)).eval().to(device)
        example = torch.zeros(shape, device=device)
        traced = torch.jit.trace(module, example, strict=True)
        target = model_files / filename
        traced.save(str(target))
        generated[filename] = {
            "path": str(target.resolve()),
            "sha256": sha256(target),
            "source_path": str(source.resolve()),
            "source_sha256": sha256(source),
            "input_shape": list(shape),
            "device": str(device),
        }

    model_root.joinpath("config.pbtxt").write_text(
        'name: "resnet50-imagenette"\n'
        "dims: 1\ndims: 3\ndims: 224\ndims: 224\n"
        f"block_profile {{ id: 0 latency: {block_latency_us} }}\n"
        f"exit_profile {{ id: 0 latency: {branch_latency_us} "
        f"accuracy: {accuracy:.9g} }}\n",
        encoding="ascii",
    )
    result = {
        "schema_version": 1,
        "kind": "pantheon-resnet50-imagenette-torchscript-adapter",
        "model": "resnet50-imagenette",
        "source_models": {
            "backbone": {"path": str(backbone.resolve()), "sha256": sha256(backbone)},
            "head": {"path": str(head.resolve()), "sha256": sha256(head)},
        },
        "model_repository": str((output / "model-repository").resolve()),
        "config_path": str((model_root / "config.pbtxt").resolve()),
        "profile": {
            "block_latency_us": block_latency_us,
            "branch_latency_us": branch_latency_us,
            "branch_accuracy": accuracy,
            "profile_source": "explicit-run-contract",
        },
        "generated_modules": generated,
        "numeric_comparison_allowed": False,
        "claim_guard": (
            "Pantheon pinned online runtime adapter; promotion requires raw "
            "common-workload input/output and accuracy verification"
        ),
    }
    (output / "adapter.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--block-latency-us", type=int, required=True)
    parser.add_argument("--branch-latency-us", type=int, required=True)
    parser.add_argument("--accuracy", type=float, required=True)
    args = parser.parse_args(argv)
    result = build(
        args.output.resolve(), args.backbone.resolve(), args.head.resolve(),
        block_latency_us=args.block_latency_us,
        branch_latency_us=args.branch_latency_us,
        accuracy=args.accuracy,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
