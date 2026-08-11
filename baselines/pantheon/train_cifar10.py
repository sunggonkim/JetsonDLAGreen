#!/usr/bin/env python3
"""Regenerate Pantheon's public CIFAR-10/ResNet50 early-exit model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


EXPECTED_COMMIT = "1caa4321fe9f9902ffacb78978f11a32a7a62f64"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)
PROFILE_COLUMNS = (
    "exit", "block_latency_ms", "branch_latency_ms", "accuracy",
    "block_mem_mib", "branch_mem_mib",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise ValueError("invalid percentile input")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def bounded(loader: Iterable[Any], maximum: int | None) -> Iterable[Any]:
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield batch


def output_logits(value: Any) -> Any:
    while value.ndim > 2 and value.shape[1] == 1:
        value = value.squeeze(1)
    if value.ndim != 2 or value.shape[1] != 10:
        raise ValueError(f"unexpected Pantheon output shape: {tuple(value.shape)}")
    return value


def evaluate(model: Any, loader: Iterable[Any], device: Any,
             maximum: int | None, exit_index: int | None = None) -> dict[str, Any]:
    import torch

    model.eval()
    if exit_index is not None:
        model.set_exit_idx(exit_index)
    correct = count = 0
    losses: list[float] = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in bounded(loader, maximum):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = output_logits(model(images))
            losses.append(float(criterion(logits, labels)))
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += labels.numel()
    if not count:
        raise ValueError("empty Pantheon evaluation")
    return {
        "samples": count,
        "accuracy": correct / count,
        "mean_loss": sum(losses) / len(losses),
    }


def train_epoch(model: Any, loader: Iterable[Any], optimizer: Any,
                scheduler: Any, device: Any, maximum: int | None,
                early_exits: bool) -> dict[str, Any]:
    import torch

    model.train()
    if early_exits:
        model.blocks.eval()
        model.branches.train()
    criterion = torch.nn.CrossEntropyLoss()
    count = correct = 0
    loss_sum = 0.0
    steps = 0
    for images, labels in bounded(loader, maximum):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if early_exits:
            outputs = model(images)
            if not isinstance(outputs, list) or not outputs:
                raise ValueError("Pantheon EEN did not return active exits")
            logits = [output_logits(value) for value in outputs]
            loss = torch.stack([criterion(value, labels) for value in logits]).sum()
            predictions = logits[-1]
        else:
            predictions = output_logits(model(images))
            loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        count += labels.numel()
        correct += int((predictions.argmax(dim=1) == labels).sum())
        loss_sum += float(loss.detach())
        steps += 1
    if not steps:
        raise ValueError("empty Pantheon training epoch")
    return {
        "steps": steps,
        "samples": count,
        "accuracy": correct / count,
        "mean_loss": loss_sum / steps,
        "last_lr": optimizer.param_groups[0]["lr"],
    }


def profile_module(module: Any, sample: Any, iterations: int) -> dict[str, float]:
    import torch

    module.eval()
    stream = torch.cuda.current_stream()
    with torch.inference_mode():
        for _ in range(20):
            module(sample)
        stream.synchronize()
        values: list[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            module(sample)
            end.record(stream)
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": percentile(values, 0.50),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
    }


def parameter_mib(module: Any) -> float:
    return sum(value.numel() * value.element_size() for value in module.parameters()) / 2**20


def export_and_profile(een: Any, dummy: Any, output: Path,
                       iterations: int) -> list[dict[str, Any]]:
    import torch

    modules = output / "modules"
    modules.mkdir()
    records: list[dict[str, Any]] = []
    intermediate = dummy
    with torch.inference_mode():
        for index, (block, branch) in enumerate(zip(een.blocks, een.branches)):
            block.eval()
            branch.eval()
            block_input = intermediate
            block_profile = profile_module(block, block_input, iterations)
            torch.jit.save(torch.jit.trace(block, block_input),
                           modules / f"block_{index:02d}.pth")
            intermediate = block(block_input)
            branch_profile = profile_module(branch, intermediate, iterations)
            torch.jit.save(torch.jit.trace(branch, intermediate),
                           modules / f"branch_{index:02d}.pth")
            records.append({
                "exit": index,
                "block": block_profile,
                "branch": branch_profile,
                "block_mem_mib": parameter_mib(block),
                "branch_mem_mib": parameter_mib(branch),
            })
    return records


def make_loaders(data_dir: Path, batch_size: int, workers: int) -> tuple[Any, Any, Any]:
    import torch
    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    full = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    train, validation = random_split(
        full, [45_000, 5_000], generator=torch.Generator().manual_seed(0)
    )
    test = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(0), **common,
    )
    validation_loader = DataLoader(validation, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test, shuffle=False, drop_last=False, **common)
    return train_loader, validation_loader, test_loader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pantheon", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--backbone-epochs", type=int, default=100)
    parser.add_argument("--exit-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--profile-iterations", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing existing Pantheon output: {args.output}")
    if min(args.backbone_epochs, args.exit_epochs, args.batch_size,
           args.profile_iterations) <= 0:
        raise ValueError("Pantheon numeric arguments must be positive")
    commit = subprocess.check_output(
        ["git", "-C", str(args.pantheon), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError("Pantheon upstream commit differs")

    offline = args.pantheon / "offline"
    sys.path.insert(0, str(offline))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from common.lr_scheduler import WarmupCosineLR
    from dnn.image_classification.resnet import resnet50
    from een.een import EEN

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Pantheon training requires exactly one visible CUDA device")
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.output.mkdir(parents=True)
    train_loader, validation_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, args.workers
    )
    device = torch.device("cuda:0")

    backbone = resnet50().to(device)
    optimizer = torch.optim.SGD(
        backbone.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, momentum=0.9, nesterov=True,
    )
    total_steps = args.backbone_epochs * (args.max_train_batches or len(train_loader))
    scheduler = WarmupCosineLR(
        optimizer, warmup_epochs=max(2, round(total_steps * 0.3)),
        max_epochs=max(3, total_steps),
    )
    history: dict[str, list[dict[str, Any]]] = {"backbone": [], "exits": []}
    started = time.monotonic_ns()
    for epoch in range(args.backbone_epochs):
        record = train_epoch(
            backbone, train_loader, optimizer, scheduler, device,
            args.max_train_batches, False,
        )
        record.update({
            "epoch": epoch,
            "validation": evaluate(
                backbone, validation_loader, device, args.max_eval_batches
            ),
        })
        history["backbone"].append(record)
        print(f"backbone epoch {epoch + 1}/{args.backbone_epochs}: {record}", flush=True)
    torch.save(backbone.state_dict(), args.output / "backbone-state.pt")

    backbone_cpu = backbone.to("cpu").eval()
    dummy_cpu = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        reference = backbone_cpu(dummy_cpu)
    een = EEN(backbone_cpu, dummy_cpu, "image_classification", num_classes=10)
    een.set_exit_idx(len(een.branches) - 1)
    with torch.inference_mode():
        reconstructed = output_logits(een(dummy_cpu))
    initial_partition_max_abs_error = float((reference - reconstructed).abs().max())
    if initial_partition_max_abs_error != 0.0:
        raise ValueError("Pantheon partition changed the full model output")
    een.set_exit_idx(-1)
    een.deactivate_branch(-1)
    een.freeze_backbone()
    een = een.to(device)
    exit_parameters = [value for value in een.parameters() if value.requires_grad]
    optimizer = torch.optim.SGD(
        exit_parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
        momentum=0.9, nesterov=True,
    )
    total_steps = args.exit_epochs * (args.max_train_batches or len(train_loader))
    scheduler = WarmupCosineLR(
        optimizer, warmup_epochs=max(2, round(total_steps * 0.3)),
        max_epochs=max(3, total_steps),
    )
    for epoch in range(args.exit_epochs):
        record = train_epoch(
            een, train_loader, optimizer, scheduler, device,
            args.max_train_batches, True,
        )
        record["epoch"] = epoch
        history["exits"].append(record)
        print(f"exit epoch {epoch + 1}/{args.exit_epochs}: {record}", flush=True)

    torch.save(een.state_dict(), args.output / "early-exit-state.pt")
    exit_metrics = [
        evaluate(een, test_loader, device, args.max_eval_batches, index)
        for index in range(len(een.branches))
    ]
    profile = export_and_profile(
        een, dummy_cpu.to(device), args.output, args.profile_iterations
    )
    for record, metrics in zip(profile, exit_metrics):
        record["accuracy"] = metrics["accuracy"]
        record["test_samples"] = metrics["samples"]
    with (args.output / "profile.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        for record in profile:
            writer.writerow({
                "exit": record["exit"],
                "block_latency_ms": record["block"]["mean_ms"],
                "branch_latency_ms": record["branch"]["mean_ms"],
                "accuracy": record["accuracy"],
                "block_mem_mib": record["block_mem_mib"],
                "branch_mem_mib": record["branch_mem_mib"],
            })

    een = een.to("cpu").eval()
    een.set_exit_idx(len(een.branches) - 1)
    reference_backbone = resnet50().eval()
    reference_backbone.load_state_dict(
        torch.load(args.output / "backbone-state.pt", map_location="cpu",
                   weights_only=True)
    )
    with torch.inference_mode():
        reference = reference_backbone(dummy_cpu)
        reconstructed = output_logits(een(dummy_cpu))
    full_output_max_abs_error = float((reference - reconstructed).abs().max())
    if full_output_max_abs_error != 0.0:
        raise ValueError("Pantheon exit training changed the frozen full model")

    source_files = [
        offline / "pretrain.py", offline / "construct.py", offline / "lit_module.py",
        offline / "een/een.py", offline / "een/branch/image_classification.py",
        offline / "een/block_extraction/computation_graph.py",
        offline / "een/block_extraction/graph_partition.py",
        offline / "dnn/image_classification/resnet.py",
        offline / "dnn/image_classification/data.py",
    ]
    formal_contract = (
        args.backbone_epochs == 100 and args.exit_epochs == 100
        and args.max_train_batches is None and args.max_eval_batches is None
        and args.batch_size == 256 and args.learning_rate == 1e-2
        and args.weight_decay == 1e-2
    )
    dataset_files = [
        args.data_dir / "cifar-10-python.tar.gz",
        *sorted((args.data_dir / "cifar-10-batches-py").glob("data_batch_*")),
        args.data_dir / "cifar-10-batches-py/test_batch",
        args.data_dir / "cifar-10-batches-py/batches.meta",
    ]
    if any(not path.is_file() for path in dataset_files):
        raise ValueError("Pantheon CIFAR-10 evidence is incomplete")
    module_files = sorted((args.output / "modules").glob("*.pth"))
    if len(module_files) != 2 * len(profile):
        raise ValueError("Pantheon exported module count differs")
    accuracy_gate_passed = exit_metrics[-1]["accuracy"] >= 0.90
    result = {
        "schema_version": 1,
        "kind": "pantheon-cifar10-resnet50-training",
        "system": "Pantheon",
        "status": "ok",
        "formal_training_contract": formal_contract,
        "accuracy_gate_passed": accuracy_gate_passed,
        "numeric_comparison_allowed": False,
        "upstream_commit": commit,
        "task": "image_classification",
        "dataset": "CIFAR-10",
        "model": "Pantheon upstream ResNet50",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "multiprocessors": torch.cuda.get_device_properties(0).multi_processor_count,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "config": vars(args) | {
            "pantheon": str(args.pantheon),
            "output": str(args.output),
            "data_dir": str(args.data_dir),
        },
        "full_output_max_abs_error": full_output_max_abs_error,
        "initial_partition_max_abs_error": initial_partition_max_abs_error,
        "exits": profile,
        "history": history,
        "elapsed_seconds": (time.monotonic_ns() - started) / 1.0e9,
        "source_sha256": {
            str(path.relative_to(args.pantheon)): sha256(path) for path in source_files
        },
        "dataset_sha256": {
            str(path.relative_to(args.data_dir)): sha256(path) for path in dataset_files
        },
        "artifacts": {
            "backbone_state_sha256": sha256(args.output / "backbone-state.pt"),
            "early_exit_state_sha256": sha256(args.output / "early-exit-state.pt"),
            "profile_sha256": sha256(args.output / "profile.csv"),
            "module_sha256": {
                path.name: sha256(path) for path in module_files
            },
        },
        "next_gate": (
            "full training and held-out accuracy" if not formal_contract
            else "Pantheon online runtime under the shared arrival trace"
        ),
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "formal_training_contract": formal_contract,
        "full_output_max_abs_error": full_output_max_abs_error,
        "exit_accuracies": [value["accuracy"] for value in exit_metrics],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
