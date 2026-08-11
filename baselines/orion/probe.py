#!/usr/bin/env python3
"""Build pinned Orion and probe native TensorRT LD_PRELOAD compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
from typing import Any


UPSTREAM_COMMIT = "20f9469764fb96d94ce23a8e70615196e9ce4ba1"
FIDELITY = "native-interposition-port"


def run(command: list[str], cwd: pathlib.Path, **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, **kwargs)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source: pathlib.Path) -> None:
    completed = run(["git", "rev-parse", "HEAD"], source, check=True)
    if completed.stdout.strip() != UPSTREAM_COMMIT:
        raise ValueError("Orion source is not the pinned upstream commit")


def cuda_toolchain(cuda_root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    include = cuda_root / "targets" / "sbsa-linux" / "include"
    library = cuda_root / "targets" / "sbsa-linux" / "lib"
    if not (cuda_root / "bin" / "nvcc").is_file() or not (include / "cublas.h").is_file():
        raise ValueError("selected CUDA root lacks nvcc or cublas.h")
    return include, library


def build_orion(source: pathlib.Path, cuda_root: pathlib.Path) -> dict[str, Any]:
    include, library = cuda_toolchain(cuda_root)
    include_argument = f"{include} -I/usr/include/aarch64-linux-gnu"
    common = [f"CUDAINCLUDE={include_argument}", f"CUDALIB={library}"]
    capture_dir = source / "src" / "cuda_capture"
    scheduler_dir = source / "src" / "scheduler"
    run(["make", "clean"], capture_dir)
    capture = run(["make", "libinttemp.so", *common], capture_dir)
    run(["make", "clean"], scheduler_dir)
    scheduler = run(
        [
            "make",
            "scheduler_eval.so",
            *common,
            f"NVCC={cuda_root / 'bin' / 'nvcc'}",
        ],
        scheduler_dir,
    )
    capture_so = capture_dir / "libinttemp.so"
    scheduler_so = scheduler_dir / "scheduler_eval.so"
    return {
        "capture_returncode": capture.returncode,
        "scheduler_returncode": scheduler.returncode,
        "capture_log": capture.stdout + capture.stderr,
        "scheduler_log": scheduler.stdout + scheduler.stderr,
        "capture_library": str(capture_so),
        "scheduler_library": str(scheduler_so),
        "capture_sha256": sha256(capture_so) if capture_so.is_file() else None,
        "scheduler_sha256": sha256(scheduler_so) if scheduler_so.is_file() else None,
    }


def classify_probe(returncode: int) -> tuple[str, str]:
    if returncode == 0:
        return (
            "interceptor-process-survived",
            "survival alone does not prove Orion scheduling; operation profiles are still required",
        )
    if returncode < 0 or returncode in (134, 139):
        return (
            "requires-orion-managed-client-integration",
            "the native TensorRT process cannot initialize Orion's thread queues via LD_PRELOAD alone",
        )
    return "probe-failed", "the injected TensorRT process returned a nonzero status"


def probe(
    source: pathlib.Path,
    cuda_root: pathlib.Path,
    benchmark: pathlib.Path,
    engine: pathlib.Path,
    mig_uuid: str,
) -> dict[str, Any]:
    verify_source(source)
    build = build_orion(source, cuda_root)
    capture_library = pathlib.Path(build["capture_library"])
    if build["capture_returncode"] != 0 or build["scheduler_returncode"] != 0:
        status, reason = "build-failed", "one or more pinned Orion libraries failed to build"
        execution = None
    else:
        command = [
            str(benchmark),
            "--engine", str(engine),
            "--model-name", "resnet50-v2",
            "--role", "benchmark",
            "--samples", "1",
            "--warmup", "0",
            "--burst-size", "1",
            "--period-ms", "20",
            "--deadline-ms", "6",
            "--priority", "high",
            "--include-transfers", "true",
            "--trace", "/tmp/orion-tensorrt-probe.csv",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = mig_uuid
        environment["LD_PRELOAD"] = str(capture_library)
        completed = subprocess.run(
            command,
            cwd=benchmark.parent,
            env=environment,
            text=True,
            capture_output=True,
            timeout=15,
        )
        status, reason = classify_probe(completed.returncode)
        execution = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return {
        "schema_version": 1,
        "system": "Orion",
        "provenance": {"upstream_commit": UPSTREAM_COMMIT, "fidelity": FIDELITY},
        "cuda_root": str(cuda_root),
        "build": build,
        "execution": execution,
        "status": status,
        "reason": reason,
        "numeric_comparison_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--cuda-root", type=pathlib.Path, default=pathlib.Path("/usr/local/cuda-13.0"))
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--engine", type=pathlib.Path, required=True)
    parser.add_argument("--mig-uuid", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = probe(
        args.source.resolve(), args.cuda_root.resolve(), args.benchmark.resolve(),
        args.engine.resolve(), args.mig_uuid
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
