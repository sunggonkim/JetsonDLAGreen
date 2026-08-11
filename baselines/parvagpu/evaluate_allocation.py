#!/usr/bin/env python3
"""Execute a feasible ParvaGPU fixed-layout allocation on Thor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time
from typing import Any


MODEL_ENGINES = {
    "resnet-producer": "resnet10-detection",
    "distilbert-background": "distilbert-sst2",
}


def state(pid: int) -> str:
    text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return text[text.rfind(")") + 2]


def wait_paused(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"worker exited before barrier: {stdout} {stderr}")
        if state(process.pid) in {"T", "t"}:
            return
        time.sleep(0.01)
    raise TimeoutError("worker start barrier timed out")


def summarize(
    allocation: dict[str, Any], results: dict[str, dict[str, Any]], slo_ms: float,
    offered_rps: int = 500,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in allocation["allocation"]:
        model = item["model"]
        result = results[model]
        p99 = float(result["release_to_completion"]["p99_ms"])
        rows.append({
            "model": model,
            "physical_segment_gpc": item["physical_segment_gpc"],
            "offered_rps": offered_rps,
            "served_rps": result["throughput_per_second"],
            "p99_ms": p99,
            "slo_ms": slo_ms,
            "slo_met": p99 <= slo_ms,
        })
    return {
        "schema_version": 1,
        "system": "ParvaGPU",
        "status": "measured-smoke",
        "all_slos_met": all(row["slo_met"] for row in rows),
        "services": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--allocation", type=pathlib.Path, required=True)
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--offered-rps", type=int, default=500)
    parser.add_argument("--slo-ms", type=float, default=3.0)
    parser.add_argument("--duration-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.offered_rps <= 0 or args.slo_ms <= 0 or args.duration_seconds <= 0:
        raise ValueError("invalid evaluation contract")
    repo = args.repo.resolve()
    allocation_path = args.allocation.resolve()
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    if allocation.get("system") != "ParvaGPU" or allocation.get("feasible") is not True:
        raise ValueError("ParvaGPU allocation is not executable")
    if sorted(item["physical_segment_gpc"] for item in allocation["allocation"]) != [1, 2]:
        raise ValueError("allocation must consume the fixed 2g+1g layout")
    mig = dict(
        line.split("=", 1)
        for line in pathlib.Path("/tmp/jdg-mps-1g/mig.env")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    uuid = {1: mig["JDG_MIG_SMALL_UUID"], 2: mig["JDG_MIG_BIG_UUID"]}
    args.result_dir.mkdir(parents=True)
    workers: list[tuple[str, subprocess.Popen[str]]] = []
    try:
        for cpu, item in enumerate(allocation["allocation"]):
            model = item["model"]
            model_name = MODEL_ENGINES[model]
            size = item["physical_segment_gpc"]
            engine = repo / f"models/engines/mig-{size}g-q100/{model_name}.engine"
            environment = os.environ.copy()
            environment.update({
                "CUDA_VISIBLE_DEVICES": uuid[size],
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100",
            })
            command = [
                "taskset", "--cpu-list", str(cpu), str(repo / "build-r39/jdg-trt-bench"),
                "--engine", str(engine), "--model-name", model_name, "--role", "pressure",
                "--duration-seconds", str(args.duration_seconds), "--burst-size", "1",
                "--period-ms", str(1000.0 / args.offered_rps), "--warmup", "20",
                "--include-transfers", "true", "--priority", "default",
                "--start-paused", "true",
            ]
            workers.append((model, subprocess.Popen(
                command, cwd=repo, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )))
        for _, process in workers:
            wait_paused(process)
        for _, process in workers:
            os.kill(process.pid, signal.SIGCONT)
        results: dict[str, dict[str, Any]] = {}
        for model, process in workers:
            stdout, stderr = process.communicate(timeout=args.duration_seconds + 30)
            (args.result_dir / f"{model}.stderr").write_text(stderr, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"{model} worker failed with {process.returncode}")
            results[model] = json.loads(stdout)
            (args.result_dir / f"{model}.json").write_text(
                json.dumps(results[model], indent=2) + "\n", encoding="utf-8"
            )
    finally:
        for _, process in workers:
            if process.poll() is None:
                process.kill()
                process.communicate()
    summary = summarize(allocation, results, args.slo_ms, args.offered_rps)
    summary["offered_rps_per_service"] = args.offered_rps
    summary["allocation"] = {
        "path": str(allocation_path),
        "sha256": hashlib.sha256(allocation_path.read_bytes()).hexdigest(),
    }
    (args.result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
