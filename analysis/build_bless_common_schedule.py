#!/usr/bin/env python3
"""Replay TensorRT kernel profiles and form the first BLESS common-workload squad."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines" / "bless"))
from scheduler import KernelProfile, RequestState, choose_configuration, form_kernel_squad  # noqa: E402


SMS_TO_SHARE = {2: 25, 4: 50, 6: 75, 8: 100}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{path} is empty or truncated")
    records = [json.loads(line) for line in raw.splitlines()]
    previous_end = 0
    for index, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or record.get("operation") != index
            or record.get("result") != 0
            or not isinstance(record.get("start_monotonic_ns"), int)
            or not isinstance(record.get("end_monotonic_ns"), int)
            or record["start_monotonic_ns"] < previous_end
            or record["end_monotonic_ns"] <= record["start_monotonic_ns"]
        ):
            raise ValueError(f"{path} record {index} differs")
        previous_end = record["end_monotonic_ns"]
    return records


def model_profiles(root: Path, model: str) -> tuple[list[KernelProfile], dict[str, str]]:
    traces = {sms: load_trace(root / f"{model}-{sms}" / "squad.jsonl") for sms in SMS_TO_SHARE}
    counts = {len(trace) for trace in traces.values()}
    if len(counts) != 1:
        raise ValueError(f"{model} launch counts differ across SM profiles")
    count = counts.pop()
    profiles: list[KernelProfile] = []
    cumulative = {share: 0.0 for share in SMS_TO_SHARE.values()}
    for operation in range(count):
        durations: dict[int, float] = {}
        cumulative_at_operation: dict[int, float] = {}
        for sms, share in SMS_TO_SHARE.items():
            record = traces[sms][operation]
            if record.get("selected_sms") != sms:
                raise ValueError(f"{model} operation {operation} selected SMs differ")
            duration = (record["end_monotonic_ns"] - record["start_monotonic_ns"]) / 1000.0
            durations[share] = duration
            cumulative[share] += duration
            cumulative_at_operation[share] = cumulative[share]
        profiles.append(
            KernelProfile(
                name=f"{model}-op-{operation}",
                cumulative_us=cumulative_at_operation,
                duration_us=durations,
                native_share=100,
            )
        )
    return profiles, {
        str(sms): sha256(root / f"{model}-{sms}" / "squad.jsonl")
        for sms in SMS_TO_SHARE
    }


def build(profile_root: Path) -> dict[str, Any]:
    resnet, resnet_hashes = model_profiles(profile_root, "resnet")
    distilbert, distilbert_hashes = model_profiles(profile_root, "distilbert")
    requests = [
        RequestState("resnet", resnet[-1].cumulative_us[100], 0.0, resnet),
        RequestState("distilbert", distilbert[-1].cumulative_us[100], 0.0, distilbert),
    ]
    squad = form_kernel_squad(requests, maximum_kernels=6)
    decision = choose_configuration(squad, allowed_shares=(25, 50, 75))
    return {
        "schema_version": 1,
        "kind": "bless-thor-common-tensorrt-profile-and-first-squad",
        "status": "profiled",
        "models": {
            "resnet": {"logical_launches": len(resnet), "trace_sha256": resnet_hashes},
            "distilbert": {"logical_launches": len(distilbert), "trace_sha256": distilbert_hashes},
        },
        "squad": [
            {"request_id": item.request_id, "kernel_index": item.kernel_index}
            for item in squad
        ],
        "configuration": {
            "shares": decision.shares,
            "predicted_us": decision.predicted_us,
            "estimator": decision.estimator,
        },
        "numeric_comparison_allowed": False,
        "remaining_gate": "execute every scheduler-selected squad on the common dependent workload",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.profile_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
