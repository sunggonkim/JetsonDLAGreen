#!/usr/bin/env python3
"""Generate EuroSys paper figures from experiment summaries."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER_FIG_DIR = ROOT / "paper" / "eurosys27" / "figures"


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_pdf_png(name: str) -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PAPER_FIG_DIR / f"{name}.pdf")
    plt.savefig(PAPER_FIG_DIR / f"{name}.png", dpi=220)
    plt.close()


def plot_p99_slowdown(full_summary: dict[str, Any], mig_summary: dict[str, Any]) -> None:
    full_cases = ["native", "native-priority", "mps-q25", "mps-q50", "mps-q100"]
    mig_cases = ["cross-mps-q25", "cross-mps-q50", "cross-mps-q100"]

    labels = ["NoMIG-Native", "NoMIG-Priority", "NoMIG-Q25", "NoMIG-Q50", "NoMIG-Q100", "CrossMIG-Q25", "CrossMIG-Q50", "CrossMIG-Q100"]
    values = [
        full_summary["cases"][c]["critical_p99_slowdown"] for c in full_cases
    ] + [
        mig_summary["cases"][c]["critical_p99_slowdown"] for c in mig_cases
    ]

    colors = ["#c0392b"] * 5 + ["#1f77b4"] * 3

    plt.figure(figsize=(8.2, 3.3))
    plt.axhline(1.10, color="#2c3e50", linestyle="--", linewidth=1.2, label="SLO target (1.10x)")
    plt.bar(labels, values, color=colors)
    plt.ylabel("Critical p99 slowdown")
    plt.xticks(rotation=22, ha="right")
    plt.ylim(0, max(values) * 1.15)
    plt.legend(frameon=False, loc="upper right")
    save_pdf_png("p99_slowdown_matrix")


def plot_deadline_miss(full_summary: dict[str, Any], mig_summary: dict[str, Any]) -> None:
    full_cases = ["native", "native-priority", "mps-q25", "mps-q50", "mps-q100"]
    mig_cases = ["cross-mps-q25", "cross-mps-q50", "cross-mps-q100"]
    labels = ["NoMIG-Native", "NoMIG-Priority", "NoMIG-Q25", "NoMIG-Q50", "NoMIG-Q100", "CrossMIG-Q25", "CrossMIG-Q50", "CrossMIG-Q100"]

    values = [
        full_summary["cases"][c]["critical_deadline_miss_rate"]["mean"] for c in full_cases
    ] + [
        mig_summary["cases"][c]["critical_deadline_miss_rate"]["mean"] for c in mig_cases
    ]

    colors = ["#e67e22"] * 5 + ["#16a085"] * 3

    plt.figure(figsize=(8.2, 3.3))
    plt.bar(labels, values, color=colors)
    plt.ylabel("Deadline miss rate")
    plt.xticks(rotation=22, ha="right")
    plt.ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    save_pdf_png("deadline_miss_matrix")


def plot_governor_tradeoff(governor_summary: dict[str, Any]) -> None:
    policies_obj = governor_summary["policies"]
    if isinstance(policies_obj, list):
        policy_map = {item["name"]: item for item in policies_obj}
        extractor = lambda item, key: float(item[key])
    else:
        policy_map = policies_obj
        extractor = lambda item, key: float(item[key]["mean"])

    preferred = ["static-q25", "static-q100", "time-division", "profiled", "joint-governor", "jdg-governor"]
    available = set(policy_map.keys())
    policy_names = [name for name in preferred if name in available]
    if not policy_names:
        raise ValueError("no supported policies found in governor summary")

    short = {
        "static-q25": "Q25",
        "static-q100": "Q100",
        "time-division": "TD",
        "profiled": "Profiled",
        # Raw IDs remain accepted as input for provenance, but the only
        # proposed-system label exposed in figures is QUIET.
        "joint-governor": "QUIET",
        "jdg-governor": "QUIET",
    }
    dmr = [extractor(policy_map[p], "deadline_miss_rate") for p in policy_names]
    gp = [extractor(policy_map[p], "pressure_goodput_per_second") for p in policy_names]

    plt.figure(figsize=(5.0, 3.8))
    for p, x, y in zip(policy_names, gp, dmr):
        is_joint = p in {"joint-governor", "jdg-governor"}
        color = "#d62728" if is_joint else "#7f8c8d"
        size = 90 if is_joint else 55
        plt.scatter(x, y, s=size, c=color)
        plt.text(x, y, short[p], fontsize=8, ha="left", va="bottom")

    plt.xlabel("Pressure goodput (req/s)")
    plt.ylabel("Deadline miss rate")
    plt.grid(alpha=0.25)
    save_pdf_png("governor_tradeoff")


def plot_partition_ladder(
    full_summary: dict[str, Any], mig_summary: dict[str, Any], green_summary: dict[str, Any]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))

    green_labels = ["Green-Compute", "Green-Memory"]
    green_values = [
        green_summary["cases"]["green-compute"]["interference_p99_slowdown"],
        green_summary["cases"]["green-memory"]["interference_p99_slowdown"],
    ]
    axes[0].axhline(1.10, color="#2c3e50", linestyle="--", linewidth=1.0)
    axes[0].bar(green_labels, green_values, color=["#4c78a8", "#72b7b2"])
    axes[0].set_title("Soft Partition (CUDA microbenchmark)")
    axes[0].set_ylabel("p99 slowdown vs isolated")
    axes[0].set_ylim(0, max(green_values) * 1.25)
    axes[0].tick_params(axis="x", rotation=15)

    dnn_labels = ["NoMIG-Native", "NoMIG-Q25", "SameMIG-Q25", "CrossMIG-Q25"]
    dnn_values = [
        full_summary["cases"]["native"]["critical_p99_slowdown"],
        full_summary["cases"]["mps-q25"]["critical_p99_slowdown"],
        mig_summary["cases"]["same-mps-q25"]["critical_p99_slowdown"],
        mig_summary["cases"]["cross-mps-q25"]["critical_p99_slowdown"],
    ]
    axes[1].axhline(1.10, color="#2c3e50", linestyle="--", linewidth=1.0)
    axes[1].bar(
        dnn_labels,
        dnn_values,
        color=["#c0392b", "#e67e22", "#6c5ce7", "#16a085"],
    )
    axes[1].set_title("Hard Partition (TensorRT multimodal)")
    axes[1].set_ylim(0, max(dnn_values) * 1.15)
    axes[1].tick_params(axis="x", rotation=20)

    save_pdf_png("partition_ladder")


def main() -> int:
    full_summary = load(ROOT / "results" / "p6-full-multimodal-20260806T140104Z" / "summary.json")
    mig_summary = load(ROOT / "results" / "p5-mig-multimodal-formal-20260806T134221Z" / "summary.json")
    green_summary = load(ROOT / "results" / "p2-full-20260806T123128Z" / "summary.json")
    p7_candidates = sorted(
        (ROOT / "results").glob("p7-multimodal-governor-*/summary.json")
    )
    if p7_candidates:
        governor_summary = load(p7_candidates[-1])
    else:
        governor_summary = load(ROOT / "results" / "p4-governor-final-summary.json")

    plot_p99_slowdown(full_summary, mig_summary)
    plot_deadline_miss(full_summary, mig_summary)
    plot_governor_tradeoff(governor_summary)
    plot_partition_ladder(full_summary, mig_summary, green_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
