#!/usr/bin/env python3
"""Generate QUIET paper figures exclusively from recorded experiment JSON."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


COLORS = {
    "static-q5": "#6b7280",
    "static-q25": "#d55e00",
    "priority-q25": "#cc79a7",
    "conservative-guard": "#56b4e9",
    "profiled-guard": "#009e73",
    "joint-governor": "#0072b2",
}
LABELS = {
    "static-q5": "MPS q5",
    "static-q25": "MPS q25",
    "priority-q25": "Priority q25",
    "conservative-guard": "Conservative",
    "profiled-guard": "Profiled",
    "joint-governor": "QUIET",
}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def load(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported schema in {path}")
    return data


def ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0
    error = T95.get(len(values) - 1, 1.96) * statistics.stdev(values) / math.sqrt(
        len(values)
    )
    return mean, error


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, output: pathlib.Path, name: str) -> None:
    fig.tight_layout(pad=0.5)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def positive_errors(mean: float, error: float) -> list[list[float]]:
    lower = min(error, mean * 0.85)
    return [[lower], [error]]


def log_floor_errors(
    means: list[float], errors: list[float], floor: float = 0.001
) -> tuple[list[float], list[list[float]]]:
    displayed = [max(value, floor) for value in means]
    lower = [
        min(error, value - floor * 0.5)
        for value, error in zip(displayed, errors)
    ]
    return displayed, [lower, errors]


def motivation(aggregate: dict[str, Any], output: pathlib.Path) -> None:
    names = ["static-q5", "static-q25", "priority-q25", "joint-governor"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    x = list(range(len(names)))
    for axis, metric, scale, ylabel in (
        (axes[0], "deadline_miss_rate", 100.0, "Deadline misses (%)"),
        (axes[1], "critical_p99_ms_max", 1.0, "Maximum epoch p99 (ms)"),
    ):
        means = [aggregate["policies"][name][metric]["mean"] * scale for name in names]
        errors = [aggregate["policies"][name][metric]["ci95"] * scale for name in names]
        axis.bar(x, means, color=[COLORS[name] for name in names], width=0.68)
        for index, (mean, error) in enumerate(zip(means, errors)):
            axis.errorbar(
                [index], [mean], yerr=positive_errors(mean, error), fmt="none",
                ecolor="black", capsize=2, linewidth=0.7,
            )
        axis.set_yscale("log")
        axis.set_xticks(x, [LABELS[name] for name in names], rotation=22, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25, which="both")
    axes[0].set_title("(a) Deadline protection")
    axes[1].set_title("(b) Tail latency")
    save(fig, output, "motivation_interference")


def tradeoff(aggregate: dict[str, Any], output: pathlib.Path) -> None:
    fig, axis = plt.subplots(figsize=(3.35, 2.45))
    for name in LABELS:
        policy = aggregate["policies"][name]
        x = policy["pressure_goodput_per_second"]["mean"]
        y = policy["deadline_miss_rate"]["mean"] * 100.0
        xerr = policy["pressure_goodput_per_second"]["ci95"]
        yerr = policy["deadline_miss_rate"]["ci95"] * 100.0
        axis.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=positive_errors(y, yerr),
            marker="o",
            markersize=5,
            capsize=2,
            color=COLORS[name],
            label=LABELS[name],
        )
    axis.set_yscale("log")
    axis.set_xlabel("Best-effort goodput (requests/s)")
    axis.set_ylabel("Deadline misses (%)")
    axis.grid(alpha=0.25, which="both")
    axis.legend(ncol=2, frameon=False, loc="best")
    save(fig, output, "main_tradeoff")


def raw_policy(run: dict[str, Any], name: str) -> dict[str, Any]:
    return next(policy for policy in run["policies"] if policy["name"] == name)


def tenant_scaling(runs: list[dict[str, Any]], output: pathlib.Path) -> None:
    names = ["static-q25", "priority-q25", "profiled-guard", "joint-governor"]
    tenants = [1, 2, 4, 6]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    for name in names:
        dmr_means: list[float] = []
        dmr_errors: list[float] = []
        goodput_means: list[float] = []
        goodput_errors: list[float] = []
        for count in tenants:
            per_run_dmr: list[float] = []
            per_run_goodput: list[float] = []
            for run in runs:
                epochs = [
                    epoch
                    for epoch in raw_policy(run, name)["epochs"]
                    if epoch["offered_tenants"] == count
                ]
                samples = len(epochs) * int(run["config"]["samples_per_epoch"])
                per_run_dmr.append(
                    sum(int(epoch["deadline_misses"]) for epoch in epochs) / samples
                )
                completions = sum(
                    sum(int(value) for value in epoch["completed_by_modality"].values())
                    for epoch in epochs
                )
                elapsed = sum(float(epoch["elapsed_seconds"]) for epoch in epochs)
                per_run_goodput.append(completions / elapsed)
            mean, error = ci(per_run_dmr)
            dmr_means.append(mean * 100.0)
            dmr_errors.append(error * 100.0)
            mean, error = ci(per_run_goodput)
            goodput_means.append(mean)
            goodput_errors.append(error)
        displayed_dmr = [max(value, 0.001) for value in dmr_means]
        dmr_yerr = [
            [min(error, value * 0.85) for value, error in zip(displayed_dmr, dmr_errors)],
            dmr_errors,
        ]
        axes[0].errorbar(
            tenants, displayed_dmr, yerr=dmr_yerr, marker="o", capsize=2,
            color=COLORS[name], label=LABELS[name],
        )
        axes[1].errorbar(
            tenants, goodput_means, yerr=goodput_errors, marker="o", capsize=2,
            color=COLORS[name], label=LABELS[name],
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Deadline misses (%)")
    axes[1].set_ylabel("Best-effort goodput (requests/s)")
    for axis in axes:
        axis.set_xlabel("Offered tenants")
        axis.set_xticks(tenants)
        axis.grid(alpha=0.25, which="both")
    axes[0].set_title("(a) Critical SLO")
    axes[1].set_title("(b) Useful pressure work")
    axes[1].legend(ncol=2, frameon=False)
    save(fig, output, "tenant_scaling")


def gate_timeline(output: pathlib.Path) -> None:
    fig, axis = plt.subplots(figsize=(3.35, 1.65))
    axis.set_xlim(0, 12)
    axis.set_ylim(-0.1, 2.4)
    axis.axis("off")
    axis.text(-0.1, 1.65, "Host", ha="right", va="center", fontweight="bold")
    axis.text(-0.1, 0.55, "GPU", ha="right", va="center", fontweight="bold")
    axis.add_patch(Rectangle((0, 1.4), 3, 0.5, color="#9ca3af"))
    axis.add_patch(Rectangle((3, 1.4), 5, 0.5, color="#f3f4f6", ec="#6b7280", hatch="//"))
    axis.add_patch(Rectangle((8, 1.4), 4, 0.5, color="#9ca3af"))
    axis.add_patch(Rectangle((0, 0.3), 4.8, 0.5, color="#e69f00"))
    axis.add_patch(Rectangle((4.8, 0.3), 3.2, 0.5, color="white", ec="#6b7280", hatch=".."))
    axis.add_patch(Rectangle((8, 0.3), 2.0, 0.5, color="#0072b2"))
    axis.add_patch(Rectangle((10, 0.3), 2.0, 0.5, color="#e69f00"))
    axis.text(1.5, 1.65, "submit", ha="center", va="center", color="white")
    axis.text(5.5, 1.65, "stopped", ha="center", va="center")
    axis.text(10, 1.65, "resume", ha="center", va="center", color="white")
    axis.text(2.4, 0.55, "BE inference", ha="center", va="center")
    axis.text(6.4, 0.55, "drained", ha="center", va="center")
    axis.text(9, 0.55, "critical", ha="center", va="center", color="white")
    axis.text(11, 0.55, "BE", ha="center", va="center")
    axis.add_patch(FancyArrowPatch((3, 2.15), (3, 1.95), arrowstyle="-|>", mutation_scale=8))
    axis.add_patch(FancyArrowPatch((8, 2.15), (8, 1.95), arrowstyle="-|>", mutation_scale=8))
    axis.text(3, 2.22, "SIGSTOP", ha="center", va="bottom")
    axis.text(8, 2.22, "release", ha="center", va="bottom")
    axis.annotate("guard $g$", xy=(5.5, 1.08), ha="center", va="center")
    axis.plot([3, 8], [1.02, 1.02], color="black", linewidth=0.7)
    save(fig, output, "gate_timeline")


def stability(runs: list[dict[str, Any]], output: pathlib.Path) -> None:
    indices = list(range(1, len(runs) + 1))
    goodput: list[float] = []
    dmr: list[float] = []
    temperature: list[float] = []
    for run in runs:
        policy = raw_policy(run, "joint-governor")
        goodput.append(float(policy["pressure_goodput_per_second"]))
        dmr.append(float(policy["deadline_miss_rate"]) * 100.0)
        temperatures = [
            float(epoch["telemetry"]["temperature_c_max"])
            for epoch in policy["epochs"]
            if epoch["telemetry"]["temperature_c_max"] is not None
        ]
        temperature.append(max(temperatures))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    series = (
        (goodput, "Goodput (req/s)"),
        (dmr, "Deadline misses (%)"),
        (temperature, r"Maximum temp. ($^\circ$C)"),
    )
    for axis, (values, label) in zip(axes, series):
        axis.plot(indices, values, marker="o", color=COLORS["joint-governor"])
        axis.axhline(statistics.fmean(values), color="#6b7280", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Repetition")
        axis.set_ylabel(label)
        axis.set_xticks(indices)
        axis.grid(alpha=0.25)
    axes[1].set_yscale("symlog", linthresh=0.001)
    save(fig, output, "stability")


def mig_oracle(mig: dict[str, Any], full: dict[str, Any], output: pathlib.Path) -> None:
    def metric(data: dict[str, Any], name: str) -> float:
        cases = data.get("cases", data.get("policies", data.get("results", [])))
        items = cases.items() if isinstance(cases, dict) else (("", case) for case in cases)
        for case_name, case in items:
            if case_name == name or case.get("name") == name or case.get("case") == name:
                for key in ("p99_slowdown", "critical_p99_slowdown"):
                    if key in case:
                        return float(case[key])
        raise KeyError(f"missing {name} in oracle input")

    # P5/P6 summaries have stable, documented schemas but different key names.
    values = [
        metric(full, "native"),
        metric(full, "mps-q25"),
        metric(mig, "same-mps-q25"),
        metric(mig, "cross-mps-q25"),
    ]
    labels = ["Full\nnative", "Full\nMPS q25", "Same MIG\nMPS q25", "Cross MIG\nMPS q25"]
    fig, axis = plt.subplots(figsize=(3.35, 2.25))
    axis.bar(range(4), values, color=["#d55e00", "#e69f00", "#56b4e9", "#009e73"])
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_xticks(range(4), labels)
    axis.set_ylabel("Critical p99 slowdown")
    axis.grid(axis="y", alpha=0.25)
    save(fig, output, "mig_oracle")


def sensitivity(data: dict[str, Any], output: pathlib.Path) -> None:
    guard = data["experiments"]["guard"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25))
    x = [point["x"] for point in guard]
    dmr = [point["metrics"]["deadline_miss_rate"]["mean"] * 100.0 for point in guard]
    dmr_error = [point["metrics"]["deadline_miss_rate"]["ci95"] * 100.0 for point in guard]
    displayed_dmr, dmr_yerr = log_floor_errors(dmr, dmr_error)
    axes[0].errorbar(x, displayed_dmr, yerr=dmr_yerr, marker="o", capsize=2, color=COLORS["joint-governor"])
    axes[0].set_yscale("log")
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=0.8, label="DMR target")
    axes[0].legend(frameon=False)
    goodput = [point["metrics"]["pressure_goodput_per_second"]["mean"] for point in guard]
    goodput_error = [point["metrics"]["pressure_goodput_per_second"]["ci95"] for point in guard]
    axes[1].errorbar(x, goodput, yerr=goodput_error, marker="o", capsize=2, color=COLORS["joint-governor"])
    for axis, ylabel in zip(axes, ("Deadline misses (%)", "Best-effort goodput (requests/s)")):
        axis.set_xlabel("Fixed guard (ms)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, which="both")
    save(fig, output, "guard_sensitivity")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex="col")
    for column, label in enumerate(("burst", "period")):
        points = data["experiments"][label]
        x = [point["x"] for point in points]
        dmr = [point["metrics"]["deadline_miss_rate"]["mean"] * 100.0 for point in points]
        dmr_error = [point["metrics"]["deadline_miss_rate"]["ci95"] * 100.0 for point in points]
        goodput = [point["metrics"]["pressure_goodput_per_second"]["mean"] for point in points]
        goodput_error = [point["metrics"]["pressure_goodput_per_second"]["ci95"] for point in points]
        displayed_dmr, dmr_yerr = log_floor_errors(dmr, dmr_error)
        axes[0, column].errorbar(x, displayed_dmr, yerr=dmr_yerr, marker="o", capsize=2, color=COLORS["joint-governor"])
        axes[1, column].errorbar(x, goodput, yerr=goodput_error, marker="o", capsize=2, color=COLORS["joint-governor"])
        axes[0, column].set_yscale("log")
        axes[0, column].axhline(0.05, color="black", linestyle="--", linewidth=0.8)
        axes[0, column].grid(alpha=0.25, which="both")
        axes[1, column].grid(alpha=0.25)
    axes[0, 0].set_ylabel("Deadline misses (%)")
    axes[1, 0].set_ylabel("Goodput (requests/s)")
    axes[1, 0].set_xlabel("Burst size (requests)")
    axes[1, 1].set_xlabel("Burst period (ms)")
    axes[0, 0].set_title("Constant mean arrival rate")
    axes[0, 1].set_title("Eight-request burst")
    save(fig, output, "burst_sensitivity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=pathlib.Path, required=True)
    parser.add_argument("--runs", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--sensitivity", type=pathlib.Path)
    parser.add_argument("--mig", type=pathlib.Path, required=True)
    parser.add_argument("--full", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    setup_style()
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = load(args.aggregate)
    runs = [load(path) for path in args.runs]
    motivation(aggregate, args.output)
    tradeoff(aggregate, args.output)
    tenant_scaling(runs, args.output)
    gate_timeline(args.output)
    stability(runs, args.output)
    mig_oracle(load(args.mig), load(args.full), args.output)
    if args.sensitivity:
        sensitivity(load(args.sensitivity), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
