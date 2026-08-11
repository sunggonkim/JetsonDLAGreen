#!/usr/bin/env python3
"""Generate the publication figures and tables for the current QUIET evidence.

The script is intentionally fail-closed: it replays the recorded SHA-256
bindings, recomputes pooled tail statistics from request-level CSV files, and
checks the claim scope before writing publication assets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THERMAL = ROOT / "results/p9-resnet50-imagenette-thermal-current-r02-20260811/summary.json"
DEFAULT_FRONTIER = ROOT / "results/p9-resnet50-imagenette-load-frontier-current-r01-20260811/frontier.json"
DEFAULT_VISION = ROOT / "results/p9-resnet50-imagenette-gate100-20260811/accuracy-gate.json"
DEFAULT_WHISPER = ROOT / "results/p9-real-whisper-asr-lex12-20260811/accuracy-gate.json"
DEFAULT_PANTHEON = ROOT / "results/p9-pantheon-resnet50-imagenette-gate100-r01-20260811/verification.json"
DEFAULT_QUIET_CAUSAL = ROOT / "results/p9-real-resnet-head-causal-current-wall-replay-20260811/quiet-causal-repeat-summary.json"
DEFAULT_MPS_CAUSAL = ROOT / "results/p9-real-resnet-head-causal-current-wall-replay-20260811/mps-causal-repeat-summary.json"
DEFAULT_DEADLINE = ROOT / "results/p9-resnet50-imagenette-calibration-current-r06-20260811/deadline-lock.json"
DEFAULT_PLAN = ROOT / "results/p9-resnet50-imagenette-quiet-plan-current-r02-20260811/quiet-plan.json"
DEFAULT_FIGURES = ROOT / "paper/eurosys27/figures"
DEFAULT_GENERATED = ROOT / "paper/eurosys27/generated"

COLORS = {
    "QUIET": "#167D91",
    "NVIDIA MPS": "#D8783D",
    "XSched": "#6E5AA8",
    "ink": "#263238",
    "muted": "#68757B",
    "blue_fill": "#DDEFF5",
    "green_fill": "#DFF1E4",
    "orange_fill": "#F8E8D8",
    "purple_fill": "#E9E2F3",
    "yellow_fill": "#FFF2C7",
    "gray_fill": "#EEF1F2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def recorded_path(raw: str) -> Path:
    """Resolve an absolute recorded path, with a repository-relative fallback."""
    path = Path(raw)
    if path.exists():
        return path
    parts = path.parts
    for marker in ("results", "paper", "models", "docs"):
        if marker in parts:
            candidate = ROOT.joinpath(*parts[parts.index(marker) :])
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"recorded artifact is unavailable: {raw}")


def portable_provenance_path(path: Path) -> str:
    """Prefer a portable repository-relative path in generated metadata."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def verify_ref(ref: dict[str, Any]) -> Path:
    path = recorded_path(str(ref["path"]))
    observed = sha256(path)
    expected = str(ref["sha256"])
    if observed != expected:
        raise ValueError(f"{path}: SHA-256 {observed} != recorded {expected}")
    return path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "legend.fontsize": 7.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": "#D7DEE1",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outputs[0])
    fig.savefig(outputs[1], dpi=240)
    plt.close(fig)
    return outputs


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = COLORS["ink"],
    fontsize: float = 8.0,
    weight: str = "normal",
    radius: float = 0.025,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=0.85,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        weight=weight,
        zorder=zorder + 1,
        linespacing=1.15,
    )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["ink"],
    style: str = "-|>",
    connectionstyle: str = "arc3",
    linewidth: float = 1.1,
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            color=color,
            linewidth=linewidth,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def add_step(ax: plt.Axes, x: float, y: float, number: int) -> None:
    ax.text(
        x,
        y,
        str(number),
        ha="center",
        va="center",
        color="white",
        fontsize=7.3,
        weight="bold",
        bbox={"boxstyle": "circle,pad=0.18", "facecolor": COLORS["ink"], "edgecolor": "none"},
        zorder=8,
    )


def draw_overview(deadline: dict[str, Any], plan: dict[str, Any], output: Path) -> list[Path]:
    selected = plan["selected_plan"]
    deadline_us = float(deadline["deadline_us"])
    slack_us = float(selected["reserved_slack_us"])
    payload = int(deadline["contract"]["payload_bytes"])

    fig, ax = plt.subplots(figsize=(12.2, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    columns = [
        (0.01, 0.18, "1  Profile and lock", COLORS["blue_fill"]),
        (0.205, 0.245, "2  Place stages and edge", COLORS["green_fill"]),
        (0.465, 0.29, "3  Execute by stage state", COLORS["orange_fill"]),
        (0.77, 0.22, "4  Promote evidence", COLORS["purple_fill"]),
    ]
    for x, width, title, fill in columns:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.13),
                width,
                0.80,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor="#FAFBFB",
                edgecolor="#7F8C8D",
                linewidth=1.0,
            )
        )
        ax.add_patch(Rectangle((x, 0.82), width, 0.11, facecolor=fill, edgecolor="none"))
        ax.text(x + width / 2, 0.875, title, ha="center", va="center", weight="bold", fontsize=9.0)

    add_box(ax, (0.032, 0.62), 0.136, 0.115, "Input + arrival traces\nJDGINT1 / JDGARR1", facecolor=COLORS["blue_fill"], fontsize=7.5)
    add_box(ax, (0.032, 0.43), 0.136, 0.115, "Profile production wall\nand edge/stage tails", facecolor=COLORS["gray_fill"], fontsize=7.5)
    add_box(ax, (0.032, 0.235), 0.136, 0.115, f"Freeze deadline\nD = {deadline_us:,.3f} us", facecolor=COLORS["yellow_fill"], fontsize=7.5, weight="bold")
    add_arrow(ax, (0.10, 0.62), (0.10, 0.55))
    add_arrow(ax, (0.10, 0.43), (0.10, 0.36))

    add_box(ax, (0.226, 0.62), 0.087, 0.13, "ResNet-50\nbackbone\n1g MIG", facecolor=COLORS["green_fill"], fontsize=7.4, weight="bold")
    add_box(ax, (0.349, 0.62), 0.077, 0.13, "trained head\n2g MIG", facecolor=COLORS["green_fill"], fontsize=7.4, weight="bold")
    add_arrow(ax, (0.313, 0.685), (0.349, 0.685), color=COLORS["QUIET"], linewidth=1.8)
    ax.text(0.331, 0.735, f"{payload:,} B", ha="center", va="bottom", color=COLORS["QUIET"], fontsize=7.0, weight="bold")
    add_box(ax, (0.226, 0.405), 0.20, 0.115, "shared mmap -> register mapped\nlocal device pointers -> TRT bind", facecolor=COLORS["blue_fill"], fontsize=7.2)
    add_box(ax, (0.226, 0.225), 0.20, 0.10, f"fixed 1g -> 2g plan\nreserved slack = {slack_us:,.3f} us", facecolor=COLORS["yellow_fill"], fontsize=7.4)
    add_arrow(ax, (0.326, 0.62), (0.326, 0.52))
    add_arrow(ax, (0.326, 0.405), (0.326, 0.325))

    add_box(ax, (0.49, 0.635), 0.104, 0.115, "pause + ACK\nbest effort", facecolor="#F8D8CF", fontsize=7.4, weight="bold")
    add_box(ax, (0.627, 0.635), 0.103, 0.115, "producer\nexecute", facecolor=COLORS["orange_fill"], fontsize=7.4, weight="bold")
    add_box(ax, (0.49, 0.405), 0.104, 0.115, "publish payload\n+ visibility", facecolor=COLORS["yellow_fill"], fontsize=7.4, weight="bold")
    add_box(ax, (0.627, 0.405), 0.103, 0.115, "resume BE;\nconsumer runs", facecolor=COLORS["green_fill"], fontsize=7.4, weight="bold")
    add_arrow(ax, (0.594, 0.692), (0.627, 0.692))
    add_arrow(ax, (0.679, 0.635), (0.55, 0.52), connectionstyle="arc3,rad=-0.15")
    add_arrow(ax, (0.594, 0.462), (0.627, 0.462))
    add_box(ax, (0.516, 0.225), 0.188, 0.09, "production wall: release -> completion", facecolor=COLORS["gray_fill"], fontsize=7.2)

    add_box(ax, (0.797, 0.635), 0.165, 0.115, "post-completion\noutput validation", facecolor=COLORS["purple_fill"], fontsize=7.4, weight="bold")
    add_box(ax, (0.797, 0.435), 0.165, 0.115, "raw replay + SHA-256\nType-7 tails + CP95", facecolor=COLORS["gray_fill"], fontsize=7.3)
    add_box(ax, (0.797, 0.235), 0.165, 0.115, "accuracy + native-port\n+ thermal gates", facecolor=COLORS["blue_fill"], fontsize=7.3)
    add_arrow(ax, (0.879, 0.635), (0.879, 0.55))
    add_arrow(ax, (0.879, 0.435), (0.879, 0.35))

    for x, y, number in [(0.184, 0.49, 1), (0.448, 0.49, 2), (0.757, 0.49, 3), (0.982, 0.49, 4)]:
        add_step(ax, x, y, number)

    ax.text(
        0.5,
        0.055,
        "Jetson AGX Thor: isolated MIG execution/local allocations over physically shared coherent SoC DRAM",
        ha="center",
        va="center",
        color=COLORS["muted"],
        fontsize=8.0,
    )
    return save_figure(fig, output)


def draw_timeline(output: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 3.55))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 4.7)
    ax.axis("off")
    lanes = [(4.0, "arrival scheduler"), (3.0, "BE / 1g MIG"), (2.0, "producer / 1g MIG"), (1.0, "consumer / 2g MIG"), (0.0, "CPU verifier")]
    for y, label in lanes:
        ax.hlines(y, 0.8, 9.7, color="#CFD8DC", linewidth=0.8, zorder=0)
        ax.text(0.68, y, label, ha="right", va="center", fontsize=7.5, color=COLORS["ink"])

    add_box(ax, (1.0, 3.76), 1.35, 0.48, "declared arrival", facecolor=COLORS["blue_fill"], fontsize=7.2, radius=0.08)
    add_arrow(ax, (2.35, 4.0), (2.78, 4.0), zorder=1)
    add_box(ax, (2.85, 3.76), 1.25, 0.48, "actual release", facecolor=COLORS["blue_fill"], fontsize=7.2, weight="bold", radius=0.08)

    ax.add_patch(Rectangle((0.9, 2.78), 2.30, 0.44, facecolor="#CFE7D7", edgecolor=COLORS["ink"], linewidth=0.7))
    ax.text(2.05, 3.0, "running", ha="center", va="center", fontsize=7.2)
    add_box(ax, (3.2, 2.76), 1.15, 0.48, "pause / drain", facecolor="#F8D8CF", fontsize=7.1, radius=0.06)
    ax.add_patch(Rectangle((4.35, 2.78), 2.10, 0.44, facecolor="#F2F3F3", edgecolor=COLORS["ink"], linewidth=0.7, hatch="////"))
    ax.text(5.40, 3.0, "quiesced", ha="center", va="center", fontsize=7.2)
    ax.add_patch(Rectangle((6.45, 2.78), 3.18, 0.44, facecolor="#CFE7D7", edgecolor=COLORS["ink"], linewidth=0.7))
    ax.text(8.04, 3.0, "resumed while consumer runs", ha="center", va="center", fontsize=7.2)

    add_box(ax, (4.35, 1.72), 2.10, 0.56, "producer TensorRT\nexecute + publish", facecolor=COLORS["orange_fill"], fontsize=7.2, weight="bold", radius=0.06)
    add_box(ax, (6.45, 0.72), 2.45, 0.56, "direct-bound consumer\nTensorRT execute", facecolor=COLORS["green_fill"], fontsize=7.2, weight="bold", radius=0.06)
    add_box(ax, (8.90, -0.24), 0.72, 0.48, "validate", facecolor=COLORS["purple_fill"], fontsize=6.9, radius=0.06)

    ax.vlines([4.35, 6.45, 8.90], -0.05, 4.25, colors=["#A9B2B6", COLORS["QUIET"], "#A9B2B6"], linestyles=[":", "--", ":"], linewidths=[0.8, 1.3, 0.8])
    ax.text(4.35, 4.43, "pause ACK", ha="center", va="bottom", fontsize=7.0)
    ax.text(6.45, 4.43, "publication = resume issued", ha="center", va="bottom", fontsize=7.0, color=COLORS["QUIET"], weight="bold")
    ax.text(8.90, 4.43, "completion", ha="center", va="bottom", fontsize=7.0)

    add_arrow(ax, (6.45, 2.0), (6.45, 1.28), color=COLORS["QUIET"], linewidth=1.5)
    ax.annotate("gate hold", xy=(4.35, 3.45), xytext=(6.45, 3.45), ha="center", va="center", arrowprops={"arrowstyle": "|-|", "color": "#B54E3C", "linewidth": 1.0}, fontsize=7.0, color="#B54E3C")
    ax.annotate("production wall", xy=(2.85, 0.35), xytext=(8.90, 0.35), ha="center", va="center", arrowprops={"arrowstyle": "|-|", "color": COLORS["ink"], "linewidth": 1.0}, fontsize=7.0)
    return save_figure(fig, output)


def locate_pipeline(record: dict[str, Any]) -> Path:
    summary_path = recorded_path(str(record["path"]))
    expected = str(record["raw_pipeline_sha256"])
    matches = [path for path in summary_path.parent.rglob("pipeline.csv") if sha256(path) == expected]
    require(len(matches) == 1, f"{summary_path}: expected one pipeline.csv matching {expected}, found {len(matches)}")
    return matches[0]


def read_latencies(path: Path) -> tuple[np.ndarray, np.ndarray]:
    latencies: list[float] = []
    misses: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"{path}: missing header")
        require("validation_excluded_end_to_end_us" in reader.fieldnames, f"{path}: missing production-wall field")
        require("deadline_miss" in reader.fieldnames, f"{path}: missing deadline_miss")
        for row in reader:
            latencies.append(float(row["validation_excluded_end_to_end_us"]))
            misses.append(int(row["deadline_miss"]))
    return np.asarray(latencies), np.asarray(misses)


def replay_thermal(thermal: dict[str, Any]) -> dict[str, np.ndarray]:
    require(thermal.get("formal") is True, "thermal aggregate is not formal")
    require(thermal.get("thermal_normalized") is True, "thermal aggregate is not thermal normalized")
    require(thermal.get("thermal_claim_allowed") is True, "thermal claim is not allowed")
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in ("NVIDIA MPS", "XSched", "QUIET")}
    miss_counts = {name: 0 for name in pooled}
    for session_ref in thermal["sessions_input"]:
        verify_ref(session_ref)
        require("systems" in session_ref, "thermal aggregate session lacks system bindings")
        for name in pooled:
            record = session_ref["systems"][name]
            pipeline = locate_pipeline(record)
            values, misses = read_latencies(pipeline)
            require(len(values) == thermal["requests_per_session"], f"{pipeline}: unexpected row count")
            pooled[name].append(values)
            miss_counts[name] += int(misses.sum())
    output = {name: np.concatenate(chunks) for name, chunks in pooled.items()}
    for name, values in output.items():
        recorded = thermal["systems"][name]
        require(len(values) == recorded["requests"], f"{name}: pooled request count mismatch")
        require(miss_counts[name] == recorded["misses"], f"{name}: replayed miss count mismatch")
        p99 = float(np.percentile(values, 99, method="linear"))
        # The aggregate stores millisecond-converted tails at three decimals,
        # while the CSV retains the pre-serialization floating-point values.
        require(math.isclose(p99, float(recorded["tail"]["p99_us"]), abs_tol=0.1), f"{name}: replayed p99 mismatch")
    return output


def draw_latency_cdf(thermal: dict[str, Any], pooled: dict[str, np.ndarray], deadline_us: float, output: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), gridspec_kw={"wspace": 0.25})
    order = ["QUIET", "NVIDIA MPS", "XSched"]
    for name in order:
        values = np.sort(pooled[name])
        cdf = np.arange(1, len(values) + 1) / len(values)
        axes[0].plot(values, cdf, color=COLORS[name], linewidth=1.7, label=name)
        survival = (len(values) - np.arange(len(values))) / len(values)
        axes[1].step(values, survival, where="post", color=COLORS[name], linewidth=1.45, label=name)

    axes[0].axvline(deadline_us, color="#B23A48", linestyle="--", linewidth=1.1)
    axes[0].set_xlabel("arrival-to-completion latency (us)")
    axes[0].set_ylabel("CDF")
    axes[0].set_xlim(500, 5000)
    axes[0].set_ylim(0, 1.01)
    axes[0].set_title("(a) Production-wall distribution")
    axes[0].legend(loc="lower right", frameon=True)

    axes[1].axvline(deadline_us, color="#B23A48", linestyle="--", linewidth=1.1, label="deadline")
    axes[1].set_xlabel("arrival-to-completion latency (us)")
    axes[1].set_ylabel("P(latency > x)")
    axes[1].set_xlim(1500, 5000)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1 / 8000, 1)
    axes[1].set_title("(b) Tail view")
    axes[1].legend(loc="upper right", frameon=True)
    return save_figure(fig, output)


def draw_session_stability(thermal: dict[str, Any], deadline_us: float, output: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7), gridspec_kw={"wspace": 0.27})
    sessions = np.arange(1, len(thermal["sessions_input"]) + 1)
    for name in ("QUIET", "NVIDIA MPS", "XSched"):
        axes[0].plot(
            sessions,
            thermal["systems"][name]["session_p99_us"],
            marker="o",
            markersize=3.8,
            linewidth=1.4,
            color=COLORS[name],
            label=name,
        )
    axes[0].axhline(deadline_us, color="#B23A48", linestyle="--", linewidth=1.0, label="deadline")
    axes[0].set_xticks(sessions)
    axes[0].set_xlabel("counterbalanced session")
    axes[0].set_ylabel("p99 latency (us)")
    axes[0].set_title("(a) Session-level tails")
    axes[0].legend(loc="best", frameon=True, ncol=2)

    soc_mean, soc_low, soc_high, tj_mean, tj_low, tj_high = [], [], [], [], [], []
    for ref in thermal["thermal_sessions"]:
        verify_ref(ref)
        metrics = ref["metrics"]["temperature_c"]
        soc = metrics["soc012"]
        tj = metrics["tj"]
        soc_mean.append(soc["mean"])
        soc_low.append(soc["mean"] - soc["min"])
        soc_high.append(soc["max"] - soc["mean"])
        tj_mean.append(tj["mean"])
        tj_low.append(tj["mean"] - tj["min"])
        tj_high.append(tj["max"] - tj["mean"])
        require(ref["thermal_condition"]["passed"] is True, "thermal session failed its sensor/range gate")
        require("VDD_GPU" in ref["metrics"]["power_mw"], "thermal session lacks VDD_GPU")
    axes[1].errorbar(sessions - 0.06, soc_mean, yerr=[soc_low, soc_high], fmt="o-", capsize=2.5, color="#3574A7", linewidth=1.3, markersize=3.7, label="soc012")
    axes[1].errorbar(sessions + 0.06, tj_mean, yerr=[tj_low, tj_high], fmt="s-", capsize=2.5, color="#B45543", linewidth=1.3, markersize=3.5, label="tj")
    axes[1].set_xticks(sessions)
    axes[1].set_xlabel("counterbalanced session")
    axes[1].set_ylabel("temperature (C)")
    axes[1].set_title("(b) Frozen thermal envelope")
    axes[1].legend(loc="best", frameon=True)
    return save_figure(fig, output)


def draw_frontier(frontier: dict[str, Any], thermal: dict[str, Any], output: Path) -> list[Path]:
    require(frontier.get("formal") is False, "load sweep must remain descriptive")
    require(frontier.get("ranking_allowed") is False, "load sweep unexpectedly allows ranking")
    target = float(frontier["dmr_target"]) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.68), gridspec_kw={"wspace": 0.28})
    for name in ("NVIDIA MPS", "QUIET"):
        points = frontier["systems"][name]["points"]
        offered = np.asarray([float(point["offered_load_rps"]) for point in points])
        cp95 = np.asarray([float(point["dmr_cp95_upper"]) * 100.0 for point in points])
        goodput = np.asarray([float(point["background_goodput_rps"]) for point in points])
        require(all(point["cp95_slo_qualified"] is False for point in points), f"{name}: descriptive point marked qualified")
        axes[0].plot(offered, cp95, marker="o", markerfacecolor="white", markersize=5.0, linewidth=1.4, color=COLORS[name], label=f"{name} (descriptive)")
        axes[1].plot(offered, goodput, marker="o", markerfacecolor="white", markersize=5.0, linewidth=1.4, color=COLORS[name], label=name)
    anchor = thermal["systems"]["QUIET"]
    axes[0].scatter([250], [float(anchor["dmr_cp95_upper"]) * 100.0], marker="*", s=95, color=COLORS["QUIET"], edgecolor=COLORS["ink"], linewidth=0.45, zorder=6, label="QUIET formal anchor")
    axes[0].axhline(target, color="#B23A48", linestyle="--", linewidth=1.0, label="0.05% target")
    axes[0].set_yscale("log")
    axes[0].set_xticks([125, 250, 375])
    axes[0].set_xlabel("offered BE load (requests/s)")
    axes[0].set_ylabel("one-sided CP95 DMR (%)")
    axes[0].set_title("(a) Confidence-bounded DMR")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot([110, 390], [110, 390], color=COLORS["muted"], linestyle="--", linewidth=1.0, label="ideal")
    axes[1].set_xticks([125, 250, 375])
    axes[1].set_xlabel("offered BE load (requests/s)")
    axes[1].set_ylabel("completed BE goodput (requests/s)")
    axes[1].set_title("(b) Background completion rate")
    axes[1].legend(loc="upper left", frameon=True)
    return save_figure(fig, output)


def draw_causal(quiet: dict[str, Any], mps: dict[str, Any], output: Path) -> list[Path]:
    require(quiet.get("formal") is False and mps.get("formal") is False, "causal pairs must remain exploratory")
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, summary, title, color in (
        (axes[0], quiet, "(a) QUIET", COLORS["QUIET"]),
        (axes[1], mps, "(b) NVIDIA MPS", COLORS["NVIDIA MPS"]),
    ):
        for row in summary["rows"]:
            ax.plot([0, 1], [row["independent_p99_us"], row["dependent_p99_us"]], color=color, alpha=0.7, linewidth=1.2, marker="o", markersize=3.7)
        ax.set_xticks([0, 1], ["replayed\nindependent", "dependent"])
        ax.set_xlim(-0.15, 1.25)
        ax.set_title(title)
        mean = float(summary["delta_p99_us"]["mean"])
        ax.text(0.04, 0.94, f"mean delta = {mean:,.1f} us", transform=ax.transAxes, va="top", fontsize=7.2, color=COLORS["ink"])
    axes[0].set_ylabel("production-wall p99 (us)")
    return save_figure(fig, output)


def generate_tables(
    thermal: dict[str, Any],
    vision: dict[str, Any],
    whisper: dict[str, Any],
    pantheon: dict[str, Any],
    output: Path,
) -> Path:
    require(vision["status"] == "passed" and vision["numeric_comparison_allowed"] is True, "vision accuracy gate failed")
    require(whisper["status"] == "passed" and whisper["numeric_comparison_allowed"] is True, "Whisper accuracy gate failed")
    require(pantheon["status"] == "passed" and pantheon["numeric_comparison_allowed"] is True, "Pantheon fidelity gate failed")
    systems = thermal["systems"]
    lines = [
        "% Generated by analysis/generate_p9_current_figures.py; do not edit by hand.",
        r"\newcommand{\PnineApplicationTable}{%",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Application-semantic gates. Accuracy is utterance exact-match for ASR; WER is reported separately. Every candidate consumes the recorded inputs and emits a post-completion output trace.}",
        r"\label{tab:application-gates}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Workload & Measured & Reference & QUIET & Delta \\",
        r"\midrule",
        f"ImageNette ResNet-50 & {vision['requests']} & {vision['reference_accuracy']:.4f} acc. & {vision['candidate_accuracy']:.4f} acc. & {vision['accuracy_delta']:+.4f} \\\\",
        f"LibriSpeech Whisper-Tiny & {whisper['requests']} & {whisper['reference_accuracy']:.4f} acc. & {whisper['candidate_accuracy']:.4f} acc. & {whisper['accuracy_delta']:+.4f} \\\\",
        f"\\quad WER & {whisper['requests']} & {whisper['reference_wer']:.4f} & {whisper['candidate_wer']:.4f} & {whisper['wer_delta']:+.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        r"}",
        "",
        r"\newcommand{\PnineFormalTable}{%",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Thermal-normalized ImageNette campaign at the frozen $2{,}255.483~\mu$s production-wall deadline. CP95 is the exact one-sided 95\% upper bound on deadline-miss ratio; the target is 0.05\%.}",
        r"\label{tab:formal-campaign}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"System & Requests & Misses & CP95 DMR & p99 ($\mu$s) & BE rps \\",
        r"\midrule",
    ]
    for name in ("QUIET", "NVIDIA MPS", "XSched"):
        row = systems[name]
        display = r"\textbf{QUIET}" if name == "QUIET" else ("XSched (native)" if name == "XSched" else name)
        cp95 = float(row["dmr_cp95_upper"]) * 100.0
        lines.append(
            f"{display} & {row['requests']:,} & {row['misses']:,} & {cp95:.4f}\\% & {row['tail']['p99_us']:,.2f} & {row['background_goodput_rps']['mean']:.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            r"}",
            "",
            r"\newcommand{\PnineComparatorTable}{%",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Native published-system evidence. Pantheon is a separate fidelity gate and is not pooled or ranked with the six-session campaign because its adapter uses a distinct 90-request, integer-$\mu$s deadline contract.}",
            r"\label{tab:native-comparators}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{lrrrrl}",
            r"\toprule",
            r"System & Requests & Accuracy & Misses & p99 ($\mu$s) & Scope \\",
            r"\midrule",
            f"XSched & {systems['XSched']['requests']:,} & 0.8345 & {systems['XSched']['misses']:,} & {systems['XSched']['tail']['p99_us']:,.2f} & formal campaign \\\\",
            f"Pantheon & {pantheon['requests']} & {pantheon['pantheon_accuracy']:.4f} & {pantheon['deadline_misses']} & {pantheon['p99_us']:,.0f} & separate native gate \\\\",
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            r"}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thermal", type=Path, default=DEFAULT_THERMAL)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--vision", type=Path, default=DEFAULT_VISION)
    parser.add_argument("--whisper", type=Path, default=DEFAULT_WHISPER)
    parser.add_argument("--pantheon", type=Path, default=DEFAULT_PANTHEON)
    parser.add_argument("--quiet-causal", type=Path, default=DEFAULT_QUIET_CAUSAL)
    parser.add_argument("--mps-causal", type=Path, default=DEFAULT_MPS_CAUSAL)
    parser.add_argument("--deadline", type=Path, default=DEFAULT_DEADLINE)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    args.figures.mkdir(parents=True, exist_ok=True)
    args.generated.mkdir(parents=True, exist_ok=True)

    inputs = {
        "thermal": args.thermal,
        "frontier": args.frontier,
        "vision": args.vision,
        "whisper": args.whisper,
        "pantheon": args.pantheon,
        "quiet_causal": args.quiet_causal,
        "mps_causal": args.mps_causal,
        "deadline": args.deadline,
        "plan": args.plan,
    }
    documents = {name: load_json(path) for name, path in inputs.items()}
    deadline = documents["deadline"]
    require(deadline["kind"] == "p9-dependent-pipeline-deadline-lock", "unexpected deadline lock kind")
    require(deadline["contract"]["production_wall_definition"] == "arrival-to-consumer-completion-excludes-correctness-validation", "unexpected production-wall definition")
    require(documents["plan"]["status"] == "selected", "QUIET plan is not selected")
    require(documents["plan"]["proposed_system"] == "QUIET", "unexpected proposed-system name")

    pooled = replay_thermal(documents["thermal"])
    deadline_us = float(deadline["deadline_us"])
    outputs: list[Path] = []
    outputs += draw_overview(deadline, documents["plan"], args.figures / "p9-quiet-overview")
    outputs += draw_timeline(args.figures / "p9-stage-timeline")
    outputs += draw_latency_cdf(documents["thermal"], pooled, deadline_us, args.figures / "p9-latency-cdf")
    outputs += draw_session_stability(documents["thermal"], deadline_us, args.figures / "p9-session-stability")
    outputs += draw_frontier(documents["frontier"], documents["thermal"], args.figures / "p9-load-frontier")
    outputs += draw_causal(documents["quiet_causal"], documents["mps_causal"], args.figures / "p9-causal-pairs")
    outputs.append(
        generate_tables(
            documents["thermal"],
            documents["vision"],
            documents["whisper"],
            documents["pantheon"],
            args.generated / "p9-current-results.tex",
        )
    )

    provenance = {
        "schema_version": 1,
        "kind": "p9-current-paper-figure-provenance",
        "claim_scope": "current thermal-normalized ImageNette headline; load and causal figures explicitly descriptive/exploratory",
        "inputs": {name: {"path": portable_provenance_path(path), "sha256": sha256(path)} for name, path in inputs.items()},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
    }
    provenance_path = args.generated / "p9-figure-provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "outputs": len(outputs), "provenance": str(provenance_path)}, indent=2))


if __name__ == "__main__":
    main()
