#!/usr/bin/env python3
"""Compact and plot the balanced nonthermal Whisper-ASR crossover result."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "results/p9-whisper-asr-mig-crossover-balanced-r01-20260814/summary.json"
DEFAULT_RAW_PROVENANCE = DEFAULT_RAW.with_name("provenance.json")
DEFAULT_INPUT_PROVENANCE = ROOT / "results/p9-whisper-asr-crossover-inputs-102-20260814/provenance.json"
DEFAULT_COMPACT = ROOT / "paper/eurosys27/generated/p9-whisper-asr-mig-crossover.json"
DEFAULT_FIGURE = ROOT / "paper/eurosys27/figures/p9-whisper-asr-mig-crossover"

SYSTEMS = (
    ("QUIET", "quiet"),
    ("NVIDIA MIG", "nvidia-mig"),
    ("NVIDIA MPS", "nvidia-mps-static-split"),
)
COLORS = {
    "QUIET": "#167D91",
    "NVIDIA MIG": "#6B7C85",
    "NVIDIA MPS": "#D8783D",
}
HATCHES = {"QUIET": "", "NVIDIA MIG": "///", "NVIDIA MPS": ".."}
LABELS = {"QUIET": "QUIET", "NVIDIA MIG": "NVIDIA\nMIG", "NVIDIA MPS": "NVIDIA\nMPS"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def compact(raw: dict[str, Any], source: Path, artifacts: dict[str, str]) -> dict[str, Any]:
    if raw.get("kind") != "p9-whisper-asr-mig-crossover":
        raise ValueError("crossover summary kind differs")
    if raw.get("evidence_class") != "exploratory-nonthermal-directional":
        raise ValueError("crossover evidence class differs")
    if raw.get("thermal_campaign") is not False:
        raise ValueError("crossover must remain explicitly nonthermal")
    if raw.get("comparator_output_contract") != "byte-identical":
        raise ValueError("crossover output contract failed")
    if int(raw.get("pipeline_slots", 0)) != 3 or float(raw.get("deadline_us", 0.0)) != 250000.0:
        raise ValueError("crossover slot/deadline contract differs")
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("crossover requires nine balanced rows")
    output_hashes = {str(row["output_sha256"]) for row in rows}
    if len(output_hashes) != 1:
        raise ValueError("crossover comparator outputs differ")

    systems: dict[str, dict[str, Any]] = {}
    for system, mode in SYSTEMS:
        group = [row for row in rows if row.get("mode") == mode]
        if {int(row["session"]) for row in group} != {1, 2, 3}:
            raise ValueError(f"{system} session coverage differs")
        if any(float(row["rate_rps"]) != 19.0 or int(row["requests"]) != 100 for row in group):
            raise ValueError(f"{system} load contract differs")
        if mode == "quiet":
            if any(int(row["gated_processes"]) != 1 for row in group):
                raise ValueError("QUIET producer gate is absent")
        elif any(int(row["gated_processes"]) != 0 for row in group):
            raise ValueError(f"{system} unexpectedly gates the BE tenant")
        misses = sum(int(row["deadline_misses"]) for row in group)
        requests = sum(int(row["requests"]) for row in group)
        system_row: dict[str, Any] = {
            "system": system,
            "mode": mode,
            "sessions": 3,
            "requests": requests,
            "misses": misses,
            "observed_dmr": misses / requests,
            "session_misses": [int(row["deadline_misses"]) for row in group],
            "mean_session_p50_us": statistics.fmean(float(row["p50_us"]) for row in group),
            "mean_session_p99_us": statistics.fmean(float(row["p99_us"]) for row in group),
            "min_session_p99_us": min(float(row["p99_us"]) for row in group),
            "max_session_p99_us": max(float(row["p99_us"]) for row in group),
            "mean_queue_p99_us": statistics.fmean(float(row["queue_p99_us"]) for row in group),
            "mean_critical_goodput_rps": statistics.fmean(float(row["request_goodput_rps"]) for row in group),
            "mean_background_goodput_rps": statistics.fmean(float(row["background_goodput_rps"]) for row in group),
            "min_background_goodput_rps": min(float(row["background_goodput_rps"]) for row in group),
            "max_background_goodput_rps": max(float(row["background_goodput_rps"]) for row in group),
            "mean_producer_us": statistics.fmean(float(row["producer_mean_us"]) for row in group),
            "mean_consumer_us": statistics.fmean(float(row["consumer_mean_us"]) for row in group),
            "output_sha256": next(iter(output_hashes)),
        }
        if mode == "quiet":
            system_row["mean_gate_hold_p99_us"] = statistics.fmean(
                float(row["gate_hold_p99_us"]) for row in group
            )
        systems[system] = system_row

    return {
        "schema_version": 1,
        "kind": "p9-whisper-asr-mig-crossover-compact",
        "evidence_class": "exploratory-nonthermal-motivation",
        "system_order": [system for system, _ in SYSTEMS],
        "workload": "Whisper-Tiny encoder-decoder on labelled LibriSpeech windows",
        "input_policy": "12-real-window cyclic performance replay; not accuracy expansion",
        "offered_rps": 19.0,
        "deadline_us": 250000.0,
        "pipeline_slots": 3,
        "background": "saturated DistilBERT-SST2 on Thor 1g",
        "treatment_order": [
            ["NVIDIA MIG", "NVIDIA MPS", "QUIET"],
            ["NVIDIA MPS", "QUIET", "NVIDIA MIG"],
            ["QUIET", "NVIDIA MIG", "NVIDIA MPS"],
        ],
        "systems": systems,
        "output_contract": "byte-identical across all nine runs",
        "source_summary": {"path": portable(source), "sha256": sha256(source)},
        "artifact_sha256": {
            portable(Path(path)): digest for path, digest in artifacts.items()
        },
        "claim_guard": (
            "This is a three-session nonthermal motivation result, not a formal deployment "
            "trace or universal ASR SLO. The inputs are real and outputs are byte-identical, "
            "but twelve labelled windows are replayed cyclically at a synthetic fixed rate."
        ),
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.1,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": "#263238",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": "#D7DEE1",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def draw(summary: dict[str, Any], stem: Path) -> list[Path]:
    order = summary["system_order"]
    if order != [system for system, _ in SYSTEMS]:
        raise ValueError("crossover figure system order differs")
    rows = [summary["systems"][system] for system in order]
    x = np.arange(len(order))
    colors = [COLORS[system] for system in order]
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.85), gridspec_kw={"wspace": 0.38})
    fig.subplots_adjust(bottom=0.29, top=0.82)

    def bars(axis: plt.Axes, values: list[float]) -> list[Any]:
        result = axis.bar(x, values, width=0.64, color=colors, edgecolor="#263238", linewidth=0.65)
        for bar, system in zip(result, order, strict=True):
            bar.set_hatch(HATCHES[system])
        axis.set_xticks(x, [LABELS[system] for system in order])
        axis.grid(axis="x", visible=False)
        return list(result)

    dmr = [100.0 * float(row["observed_dmr"]) for row in rows]
    dmr_bars = bars(axes[0], dmr)
    axes[0].set_ylim(0, 65)
    axes[0].set_ylabel("observed DMR (%)")
    axes[0].set_title("(a) Deadline misses")
    for bar, row, value in zip(dmr_bars, rows, dmr, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            max(value, 1.5) + 1.5,
            f"{row['misses']}/{row['requests']}",
            ha="center", va="bottom", fontsize=7.2,
        )

    p99 = [float(row["mean_session_p99_us"]) / 1000.0 for row in rows]
    low = [value - float(row["min_session_p99_us"]) / 1000.0 for value, row in zip(p99, rows, strict=True)]
    high = [float(row["max_session_p99_us"]) / 1000.0 - value for value, row in zip(p99, rows, strict=True)]
    p99_bars = bars(axes[1], p99)
    axes[1].errorbar(x, p99, yerr=[low, high], fmt="none", ecolor="#263238", capsize=3, linewidth=0.8)
    axes[1].axhline(250.0, color="#A52A3A", linestyle="--", linewidth=1.05)
    axes[1].set_ylim(0, 470)
    axes[1].set_ylabel("mean session p99 (ms)")
    axes[1].set_title("(b) Production-wall tail")
    axes[1].text(
        -0.48, 259, "250 ms deadline", color="#A52A3A", fontsize=7.0,
        ha="left", va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )
    for bar, value in zip(p99_bars, p99, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 17, f"{value:.0f}", ha="center", va="bottom", fontsize=7.1)

    be = [float(row["mean_background_goodput_rps"]) for row in rows]
    be_low = [value - float(row["min_background_goodput_rps"]) for value, row in zip(be, rows, strict=True)]
    be_high = [float(row["max_background_goodput_rps"]) - value for value, row in zip(be, rows, strict=True)]
    be_bars = bars(axes[2], be)
    axes[2].errorbar(x, be, yerr=[be_low, be_high], fmt="none", ecolor="#263238", capsize=3, linewidth=0.8)
    axes[2].set_ylim(0, 1030)
    axes[2].set_ylabel("completed BE (requests/s)")
    axes[2].set_title("(c) Background goodput")
    for bar, value in zip(be_bars, be, strict=True):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 25, f"{value:.0f}", ha="center", va="bottom", fontsize=7.1)

    fig.text(
        0.5, 0.035,
        "Whisper-Tiny, 19 requests/s, 3 slots, 3 x 100 requests; whiskers show session min-max. "
        "Real inputs with cyclic replay; nonthermal exploratory motivation.",
        ha="center", va="bottom", fontsize=7.3, color="#5F6B70",
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outputs[1], dpi=240)
    plt.close(fig)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--raw-provenance", type=Path, default=DEFAULT_RAW_PROVENANCE)
    parser.add_argument("--input-provenance", type=Path, default=DEFAULT_INPUT_PROVENANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    raw_path = args.raw.resolve()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    artifacts = json.loads(args.raw_provenance.resolve().read_text(encoding="utf-8"))
    input_provenance_path = args.input_provenance.resolve()
    input_provenance = json.loads(input_provenance_path.read_text(encoding="utf-8"))
    if input_provenance.get("coverage_policy") != "cyclic-performance-replay-not-accuracy-expansion":
        raise ValueError("crossover input replay policy differs")
    recorded_input = str(Path(str(input_provenance["output_trace"])).resolve())
    if artifacts.get(recorded_input) != input_provenance.get("output_trace_sha256"):
        raise ValueError("crossover input trace provenance differs")
    artifacts[str(input_provenance_path)] = sha256(input_provenance_path)
    artifacts[str(ROOT / "scripts/repeat_jdgint_trace.py")] = sha256(
        ROOT / "scripts/repeat_jdgint_trace.py"
    )
    artifacts[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    summary = compact(raw, raw_path, artifacts)
    outputs = draw(summary, args.figure.resolve())
    summary["figure_sha256"] = {portable(path): sha256(path) for path in outputs}
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": portable(args.output), "figures": [portable(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
