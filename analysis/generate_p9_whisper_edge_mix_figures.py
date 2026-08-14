#!/usr/bin/env python3
"""Validate, compact, and plot the nonthermal Whisper edge-mix campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = (
    ROOT / "results/p9-whisper-mixed-regime-campaign-r01-20260814/campaign.json"
)
DEFAULT_INPUT_PROVENANCE = (
    ROOT / "results/p9-whisper-asr-crossover-inputs-102-20260814/provenance.json"
)
DEFAULT_OUTPUT = ROOT / "paper/eurosys27/generated/p9-whisper-edge-mix-regimes.json"
DEFAULT_TABLE = ROOT / "paper/eurosys27/generated/p9-whisper-edge-mix-results.tex"
DEFAULT_FRONTIER = ROOT / "paper/eurosys27/figures/p9-whisper-edge-mix-frontier"
DEFAULT_BALANCED = ROOT / "paper/eurosys27/figures/p9-whisper-edge-mix-balanced"

SYSTEMS = (
    ("QUIET", "quiet"),
    ("NVIDIA MIG", "nvidia-mig"),
    ("NVIDIA MPS", "nvidia-mps-static-split"),
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    label: str
    short_label: str
    balanced_rate_rps: float
    background_workers: int
    failure_system: str
    failure_mode: str
    directional_rates_rps: tuple[float, ...]


SCENARIOS = (
    Scenario(
        "mig-placement-nlp",
        "MIG placement: NLP",
        "NLP\nMIG@19",
        19.0,
        1,
        "NVIDIA MIG",
        "nvidia-mig",
        (15.0, 17.0, 19.0, 20.0, 21.0),
    ),
    Scenario(
        "mig-placement-vision",
        "MIG placement: vision",
        "Vision\nMIG@21",
        21.0,
        1,
        "NVIDIA MIG",
        "nvidia-mig",
        (15.0, 17.0, 19.0, 20.0, 21.0),
    ),
    Scenario(
        "mps-interference-speech-20",
        "MPS interference: 20 speech",
        "Speech20\nMPS@17",
        17.0,
        20,
        "NVIDIA MPS",
        "nvidia-mps-static-split",
        (15.0, 17.0, 18.0, 19.0),
    ),
    Scenario(
        "mps-interference-vision-24",
        "MPS interference: 24 vision",
        "Vision24\nMPS@18",
        18.0,
        24,
        "NVIDIA MPS",
        "nvidia-mps-static-split",
        (15.0, 17.0, 18.0, 19.0, 20.0),
    ),
)
RATES = tuple(
    sorted({rate for scenario in SCENARIOS for rate in scenario.directional_rates_rps})
)
COLORS = {
    "QUIET": "#167D91",
    "NVIDIA MIG": "#6B7C85",
    "NVIDIA MPS": "#D8783D",
}
MARKERS = {"QUIET": "o", "NVIDIA MIG": "s", "NVIDIA MPS": "^"}
HATCHES = {"QUIET": "", "NVIDIA MIG": "///", "NVIDIA MPS": ".."}


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


def validate_raw_summary(
    raw: dict[str, Any],
    *,
    scenario_id: str,
    phase: str,
    balanced_rate: float,
    workers: int,
    directional_rates: tuple[float, ...],
) -> dict[str, Any]:
    expected_design = (
        "directional-sweep" if phase == "directional" else "balanced-repeated"
    )
    expected_rates = (
        set(directional_rates) if phase == "directional" else {balanced_rate}
    )
    expected_sessions = {1} if phase == "directional" else {1, 2, 3}
    expected_rows = len(expected_rates) * len(SYSTEMS) * len(expected_sessions)
    if (
        raw.get("kind") != "p9-whisper-asr-mig-crossover"
        or raw.get("evidence_class") != "exploratory-nonthermal-directional"
        or raw.get("thermal_campaign") is not False
        or raw.get("input_policy")
        != "cyclic-performance-replay-not-accuracy-expansion"
        or raw.get("study_design") != expected_design
        or raw.get("comparator_output_contract") != "byte-identical"
        or int(raw.get("pipeline_slots", 0)) != 3
        or float(raw.get("deadline_us", 0.0)) != 250000.0
        or int(raw.get("background_workers", 0)) != workers
        or raw.get("scenario", {}).get("id") != scenario_id
        or int(raw.get("scenario", {}).get("background_workers", 0)) != workers
    ):
        raise ValueError(f"{phase} contract differs for {scenario_id}")
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError(f"{phase} row count differs for {scenario_id}")
    if (
        {float(row["rate_rps"]) for row in rows} != expected_rates
        or {int(row["session"]) for row in rows} != expected_sessions
        or {str(row["mode"]) for row in rows}
        != {mode for _, mode in SYSTEMS}
        or any(int(row["requests"]) != 100 for row in rows)
        or any(int(row.get("background_workers", 0)) != workers for row in rows)
        or len({str(row["output_sha256"]) for row in rows}) != 1
    ):
        raise ValueError(f"{phase} matrix differs for {scenario_id}")
    for row in rows:
        expected_gated = workers if row["mode"] == "quiet" else 0
        if int(row.get("gated_processes", -1)) != expected_gated:
            raise ValueError(f"{phase} gate coverage differs for {scenario_id}")
    return raw


def aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["session"]))
    requests = sum(int(row["requests"]) for row in ordered)
    misses = sum(int(row["deadline_misses"]) for row in ordered)
    p99_values = [float(row["p99_us"]) for row in ordered]
    return {
        "sessions": len(ordered),
        "requests": requests,
        "misses": misses,
        "observed_dmr": misses / requests,
        "session_misses": [int(row["deadline_misses"]) for row in ordered],
        "mean_session_p50_us": statistics.fmean(
            float(row["p50_us"]) for row in ordered
        ),
        "mean_session_p99_us": statistics.fmean(p99_values),
        "min_session_p99_us": min(p99_values),
        "max_session_p99_us": max(p99_values),
        "mean_queue_p99_us": statistics.fmean(
            float(row["queue_p99_us"]) for row in ordered
        ),
        "mean_critical_goodput_rps": statistics.fmean(
            float(row["request_goodput_rps"]) for row in ordered
        ),
        "mean_background_goodput_rps": statistics.fmean(
            float(row["background_goodput_rps"]) for row in ordered
        ),
        "mean_producer_us": statistics.fmean(
            float(row["producer_mean_us"]) for row in ordered
        ),
        "mean_consumer_us": statistics.fmean(
            float(row["consumer_mean_us"]) for row in ordered
        ),
        "output_sha256": str(ordered[0]["output_sha256"]),
    }


def compact_raw(
    directional: dict[str, Any],
    balanced: dict[str, Any],
    rates: tuple[float, ...],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {system: {} for system, _ in SYSTEMS}
    for system, mode in SYSTEMS:
        points: list[dict[str, Any]] = []
        for rate in rates:
            group = [
                row
                for row in directional["rows"]
                if row["mode"] == mode and float(row["rate_rps"]) == rate
            ]
            if len(group) != 1:
                raise ValueError("directional group is not one request session")
            point = aggregate_group(group)
            point["rate_rps"] = rate
            points.append(point)
        balanced_group = [row for row in balanced["rows"] if row["mode"] == mode]
        if len(balanced_group) != 3:
            raise ValueError("balanced group does not contain three sessions")
        result[system] = {
            "mode": mode,
            "directional": points,
            "balanced": aggregate_group(balanced_group),
        }
    return result


def first_target_only_failure(
    systems: dict[str, dict[str, Any]],
    target_system: str,
    rates: tuple[float, ...],
) -> float:
    """Return the first rate where only the selected vendor baseline misses."""
    if target_system not in {"NVIDIA MIG", "NVIDIA MPS"}:
        raise ValueError("failure target must be NVIDIA MIG or NVIDIA MPS")
    by_system = {
        system: {
            float(point["rate_rps"]): int(point["misses"])
            for point in values["directional"]
        }
        for system, values in systems.items()
    }
    for rate in rates:
        if by_system[target_system][rate] > 0 and all(
            by_system[system][rate] == 0
            for system in by_system
            if system != target_system
        ):
            return rate
    raise ValueError("scenario lacks a target-only crossover")


def load_campaign(campaign_path: Path, input_provenance_path: Path) -> dict[str, Any]:
    campaign_path = campaign_path.resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    raw_rate_map = campaign.get("directional_rates_rps", {})
    if not isinstance(raw_rate_map, dict):
        raise ValueError("edge-mix directional-rate map differs")
    rate_map = {
        str(scenario_id): tuple(float(rate) for rate in rates)
        for scenario_id, rates in raw_rate_map.items()
    }
    expected_rate_map = {
        scenario.scenario_id: scenario.directional_rates_rps
        for scenario in SCENARIOS
    }
    if (
        campaign.get("kind") != "p9-whisper-edge-mix-campaign"
        or campaign.get("evidence_class")
        != "exploratory-nonthermal-motivation"
        or campaign.get("thermal_campaign") is not False
        or campaign.get("selection_policy")
        != "two-placement-and-two-interference-regimes"
        or campaign.get("balanced_rate_policy")
        != "first-directional-rate-with-target-baseline-misses-and-other-baselines-zero"
        or rate_map != expected_rate_map
        or int(campaign.get("balanced_sessions", 0)) != 3
        or int(campaign.get("requests_per_run", 0)) != 100
    ):
        raise ValueError("edge-mix campaign contract differs")

    scenario_order = campaign.get("scenario_order", [])
    expected_order = [scenario.scenario_id for scenario in SCENARIOS]
    if (
        not isinstance(scenario_order, list)
        or [item.get("scenario_id") for item in scenario_order] != expected_order
    ):
        raise ValueError("edge-mix scenario order differs")
    for specification, item in zip(SCENARIOS, scenario_order, strict=True):
        if (
            item.get("failure_target") != specification.failure_mode
            or tuple(float(rate) for rate in item.get("directional_rates_rps", []))
            != specification.directional_rates_rps
            or float(item.get("balanced_rate_rps", -1.0))
            != specification.balanced_rate_rps
            or len(item.get("additional_backgrounds", [])) + 1
            != specification.background_workers
        ):
            raise ValueError(
                f"edge-mix scenario contract differs for {specification.scenario_id}"
            )
    for binding_name in ("runner", "campaign_driver"):
        binding = campaign.get(binding_name, {})
        path = Path(str(binding.get("path", ""))).resolve()
        if not path.is_file() or sha256(path) != binding.get("sha256"):
            raise ValueError(f"edge-mix {binding_name} binding differs")

    input_provenance_path = input_provenance_path.resolve()
    input_provenance = json.loads(
        input_provenance_path.read_text(encoding="utf-8")
    )
    input_trace = Path(str(input_provenance.get("output_trace", ""))).resolve()
    if (
        input_provenance.get("kind") != "jdgint1-cyclic-performance-replay"
        or input_provenance.get("coverage_policy")
        != "cyclic-performance-replay-not-accuracy-expansion"
        or int(input_provenance.get("source_records", 0)) != 12
        or int(input_provenance.get("output_records", 0)) != 102
        or not input_trace.is_file()
        or sha256(input_trace) != input_provenance.get("output_trace_sha256")
        or campaign.get("input_trace", {}).get("sha256")
        != input_provenance.get("output_trace_sha256")
    ):
        raise ValueError("edge-mix input replay contract differs")

    records = {
        (str(item["phase"]), str(item["scenario_id"])): item
        for item in campaign.get("summaries", [])
    }
    if len(records) != len(SCENARIOS) * 2:
        raise ValueError("edge-mix summary index differs")
    scenario_results: dict[str, Any] = {}
    artifact_sha256: dict[str, str] = {}
    output_hashes: set[str] = set()
    for specification in SCENARIOS:
        scenario_id = specification.scenario_id
        raw_by_phase: dict[str, dict[str, Any]] = {}
        sources: dict[str, Any] = {}
        for phase in ("directional", "balanced"):
            record = records.get((phase, scenario_id))
            if record is None:
                raise ValueError(f"missing {phase} summary for {scenario_id}")
            source = Path(str(record["path"])).resolve()
            if not source.is_file() or sha256(source) != record.get("sha256"):
                raise ValueError(f"{phase} source hash differs for {scenario_id}")
            raw = validate_raw_summary(
                json.loads(source.read_text(encoding="utf-8")),
                scenario_id=scenario_id,
                phase=phase,
                balanced_rate=specification.balanced_rate_rps,
                workers=specification.background_workers,
                directional_rates=specification.directional_rates_rps,
            )
            provenance_path = source.with_name("provenance.json")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            for path_text, digest in provenance.items():
                path = Path(path_text).resolve()
                key = portable(path)
                previous = artifact_sha256.get(key)
                if previous is not None and previous != digest:
                    raise ValueError(f"artifact hash changed across summaries: {path}")
                if previous is None and (
                    not path.is_file() or sha256(path) != digest
                ):
                    raise ValueError(f"artifact hash differs: {path}")
                artifact_sha256[key] = str(digest)
            raw_by_phase[phase] = raw
            output_hashes.update(str(row["output_sha256"]) for row in raw["rows"])
            sources[phase] = {
                "summary": {"path": portable(source), "sha256": sha256(source)},
                "provenance": {
                    "path": portable(provenance_path),
                    "sha256": sha256(provenance_path),
                },
            }
        systems = compact_raw(
            raw_by_phase["directional"],
            raw_by_phase["balanced"],
            specification.directional_rates_rps,
        )
        selected_rate = first_target_only_failure(
            systems,
            specification.failure_system,
            specification.directional_rates_rps,
        )
        if selected_rate != specification.balanced_rate_rps:
            raise ValueError(f"balanced crossover selection differs for {scenario_id}")
        balanced_rows = {
            system: systems[system]["balanced"] for system, _ in SYSTEMS
        }
        if int(balanced_rows[specification.failure_system]["misses"]) <= 0 or any(
            int(row["misses"]) != 0
            for system, row in balanced_rows.items()
            if system != specification.failure_system
        ):
            raise ValueError(f"balanced target-only failure differs for {scenario_id}")
        scenario_results[scenario_id] = {
            "label": specification.label,
            "short_label": specification.short_label,
            "failure_target": specification.failure_system,
            "failure_class": (
                "placement" if specification.failure_system == "NVIDIA MIG"
                else "shared-instance-interference"
            ),
            "background_workers": specification.background_workers,
            "background_models": raw_by_phase["balanced"]["scenario"][
                "background_models"
            ],
            "deployment_scope": raw_by_phase["balanced"]["scenario"][
                "deployment_scope"
            ],
            "directional_rates_rps": list(
                specification.directional_rates_rps
            ),
            "balanced_rate_rps": specification.balanced_rate_rps,
            "systems": systems,
            "sources": sources,
        }
    if len(output_hashes) != 1:
        raise ValueError("foreground outputs differ across the edge-mix campaign")

    system_totals = {
        system: {
            "requests": sum(
                scenario["systems"][system]["balanced"]["requests"]
                for scenario in scenario_results.values()
            ),
            "misses": sum(
                scenario["systems"][system]["balanced"]["misses"]
                for scenario in scenario_results.values()
            ),
        }
        for system, _ in SYSTEMS
    }
    if system_totals["QUIET"]["misses"] != 0:
        raise ValueError("QUIET has misses in a selected balanced regime")
    return {
        "schema_version": 2,
        "kind": "p9-whisper-edge-mix-regimes-compact",
        "evidence_class": "exploratory-nonthermal-motivation",
        "thermal_campaign": False,
        "system_order": [system for system, _ in SYSTEMS],
        "scenario_order": expected_order,
        "directional_rates_rps": {
            scenario.scenario_id: list(scenario.directional_rates_rps)
            for scenario in SCENARIOS
        },
        "deadline_us": 250000.0,
        "pipeline_slots": 3,
        "foreground": "Whisper-Tiny encoder-decoder on labelled LibriSpeech windows",
        "input_policy": (
            "12 labelled windows cyclically replayed to 102 records; performance only, "
            "not an accuracy expansion"
        ),
        "background_policy": (
            "saturated queued TensorRT workers using real DistilBERT-SST2, "
            "ResNet10-detection, and Whisper-Tiny encoder plans; high-fanout "
            "interference regimes use 20 speech or 24 vision workers"
        ),
        "balanced_rate_policy": campaign["balanced_rate_policy"],
        "scenarios": scenario_results,
        "output_contract": {
            "kind": "byte-identical-across-all-runs",
            "sha256": next(iter(output_hashes)),
        },
        "balanced_totals": {
            "all_system_requests": sum(
                values["requests"] for values in system_totals.values()
            ),
            "by_system": system_totals,
        },
        "campaign": {"path": portable(campaign_path), "sha256": sha256(campaign_path)},
        "input_provenance": {
            "path": portable(input_provenance_path),
            "sha256": sha256(input_provenance_path),
        },
        "artifact_sha256": artifact_sha256,
        "claim_guard": (
            "Two low-fanout cases expose MIG placement loss; two high-fanout cases "
            "expose static-MPS producer interference. Fixed-rate arrivals, saturated "
            "queues, and cyclic input replay are controlled stress conditions rather "
            "than a field trace. This is nonthermal exploratory motivation, not a "
            "formal deployment or thermal claim."
        ),
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.7,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.2,
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


def save(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outputs[1], dpi=240)
    plt.close(fig)
    return outputs


def render_balanced_table(summary: dict[str, Any], output: Path) -> Path:
    """Render the fixed-order repeated-session table consumed by the paper."""
    lines = [
        "% Generated by analysis/generate_p9_whisper_edge_mix_figures.py; do not edit.",
        r"\newcommand{\PnineEdgeMixTable}{%",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        (
            r"\caption{Balanced nonthermal edge-mix crossovers.  Each row aggregates "
            r"three rotated 100-request sessions at the first directional rate where "
            r"the named vendor baseline misses while the other baseline and QUIET "
            r"remain at zero.}"
        ),
        r"\label{tab:edge-mix-balanced}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        (
            r"Failure regime at foreground rate & System & Requests & Misses & DMR & "
            r"p99 (ms) & Queue p99 (ms) & Critical rps & BE rps \\"
        ),
        r"\midrule",
    ]
    for scenario_index, specification in enumerate(SCENARIOS):
        scenario = summary["scenarios"][specification.scenario_id]
        rate = specification.balanced_rate_rps
        if float(scenario["balanced_rate_rps"]) != rate:
            raise ValueError(
                f"table rate differs for {specification.scenario_id}"
            )
        for system, _ in SYSTEMS:
            row = scenario["systems"][system]["balanced"]
            system_label = r"\textbf{QUIET}" if system == "QUIET" else system
            lines.append(
                f"{specification.label} @ {rate:g} rps & {system_label} & "
                f"{int(row['requests']):,} & {int(row['misses']):,} & "
                f"{100.0 * float(row['observed_dmr']):.3f}\\% & "
                f"{float(row['mean_session_p99_us']) / 1000.0:.3f} & "
                f"{float(row['mean_queue_p99_us']) / 1000.0:.3f} & "
                f"{float(row['mean_critical_goodput_rps']):.3f} & "
                f"{float(row['mean_background_goodput_rps']):.3f} \\\\"
            )
        if scenario_index + 1 != len(SCENARIOS):
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            r"}",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def draw_frontier(summary: dict[str, Any], stem: Path) -> list[Path]:
    configure_style()
    fig, axes = plt.subplots(
        len(SCENARIOS), 2, figsize=(8.0, 8.15), sharex=True,
        gridspec_kw={"hspace": 0.42, "wspace": 0.28},
    )
    handles: list[Any] = []
    for row_index, specification in enumerate(SCENARIOS):
        scenario = summary["scenarios"][specification.scenario_id]
        rates = specification.directional_rates_rps
        balanced_rate = specification.balanced_rate_rps
        maximum_p99 = 250.0
        for system, _ in SYSTEMS:
            points = scenario["systems"][system]["directional"]
            dmr = [100.0 * float(point["observed_dmr"]) for point in points]
            p99 = [float(point["mean_session_p99_us"]) / 1000.0 for point in points]
            line = axes[row_index, 0].plot(
                rates, dmr, color=COLORS[system], marker=MARKERS[system],
                linewidth=1.55, markersize=4.2, label=system,
            )[0]
            axes[row_index, 1].plot(
                rates, p99, color=COLORS[system], marker=MARKERS[system],
                linewidth=1.55, markersize=4.2, label=system,
            )
            maximum_p99 = max(maximum_p99, *p99)
            if row_index == 0:
                handles.append(line)
        axes[row_index, 0].axvline(
            balanced_rate, color="#A52A3A", linestyle=":", linewidth=0.9
        )
        axes[row_index, 1].axvline(
            balanced_rate, color="#A52A3A", linestyle=":", linewidth=0.9
        )
        axes[row_index, 1].axhline(
            250.0, color="#A52A3A", linestyle="--", linewidth=0.95
        )
        axes[row_index, 0].set_ylim(0, 100)
        axes[row_index, 1].set_ylim(
            0, max(350.0, math.ceil(maximum_p99 * 1.08 / 50.0) * 50.0)
        )
        axes[row_index, 0].set_title(
            f"{specification.label}: deadline misses"
        )
        axes[row_index, 1].set_title(
            f"{specification.label}: production-wall p99"
        )
        axes[row_index, 0].set_ylabel("observed DMR (%)")
        axes[row_index, 1].set_ylabel("p99 (ms)")
        axes[row_index, 0].grid(axis="x", visible=False)
        axes[row_index, 1].grid(axis="x", visible=False)
        annotation_at_right_edge = balanced_rate == max(rates)
        annotation_x = (
            balanced_rate - 0.08
            if annotation_at_right_edge
            else balanced_rate + 0.08
        )
        annotation_alignment = "right" if annotation_at_right_edge else "left"
        axes[row_index, 0].text(
            annotation_x,
            95,
            f"{specification.failure_system.removeprefix('NVIDIA ')} fails "
            f"@ {balanced_rate:g}",
            color="#A52A3A", fontsize=6.7, ha=annotation_alignment, va="top",
        )
    for axis in axes[-1, :]:
        axis.set_xticks(RATES)
        axis.set_xlabel("foreground offered rate (requests/s)")
    fig.legend(
        handles, [system for system, _ in SYSTEMS], loc="upper center",
        bbox_to_anchor=(0.5, 1.006), ncol=3, frameon=False,
    )
    fig.subplots_adjust(top=0.955, bottom=0.085)
    fig.text(
        0.5, 0.014,
        "Each point is 100 requests. Every panel repeats QUIET, NVIDIA MIG, and NVIDIA MPS; "
        "dotted lines mark the first target-baseline-only failure rate. "
        "Nonthermal exploratory stress.",
        ha="center", va="bottom", fontsize=7.0, color="#5F6B70",
    )
    return save(fig, stem)


def draw_balanced(summary: dict[str, Any], stem: Path) -> list[Path]:
    configure_style()
    order = [system for system, _ in SYSTEMS]
    labels = [scenario.short_label for scenario in SCENARIOS]
    x = np.arange(len(SCENARIOS))
    width = 0.24
    fig, axes = plt.subplots(
        1, 3, figsize=(9.0, 3.25), gridspec_kw={"wspace": 0.34}
    )
    handles: list[Any] = []
    all_dmr: list[float] = []
    all_p99: list[float] = []
    all_retention: list[float] = []
    for system_index, system in enumerate(order):
        offset = (system_index - 1) * width
        rows = [
            summary["scenarios"][scenario_id]["systems"][system]["balanced"]
            for scenario_id in (scenario.scenario_id for scenario in SCENARIOS)
        ]
        dmr = [100.0 * float(row["observed_dmr"]) for row in rows]
        p99 = [float(row["mean_session_p99_us"]) / 1000.0 for row in rows]
        retention = []
        for scenario, row in zip(SCENARIOS, rows, strict=True):
            mig = summary["scenarios"][scenario.scenario_id]["systems"][
                "NVIDIA MIG"
            ]["balanced"]["mean_background_goodput_rps"]
            retention.append(
                100.0
                * float(row["mean_background_goodput_rps"])
                / float(mig)
            )
        all_dmr.extend(dmr)
        all_p99.extend(p99)
        all_retention.extend(retention)
        bars = axes[0].bar(
            x + offset, dmr, width, color=COLORS[system], edgecolor="#263238",
            linewidth=0.6, hatch=HATCHES[system], label=system,
        )
        axes[1].bar(
            x + offset, p99, width, color=COLORS[system], edgecolor="#263238",
            linewidth=0.6, hatch=HATCHES[system],
        )
        axes[2].bar(
            x + offset, retention, width, color=COLORS[system], edgecolor="#263238",
            linewidth=0.6, hatch=HATCHES[system],
        )
        handles.append(bars[0])
        for position, value, row in zip(x + offset, dmr, rows, strict=True):
            if row["misses"] > 0:
                axes[0].text(
                    position, value + 1.6, f"{row['misses']}/300", ha="center",
                    va="bottom", fontsize=6.4, rotation=90,
                )
            else:
                axes[0].text(
                    position, 0.8, "0", color=COLORS[system], ha="center",
                    va="bottom", fontsize=6.5, fontweight="bold",
                )
    axes[0].set_ylim(
        0,
        min(110.0, max(10.0, math.ceil(max(all_dmr) * 1.2 / 10.0) * 10.0)),
    )
    axes[0].set_ylabel("observed DMR (%)")
    axes[0].set_title("(a) Deadline misses")
    axes[1].axhline(250.0, color="#A52A3A", linestyle="--", linewidth=0.95)
    axes[1].set_ylim(
        0, max(300.0, math.ceil(max(all_p99) * 1.12 / 50.0) * 50.0)
    )
    axes[1].set_ylabel("mean session p99 (ms)")
    axes[1].set_title("(b) Production-wall tail")
    axes[2].set_ylim(
        0, max(110.0, math.ceil(max(all_retention) * 1.08 / 10.0) * 10.0)
    )
    axes[2].set_ylabel("BE goodput vs. MIG (%)")
    axes[2].set_title("(c) Background retention")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=6.8, pad=4)
        axis.grid(axis="x", visible=False)
    fig.legend(
        handles, order, loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=3, frameon=False,
    )
    fig.subplots_adjust(top=0.83, bottom=0.28)
    fig.text(
        0.5, 0.035,
        "Three rotated 100-request sessions per system at 19/21/17/18 requests/s. "
        "All foreground output traces are byte-identical; nonthermal exploratory motivation.",
        ha="center", va="bottom", fontsize=7.0, color="#5F6B70",
    )
    return save(fig, stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument(
        "--input-provenance", type=Path, default=DEFAULT_INPUT_PROVENANCE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--balanced", type=Path, default=DEFAULT_BALANCED)
    args = parser.parse_args()
    summary = load_campaign(args.campaign, args.input_provenance)
    summary["generator"] = {
        "path": portable(Path(__file__)), "sha256": sha256(Path(__file__))
    }
    table = render_balanced_table(summary, args.table.resolve())
    figures = [
        *draw_frontier(summary, args.frontier.resolve()),
        *draw_balanced(summary, args.balanced.resolve()),
    ]
    summary["table_sha256"] = {portable(table): sha256(table)}
    summary["figure_sha256"] = {portable(path): sha256(path) for path in figures}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": portable(output),
                "table": portable(table),
                "figures": [portable(path) for path in figures],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
