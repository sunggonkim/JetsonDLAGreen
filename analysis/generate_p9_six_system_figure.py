#!/usr/bin/env python3
"""Regenerate the fixed-roster ImageNette comparator summary and figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

try:
    from analysis.summarize_p9_six_system_imagenette_gate import (
        DEFAULT_COMMON,
        DEFAULT_LOCK,
        DEFAULT_MIG,
        DEFAULT_MPS,
        DEFAULT_ORION,
        DEFAULT_PANTHEON,
        DEFAULT_QUIET,
        DEFAULT_XSCHED,
        SYSTEM_ORDER,
        summarize,
    )
except ModuleNotFoundError:  # direct execution from analysis/
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.summarize_p9_six_system_imagenette_gate import (  # type: ignore
        DEFAULT_COMMON,
        DEFAULT_LOCK,
        DEFAULT_MIG,
        DEFAULT_MPS,
        DEFAULT_ORION,
        DEFAULT_PANTHEON,
        DEFAULT_QUIET,
        DEFAULT_XSCHED,
        SYSTEM_ORDER,
        summarize,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "paper/eurosys27/generated/p9-six-system-imagenette-gate.json"
DEFAULT_FIGURE = ROOT / "paper/eurosys27/figures/p9-six-system-imagenette-gate"
DEFAULT_PROVENANCE = ROOT / "paper/eurosys27/generated/p9-six-system-imagenette-gate-provenance.json"
COLORS = {
    "QUIET": "#167D91",
    "NVIDIA MIG": "#6B7C85",
    "NVIDIA MPS": "#D8783D",
    "XSched": "#6E5AA8",
    "Orion": "#B54E3C",
    "Pantheon": "#3C8D63",
}
HATCHES = {
    "QUIET": "",
    "NVIDIA MIG": "///",
    "NVIDIA MPS": "..",
    "XSched": "xx",
    "Orion": "--",
    "Pantheon": "\\\\",
}
LABELS = {
    "QUIET": "QUIET",
    "NVIDIA MIG": "NVIDIA\nMIG",
    "NVIDIA MPS": "NVIDIA\nMPS",
    "XSched": "XSched",
    "Orion": "Orion",
    "Pantheon": "Pantheon",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.3,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.2,
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
    })


def _bars(ax: plt.Axes, values: list[float]) -> tuple[np.ndarray, list[object]]:
    x = np.arange(len(SYSTEM_ORDER))
    bars = ax.bar(
        x,
        values,
        width=0.68,
        color=[COLORS[name] for name in SYSTEM_ORDER],
        edgecolor="#263238",
        linewidth=0.65,
    )
    for bar, name in zip(bars, SYSTEM_ORDER, strict=True):
        bar.set_hatch(HATCHES[name])
    ax.set_xticks(x, [LABELS[name] for name in SYSTEM_ORDER])
    ax.grid(axis="x", visible=False)
    return x, list(bars)


def draw(summary: dict[str, object], stem: Path) -> list[Path]:
    if summary.get("system_order") != list(SYSTEM_ORDER):
        raise ValueError("six-system figure order differs")
    systems = summary.get("systems")
    if not isinstance(systems, dict) or tuple(systems) != SYSTEM_ORDER:
        raise ValueError("six-system figure rows differ")
    rows = [systems[name] for name in SYSTEM_ORDER]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("six-system figure row is malformed")

    _style()
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), gridspec_kw={"hspace": 0.48})
    deadline = float(summary["deadline_us"])

    p99 = [float(row["p99_us"]) for row in rows]
    _, bars = _bars(axes[0], p99)
    axes[0].axhline(deadline, color="#A52A3A", linestyle="--", linewidth=1.15, label=f"deadline = {deadline:,.3f} us")
    axes[0].set_ylim(0, max(p99) * 1.17)
    axes[0].set_ylabel("p99 latency (us)")
    axes[0].set_title("(a) Production-wall tail latency — same six systems")
    axes[0].legend(loc="upper left", frameon=True, fontsize=7.5)
    for bar, value in zip(bars, p99, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + max(p99) * 0.022, f"{value:,.0f}", ha="center", va="bottom", fontsize=7.1)

    dmr = [100.0 * float(row["observed_dmr"]) for row in rows]
    _, bars = _bars(axes[1], dmr)
    axes[1].set_ylim(0, 112)
    axes[1].set_ylabel("observed DMR (%)")
    axes[1].set_title("(b) Deadline misses — same six systems")
    for bar, row, value in zip(bars, rows, dmr, strict=True):
        y = max(value, 2.0) + 2.0
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{int(row['misses'])}/{int(row['requests'])}",
            ha="center",
            va="bottom",
            fontsize=7.1,
        )

    goodput = [float(row["background_goodput_rps"]) for row in rows]
    _, bars = _bars(axes[2], goodput)
    axes[2].set_ylim(0, max(goodput) * 1.20)
    axes[2].set_ylabel("completed BE (requests/s)")
    axes[2].set_title("(c) Background goodput — same six systems")
    for bar, name, value in zip(bars, SYSTEM_ORDER, goodput, strict=True):
        text = "no BE\nslice" if name == "NVIDIA MIG" else f"{value:.1f}"
        y = max(value, max(goodput) * 0.035) + max(goodput) * 0.025
        axes[2].text(bar.get_x() + bar.get_width() / 2, y, text, ha="center", va="bottom", fontsize=7.0)

    fig.text(
        0.5,
        0.005,
        "90 labelled ImageNette requests/system; one common input + arrival + deadline lock. "
        "Directional coverage gate, not a formal ranking.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#5F6B70",
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outputs[1], dpi=240)
    plt.close(fig)
    return outputs


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, default=DEFAULT_COMMON)
    parser.add_argument("--deadline-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--quiet", type=Path, default=DEFAULT_QUIET)
    parser.add_argument("--mig", type=Path, default=DEFAULT_MIG)
    parser.add_argument("--mps", type=Path, default=DEFAULT_MPS)
    parser.add_argument("--xsched", type=Path, default=DEFAULT_XSCHED)
    parser.add_argument("--orion", type=Path, default=DEFAULT_ORION)
    parser.add_argument("--pantheon", type=Path, default=DEFAULT_PANTHEON)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args(argv)
    inputs = {
        "common": args.common.resolve(), "deadline_lock": args.deadline_lock.resolve(),
        "quiet": args.quiet.resolve(), "mig": args.mig.resolve(), "mps": args.mps.resolve(),
        "xsched": args.xsched.resolve(), "orion": args.orion.resolve(),
        "pantheon": args.pantheon.resolve(),
    }
    summary = summarize(
        common_path=inputs["common"], deadline_path=inputs["deadline_lock"],
        quiet_path=inputs["quiet"], mig_path=inputs["mig"], mps_path=inputs["mps"],
        xsched_path=inputs["xsched"], orion_path=inputs["orion"],
        pantheon_path=inputs["pantheon"],
    )
    args.summary.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary.resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    outputs = draw(summary, args.figure.resolve())
    provenance = {
        "schema_version": 1,
        "kind": "p9-six-system-imagenette-gate-figure-provenance",
        "inputs": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            str(args.summary.resolve().relative_to(ROOT)): sha256(args.summary),
            **{str(path.relative_to(ROOT)): sha256(path) for path in outputs},
        },
    }
    args.provenance.resolve().write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "summary": str(args.summary), "outputs": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
