#!/usr/bin/env python3
"""Generate paper tables from replay-verified P9 aggregate evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


SYSTEM_ORDER = ("NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "gpulet", "QUIET")
HEADLINE_SYSTEM_ORDER = ("NVIDIA MIG", "NVIDIA MPS", "Orion", "XSched", "Pantheon", "QUIET")
FORBIDDEN_PRESENTATION_NAMES = ("mig-governor", "joint-governor", "jdg-governor")
ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_manifest() -> dict[str, Any]:
    path = ROOT / "docs" / "p9-comparator-manifest.json"
    value = load_json(path)
    if value.get("proposed_system") != "QUIET" or not isinstance(value.get("rows"), dict):
        raise ValueError("comparator manifest is malformed")
    return value


def manifest_eligibility(manifest: dict[str, Any], system: str) -> tuple[bool, str]:
    entry = manifest["rows"].get(system)
    if isinstance(entry, dict):
        allowed, status = entry.get("numeric_comparison_allowed"), entry.get("status")
        if not isinstance(allowed, bool) or not isinstance(status, str):
            raise ValueError(f"manifest contract is malformed for {system}")
        return allowed, status
    if system in manifest.get("structural_controls", []):
        return False, "structural-only"
    raise ValueError(f"system {system!r} is absent from comparator manifest")


def validate(aggregate: dict[str, Any], heldout: dict[str, Any]) -> None:
    if aggregate.get("kind") != "p9-common-sota-williams-aggregate":
        raise ValueError("unexpected common-workload aggregate kind")
    if aggregate.get("proposed_system") != "QUIET":
        raise ValueError("the sole proposed presentation name must be QUIET")
    if aggregate.get("scope") != "order-balanced-raw-replayed-nonthermal-campaign":
        raise ValueError("the table requires raw-replayed order-balanced evidence")
    systems = aggregate.get("systems")
    headline = aggregate.get("headline_systems")
    if headline is not None:
        declared_order = aggregate.get("headline_system_order")
        if (
            not isinstance(headline, dict)
            or (
                tuple(declared_order) != HEADLINE_SYSTEM_ORDER
                if declared_order is not None
                else tuple(headline) != HEADLINE_SYSTEM_ORDER
            )
            or any(name not in headline for name in HEADLINE_SYSTEM_ORDER)
        ):
            raise ValueError("headline system order differs")
    elif not isinstance(systems, dict) or tuple(systems) != SYSTEM_ORDER:
        raise ValueError("common-workload system order differs")
    if heldout.get("kind") != "p9-quiet-frozen-plan-heldout-load-sweep":
        raise ValueError("unexpected held-out aggregate kind")
    if heldout.get("proposed_system") != "QUIET":
        raise ValueError("held-out proposed presentation name differs")
    rendered = json.dumps((aggregate, heldout), sort_keys=True)
    if any(name in rendered for name in FORBIDDEN_PRESENTATION_NAMES):
        raise ValueError("an internal policy identifier leaked into paper evidence")


def format_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest = load_manifest()
    source = aggregate.get("headline_systems", aggregate["systems"])
    order = HEADLINE_SYSTEM_ORDER if "headline_systems" in aggregate else SYSTEM_ORDER
    for system in order:
        value = source[system]
        manifest_allowed, manifest_status = manifest_eligibility(manifest, system)
        allowed = value.get("numeric_comparison_allowed", manifest_allowed)
        if not isinstance(allowed, bool):
            allowed = manifest_allowed
        allowed = allowed and manifest_allowed
        status = value.get("comparison_status", manifest_status)
        if not isinstance(status, str) or not status:
            status = manifest_status
        slo_qualified = bool(value.get("slo_confidence_qualified", False))
        if "requests" not in value:
            rows.append({
                "system": system,
                "requests": None,
                "misses": None,
                "dmr_percent": None,
                "cp95_percent": None,
                "p99_us": None,
                "p999_us": None,
                "maximum_us": None,
                "background_goodput_rps": None,
                "qualified": False,
                "status": status,
            })
            continue
        if not allowed or not slo_qualified:
            if allowed and not slo_qualified:
                status = f"{status};slo-infeasible"
            rows.append({
                "system": system,
                "requests": None,
                "misses": None,
                "dmr_percent": None,
                "cp95_percent": None,
                "p99_us": None,
                "p999_us": None,
                "maximum_us": None,
                "background_goodput_rps": None,
                "qualified": False,
                "status": status,
            })
            continue
        rows.append({
            "system": system,
            "requests": int(value["requests"]),
            "misses": int(value["misses"]),
            "dmr_percent": 100.0 * float(value["observed_dmr"]),
            "cp95_percent": 100.0 * float(value["dmr_cp95_upper"]),
            "p99_us": float(value["pooled_p99_us"]),
            "p999_us": float(value["pooled_p999_us"]),
            "maximum_us": float(value["maximum_us"]),
            "background_goodput_rps": float(value["background_goodput_rps_mean"]),
            "qualified": bool(value["slo_confidence_qualified"]),
            "status": status,
        })
    return rows


def row_is_numeric(row: dict[str, Any]) -> bool:
    """Return whether a row contains an eligible numeric comparison."""
    return row.get("requests") is not None and bool(row.get("qualified"))


def latex(aggregate: dict[str, Any], heldout: dict[str, Any]) -> str:
    rows = format_rows(aggregate)
    quiet = next(row for row in rows if row["system"] == "QUIET")
    lines = [
        "% Generated by analysis/generate_p9_sota_tables.py; do not edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Common dependent TensorRT workload. All rows use six Williams sequences and a frozen $770.605$-$\\mu$s deadline. Results are raw-replayed but not thermally normalized.}",
        "\\label{tab:p9-sota-common}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "System & Misses & DMR (\\%) & p99 ($\\mu$s) & p99.9 ($\\mu$s) & BE req/s \\\\",
        "\\midrule",
    ]
    for row in rows:
        label = "\\textbf{QUIET}" if row["system"] == "QUIET" else row["system"]
        if row["requests"] is None:
            lines.append(f"{label} & \\multicolumn{{5}}{{c}}{{\\emph{{{row['status']}}}}} \\\\")
        else:
            lines.append(
                f"{label} & {row['misses']}/{row['requests']} & {row['dmr_percent']:.4f} & "
                f"{row['p99_us']:.2f} & {row['p999_us']:.2f} & "
                f"{row['background_goodput_rps']:.2f} \\\\"
            )
    lines.extend((
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Held-out offered-load sweep with the frozen QUIET plan. These 1,100-request points locate the empirical boundary; individually they do not certify the 0.05\\% DMR objective.}",
        "\\label{tab:p9-quiet-heldout}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "Offered & Goodput & Misses & p99 ($\\mu$s) & Max ($\\mu$s) \\\\",
        "\\midrule",
    ))
    for row in heldout["loads"]:
        lines.append(
            f"{float(row['offered_rps']):.0f} & {float(row['background_goodput_rps']):.2f} & "
            f"{int(row['misses'])}/{int(row['requests'])} & {float(row['p99_us']):.2f} & "
            f"{float(row['maximum_us']):.2f} \\\\"
        )
    lines.extend((
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
        f"\\newcommand{{\\quietPooledPnn}}{{{quiet['p99_us']:.2f}\\xspace}}",
        f"\\newcommand{{\\quietPooledPnnn}}{{{quiet['p999_us']:.2f}\\xspace}}",
        f"\\newcommand{{\\quietDmrUpper}}{{{quiet['cp95_percent']:.4f}\\%\\xspace}}",
        "",
    ))
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args(argv)
    aggregate, heldout = load_json(args.aggregate), load_json(args.heldout)
    validate(aggregate, heldout)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text(latex(aggregate, heldout), encoding="utf-8")
    write_csv(args.csv_output, format_rows(aggregate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
