#!/usr/bin/env python3
"""Run the predeclared nonthermal Whisper edge-tenant regime matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    label: str
    description: str
    background_model_name: str
    background_engine: str
    deployment_scope: str
    additional_backgrounds: tuple[str, ...]
    balanced_rate_rps: float


SCENARIOS = (
    Scenario(
        "speech-plus-nlp",
        "Speech + NLP",
        "interactive ASR with queued intent and text classification",
        "distilbert-sst2",
        "models/engines/mig-1g-q100/distilbert-sst2.engine",
        "multi-channel-robot-or-edge-gateway-stress",
        (),
        19.0,
    ),
    Scenario(
        "speech-plus-vision",
        "Speech + Vision",
        "interactive ASR with queued camera perception",
        "resnet10-detection",
        "models/engines/mig-1g-q100/resnet10-detection.engine",
        "multi-sensor-robot-or-video-gateway-stress",
        (),
        21.0,
    ),
    Scenario(
        "speech-plus-speech",
        "Speech + Speech",
        "interactive ASR with a queued secondary speech encoder",
        "whisper-tiny-encoder",
        "models/engines/mig-1g-q100/whisper-tiny-encoder.engine",
        "multi-model-multi-channel-speech-gateway-stress",
        (),
        21.0,
    ),
    Scenario(
        "speech-plus-multimodal",
        "Speech + Multimodal",
        "interactive ASR with queued NLP, camera perception, and secondary speech encoding",
        "distilbert-sst2",
        "models/engines/mig-1g-q100/distilbert-sst2.engine",
        "multimodal-robot-or-edge-gateway-stress",
        (
            "resnet10-detection=models/engines/mig-1g-q100/resnet10-detection.engine",
            "whisper-tiny-encoder=models/engines/mig-1g-q100/whisper-tiny-encoder.engine",
        ),
        20.0,
    ),
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_for(
    args: argparse.Namespace,
    scenario: Scenario,
    phase: str,
    output: pathlib.Path,
) -> list[str]:
    if phase == "directional":
        rates = args.directional_rates
        sessions = 1
    elif phase == "balanced":
        rates = [scenario.balanced_rate_rps]
        sessions = args.balanced_sessions
    else:
        raise ValueError("phase must be directional or balanced")
    command = [
        sys.executable,
        str(args.runner),
        "--result-dir",
        str(output),
        "--input-trace",
        str(args.input_trace),
        "--background-engine",
        str(args.repo / scenario.background_engine),
        "--background-model-name",
        scenario.background_model_name,
        "--scenario-id",
        scenario.scenario_id,
        "--scenario-label",
        scenario.label,
        "--scenario-description",
        scenario.description,
        "--deployment-scope",
        scenario.deployment_scope,
        "--sessions",
        str(sessions),
        "--requests",
        str(args.requests),
        "--warmup",
        str(args.warmup),
        "--rates",
    ]
    command.extend(str(rate) for rate in rates)
    for background in scenario.additional_backgrounds:
        command.extend(("--additional-background", background))
    return command


def validate_summary(
    path: pathlib.Path,
    scenario: Scenario,
    phase: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_design = (
        "directional-sweep" if phase == "directional" else "balanced-repeated"
    )
    if (
        value.get("kind") != "p9-whisper-asr-mig-crossover"
        or value.get("thermal_campaign") is not False
        or value.get("study_design") != expected_design
        or value.get("scenario", {}).get("id") != scenario.scenario_id
        or value.get("comparator_output_contract") != "byte-identical"
    ):
        raise ValueError(f"invalid {phase} summary for {scenario.scenario_id}")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty {phase} summary for {scenario.scenario_id}")
    expected_rates = (
        set(args.directional_rates)
        if phase == "directional" else {scenario.balanced_rate_rps}
    )
    expected_sessions = 1 if phase == "directional" else args.balanced_sessions
    expected_rows = len(expected_rates) * 3 * expected_sessions
    if (
        len(rows) != expected_rows
        or {float(row["rate_rps"]) for row in rows} != expected_rates
        or {str(row["mode"]) for row in rows}
        != {"nvidia-mig", "nvidia-mps-static-split", "quiet"}
        or {int(row["session"]) for row in rows}
        != set(range(1, expected_sessions + 1))
        or any(int(row["requests"]) != args.requests for row in rows)
        or len({str(row["output_sha256"]) for row in rows}) != 1
    ):
        raise ValueError(f"incomplete {phase} matrix for {scenario.scenario_id}")
    return value


def first_mig_only_failure(summary: dict[str, Any]) -> float:
    """Find the first rate where MIG misses and both split modes remain clean."""
    rows = summary.get("aggregate")
    if not isinstance(rows, list):
        raise ValueError("directional aggregate is missing")
    rates = sorted({float(row["rate_rps"]) for row in rows})
    for rate in rates:
        misses = {
            str(row["mode"]): int(row["deadline_misses"])
            for row in rows
            if float(row["rate_rps"]) == rate
        }
        if (
            misses.get("nvidia-mig", 0) > 0
            and misses.get("nvidia-mps-static-split") == 0
            and misses.get("quiet") == 0
        ):
            return rate
    raise ValueError("directional sweep has no MIG-only failure crossover")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument("--repo", type=pathlib.Path, default=repo)
    parser.add_argument(
        "--runner",
        type=pathlib.Path,
        default=repo / "scripts/run_p9_whisper_asr_mig_crossover.py",
    )
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--input-trace", type=pathlib.Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("directional", "balanced", "all"),
        default="all",
    )
    parser.add_argument(
        "--directional-rates",
        type=float,
        nargs="+",
        default=(15.0, 17.0, 19.0, 20.0, 21.0),
    )
    parser.add_argument("--balanced-sessions", type=int, default=3)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.runner = args.runner.resolve()
    args.result_dir = args.result_dir.resolve()
    args.input_trace = args.input_trace.resolve()
    if not args.repo.is_dir() or not args.runner.is_file() or not args.input_trace.is_file():
        raise ValueError("repo, runner, and input trace must exist")
    if (
        args.requests < 100
        or args.warmup < 0
        or args.balanced_sessions != 3
        or set(args.directional_rates) != {15.0, 17.0, 19.0, 20.0, 21.0}
    ):
        raise ValueError("campaign request/session/rate contract differs")
    for scenario in SCENARIOS:
        if not (args.repo / scenario.background_engine).is_file():
            raise ValueError(f"missing background engine for {scenario.scenario_id}")
        for specification in scenario.additional_backgrounds:
            _, separator, engine = specification.partition("=")
            if not separator or not (args.repo / engine).is_file():
                raise ValueError(
                    f"missing additional background engine for {scenario.scenario_id}"
                )

    phases = (
        ("directional", "balanced") if args.phase == "all" else (args.phase,)
    )
    args.result_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    for phase in phases:
        for scenario in SCENARIOS:
            output = args.result_dir / phase / scenario.scenario_id
            summary_path = output / "summary.json"
            if summary_path.is_file():
                value = validate_summary(summary_path, scenario, phase, args)
            else:
                if output.exists():
                    raise ValueError(f"incomplete result directory exists: {output}")
                command = command_for(args, scenario, phase, output)
                subprocess.run(
                    command, cwd=args.repo, check=True, stdout=subprocess.DEVNULL
                )
                value = validate_summary(summary_path, scenario, phase, args)
            loaded[(phase, scenario.scenario_id)] = value
            summaries.append(
                {
                    "phase": phase,
                    "scenario_id": scenario.scenario_id,
                    "path": str(summary_path),
                    "sha256": sha256(summary_path),
                    "rows": len(value["rows"]),
                }
            )

    if set(phases) == {"directional", "balanced"}:
        for scenario in SCENARIOS:
            calibrated = first_mig_only_failure(
                loaded[("directional", scenario.scenario_id)]
            )
            if calibrated != scenario.balanced_rate_rps:
                raise ValueError(
                    f"balanced rate is not the first MIG-only crossover for {scenario.scenario_id}"
                )

    campaign = {
        "schema_version": 1,
        "kind": "p9-whisper-edge-mix-campaign",
        "evidence_class": "exploratory-nonthermal-motivation",
        "thermal_campaign": False,
        "selection_policy": "four-predeclared-real-model-edge-tenant-categories",
        "balanced_rate_policy": (
            "first-directional-rate-with-MIG-misses-and-both-split-modes-zero"
        ),
        "input_policy": "real-labelled-windows-cyclic-performance-replay",
        "scenario_order": [asdict(scenario) for scenario in SCENARIOS],
        "directional_rates_rps": list(args.directional_rates),
        "balanced_rate_rps": {
            scenario.scenario_id: scenario.balanced_rate_rps
            for scenario in SCENARIOS
        },
        "balanced_sessions": args.balanced_sessions,
        "requests_per_run": args.requests,
        "summaries": summaries,
        "runner": {"path": str(args.runner), "sha256": sha256(args.runner)},
        "campaign_driver": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256(pathlib.Path(__file__).resolve()),
        },
        "input_trace": {
            "path": str(args.input_trace),
            "sha256": sha256(args.input_trace),
        },
        "claim_guard": (
            "The model combinations match plausible Jetson services, but fixed-rate "
            "arrival and saturated background queues are stress regimes rather than "
            "field traces. No thermal claim is made."
        ),
    }
    campaign_path = args.result_dir / "campaign.json"
    campaign_path.write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(campaign, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
