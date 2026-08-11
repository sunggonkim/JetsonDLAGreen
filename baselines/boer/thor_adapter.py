#!/usr/bin/env python3
"""Algorithm-preserving BOER search adapter for the fixed Thor MIG layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


UPSTREAM_COMMIT = "df54815de3b1c9059f873a17c13f7d5203eedd3e"
FIDELITY = "upstream-ei-discrete-thor-adapter"
INITIAL_RANDOM_POINTS = 6
MAX_OBSERVATIONS = 20
NO_IMPROVEMENT_LIMIT = 5
EI_XI = 0.2
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    sm_percent: int
    offered_rps: int


@dataclass(frozen=True)
class Observation:
    candidate: Candidate
    objective: float
    feasible: bool
    metrics: dict[str, Any]
    source: str


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_spec(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("BOER spec must use schema_version 1")
    if value.get("system") != "BOER" or value.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("BOER spec does not bind the pinned artifact")
    contract = value.get("contract")
    if not isinstance(contract, dict) or contract.get("pressure_layout") != "1g+2g":
        raise ValueError("BOER Thor port requires fixed 1g+2g pressure layout")
    command = value.get("evaluator_command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise ValueError("BOER evaluator_command must be a nonempty string list")
    repo = REPO_ROOT
    lock_path_value = contract.get("deadline_lock_path")
    if lock_path_value is not None:
        lock_path = (repo / lock_path_value).resolve()
        if sha256(lock_path) != contract.get("deadline_lock_sha256"):
            raise ValueError("BOER deadline lock hash differs")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if (
            lock.get("kind") != "p9-dependent-pipeline-deadline-lock"
            or finite_number(lock.get("deadline_us"), "locked deadline")
            != finite_number(contract.get("deadline_us"), "contract deadline")
        ):
            raise ValueError("BOER contract differs from deadline lock")
    for point in value.get("static_capacity_profile", []):
        if set(point) != {
            "sm_percent",
            "max_rps",
            "evidence_path",
            "evidence_sha256",
        }:
            raise ValueError("BOER capacity point lacks raw evidence binding")
        evidence = (repo / point["evidence_path"]).resolve()
        if sha256(evidence) != point["evidence_sha256"]:
            raise ValueError("BOER capacity evidence hash differs")
        raw = json.loads(evidence.read_text(encoding="utf-8"))
        expected_model = contract.get("producer_model", "resnet10-detection")
        if (
            raw.get("model") != expected_model
            or raw.get("role") != "benchmark"
            or raw.get("execution_environment", {}).get(
                "mps_active_thread_percentage"
            )
            != point["sm_percent"]
            or raw.get("completed_requests") != 1000
            or finite_number(raw.get("throughput_per_second"), "profile RPS")
            != finite_number(point["max_rps"], "max_rps")
        ):
            raise ValueError("BOER capacity evidence content differs")
    return value


def candidates_from_spec(spec: dict[str, Any]) -> list[Candidate]:
    result: list[Candidate] = []
    seen: set[str] = set()
    for raw in spec.get("candidates", []):
        if not isinstance(raw, dict):
            raise ValueError("BOER candidate must be an object")
        candidate_id = raw.get("id")
        sm = raw.get("sm_percent")
        rps = raw.get("offered_rps")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError("BOER candidate id is empty or duplicated")
        if isinstance(sm, bool) or not isinstance(sm, int) or sm < 10 or sm > 90:
            raise ValueError("BOER SM percentage must be an integer in [10, 90]")
        if isinstance(rps, bool) or not isinstance(rps, int) or rps <= 0:
            raise ValueError("BOER offered RPS must be positive")
        seen.add(candidate_id)
        result.append(Candidate(candidate_id, sm, rps))
    if len(result) < INITIAL_RANDOM_POINTS:
        raise ValueError("BOER search needs at least six candidate points")
    return result


def static_capacity_line(points: list[dict[str, Any]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("BOER static pruning needs at least two capacity points")
    x = np.asarray([finite_number(point.get("sm_percent"), "sm_percent") for point in points])
    y = np.asarray([finite_number(point.get("max_rps"), "max_rps") for point in points])
    if len(set(x.tolist())) < 2:
        raise ValueError("BOER capacity points need distinct SM percentages")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def statically_feasible(candidate: Candidate, line: tuple[float, float]) -> bool:
    slope, intercept = line
    return candidate.offered_rps < candidate.sm_percent * slope + intercept


def dynamically_feasible(candidate: Candidate, failures: list[Candidate]) -> bool:
    return not any(
        candidate.sm_percent <= failed.sm_percent
        and candidate.offered_rps >= failed.offered_rps
        for failed in failures
    )


def boer_objective(metrics: dict[str, Any], demands: tuple[float, float]) -> float:
    if metrics["feasible"] == 0.0:
        return 0.5 * min(1.0, metrics["slo_limit_ms"] / metrics["worst_p99_ms"])
    served0 = min(metrics["served_rps_0"], demands[0]) / demands[0]
    served1 = min(metrics["served_rps_1"], demands[1]) / demands[1]
    return 0.5 + 0.25 * (served0 + served1)


def encode(candidate: Candidate, domain: list[Candidate]) -> np.ndarray:
    sm_values = [item.sm_percent for item in domain]
    rps_values = [item.offered_rps for item in domain]
    return np.asarray(
        [
            (candidate.sm_percent - min(sm_values)) / (max(sm_values) - min(sm_values) or 1),
            (candidate.offered_rps - min(rps_values)) / (max(rps_values) - min(rps_values) or 1),
        ],
        dtype=float,
    )


def matern52(a: np.ndarray, b: np.ndarray, length_scale: float = 0.35) -> np.ndarray:
    distance = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    scaled = math.sqrt(5.0) * distance / length_scale
    return (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)


def _normal_pdf(value: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(value / math.sqrt(2.0)))


def select_expected_improvement(
    remaining: list[Candidate], observations: list[Observation], domain: list[Candidate]
) -> Candidate:
    """Select the next discrete point using BOER's upstream EI rule.

    The published implementation optimizes continuous SM/RPS bounds with
    ``UtilityFunction(kind='ei', xi=0.2)``.  Thor evaluates integer quota/RPS
    points, so this adapter applies the same acquisition function to the
    explicitly supplied discrete candidate domain; it does not silently
    substitute a UCB policy.
    """
    train_x = np.stack([encode(item.candidate, domain) for item in observations])
    train_y = np.asarray([item.objective for item in observations], dtype=float)
    test_x = np.stack([encode(item, domain) for item in remaining])
    inverse = np.linalg.inv(matern52(train_x, train_x) + np.eye(len(train_x)) * 1e-6)
    cross = matern52(test_x, train_x)
    mean = cross @ inverse @ train_y
    variance = np.maximum(1e-12, 1.0 - np.sum((cross @ inverse) * cross, axis=1))
    std = np.sqrt(variance)
    incumbent = float(np.max(train_y))
    improvement = mean - incumbent - EI_XI
    z = improvement / std
    expected = improvement * _normal_cdf(z) + std * _normal_pdf(z)
    expected[std <= 1e-9] = 0.0
    return remaining[int(np.argmax(expected))]


def bind_evaluator_evidence(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("result_dir")
    if not isinstance(value, str) or not value:
        raise ValueError("BOER hardware evaluator omitted result_dir")
    directory = pathlib.Path(value).resolve()
    if not directory.is_dir():
        raise ValueError("BOER hardware evidence directory is missing")
    files = sorted(path for path in directory.iterdir() if path.is_file())
    required = {"pipeline.csv", "pipeline.json", "background.json"}
    if not required.issubset(path.name for path in files):
        raise ValueError("BOER hardware evidence is incomplete")
    return {
        "result_dir": str(directory),
        "sha256": {path.name: sha256(path) for path in files},
    }


def evaluate_subprocess(spec: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "BOER_CANDIDATE_ID": candidate.candidate_id,
            "BOER_SM_PERCENT": str(candidate.sm_percent),
            "BOER_OFFERED_RPS": str(candidate.offered_rps),
        }
    )
    completed = subprocess.run(
        spec["evaluator_command"], check=True, capture_output=True, text=True, env=environment
    )
    raw = json.loads(completed.stdout)
    if not isinstance(raw, dict):
        raise ValueError("BOER evaluator output must be a JSON object")
    keys = ("feasible", "slo_limit_ms", "worst_p99_ms", "served_rps_0", "served_rps_1")
    metrics = {key: finite_number(raw.get(key), key) for key in keys}
    for key in ("deadline_miss_rate", "dmr_target"):
        if key in raw:
            metrics[key] = finite_number(raw[key], key)
    if metrics["feasible"] not in (0.0, 1.0):
        raise ValueError("BOER feasible must be 0 or 1")
    if min(metrics[key] for key in keys[1:]) <= 0.0:
        raise ValueError("BOER evaluator metrics must be positive")
    metrics["evidence"] = bind_evaluator_evidence(raw)
    return metrics


def run_search(
    spec: dict[str, Any],
    evaluator: Callable[[dict[str, Any], Candidate], dict[str, Any]] = evaluate_subprocess,
) -> dict[str, Any]:
    domain = candidates_from_spec(spec)
    line = static_capacity_line(spec.get("static_capacity_profile", []))
    raw_demands = spec.get("tenant_demands_rps")
    if not isinstance(raw_demands, list) or len(raw_demands) != 2:
        raise ValueError("BOER requires exactly two tenant demands")
    demands = tuple(finite_number(value, "tenant demand") for value in raw_demands)
    if min(demands) <= 0.0:
        raise ValueError("BOER tenant demands must be positive")

    rng = np.random.default_rng(int(spec.get("seed", 4)))
    remaining = list(domain)
    sampled = rng.choice(len(remaining), size=INITIAL_RANDOM_POINTS, replace=False)
    initial_ids = {remaining[int(index)].candidate_id for index in sampled}
    initial = [item for item in remaining if item.candidate_id in initial_ids]
    observations: list[Observation] = []
    dynamic_failures: list[Candidate] = []
    best = -math.inf
    no_improvement = 0

    while remaining and len(observations) < MAX_OBSERVATIONS:
        candidate = initial.pop(0) if initial else select_expected_improvement(
            remaining, observations, domain
        )
        remaining.remove(candidate)
        if not statically_feasible(candidate, line):
            capacity = candidate.sm_percent * line[0] + line[1]
            observation = Observation(
                candidate, max(0.0, min(0.5, capacity / candidate.offered_rps / 2.0)),
                False, {}, "static-prune"
            )
        elif not dynamically_feasible(candidate, dynamic_failures):
            observation = Observation(candidate, 0.0, False, {}, "dynamic-prune")
        else:
            metrics = evaluator(spec, candidate)
            feasible = metrics["feasible"] == 1.0
            observation = Observation(
                candidate, boer_objective(metrics, demands), feasible, metrics, "hardware"
            )
            if not feasible:
                dynamic_failures.append(candidate)
        observations.append(observation)
        if observation.objective > best:
            best, no_improvement = observation.objective, 0
        else:
            no_improvement += 1
        if len(observations) >= INITIAL_RANDOM_POINTS and no_improvement >= NO_IMPROVEMENT_LIMIT:
            break

    measured = [item for item in observations if item.source == "hardware" and item.feasible]
    if not measured and not spec.get("allow_no_feasible", False):
        raise RuntimeError("BOER did not measure a feasible Thor configuration")
    selected = max(measured, key=lambda item: item.objective) if measured else None
    result = {
        "schema_version": 1,
        "system": "BOER",
        "status": "selected" if selected is not None else "no-feasible-configuration",
        "provenance": {"upstream_commit": UPSTREAM_COMMIT, "fidelity": FIDELITY},
        "contract": spec["contract"],
        "search": {
            "initial_random_points": INITIAL_RANDOM_POINTS,
            "maximum_observations": MAX_OBSERVATIONS,
            "no_improvement_limit": NO_IMPROVEMENT_LIMIT,
            "acquisition": "expected-improvement",
            "xi": EI_XI,
            "candidate_domain": "discrete-explicit",
            "upstream_domain": "continuous",
            "numeric_comparison_allowed": False,
            "seed": int(spec.get("seed", 4)),
            "static_capacity_slope": line[0],
            "static_capacity_intercept": line[1],
            "static_capacity_profile": spec.get("static_capacity_profile", []),
        },
        "selected": None,
        "observations": [
            {
                "id": item.candidate.candidate_id,
                "sm_percent": item.candidate.sm_percent,
                "complement_sm_percent": 100 - item.candidate.sm_percent,
                "offered_rps": item.candidate.offered_rps,
                "objective": item.objective,
                "feasible": item.feasible,
                "source": item.source,
                "metrics": item.metrics,
            }
            for item in observations
        ],
    }
    if selected is not None:
        result["selected"] = {
            "id": selected.candidate.candidate_id,
            "sm_percent": selected.candidate.sm_percent,
            "complement_sm_percent": 100 - selected.candidate.sm_percent,
            "offered_rps": selected.candidate.offered_rps,
            "objective": selected.objective,
            "metrics": selected.metrics,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = run_search(load_spec(args.spec))
    result["provenance"]["spec_path"] = str(args.spec.resolve())
    result["provenance"]["spec_sha256"] = sha256(args.spec.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
