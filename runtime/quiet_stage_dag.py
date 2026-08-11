#!/usr/bin/env python3
"""Select a communication-aware QUIET plan from measured stage profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
from runtime.quiet_dag_contract import validate_dag


SCHEMA_VERSION = 1
SYSTEM_NAME = "QUIET"

TRANSPORT_ALIASES = {
    "registered-shared-sysmem-direct-binding": "registered-direct",
    "full-coherent-registered-system-memory": "registered-direct",
    "registered-direct": "registered-direct",
    "pinned-shared-sysmem-d2h-h2d": "pinned-bounce",
    "pinned-bounce": "pinned-bounce",
    "pageable-shared-sysmem-d2h-h2d": "pageable-bounce",
    "pageable-bounce": "pageable-bounce",
}

PLACEMENT_VARIANTS = {
    "fixed-1g-producer-2g-consumer": {
        "producer_slice": "1g",
        "consumer_slice": "2g",
        "producer_uuid_key": "JDG_MIG_SMALL_UUID",
        "consumer_uuid_key": "JDG_MIG_BIG_UUID",
    },
    "fixed-2g-producer-1g-consumer": {
        "producer_slice": "2g",
        "consumer_slice": "1g",
        "producer_uuid_key": "JDG_MIG_BIG_UUID",
        "consumer_uuid_key": "JDG_MIG_SMALL_UUID",
    },
}


def finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_transport(value: Any) -> str:
    if not isinstance(value, str) or value not in TRANSPORT_ALIASES:
        raise ValueError(f"unsupported QUIET plan transport: {value}")
    return TRANSPORT_ALIASES[value]


def _placement_label(value: Any, label: str) -> tuple[str, int]:
    if not isinstance(value, str):
        raise ValueError(f"{label} placement label is malformed")
    match = re.fullmatch(r"(1g|2g)-q(10|25|50|75|90|100)", value)
    if match is None:
        raise ValueError(f"{label} placement label is not quota-bound")
    return match.group(1), int(match.group(2))


def _candidate_quotas(candidate_id: Any) -> tuple[int, int]:
    if not isinstance(candidate_id, str):
        raise ValueError("QUIET candidate_id is malformed")
    match = re.search(
        r"q(10|25|50|75|90|100)-q(10|25|50|75|90|100)(?:@|$)",
        candidate_id,
    )
    if match is None:
        raise ValueError("QUIET candidate_id does not bind producer/background quotas")
    return int(match.group(1)), int(match.group(2))


def actuate_selected_plan(
    plan: dict[str, Any],
    mig: dict[str, str],
    placement_variant: str,
    requested_transport: str,
    requested_producer_quota: int,
    requested_background_quota: int,
    requested_scope: str,
    workload: str,
    dependency_mode: str,
) -> dict[str, Any]:
    """Bind a selected plan to launch-time resources before CUDA is touched."""
    placement = PLACEMENT_VARIANTS.get(placement_variant)
    if placement is None:
        raise ValueError("unknown QUIET placement variant")
    selected = plan.get("selected_plan")
    if (
        plan.get("proposed_system") != SYSTEM_NAME
        or plan.get("status") != "selected"
        or not isinstance(selected, dict)
        or selected.get("feasible") is not True
    ):
        raise ValueError("QUIET plan is not a selected feasible plan")
    plan_placement = selected.get("placement")
    expected_placement = {
        "producer": f"{placement['producer_slice']}-q{requested_producer_quota}",
        "consumer": f"{placement['consumer_slice']}-q100",
    }
    if plan_placement != expected_placement:
        raise ValueError("QUIET plan placement/quota differs from CLI")
    selected_variant = selected.get("placement_variant", placement_variant)
    if selected_variant != placement_variant:
        raise ValueError("QUIET plan placement variant differs from CLI")
    producer_slice, producer_quota = _placement_label(
        plan_placement.get("producer"), "producer"
    )
    consumer_slice, consumer_quota = _placement_label(
        plan_placement.get("consumer"), "consumer"
    )
    if (
        producer_slice != placement["producer_slice"]
        or consumer_slice != placement["consumer_slice"]
        or producer_quota != requested_producer_quota
        or consumer_quota != 100
    ):
        raise ValueError("QUIET plan slice/quota binding differs from CLI")
    producer_candidate_quota, background_quota = _candidate_quotas(
        selected.get("candidate_id")
    )
    if (
        producer_candidate_quota != requested_producer_quota
        or background_quota != requested_background_quota
    ):
        raise ValueError("QUIET plan candidate quotas differ from CLI")
    edges = selected.get("dag", {}).get("edges")
    if not isinstance(edges, list) or len(edges) != 1:
        raise ValueError("QUIET plan must have exactly one executable edge")
    planned_transport = canonical_transport(edges[0].get("transport"))
    cli_transport = canonical_transport(requested_transport)
    if planned_transport != cli_transport:
        raise ValueError("QUIET plan transport differs from CLI")
    scope = selected.get("protection_scope", "producer")
    if scope != requested_scope:
        raise ValueError("QUIET plan protection scope differs from CLI")
    if dependency_mode != "dependent":
        raise ValueError("QUIET plan actuation requires dependent mode")
    if not isinstance(workload, str) or not workload:
        raise ValueError("QUIET plan workload is missing")
    producer_uuid = mig[placement["producer_uuid_key"]]
    consumer_uuid = mig[placement["consumer_uuid_key"]]
    manifest = {
        "system": SYSTEM_NAME,
        "candidate_id": selected["candidate_id"],
        "placement_variant": placement_variant,
        "producer_uuid": producer_uuid,
        "consumer_uuid": consumer_uuid,
        "transport": planned_transport,
        "producer_quota_percent": producer_quota,
        "consumer_quota_percent": consumer_quota,
        "background_quota_percent": background_quota,
        "protection_scope": scope,
        "workload": workload,
        "dependency_mode": dependency_mode,
        "tail_bound_method": selected.get(
            "tail_bound_method", "legacy-component-p99-sum"
        ),
        "tail_bound_promotable": selected.get("tail_bound_promotable", False),
        "admission": {
            "uuid_bound": True,
            "transport_bound": True,
            "quota_bound": True,
            "protection_scope_bound": True,
            "tail_bound_bound": True,
            "before_cuda_context": True,
            "best_effort_admitted": background_quota > 0,
        },
    }
    return manifest


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    profile_path: Path
    placement: dict[str, str]
    placement_variant: str
    background_goodput_rps: float
    pre_release_guard_p99_us: float
    reservation_margin_us: float
    protection_scope: str


def parse_candidate(raw: dict[str, Any], root: Path) -> Candidate:
    expected = {
        "candidate_id",
        "profile_path",
        "placement",
        "placement_variant",
        "background_goodput_rps",
        "pre_release_guard_p99_us",
        "reservation_margin_us",
        "protection_scope",
    }
    # Specs emitted before placement search existed remain readable, but every
    # newly generated candidate carries an explicit variant.
    # Candidate specs may carry replay metadata emitted by
    # build_p9_quiet_candidate_spec.py.  These fields are descriptive and are
    # still checked by the builder; the runtime selector consumes only the
    # scheduling fields above.
    metadata = {
        "summary", "profile_sha256", "background_quota_percent",
        "pipeline_requests", "deadline_misses", "observed_p99_us",
        "slo_qualified", "workload", "background_period_ms",
        "background_offered_rps", "deadline_mode", "deadline_us",
        "deadline_lock_sha256", "checksum_mode", "correctness_validated",
        "unique_payload_checksums", "unique_policy_output_checksums",
    }
    unknown = set(raw) - expected - metadata
    missing = expected - set(raw)
    if unknown or missing - {"placement_variant", "protection_scope"}:
        raise ValueError(
            "candidate keys differ: "
            f"{sorted(unknown | (missing - {'placement_variant', 'protection_scope'}))}"
        )
    candidate_id = raw["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id must be a nonempty internal identifier")
    placement = raw["placement"]
    if (
        not isinstance(placement, dict)
        or set(placement) != {"producer", "consumer"}
        or not all(isinstance(value, str) and value for value in placement.values())
    ):
        raise ValueError("placement must map producer and consumer to instances")
    placement_variant = raw.get(
        "placement_variant", "fixed-1g-producer-2g-consumer"
    )
    if not isinstance(placement_variant, str) or not placement_variant:
        raise ValueError("placement_variant must be a nonempty string")
    protection_scope = raw.get("protection_scope", "producer")
    if protection_scope not in {"producer", "consumer", "pipeline"}:
        raise ValueError("protection_scope must be producer, consumer, or pipeline")
    profile_path = Path(raw["profile_path"])
    if not profile_path.is_absolute():
        profile_path = (root / profile_path).resolve()
    return Candidate(
        candidate_id=candidate_id,
        profile_path=profile_path,
        placement=dict(placement),
        placement_variant=placement_variant,
        background_goodput_rps=finite_nonnegative(
            raw["background_goodput_rps"], "background_goodput_rps"
        ),
        pre_release_guard_p99_us=finite_nonnegative(
            raw["pre_release_guard_p99_us"], "pre_release_guard_p99_us"
        ),
        reservation_margin_us=finite_nonnegative(
            raw["reservation_margin_us"], "reservation_margin_us"
        ),
        protection_scope=protection_scope,
    )


def load_pipeline_profile(
    candidate: Candidate, deadline_us: float, critical_lookahead_us: float
) -> dict[str, Any]:
    raw_bytes = candidate.profile_path.read_bytes()
    profile = json.loads(raw_bytes)
    if profile.get("status") != "ok":
        raise ValueError(f"profile is not successful: {candidate.profile_path}")
    pipeline = profile.get("pipeline")
    stage_names = {
        "resnet10-layer7-cov-to-control-mlp": ("perception", "control"),
        "resnet10-backbone-to-learned-detection-head": (
            "backbone", "detection-head"
        ),
        "resnet50-backbone-to-classification-head": (
            "backbone", "classification-head"
        ),
        "whisper-last-hidden-state-to-projection-mlp": (
            "audio-encoder", "projection"
        ),
    }.get(pipeline)
    if stage_names is None:
        raise ValueError("unexpected pipeline workload")
    if int(profile.get("checksum_failures", -1)) != 0:
        raise ValueError("payload checksum failure in profile")
    if positive_int(profile.get("iterations"), "iterations") < 2:
        raise ValueError("profile needs at least two measured requests")
    if positive_int(profile.get("unique_payload_checksums"), "payload checksums") < 2:
        raise ValueError("profile does not prove changing payloads")
    if positive_int(
        profile.get("unique_policy_output_checksums"), "policy output checksums"
    ) < 2:
        raise ValueError("profile does not prove payload-dependent outputs")

    stage = profile.get("stage_latency_us")
    if not isinstance(stage, dict):
        raise ValueError("profile lacks stage_latency_us")
    producer_us = finite_nonnegative(stage.get("producer_compute_p99"), "producer p99")
    precise_transport = stage.get("transport_notification_p99")
    if precise_transport is not None:
        precise_transport = stage.get("edge_transport_p99", (
            finite_nonnegative(
                stage.get("producer_handoff_copy_p99", 0), "producer handoff copy"
            )
            + finite_nonnegative(precise_transport, "transport notification")
            + finite_nonnegative(
                stage.get("consumer_handoff_copy_p99", 0), "consumer handoff copy"
            )
        ))
    transport_us = finite_nonnegative(
        precise_transport
        if precise_transport is not None
        else stage.get("transport_ready_p99"),
        "transport p99",
    )
    consumer_us = finite_nonnegative(stage.get("consumer_compute_p99"), "consumer p99")
    verification_us = finite_nonnegative(
        stage.get("output_verification_p99"), "verification p99"
    )
    validation_excluded = stage.get("validation_excluded_end_to_end_p99")
    profile_deadline_mode = profile.get("deadline_mode")
    if profile_deadline_mode is None:
        profile_deadline_mode = "validation-excluded" if validation_excluded is not None else "wall"
    if profile_deadline_mode not in {"wall", "validation-excluded"}:
        raise ValueError("profile deadline mode is unsupported")
    observed_value = (
        validation_excluded
        if profile_deadline_mode == "validation-excluded"
        else profile.get("end_to_end_us", {}).get("p99")
    )
    observed_p99_us = finite_nonnegative(observed_value, "observed end-to-end p99")
    joint_tail = profile.get("joint_tail_p99_us")
    explicit_joint_tail = joint_tail is not None
    if (
        joint_tail is None
        and profile_deadline_mode == "wall"
        and profile.get("risk_budget_us") is None
        and profile.get("pipeline") in {
            "resnet10-backbone-to-learned-detection-head",
            "resnet50-backbone-to-classification-head",
        }
    ):
        # The production wall is measured per request in end_to_end_us.  Use
        # that observed joint tail when the profiler did not emit the newer
        # explicit field; summing component p99 values would double-count
        # validation/execution overlap and is not a valid admission bound.
        joint_tail = profile.get("end_to_end_us", {}).get("p99")
    risk_budget = profile.get("risk_budget_us")
    if joint_tail is not None and risk_budget is not None:
        raise ValueError("profile cannot provide both joint tail and risk budget")
    if joint_tail is not None:
        response_reservation_us = (
            finite_nonnegative(joint_tail, "joint_tail_p99_us")
            + candidate.reservation_margin_us
        )
        tail_bound_method = (
            "joint-request-p99" if explicit_joint_tail
            else "measured-production-wall-p99"
        )
        tail_bound_promotable = True
    elif risk_budget is not None:
        if not isinstance(risk_budget, dict) or not risk_budget:
            raise ValueError("risk_budget_us must be a nonempty object")
        response_reservation_us = candidate.reservation_margin_us
        for name, value in risk_budget.items():
            response_reservation_us += finite_nonnegative(
                value, f"risk_budget_us.{name}"
            )
        tail_bound_method = "explicit-risk-budget"
        tail_bound_promotable = True
    else:
        # Preserve old characterization profiles, but make the unsafe
        # statistical assumption visible in every emitted plan.  This path is
        # never a formal SLO promotion bound.
        response_reservation_us = (
            producer_us
            + transport_us
            + consumer_us
            + verification_us
            + candidate.reservation_margin_us
        )
        tail_bound_method = "legacy-component-p99-sum"
        tail_bound_promotable = False
    uncovered_guard_us = max(
        0.0, candidate.pre_release_guard_p99_us - critical_lookahead_us
    )
    arrival_to_completion_reservation_us = (
        response_reservation_us + uncovered_guard_us
    )
    dag = {
        "stages": [
            {"id": stage_names[0], "p99_us": producer_us},
            {"id": stage_names[1], "p99_us": consumer_us},
        ],
        "edges": [
            {
                "source": stage_names[0],
                "target": stage_names[1],
                "payload_bytes": positive_int(profile.get("payload_bytes"), "payload_bytes"),
                "transport": profile.get("transport"),
                "p99_us": transport_us,
            }
        ],
        "output_verification_p99_us": verification_us,
    }
    return {
        "candidate_id": candidate.candidate_id,
        "profile": {
            "path": str(candidate.profile_path),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "iterations": profile["iterations"],
        },
        "placement": candidate.placement,
        "placement_variant": candidate.placement_variant,
        "protection_scope": candidate.protection_scope,
        "dag": dag,
        "dag_contract": validate_dag(dag),
        "tail_bound_method": tail_bound_method,
        "tail_bound_promotable": tail_bound_promotable,
        "joint_tail_p99_us": (
            finite_nonnegative(joint_tail, "joint_tail_p99_us")
            if joint_tail is not None
            else None
        ),
        "risk_budget_us": risk_budget if risk_budget is not None else None,
        "response_reservation_us": response_reservation_us,
        "pre_release_guard_p99_us": candidate.pre_release_guard_p99_us,
        "release_lead_time_us": candidate.pre_release_guard_p99_us,
        "critical_lookahead_us": critical_lookahead_us,
        "uncovered_guard_us": uncovered_guard_us,
        "arrival_to_completion_reservation_us": (
            arrival_to_completion_reservation_us
        ),
        "observed_end_to_end_p99_us": observed_p99_us,
        "payload_validation_excluded": profile_deadline_mode == "validation-excluded",
        "deadline_us": deadline_us,
        "reserved_slack_us": deadline_us - arrival_to_completion_reservation_us,
        "background_goodput_rps": candidate.background_goodput_rps,
        "feasible": arrival_to_completion_reservation_us <= deadline_us
        and observed_p99_us <= deadline_us,
    }


def select_plan(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported QUIET planner schema")
    if spec.get("system") != SYSTEM_NAME:
        raise ValueError("the only public proposed-system name is QUIET")
    deadline_us = finite_nonnegative(spec.get("deadline_us"), "deadline_us")
    if deadline_us == 0.0:
        raise ValueError("deadline_us must be positive")
    deadline_lock = spec.get("deadline_lock")
    if deadline_lock is not None and (
        not isinstance(deadline_lock, dict)
        or set(deadline_lock) != {"path", "sha256"}
        or not isinstance(deadline_lock["path"], str)
        or not isinstance(deadline_lock["sha256"], str)
        or len(deadline_lock["sha256"]) != 64
    ):
        raise ValueError("deadline_lock provenance is malformed")
    critical_lookahead_us = finite_nonnegative(
        spec.get("critical_lookahead_us"), "critical_lookahead_us"
    )
    raw_candidates = spec.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("at least one candidate is required")
    candidates = [parse_candidate(item, root) for item in raw_candidates]
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("candidate_id values must be unique")
    candidate_search = spec.get("candidate_search")
    if candidate_search is not None:
        if not isinstance(candidate_search, dict):
            raise ValueError("candidate_search must be an object")
        expected_count = candidate_search.get("candidate_count")
        if expected_count != len(candidates):
            raise ValueError("candidate_search count differs from candidates")
        expected_multi = len(candidates) >= 2
        if candidate_search.get("multi_candidate_evaluated") is not expected_multi:
            raise ValueError("candidate_search multi-candidate flag differs")
        variants = {item.placement_variant for item in candidates}
        expected_placement_search = len(variants) >= 2
        expected_status = (
            "multi-candidate-placement-and-quota-search"
            if expected_multi and expected_placement_search
            else "multi-candidate-quota-search-only"
            if expected_multi
            else "single-candidate-characterization"
        )
        if candidate_search.get("claim_status") != expected_status:
            raise ValueError("candidate_search claim status differs")
        if candidate_search.get("placement_variant_count") != len(variants):
            raise ValueError("candidate_search placement count differs")
        if candidate_search.get("placement_search_evaluated") is not expected_placement_search:
            raise ValueError("candidate_search placement flag differs")

    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            evaluated.append(
                load_pipeline_profile(candidate, deadline_us, critical_lookahead_us)
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            evaluated.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "placement": candidate.placement,
                    "placement_variant": candidate.placement_variant,
                    "protection_scope": candidate.protection_scope,
                    "background_goodput_rps": candidate.background_goodput_rps,
                    "feasible": False,
                    "error": str(error),
                }
            )
    feasible = [item for item in evaluated if item["feasible"]]
    selected = max(
        feasible,
        key=lambda item: (
            item["background_goodput_rps"],
            item["reserved_slack_us"],
        ),
        default=None,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "proposed_system": SYSTEM_NAME,
        "objective": "maximize-background-goodput-subject-to-dependent-deadline",
        "deadline_us": deadline_us,
        "deadline_lock": deadline_lock,
        "critical_lookahead_us": critical_lookahead_us,
        "candidate_search": {
            "candidate_count": len(candidates),
            "multi_candidate_evaluated": len(candidates) >= 2,
            "placement_variant_count": len({item.placement_variant for item in candidates}),
            "placement_search_evaluated": len({item.placement_variant for item in candidates}) >= 2,
            "claim_status": (
                "multi-candidate-placement-and-quota-search"
                if len(candidates) >= 2 and len({item.placement_variant for item in candidates}) >= 2
                else "multi-candidate-quota-search-only"
                if len(candidates) >= 2
                else "single-candidate-characterization"
            ),
        },
        "status": "selected" if selected is not None else "no-feasible-plan",
        "selected_plan": selected,
        "candidates": evaluated,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result = select_plan(spec, spec_path.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
