#!/usr/bin/env python3
"""Validate the structural DAG contract used by QUIET planning artifacts.

This module validates topology and precedence only.  It does not promote a
synthetic graph to an application result; application promotion still needs a
real learned workload, output trace, and accuracy gate.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def validate_dag(dag: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dag, dict):
        raise ValueError("dag must be an object")
    stages = dag.get("stages")
    edges = dag.get("edges")
    if not isinstance(stages, list) or not stages:
        raise ValueError("dag stages must be a nonempty list")
    if not isinstance(edges, list) or not edges:
        raise ValueError("dag edges must be a nonempty list")

    stage_ids: list[str] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"dag stage {index} must be an object")
        stage_ids.append(_text(stage.get("id"), f"dag stage {index} id"))
    if len(set(stage_ids)) != len(stage_ids):
        raise ValueError("dag stage ids must be unique")
    known = set(stage_ids)

    pairs: list[tuple[str, str]] = []
    indegree = Counter({stage_id: 0 for stage_id in stage_ids})
    outgoing: dict[str, list[str]] = {stage_id: [] for stage_id in stage_ids}
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ValueError(f"dag edge {index} must be an object")
        source = _text(edge.get("source"), f"dag edge {index} source")
        target = _text(edge.get("target"), f"dag edge {index} target")
        if source not in known or target not in known:
            raise ValueError(f"dag edge {index} references an unknown stage")
        if source == target:
            raise ValueError("dag cannot contain self-edges")
        pair = (source, target)
        if pair in pairs:
            raise ValueError("dag edges must be unique")
        pairs.append(pair)
        outgoing[source].append(target)
        indegree[target] += 1

    ready = deque(stage_id for stage_id in stage_ids if indegree[stage_id] == 0)
    topological: list[str] = []
    while ready:
        stage_id = ready.popleft()
        topological.append(stage_id)
        for target in outgoing[stage_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(topological) != len(stage_ids):
        raise ValueError("dag precedence graph contains a cycle")

    outdegrees = [len(outgoing[stage_id]) for stage_id in stage_ids]
    indegrees = [
        sum(1 for source, target in pairs if target == stage_id)
        for stage_id in stage_ids
    ]
    source_count = sum(value > 0 for value in outdegrees)
    sink_count = sum(value == 0 for value in outdegrees)
    has_fan_out = max(outdegrees, default=0) > 1
    has_fan_in = max(indegrees, default=0) > 1
    is_chain = all(value <= 1 for value in outdegrees) and all(
        value <= 1 for value in indegrees
    )
    if has_fan_out and has_fan_in:
        topology = "fan-out-fan-in"
    elif has_fan_out:
        topology = "fan-out"
    elif has_fan_in:
        topology = "fan-in"
    elif is_chain and len(stage_ids) == 2:
        topology = "two-stage-chain"
    elif is_chain and len(stage_ids) >= 3:
        topology = "three-or-more-stage-chain"
    else:
        topology = "general-dag"

    multi_stage_validated = len(stage_ids) >= 3 and topology in {
        "three-or-more-stage-chain", "fan-out", "fan-in", "fan-out-fan-in",
    }
    return {
        "schema_version": 1,
        "validated": True,
        "stage_count": len(stage_ids),
        "edge_count": len(pairs),
        "stage_ids": stage_ids,
        "topological_order": topological,
        "topology": topology,
        "multi_stage_validation": "passed" if multi_stage_validated else "pending",
        "general_dag_claim_allowed": multi_stage_validated,
        "source_count": source_count,
        "sink_count": sink_count,
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dag", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_dag(json.loads(args.dag.read_text())), indent=2))
