import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.summarize_p9_active_williams_repeats import summarize
from scripts.run_p9_common_sota_williams import active_williams_orders


def _write(path: Path, value: dict) -> str:
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _fixture(root: Path, index: int, workload: str = "resnet-control") -> Path:
    lock = root / "deadline-lock.json"
    plan = root / "selection-forward.json"
    if not lock.exists():
        _write(lock, {"kind": "test-lock"})
        _write(plan, {"proposed_system": "QUIET", "status": "selected"})
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    source_items = []
    result_rows = []
    for position, system in enumerate(active_williams_orders()[index]):
        misses = 40 + index if system == "NVIDIA MPS" else (100 if system == "XSched" else 0)
        p99 = 850.0 + index if system == "NVIDIA MPS" else (1300.0 + index if system == "XSched" else 650.0 + index)
        goodput = 249.0 - index if system == "NVIDIA MPS" else (33.0 + index if system == "XSched" else 248.0 - index)
        if system == "XSched":
            source = {
                "kind": (
                    "xsched-thor-resnet-control-numeric-smoke-verification"
                    if workload == "resnet-control"
                    else f"xsched-dependent-{workload}-numeric-smoke-verification"
                ),
                "requests": 100, "misses": misses, "p99_us": p99,
                "background_goodput_rps": goodput, "correctness_validated": True,
                "deadline_lock": {"sha256": lock_sha},
            }
            if workload != "resnet-control":
                source["workload"] = workload
        else:
            source = {
                "kind": "p9-dependent-small-stress-smoke",
                "deadline_lock": {"sha256": lock_sha},
                "results": [{"system": system, "pipeline_requests": 100,
                              "deadline_misses": misses, "pipeline_p99_us": p99,
                              "background_goodput_rps": goodput,
                              "correctness_validated": True}],
            }
        source_path = root / f"source-{index}-{position}.json"
        source_sha = _write(source_path, source)
        source_items.append({"system": system, "path": str(source_path), "sha256": source_sha})
        result_rows.append({"system": system, "requests": 100, "misses": misses,
                            "p99_us": p99, "background_goodput_rps": goodput,
                            "deadline_mode": "wall",
                            "latency_contract": "production-wall-arrival-to-completion",
                            "production_wall_definition": (
                                "arrival-to-consumer-completion-excludes-correctness-validation"
                            ),
                            "correctness_validated": True})
    run = {
        "kind": "p9-common-sota-williams-sequence", "proposed_system": "QUIET",
        "sequence_index": index, "execution_order": list(active_williams_orders()[index]),
        "active_only": True,
        "active_exploratory_systems": ["NVIDIA MPS", "XSched", "QUIET"],
        "numeric_frontier_systems": ["NVIDIA MPS", "QUIET"],
        "workload": workload, "placement_variant": "fixed-1g-producer-2g-consumer",
        "deadline_mode": "wall", "background_period_ms": 4.0,
        "background_offered_rps": 250.0, "requests_per_system": 100,
        "deadline_lock": {"path": str(lock), "sha256": lock_sha},
        "quiet_plan": {"path": str(plan), "sha256": plan_sha},
        "results": result_rows, "inputs": source_items,
    }
    path = root / f"run-{index}.json"
    _write(path, run)
    return path


class ActiveWilliamsRepeatsTest(unittest.TestCase):
    def test_replays_all_three_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = summarize([_fixture(root, index) for index in range(3)])
        self.assertEqual(value["kind"], "p9-active-williams-production-wall-repeats")
        self.assertEqual(value["systems"]["QUIET"]["repeat_count"], 3)
        self.assertEqual(value["systems"]["QUIET"]["total_deadline_misses"], 0)
        self.assertEqual(value["systems"]["XSched"]["total_deadline_misses"], 300)
        self.assertFalse(value["formal"])
        self.assertFalse(value["ranking_allowed"])
        self.assertEqual(value["paired_session_statistics"]["NVIDIA MPS"]["p99_delta_us_quiet_minus_baseline"]["t95"]["n"], 3)

    def test_replays_current_resnet50_workload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = summarize([
                _fixture(root, index, "resnet50-classification")
                for index in range(3)
            ])
        self.assertEqual(value["workload"], "resnet50-classification")
        self.assertEqual(value["systems"]["XSched"]["repeat_count"], 3)

    def test_rejects_duplicate_sequence_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [_fixture(root, index) for index in range(3)]
            duplicate = json.loads(paths[2].read_text())
            duplicate["sequence_index"] = 1
            paths[2].write_text(json.dumps(duplicate) + "\n")
            with self.assertRaisesRegex(ValueError, "active Williams contract"):
                summarize(paths)

    def test_rejects_reused_source_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [_fixture(root, index) for index in range(3)]
            first = json.loads(paths[0].read_text())
            second = json.loads(paths[1].read_text())
            reused = next(item for item in first["inputs"] if item["system"] == second["inputs"][0]["system"])
            second["inputs"][0] = reused
            reused_system = reused["system"]
            reused_row = next(row for row in first["results"] if row["system"] == reused_system)
            current_row = next(row for row in second["results"] if row["system"] == reused_system)
            current_row.update({key: reused_row[key] for key in ("requests", "misses", "p99_us", "background_goodput_rps")})
            paths[1].write_text(json.dumps(second) + "\n")
            with self.assertRaisesRegex(ValueError, "reuses source evidence"):
                summarize(paths)


if __name__ == "__main__":
    unittest.main()
