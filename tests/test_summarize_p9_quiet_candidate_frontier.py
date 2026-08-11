import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "candidate_frontier", ROOT / "analysis/summarize_p9_quiet_candidate_frontier.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QuietCandidateFrontierTest(unittest.TestCase):
    def _write(self, root: Path, index: int, placement: str, quota: int, rep: int) -> Path:
        path = root / f"{index}.json"
        trace_path = root / f"trace-{index}.csv"
        trace_path.write_text("request_id,latency_us\n0,1.0\n", encoding="utf-8")
        trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        path.write_text(json.dumps({
            "kind": "p9-dependent-small-stress-smoke",
            "workload": "resnet-control",
            "latency_contract": "production-wall-arrival-to-completion",
            "deadline_mode": "wall",
            "production_wall_definition": "arrival-to-consumer-completion-excludes-correctness-validation",
            "checksum_mode": "inline", "placement_variant": placement,
            "producer_quota_percent": 100, "background_quota_percent": quota,
            "deadline_us": 773.730452, "background_period_ms": 4.0,
            "deadline_lock": {"sha256": "a" * 64},
            "results": [{"system": "QUIET", "correctness_validated": True,
                         "placement_variant": placement, "producer_quota_percent": 100,
                         "background_quota_percent": quota, "pipeline_requests": 100,
                         "deadline_misses": 0, "pipeline_p99_us": 450.0 + rep,
                         "background_goodput_rps": 248.0,
                         "request_trace": {"path": str(trace_path), "sha256": trace_sha}}],
        }) + "\n")
        return path

    def test_requires_all_points_and_reports_cp95(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            index = 0
            for placement in MODULE.PLACEMENTS:
                for quota in MODULE.QUOTAS:
                    for rep in range(3):
                        paths.append(self._write(root, index, placement, quota, rep))
                        index += 1
            result = MODULE.summarize(paths)
        self.assertEqual(len(result["points"]), 6)
        self.assertFalse(result["points"][0]["cp95_slo_qualified"])
        self.assertIsNone(result["selected_cp95_slo_point"])

    def test_rejects_missing_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            index = 0
            for placement in MODULE.PLACEMENTS:
                for quota in MODULE.QUOTAS:
                    for rep in range(3 if index != 0 else 2):
                        paths.append(self._write(root, index, placement, quota, rep))
                        index += 1
            with self.assertRaisesRegex(ValueError, "exactly 18"):
                MODULE.summarize(paths)


if __name__ == "__main__":
    unittest.main()
