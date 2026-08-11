import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_causal_pair", ROOT / "analysis" / "validate_p9_causal_pair.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def summary(scenario: str, *, deadline: float = 5.0, workload: str = "resnet") -> dict:
    config = {
        "scenario": scenario,
        "epochs": 2,
        "samples_per_epoch": 80,
        "warmup": 100,
        "burst_size": 8,
        "period_ms": 20.0,
        "dmr_target": 0.0005,
        "includes_transfers": True,
        "worker_max_inflight": 1,
        "trace": [["audio", "language"]],
        "dependency_semantics": "none" if scenario == "independent" else "audio-completion-token-precedes-language-inference",
        "experiment_label": scenario,
    }
    return {
        "schema_version": 4,
        "config": config,
        "deadline_ms": deadline,
        "artifacts": {"benchmark_sha256": "a" * 64},
        "hardware": {"gpu": "thor", "mig": "fixed"},
        "mig": {"critical_uuid": "critical", "resident_uuid": "resident"},
        "policies": [{
            "name": "mig-governor",
            "critical_requests": 80,
            "deadline_misses": 0 if scenario == "independent" else 4,
            "deadline_miss_rate": 0.0 if scenario == "independent" else 0.05,
            "critical_p99_ms_max": 4.0 if scenario == "independent" else 5.0,
            "pressure_goodput_per_second": 100.0 if scenario == "independent" else 80.0,
            "telemetry_unhealthy_epochs": 0,
            "rejected_tenants": 0,
            "epochs": [{
                "epoch": 0,
                "dependency_edges": [] if scenario == "independent" else [{
                    "upstream_tenant_id": 0,
                    "downstream_tenant_id": 1,
                    "payload_bytes": 14720,
                    "transport": "registered-shared-sysmem-direct-binding",
                }],
            }],
        }],
    }


class CausalPairTest(unittest.TestCase):
    def write_pair(self, independent: dict, dependent: dict) -> tuple[Path, Path, Path]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        ip = root / "independent.json"
        dp = root / "dependent.json"
        out = root / "pair.json"
        ip.write_text(json.dumps(independent) + "\n")
        dp.write_text(json.dumps(dependent) + "\n")
        return ip, dp, out

    def test_accepts_same_workload_edge_toggle(self) -> None:
        ip, dp, _ = self.write_pair(summary("independent"), summary("dependent"))
        result = MODULE.validate_pair(ip, dp)
        self.assertEqual(result["proposed_system"], "QUIET")
        self.assertEqual(result["scope"], "same-workload-edge-toggle")
        self.assertEqual(result["edge_evidence"]["dependent"][0]["payload_bytes"], 14720)
        self.assertAlmostEqual(result["delta_dependent_minus_independent"]["critical_p99_ms"], 1.0)

    def test_rejects_model_or_deadline_confound(self) -> None:
        independent = summary("independent", deadline=5.0)
        dependent = summary("dependent", deadline=5.1)
        ip, dp, _ = self.write_pair(independent, dependent)
        with self.assertRaisesRegex(ValueError, "deadlines differ"):
            MODULE.validate_pair(ip, dp)

    def test_rejects_config_difference_outside_edge_contract(self) -> None:
        independent = summary("independent")
        dependent = summary("dependent")
        dependent["config"]["period_ms"] = 25.0
        ip, dp, _ = self.write_pair(independent, dependent)
        with self.assertRaisesRegex(ValueError, "workload contract differs"):
            MODULE.validate_pair(ip, dp)

    def test_rejects_control_only_dependency(self) -> None:
        independent = summary("independent")
        dependent = summary("dependent")
        dependent["policies"][0]["epochs"][0]["dependency_edges"][0].pop("payload_bytes")
        ip, dp, _ = self.write_pair(independent, dependent)
        with self.assertRaisesRegex(ValueError, "positive payload"):
            MODULE.validate_pair(ip, dp)


if __name__ == "__main__":
    unittest.main()
