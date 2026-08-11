import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p9_wall_frontier_table", ROOT / "analysis/generate_p9_wall_frontier_table.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _point(system: str, *, evidence: dict) -> dict:
    return {
        "system": system,
        "offered_rps": 250.0,
        "requests": 100,
        "deadline_misses": 0,
        "dmr": 0.0,
        "p99_us": 700.0,
        "background_goodput_rps": 249.0,
        "deadline_lock_sha256": "a" * 64,
        "correctness_evidence": evidence,
    }


class WallFrontierTableTest(unittest.TestCase):
    def _artifact(self, points: dict[str, dict]) -> dict:
        return {
            "kind": "p9-common-production-wall-load-frontier",
            "proposed_system": "QUIET",
            "numeric_frontier_systems": ["NVIDIA MPS", "QUIET"],
            "exploratory_systems": sorted(set(points) - {"NVIDIA MPS", "QUIET"}),
            "ranking_allowed": False,
            "production_wall_definition": MODULE.PRODUCTION_WALL_DEFINITION_V2,
            "correctness_validation_placement": MODULE.CORRECTNESS_PLACEMENT_V2,
            "frontier": {
                system: {"points": [point]} for system, point in points.items()
            },
        }

    def test_load_rechecks_inline_and_external_correctness_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verification = root / "verification.json"
            verification.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(verification.read_bytes()).hexdigest()
            value = self._artifact(
                {
                    "QUIET": _point(
                        "QUIET",
                        evidence={
                            "mode": "inline",
                            "source": "row",
                            "checksum_failures": 0,
                            "unique_payload_checksums": 4,
                            "unique_policy_output_checksums": 4,
                        },
                    ),
                    "NVIDIA MPS": _point(
                        "NVIDIA MPS",
                        evidence={
                            "mode": "inline",
                            "source": "row",
                            "checksum_failures": 0,
                            "unique_payload_checksums": 4,
                            "unique_policy_output_checksums": 4,
                        },
                    ),
                    "XSched": _point(
                        "XSched",
                        evidence={
                            "mode": "inline",
                            "source": "sota_verification",
                            "checksum_failures": 0,
                            "unique_payload_checksums": 4,
                            "unique_policy_output_checksums": 4,
                            "path": str(verification),
                            "sha256": digest,
                        },
                    ),
                }
            )
            artifact = root / "frontier.json"
            artifact.write_text(json.dumps(value) + "\n", encoding="utf-8")
            loaded = MODULE.load(artifact)
            self.assertEqual(MODULE.render(loaded).count("\\textbf{QUIET}"), 1)

    def test_rejects_tampered_dmr_or_verification_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verification = root / "verification.json"
            verification.write_text("{}\n", encoding="utf-8")
            evidence = {
                "mode": "inline",
                "source": "sota_verification",
                "checksum_failures": 0,
                "unique_payload_checksums": 4,
                "unique_policy_output_checksums": 4,
                "path": str(verification),
                "sha256": "0" * 64,
            }
            value = self._artifact({system: _point(system, evidence=evidence) for system in ("QUIET", "NVIDIA MPS", "XSched")})
            value["frontier"]["QUIET"]["points"][0]["dmr"] = 0.5
            artifact = root / "frontier.json"
            artifact.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "DMR"):
                MODULE.load(artifact)
            value["frontier"]["QUIET"]["points"][0]["dmr"] = 0.0
            artifact.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verification file changed"):
                MODULE.load(artifact)

    def test_rejects_mixed_deadline_lock_shas(self):
        evidence = {
            "mode": "inline", "source": "row", "checksum_failures": 0,
            "unique_payload_checksums": 4, "unique_policy_output_checksums": 4,
        }
        value = self._artifact({
            system: _point(system, evidence=evidence)
            for system in ("QUIET", "NVIDIA MPS", "XSched")
        })
        value["frontier"]["XSched"]["points"][0]["deadline_lock_sha256"] = "b" * 64
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "frontier.json"
            artifact.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one exact deadline lock"):
                MODULE.load(artifact)

    def test_manifest_is_the_only_numeric_table_boundary(self):
        evidence = {
            "mode": "inline", "source": "row", "checksum_failures": 0,
            "unique_payload_checksums": 4, "unique_policy_output_checksums": 4,
        }
        value = self._artifact({
            system: _point(system, evidence=evidence)
            for system in ("QUIET", "NVIDIA MPS", "XSched")
        })
        value["frontier"]["Static full gating"] = {
            "numeric_comparison_allowed": False,
            "points": [_point("Static full gating", evidence=evidence)],
        }
        value["exploratory_systems"] = ["Static full gating", "XSched"]
        rendered = MODULE.render(value)
        self.assertIn("NVIDIA MPS", rendered)
        self.assertNotIn("XSched", rendered)
        self.assertIn("\\textbf{QUIET}", rendered)
        self.assertNotIn("Static full gating", rendered)
        self.assertNotIn("NVIDIA MIG", rendered)

    def test_formal_frontier_requires_application_accuracy_gate(self):
        evidence = {
            "mode": "inline", "source": "row", "checksum_failures": 0,
            "unique_payload_checksums": 4, "unique_policy_output_checksums": 4,
        }
        value = self._artifact({
            system: _point(system, evidence=evidence)
            for system in ("QUIET", "NVIDIA MPS", "XSched")
        })
        value["formal"] = True
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "formal-frontier.json"
            artifact.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "application-accuracy"):
                MODULE.load(artifact)

    def test_rejects_pre_v2_production_wall_artifact(self):
        evidence = {
            "mode": "inline", "source": "row", "checksum_failures": 0,
            "unique_payload_checksums": 4, "unique_policy_output_checksums": 4,
        }
        value = self._artifact({
            system: _point(system, evidence=evidence)
            for system in ("QUIET", "NVIDIA MPS", "XSched")
        })
        value.pop("production_wall_definition")
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "legacy-frontier.json"
            artifact.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale production-wall"):
                MODULE.load(artifact)


if __name__ == "__main__":
    unittest.main()
