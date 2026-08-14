import copy
import importlib.util
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis/generate_p9_whisper_edge_mix_figures.py"
SPEC = importlib.util.spec_from_file_location("whisper_edge_mix_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(
    rate: float,
    mode: str,
    session: int,
    *,
    misses: int,
    workers: int = 1,
) -> dict[str, object]:
    return {
        "rate_rps": rate,
        "mode": mode,
        "session": session,
        "requests": 100,
        "deadline_misses": misses,
        "p50_us": 100000.0,
        "p99_us": 300000.0 if misses else 150000.0,
        "queue_p99_us": 100000.0 if misses else 1000.0,
        "request_goodput_rps": rate - 0.2,
        "background_goodput_rps": 800.0,
        "producer_mean_us": 9000.0,
        "consumer_mean_us": 50000.0,
        "output_sha256": "a" * 64,
        "background_workers": workers,
        "gated_processes": workers if mode == "quiet" else 0,
    }


def summary(specification: object, phase: str) -> dict[str, object]:
    if phase == "directional":
        rows = [
            row(
                rate,
                mode,
                1,
                misses=(
                    25
                    if mode == specification.failure_mode
                    and rate >= specification.balanced_rate_rps
                    else 0
                ),
                workers=specification.background_workers,
            )
            for rate in specification.directional_rates_rps
            for _, mode in MODULE.SYSTEMS
        ]
        design = "directional-sweep"
    else:
        rows = [
            row(
                specification.balanced_rate_rps,
                mode,
                session_id,
                misses=(25 if mode == specification.failure_mode else 0),
                workers=specification.background_workers,
            )
            for session_id in (1, 2, 3)
            for _, mode in MODULE.SYSTEMS
        ]
        design = "balanced-repeated"
    return {
        "kind": "p9-whisper-asr-mig-crossover",
        "evidence_class": "exploratory-nonthermal-directional",
        "thermal_campaign": False,
        "input_policy": "cyclic-performance-replay-not-accuracy-expansion",
        "study_design": design,
        "comparator_output_contract": "byte-identical",
        "pipeline_slots": 3,
        "deadline_us": 250000.0,
        "background_workers": specification.background_workers,
        "scenario": {
            "id": specification.scenario_id,
            "background_workers": specification.background_workers,
        },
        "rows": rows,
    }


class WhisperEdgeMixFigureTest(unittest.TestCase):
    def test_fixed_public_order_and_both_target_crossovers(self) -> None:
        for specification in (MODULE.SCENARIOS[0], MODULE.SCENARIOS[2]):
            with self.subTest(scenario=specification.scenario_id):
                directional = MODULE.validate_raw_summary(
                    summary(specification, "directional"),
                    scenario_id=specification.scenario_id,
                    phase="directional",
                    balanced_rate=specification.balanced_rate_rps,
                    workers=specification.background_workers,
                    directional_rates=specification.directional_rates_rps,
                )
                balanced = MODULE.validate_raw_summary(
                    summary(specification, "balanced"),
                    scenario_id=specification.scenario_id,
                    phase="balanced",
                    balanced_rate=specification.balanced_rate_rps,
                    workers=specification.background_workers,
                    directional_rates=specification.directional_rates_rps,
                )
                compact = MODULE.compact_raw(
                    directional,
                    balanced,
                    specification.directional_rates_rps,
                )
                self.assertEqual(
                    list(compact), ["QUIET", "NVIDIA MIG", "NVIDIA MPS"]
                )
                self.assertEqual(
                    MODULE.first_target_only_failure(
                        compact,
                        specification.failure_system,
                        specification.directional_rates_rps,
                    ),
                    specification.balanced_rate_rps,
                )
                self.assertEqual(compact["QUIET"]["balanced"]["misses"], 0)
                self.assertEqual(
                    compact[specification.failure_system]["balanced"]["misses"],
                    75,
                )

    def test_gate_coverage_must_match_background_worker_count(self) -> None:
        specification = MODULE.SCENARIOS[0]
        value = summary(specification, "directional")
        broken = copy.deepcopy(value)
        for item in broken["rows"]:
            if item["mode"] == "quiet":
                item["gated_processes"] = 0
                break
        with self.assertRaisesRegex(ValueError, "gate coverage differs"):
            MODULE.validate_raw_summary(
                broken,
                scenario_id=specification.scenario_id,
                phase="directional",
                balanced_rate=specification.balanced_rate_rps,
                workers=specification.background_workers,
                directional_rates=specification.directional_rates_rps,
            )

    def test_output_trace_must_be_identical(self) -> None:
        specification = MODULE.SCENARIOS[0]
        value = summary(specification, "balanced")
        value["rows"][0]["output_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "matrix differs"):
            MODULE.validate_raw_summary(
                value,
                scenario_id=specification.scenario_id,
                phase="balanced",
                balanced_rate=specification.balanced_rate_rps,
                workers=specification.background_workers,
                directional_rates=specification.directional_rates_rps,
            )

    def test_balanced_table_repeats_fixed_system_order(self) -> None:
        scenarios = {}
        for specification in MODULE.SCENARIOS:
            systems = {}
            for system_index, (system, mode) in enumerate(MODULE.SYSTEMS):
                systems[system] = {
                    "mode": mode,
                    "balanced": {
                        "requests": 300,
                        "misses": (
                            75 if system == specification.failure_system else 0
                        ),
                        "observed_dmr": (
                            0.25 if system == specification.failure_system else 0.0
                        ),
                        "mean_session_p99_us": (
                            300000.0
                            if system == specification.failure_system
                            else 150000.0
                        ),
                        "mean_queue_p99_us": (
                            100000.0
                            if system == specification.failure_system
                            else 1000.0
                        ),
                        "mean_critical_goodput_rps": (
                            specification.balanced_rate_rps - 0.2
                        ),
                        "mean_background_goodput_rps": 800.0 + system_index,
                    },
                }
            scenarios[specification.scenario_id] = {
                "label": specification.label,
                "background_workers": specification.background_workers,
                "balanced_rate_rps": specification.balanced_rate_rps,
                "systems": systems,
            }
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "table.tex"
            MODULE.render_balanced_table({"scenarios": scenarios}, output)
            rendered = output.read_text(encoding="utf-8")
        self.assertIn(r"\newcommand{\PnineEdgeMixTable}", rendered)
        self.assertEqual(rendered.count(r"\textbf{QUIET}"), 4)
        self.assertEqual(rendered.count("NVIDIA MIG"), 4)
        self.assertEqual(rendered.count("NVIDIA MPS"), 4)
        for block in rendered.split(r"\midrule")[1:5]:
            self.assertLess(block.index(r"\textbf{QUIET}"), block.index("NVIDIA MIG"))
            self.assertLess(block.index("NVIDIA MIG"), block.index("NVIDIA MPS"))


if __name__ == "__main__":
    unittest.main()
