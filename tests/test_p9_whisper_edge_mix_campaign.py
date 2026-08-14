import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p9_whisper_edge_mix_campaign.py"
SPEC = importlib.util.spec_from_file_location("whisper_edge_mix_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def arguments() -> argparse.Namespace:
    return argparse.Namespace(
        repo=ROOT,
        runner=ROOT / "scripts/run_p9_whisper_asr_mig_crossover.py",
        input_trace=ROOT / "inputs.bin",
        balanced_sessions=3,
        requests=100,
        warmup=2,
    )


class WhisperEdgeMixCampaignTest(unittest.TestCase):
    def test_scenarios_are_predeclared_real_model_categories(self) -> None:
        self.assertEqual(
            [scenario.scenario_id for scenario in MODULE.SCENARIOS],
            [
                "mig-placement-nlp",
                "mig-placement-vision",
                "mps-interference-speech-20",
                "mps-interference-vision-24",
            ],
        )
        self.assertEqual(
            {
                pathlib.Path(scenario.background_engine).name
                for scenario in MODULE.SCENARIOS
            },
            {
                "distilbert-sst2.engine",
                "resnet10-detection.engine",
                "whisper-tiny-encoder.engine",
            },
        )
        self.assertEqual(
            [scenario.balanced_rate_rps for scenario in MODULE.SCENARIOS],
            [19.0, 21.0, 17.0, 18.0],
        )
        self.assertEqual(
            [scenario.failure_target for scenario in MODULE.SCENARIOS],
            [
                "nvidia-mig",
                "nvidia-mig",
                "nvidia-mps-static-split",
                "nvidia-mps-static-split",
            ],
        )
        self.assertEqual(
            [len(scenario.additional_backgrounds) + 1 for scenario in MODULE.SCENARIOS],
            [1, 1, 20, 24],
        )

    def test_command_freezes_scenario_metadata_and_directional_rates(self) -> None:
        args = arguments()
        command = MODULE.command_for(
            args, MODULE.SCENARIOS[1], "directional", ROOT / "result"
        )
        self.assertEqual(
            command[-5:], ["15.0", "17.0", "19.0", "20.0", "21.0"]
        )
        self.assertIn("mig-placement-vision", command)
        self.assertIn("resnet10-detection", command)
        self.assertIn("--deployment-scope", command)

    def test_high_fanout_command_launches_twenty_four_background_workers(self) -> None:
        command = MODULE.command_for(
            arguments(), MODULE.SCENARIOS[-1], "balanced", ROOT / "result"
        )
        self.assertEqual(command.count("--additional-background"), 23)
        self.assertIn("18.0", command)

    def test_balanced_summary_requires_all_modes_and_sessions(self) -> None:
        args = arguments()
        scenario = MODULE.SCENARIOS[0]
        rows = [
            {
                "rate_rps": 19.0,
                "mode": mode,
                "session": session,
                "requests": 100,
                "output_sha256": "a" * 64,
            }
            for session in (1, 2, 3)
            for mode in ("nvidia-mig", "nvidia-mps-static-split", "quiet")
        ]
        value = {
            "kind": "p9-whisper-asr-mig-crossover",
            "thermal_campaign": False,
            "study_design": "balanced-repeated",
            "scenario": {"id": scenario.scenario_id},
            "comparator_output_contract": "byte-identical",
            "rows": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "summary.json"
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            result = MODULE.validate_summary(path, scenario, "balanced", args)
            self.assertEqual(len(result["rows"]), 9)
            value["rows"].pop()
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete balanced matrix"):
                MODULE.validate_summary(path, scenario, "balanced", args)

    def test_balanced_rate_is_first_target_only_failure(self) -> None:
        aggregate = []
        for rate, mig_misses in ((17.0, 0), (19.0, 42), (20.0, 77)):
            for mode in (
                "nvidia-mig",
                "nvidia-mps-static-split",
                "quiet",
            ):
                aggregate.append(
                    {
                        "rate_rps": rate,
                        "mode": mode,
                        "deadline_misses": mig_misses if mode == "nvidia-mig" else 0,
                    }
                )
        self.assertEqual(
            MODULE.first_target_only_failure(
                {"aggregate": aggregate}, "nvidia-mig"
            ),
            19.0,
        )
        for item in aggregate:
            if item["mode"] == "nvidia-mig":
                item["deadline_misses"] = 0
            elif item["mode"] == "nvidia-mps-static-split":
                item["deadline_misses"] = 42 if item["rate_rps"] >= 19.0 else 0
        self.assertEqual(
            MODULE.first_target_only_failure(
                {"aggregate": aggregate}, "nvidia-mps-static-split"
            ),
            19.0,
        )


if __name__ == "__main__":
    unittest.main()
