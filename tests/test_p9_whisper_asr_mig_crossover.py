import argparse
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p9_whisper_asr_mig_crossover.py"
SPEC = importlib.util.spec_from_file_location("whisper_crossover", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WhisperAsrMigCrossoverTest(unittest.TestCase):
    def test_modes_change_only_placement_and_protection(self) -> None:
        by_name = {mode.name: mode for mode in MODULE.MODES}
        self.assertEqual(
            set(by_name),
            {"nvidia-mig", "nvidia-mps-static-split", "quiet"},
        )
        self.assertEqual(by_name["nvidia-mig"].producer_device, "big")
        self.assertFalse(by_name["nvidia-mig"].gate_background)
        self.assertEqual(
            by_name["nvidia-mps-static-split"].producer_device, "small"
        )
        self.assertFalse(by_name["nvidia-mps-static-split"].gate_background)
        self.assertEqual(by_name["quiet"].producer_device, "small")
        self.assertTrue(by_name["quiet"].gate_background)

    def test_aggregate_preserves_deadline_miss_counts(self) -> None:
        rows = [
            {
                "rate_rps": 20.0, "mode": "quiet", "requests": 100,
                "deadline_misses": miss, "p50_us": 1.0, "p99_us": 2.0,
                "queue_p99_us": 0.5, "request_goodput_rps": 20.0,
                "background_goodput_rps": 100.0, "producer_mean_us": 3.0,
                "consumer_mean_us": 4.0,
            }
            for miss in (2, 3)
        ]
        result = MODULE.aggregate(rows)
        self.assertEqual(result[0]["sessions"], 2)
        self.assertEqual(result[0]["requests"], 200)
        self.assertEqual(result[0]["deadline_misses"], 5)

    def test_scenario_metadata_binds_real_edge_mix(self) -> None:
        args = argparse.Namespace(
            scenario_id="speech-plus-vision",
            scenario_label="Speech + Vision",
            scenario_description="ASR with queued camera analytics",
            background_model_name="resnet10-detection",
            background_engine=ROOT / "models/example.engine",
            backgrounds=[
                MODULE.Background(
                    "resnet10-detection", ROOT / "models/example.engine"
                ),
                MODULE.Background(
                    "distilbert-sst2", ROOT / "models/example-nlp.engine"
                ),
            ],
            background_period_ms=0.0,
            deployment_scope="multi-sensor-robot-or-edge-gateway-stress",
        )
        result = MODULE.scenario_metadata(args)
        self.assertEqual(result["id"], "speech-plus-vision")
        self.assertEqual(
            result["background_models"],
            ["resnet10-detection", "distilbert-sst2"],
        )
        self.assertEqual(result["background_workers"], 2)
        self.assertEqual(result["background_release"], "saturated-backlog")
        self.assertIn("multi-sensor", result["deployment_scope"])

    def test_additional_background_requires_explicit_model_and_engine(self) -> None:
        model, engine = MODULE.parse_additional_background(
            "whisper-tiny-encoder=models/whisper.engine"
        )
        self.assertEqual(model, "whisper-tiny-encoder")
        self.assertEqual(engine, pathlib.Path("models/whisper.engine"))
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.parse_additional_background("models/whisper.engine")


if __name__ == "__main__":
    unittest.main()
