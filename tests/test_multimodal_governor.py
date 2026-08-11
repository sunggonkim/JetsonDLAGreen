#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "multimodal_governor", ROOT / "runtime" / "multimodal_governor.py"
)
assert SPEC is not None and SPEC.loader is not None
GOVERNOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GOVERNOR
SPEC.loader.exec_module(GOVERNOR)


class PolicyTest(unittest.TestCase):
    def test_static_policy_admits_all_tenants(self) -> None:
        state = GOVERNOR.FeedbackState()
        offered = ("language", "audio", "language")
        self.assertEqual(
            GOVERNOR.policy_action("static-q25", offered, state),
            [("language", 25), ("audio", 25), ("language", 25)],
        )

    def test_profiled_policy_uses_modality_quota_and_cap(self) -> None:
        state = GOVERNOR.FeedbackState()
        offered = ("language", "audio", "language")
        self.assertEqual(
            GOVERNOR.policy_action("profiled", offered, state),
            [("language", 50), ("audio", 100)],
        )

    def test_feedback_uses_additive_increase_and_fast_decrease(self) -> None:
        state = GOVERNOR.FeedbackState(admission_limit=4)
        GOVERNOR.update_feedback(state, violated=True)
        self.assertEqual(state.admission_limit, 3)
        GOVERNOR.update_feedback(state, violated=False)
        self.assertEqual(state.admission_limit, 3)
        GOVERNOR.update_feedback(state, violated=False)
        self.assertEqual(state.admission_limit, 4)

    def test_cpu_list_expansion(self) -> None:
        self.assertEqual(GOVERNOR.expand_cpu_list("0-2,5"), [0, 1, 2, 5])
        with self.assertRaises(ValueError):
            GOVERNOR.expand_cpu_list("2-0")

    def test_pressure_completed_key_compatibility(self) -> None:
        self.assertEqual(
            GOVERNOR.pressure_completed({"completed_launches": 11}), 11
        )
        self.assertEqual(
            GOVERNOR.pressure_completed({"completed_requests": 7}), 7
        )
        with self.assertRaises(KeyError):
            GOVERNOR.pressure_completed({})


if __name__ == "__main__":
    unittest.main()
