#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "full_gpu_governor", ROOT / "runtime" / "full_gpu_governor.py"
)
assert SPEC is not None and SPEC.loader is not None
GOVERNOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GOVERNOR
SPEC.loader.exec_module(GOVERNOR)


class FullGpuGovernorTest(unittest.TestCase):
    def test_joint_admission_limit(self) -> None:
        state = GOVERNOR.FeedbackState(admission_limit=2)
        offered = ("language", "audio", "language")
        self.assertEqual(
            GOVERNOR.policy_action("joint-governor", offered, state),
            [("language", 25), ("audio", 25)],
        )

    def test_modality_guard_uses_slowest_admitted_model(self) -> None:
        state = GOVERNOR.FeedbackState(guard_adjustment_ms=0.25)
        actions = [("language", 25), ("audio", 25)]
        self.assertEqual(
            GOVERNOR.guard_for("profiled-guard", actions, state), 2.5
        )
        self.assertEqual(
            GOVERNOR.guard_for("joint-governor", actions, state), 2.25
        )
        self.assertEqual(
            GOVERNOR.guard_for(
                "joint-governor", actions, state, None, 1.75
            ),
            1.75,
        )

    def test_language_profile_preserves_tail_floor(self) -> None:
        state = GOVERNOR.FeedbackState(guard_adjustment_ms=0.0)
        self.assertEqual(
            GOVERNOR.guard_for(
                "joint-governor", [("language", 25)], state
            ),
            1.5,
        )

    def test_feedback_fast_increase_and_slow_decrease(self) -> None:
        state = GOVERNOR.FeedbackState(
            admission_limit=4, guard_adjustment_ms=0.25
        )
        GOVERNOR.update_feedback(
            state, violated=True, critical_p99_ms=6.0, deadline_ms=5.0
        )
        self.assertEqual(state.admission_limit, 3)
        self.assertEqual(state.guard_adjustment_ms, 0.75)
        for _ in range(2):
            GOVERNOR.update_feedback(
                state, violated=False, critical_p99_ms=4.0, deadline_ms=5.0
            )
        self.assertEqual(state.admission_limit, 4)
        self.assertEqual(state.guard_adjustment_ms, 0.5)

    def test_feedback_never_undercuts_profile(self) -> None:
        state = GOVERNOR.FeedbackState(guard_adjustment_ms=0.0)
        for _ in range(4):
            GOVERNOR.update_feedback(
                state, violated=False, critical_p99_ms=4.0, deadline_ms=5.0
            )
        self.assertEqual(state.guard_adjustment_ms, 0.0)

    def test_dmon_parser(self) -> None:
        text = """
# gpu pwr gtemp mtemp sm mem
0 12 52 - 80 31
0 14 53 - 60 21
"""
        parsed = GOVERNOR.parse_dmon(text)
        self.assertEqual(parsed["samples"], 2)
        self.assertEqual(parsed["power_w_mean"], 13.0)
        self.assertEqual(parsed["temperature_c_max"], 53.0)
        self.assertEqual(parsed["sm_utilization_mean"], 70.0)
        self.assertEqual(parsed["memory_utilization_mean"], 26.0)

    def test_cpu_list_validation(self) -> None:
        self.assertEqual(GOVERNOR.expand_cpu_list("0-2,5"), [0, 1, 2, 5])
        with self.assertRaises(ValueError):
            GOVERNOR.expand_cpu_list("2-0")


if __name__ == "__main__":
    unittest.main()
