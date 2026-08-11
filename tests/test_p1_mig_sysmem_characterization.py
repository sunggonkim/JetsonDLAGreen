import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p1_mig_sysmem_characterization.sh"
SOURCE = SCRIPT.read_text(encoding="utf-8")


class P1MigSysmemCharacterizationContractTest(unittest.TestCase):
    def test_sweep_binds_required_dimensions(self) -> None:
        for value in (
            "DIRECTIONS",
            "CACHE_STATES",
            "PRESSURE_MODES",
            "SIZES",
            "cache-state",
            "cache-flush-bytes",
            "p2p-ipc-negative-control",
        ):
            self.assertIn(value, SOURCE)

    def test_full_visibility_and_read_validation_is_retained(self) -> None:
        self.assertIn("validate_result", SOURCE)
        self.assertIn('"mismatches"', SOURCE)
        self.assertIn('"transport_description"', SOURCE)
        self.assertIn("negative-controls", SOURCE)

    def test_pressure_runs_are_external_to_the_measured_process(self) -> None:
        self.assertIn('"${PRESSURE_BENCH}" --role pressure', SOURCE)
        self.assertIn('CUDA_VISIBLE_DEVICES="${uuid}"', SOURCE)
        self.assertIn('taskset --cpu-list "${CONTROL_CPU}" "${HANDOFF}"', SOURCE)


if __name__ == "__main__":
    unittest.main()
