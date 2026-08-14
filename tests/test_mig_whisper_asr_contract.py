import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "benchmarks/mig_whisper_asr.cpp").read_text(encoding="utf-8")


class MigWhisperAsrContractTest(unittest.TestCase):
    def test_three_slot_credit_window_is_explicit_and_bounded(self) -> None:
        self.assertIn("int pipeline_slots{1};", SOURCE)
        self.assertIn('"pipeline-slots must be 1 or 3"', SOURCE)
        self.assertIn("index >= options.pipeline_slots", SOURCE)
        self.assertIn("while (acknowledgements < total)", SOURCE)
        self.assertIn("activation_slot(activation.device(), transfer.slot)", SOURCE)
        self.assertIn("Whisper activation slot ownership differs", SOURCE)

    def test_operational_arrival_and_queueing_are_reported(self) -> None:
        self.assertIn("wait_until_ns(transfer.arrival_ns)", SOURCE)
        self.assertIn('"--arrival-period-us"', SOURCE)
        self.assertIn("declared_arrival_ns,release_ns", SOURCE)
        self.assertIn("queue_p99_us", SOURCE)
        self.assertIn("request_goodput_rps", SOURCE)

    def test_warmup_misses_are_not_counted(self) -> None:
        warmup = SOURCE.index("if (item.transfer.warmup == 0U)")
        miss = SOURCE.index("misses += miss ? 1U : 0U;", warmup)
        self.assertGreater(miss, warmup)

    def test_producer_gate_resumes_at_publication_boundary(self) -> None:
        infer = SOURCE.index("encoder.infer(input.sample(index)")
        done = SOURCE.index("transfer.producer_done_ns = monotonic_ns();", infer)
        resume = SOURCE.index("resume_processes(options.gate_pids", done)
        publish = SOURCE.index("write_all(transfer_fd", resume)
        self.assertLess(infer, done)
        self.assertLess(done, resume)
        self.assertLess(resume, publish)
        self.assertIn("gate_hold_p99_us", SOURCE)


if __name__ == "__main__":
    unittest.main()
