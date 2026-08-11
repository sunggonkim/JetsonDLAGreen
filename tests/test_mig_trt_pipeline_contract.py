import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "benchmarks/mig_trt_pipeline.cpp").read_text(encoding="utf-8")


class MigTrtPipelineContractTest(unittest.TestCase):
    def test_dependent_transfer_precedes_producer_checksum(self) -> None:
        send = SOURCE.index("write_all(transfer_fd, &transfer, sizeof(transfer));")
        checksum = SOURCE.index(
            "transfer.producer_checksum = checksum(payload.host(), payload.bytes());"
        )
        self.assertLess(send, checksum)

    def test_producer_scope_resumes_before_checksum(self) -> None:
        producer_resume = SOURCE.index(
            "if (options.gate_scope == GateScope::kProducer) {\n"
            "        resume_processes(options.gate_pids, transfer.resume_issued_ns,"
        )
        checksum = SOURCE.index(
            "transfer.producer_checksum = checksum(payload.host(), payload.bytes());"
        )
        self.assertLess(producer_resume, checksum)

    def test_consumer_wall_boundary_precedes_output_validation(self) -> None:
        completion = SOURCE.index("result.consumer_done_ns = monotonic_ns();")
        output_checksum = SOURCE.index(
            "result.consumer_output_checksum = runner.output_checksum();"
        )
        self.assertLess(completion, output_checksum)
        self.assertIn("production_wall_definition", SOURCE)
        self.assertIn("correctness_validation_placement", SOURCE)

    def test_stage_breakdown_uses_completion_safe_timestamps(self) -> None:
        self.assertIn(
            "result.consumer_start_ns -\n                      result.transfer.producer_done_ns",
            SOURCE,
        )
        self.assertIn("direct_binding(options.transport)", SOURCE)
        self.assertIn(
            "result.consumer_payload_verification_done_ns -\n                                result.consumer_compute_done_ns",
            SOURCE,
        )
        self.assertIn(
            "full-coherent registered system-memory activation edge", SOURCE
        )

    def test_parent_merges_producer_metadata_for_all_dependency_modes(self) -> None:
        merge = "result.transfer = producer_metadata[result.transfer.iteration];"
        self.assertEqual(SOURCE.count(merge), 1)
        block = SOURCE[SOURCE.index("for (Result& result : collected)") :]
        self.assertIn(merge, block)

    def test_real_detection_head_workload_has_learned_split_contract(self) -> None:
        self.assertIn('value == "resnet-detection-head"', SOURCE)
        self.assertIn('return "Layer6_relu_Y"', SOURCE)
        self.assertIn("kResnetDetectionPayloadBytes", SOURCE)
        self.assertIn("[1,512,23,40]", SOURCE)

    def test_real_resnet50_classification_workload_has_image_net_split_contract(self) -> None:
        self.assertIn('value == "resnet50-classification"', SOURCE)
        self.assertIn('return "gpu_0/res4_5_branch2c_bn_2"', SOURCE)
        self.assertIn("kResnet50ClassificationPayloadBytes", SOURCE)
        self.assertIn("[1,1024,14,14]", SOURCE)

    def test_post_completion_output_trace_is_flushed_before_child_exit(self) -> None:
        record = SOURCE.index("runner.write_output_trace_record(")
        flush = SOURCE.index("application_output_trace.flush();", record)
        child_exit = SOURCE.index("_exit(0);", flush)
        self.assertLess(record, flush)
        self.assertLess(flush, child_exit)

    def test_real_input_trace_binds_cuda_inputs_and_pipeline_hashes(self) -> None:
        self.assertIn("JDGINT1", SOURCE)
        self.assertIn("--producer-input-trace", SOURCE)
        self.assertIn("cudaMemcpyAsync(producer input trace)", SOURCE)
        self.assertIn("input_sha256", SOURCE)
        self.assertIn("--producer-input-trace PATH", SOURCE)

    def test_independent_arm_requires_byte_identical_activation_replay(self) -> None:
        self.assertIn("JDGACT1", SOURCE)
        self.assertIn("ActivationReplayTrace", SOURCE)
        self.assertIn("--activation-replay-trace", SOURCE)
        self.assertIn("--capture-activation-trace", SOURCE)
        self.assertIn("runner.bind_direct_handoff(", SOURCE)
        self.assertIn("live producer activation differs from activation replay trace", SOURCE)
        self.assertIn(
            "result.consumer_checksum == result.transfer.producer_checksum",
            SOURCE,
        )
        self.assertNotIn("fill_independent_input", SOURCE)
        self.assertNotIn("independent-local-input-output-checksum", SOURCE)

    def test_activation_capture_is_outside_measured_pipeline(self) -> None:
        capture = SOURCE.index("capture_activation_trace(options);")
        mapping = SOURCE.index("const std::size_t bytes = payload_bytes(options.workload);", capture)
        self.assertLess(capture, mapping)
        self.assertIn("producer-activation-replay-output-oracle", SOURCE)

    def test_independent_correctness_does_not_require_output_diversity(self) -> None:
        """A classifier may validly emit one output for distinct inputs."""
        gate = SOURCE[SOURCE.index("if (options.checksum_mode != ChecksumMode::kOff)") :]
        self.assertIn("unique_payload_checksums >= 2U", gate)
        self.assertNotIn("unique_output_checksums >= 2U", gate)

    def test_operational_arrival_and_event_contracts_are_explicit(self) -> None:
        self.assertIn("JDGARR1", SOURCE)
        self.assertIn("--arrival-trace PATH", SOURCE)
        self.assertIn("--event-trace-csv PATH", SOURCE)
        self.assertIn("arrival_schedule_mode", SOURCE)
        self.assertIn("declared_arrival_ns", SOURCE)
        self.assertIn("resume_observed_ns", SOURCE)


if __name__ == "__main__":
    unittest.main()
