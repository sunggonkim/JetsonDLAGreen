import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "benchmarks/mig_sysmem_ring.cu").read_text(encoding="utf-8")


class MigSysmemRingContractTest(unittest.TestCase):
    def test_three_slot_sequence_and_ownership_protocol_is_present(self) -> None:
        self.assertIn("kRingSlots = 3", SOURCE)
        self.assertIn("std::atomic<std::uint64_t> sequence", SOURCE)
        self.assertIn("kReady", SOURCE)
        self.assertIn("kConsuming", SOURCE)
        self.assertIn("compare_exchange_strong", SOURCE)
        self.assertIn("ticket + 1", SOURCE)
        self.assertIn("ticket + kRingSlots", SOURCE)

    def test_direct_registered_edge_is_used_for_each_endpoint(self) -> None:
        self.assertGreaterEqual(SOURCE.count("cudaHostRegisterMapped"), 2)
        self.assertGreaterEqual(SOURCE.count("cudaHostGetDevicePointer"), 2)
        self.assertIn("full-coherent registered system-memory activation edge", SOURCE)

    def test_backpressure_timeout_and_stale_reclaim_are_observable(self) -> None:
        self.assertIn("backpressure_events", SOURCE)
        self.assertIn("timeout_events", SOURCE)
        self.assertIn("stale_reclaims", SOURCE)
        self.assertIn("consumer-delay-us", SOURCE)
        self.assertIn("fail-consumer-after", SOURCE)
        self.assertIn("fault-ok", SOURCE)

    def test_external_process_and_event_trace_contract_is_present(self) -> None:
        self.assertGreaterEqual(SOURCE.count("fork()"), 2)
        self.assertIn("producer_publish_ns", SOURCE)
        self.assertIn("consumer_start_ns", SOURCE)
        self.assertIn("consumer_done_ns", SOURCE)
        self.assertIn("\\\"events\\\":[", SOURCE)

    def test_no_cross_mig_cuda_ipc_path_is_added(self) -> None:
        self.assertNotIn("cudaIpcOpenMemHandle", SOURCE)
        self.assertNotIn("cudaDeviceCanAccessPeer", SOURCE)


if __name__ == "__main__":
    unittest.main()
