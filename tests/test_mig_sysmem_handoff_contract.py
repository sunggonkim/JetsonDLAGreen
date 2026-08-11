import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "benchmarks/mig_sysmem_handoff.cu").read_text(encoding="utf-8")


class MigSysmemHandoffContractTest(unittest.TestCase):
    def test_registered_edge_preserves_direct_mapping_contract(self) -> None:
        self.assertIn("cudaHostRegisterMapped", SOURCE)
        self.assertIn("cudaHostGetDevicePointer", SOURCE)
        self.assertIn("full-coherent registered system-memory activation edge", SOURCE)
        self.assertIn("producer_visibility_done_ns", SOURCE)
        self.assertIn("consumer_read_start_ns", SOURCE)
        self.assertIn("consumer_read_done_ns", SOURCE)

    def test_controls_are_explicitly_distinguished(self) -> None:
        for mode in (
            "pageable-direct-control",
            "pinned-bounce",
            "pageable-bounce",
            "managed-uvm-control",
            "host-materialize-control",
        ):
            self.assertIn(mode, SOURCE)
        self.assertIn("cudaHostRegisterDefault", SOURCE)
        self.assertIn("cudaMallocManaged", SOURCE)
        self.assertIn("host materialize and private consumer copy control", SOURCE)

    def test_same_instance_requires_a_shared_mps_control_surface(self) -> None:
        self.assertIn(
            "options.producer_uuid == options.consumer_uuid &&\n"
            "      options.producer_mps_pipe.empty()",
            SOURCE,
        )
        self.assertIn("same-instance-mps", SOURCE)

    def test_ipc_negative_control_never_attempts_ipc(self) -> None:
        self.assertIn("p2p-ipc-negative-control", SOURCE)
        self.assertIn("p2p_ipc_attempted\\\":false", SOURCE)
        self.assertIn("cross-MIG CUDA P2P/IPC is intentionally not", SOURCE)
        self.assertNotIn("cudaIpcOpenMemHandle", SOURCE)
        self.assertNotIn("cudaDeviceCanAccessPeer", SOURCE)

    def test_benchmark_emits_both_endpoint_capabilities(self) -> None:
        self.assertIn("producer_capabilities", SOURCE)
        self.assertIn("consumer_capabilities", SOURCE)
        self.assertIn("pageableMemoryAccessUsesHostPageTables", SOURCE)
        self.assertIn("concurrentManagedAccess", SOURCE)

    def test_cache_state_is_an_explicit_precondition_not_timed_transport(self) -> None:
        self.assertIn("enum class CacheState", SOURCE)
        self.assertIn('cache_state_name', SOURCE)
        self.assertIn('cache_state', SOURCE)
        self.assertIn('cache_flush_bytes', SOURCE)
        self.assertIn("prepare_cache(options, cache_scratch", SOURCE)
        self.assertIn("command.producer_start_ns = monotonic_ns();", SOURCE)
        self.assertIn("result.consumer_read_start_ns = monotonic_ns();", SOURCE)


if __name__ == "__main__":
    unittest.main()
