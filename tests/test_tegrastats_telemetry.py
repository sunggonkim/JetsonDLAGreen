#!/usr/bin/env python3
import importlib.util
import io
import json
import math
import pathlib
import sys
import tempfile
import threading
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tegrastats_telemetry", ROOT / "runtime" / "tegrastats_telemetry.py"
)
assert SPEC is not None and SPEC.loader is not None
TELEMETRY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TELEMETRY
SPEC.loader.exec_module(TELEMETRY)


LOADED_LINE = (
    "08-08-2026 21:00:39 RAM 8353/125749MB (lfb 31x4MB) "
    "CPU [90%@2601,75%@2601,off,7%@2601] EMC_FREQ 34%@4266 "
    "GR3D_FREQ 87%@[1574,1574,1574] NVENC0_FREQ @1691 VIC off "
    "cpu@58.25C/61.75C tj@66.625C/66.625C gpu@56.937C/66.625C "
    "soc012@55.375C/58.781C soc345@55.812C/60.031C "
    "VDD_GPU 37939mW/8660mW/37939mW "
    "VDD_CPU_SOC_MSS 30059mW/12281mW/30059mW "
    "VIN_SYS_5V0 15486mW/7424mW/15486mW "
    "VIN 124486mW/36487mW/124486mW FUTURE_TOKEN value"
)

IDLE_LINE = (
    "08-08-2026 20:58:23 RAM 3429/125749MB (lfb 120x4MB) "
    "CPU [0%,5%,off]@2601 EMC_FREQ @4266 "
    "GR3D_FREQ @[1574,1574,1574] cpu@49.75C/49.75C "
    "tj@51.218C/51.218C gpu@51.218C/51.218C "
    "VDD_GPU 6330mW/6330mW/6330mW VIN 28906mW/14453mW/28906mW"
)


class TegrastatsParserTest(unittest.TestCase):
    def test_parses_loaded_r39_line_and_ignores_unknown_tokens(self) -> None:
        parsed = TELEMETRY.parse_tegrastats_line(LOADED_LINE)
        self.assertEqual(parsed.ram.used_mb, 8353.0)
        self.assertEqual(parsed.ram.largest_free_block_count, 31)
        self.assertEqual(parsed.cpu[0].utilization_pct, 90.0)
        self.assertEqual(parsed.cpu[0].frequency_mhz, 2601.0)
        self.assertFalse(parsed.cpu[2].online)
        self.assertEqual(parsed.emc.utilization_pct, 34.0)
        self.assertEqual(parsed.emc.frequencies_mhz, (4266.0,))
        self.assertEqual(parsed.gr3d.utilization_pct, 87.0)
        self.assertEqual(
            parsed.gr3d.frequencies_mhz, (1574.0, 1574.0, 1574.0)
        )
        self.assertEqual(parsed.temperatures_c["tj"], 66.625)
        self.assertEqual(parsed.power["VIN"].current_mw, 124486.0)
        self.assertEqual(
            parsed.power["VIN"].reported_extra_mw,
            (36487.0, 124486.0),
        )

    def test_clock_only_variants_do_not_invent_zero_utilization(self) -> None:
        parsed = TELEMETRY.parse_tegrastats_line(IDLE_LINE)
        self.assertIsNone(parsed.emc.utilization_pct)
        self.assertEqual(parsed.emc.frequencies_mhz, (4266.0,))
        self.assertIsNone(parsed.gr3d.utilization_pct)
        self.assertEqual(
            parsed.gr3d.frequencies_mhz, (1574.0, 1574.0, 1574.0)
        )
        self.assertEqual(parsed.cpu[0].frequency_mhz, 2601.0)
        self.assertEqual(parsed.cpu[1].utilization_pct, 5.0)

    def test_all_documented_clock_variants_parse_independently(self) -> None:
        cases = (
            ("EMC_FREQ 95% GR3D_FREQ 99%", 95.0, (), 99.0, ()),
            (
                "EMC_FREQ @1600 GR3D_FREQ @[1098,1098,1098]",
                None,
                (1600.0,),
                None,
                (1098.0, 1098.0, 1098.0),
            ),
            (
                "EMC_FREQ 95%@1600 GR3D_FREQ 99%@[1098,1098,1098]",
                95.0,
                (1600.0,),
                99.0,
                (1098.0, 1098.0, 1098.0),
            ),
        )
        for line, emc_util, emc_freq, gr3d_util, gr3d_freq in cases:
            with self.subTest(line=line):
                parsed = TELEMETRY.parse_tegrastats_line(line)
                self.assertEqual(parsed.emc.utilization_pct, emc_util)
                self.assertEqual(parsed.emc.frequencies_mhz, emc_freq)
                self.assertEqual(parsed.gr3d.utilization_pct, gr3d_util)
                self.assertEqual(parsed.gr3d.frequencies_mhz, gr3d_freq)

    def test_partial_or_malformed_line_returns_partial_parse(self) -> None:
        parsed = TELEMETRY.parse_tegrastats_line(
            "warning RAM 200/100MB CPU [bad,off] UNKNOWN 1"
        )
        self.assertIsNone(parsed.ram)
        self.assertIsNone(parsed.emc)
        self.assertEqual(len(parsed.cpu), 2)
        self.assertIsNone(parsed.cpu[0].online)
        self.assertFalse(parsed.cpu[1].online)
        self.assertFalse(parsed.temperatures_c)
        self.assertFalse(parsed.power)

    def test_invalid_cpu_utilization_does_not_make_cpu_field_valid(self) -> None:
        line = LOADED_LINE.replace(
            "CPU [90%@2601,75%@2601,off,7%@2601]",
            "CPU [101%@2601,bad,off]",
        )
        sample = TELEMETRY.TelemetrySample(
            100,
            line,
            TELEMETRY.parse_tegrastats_line(line),
            1000.0,
        )
        result = TELEMETRY.aggregate_samples([sample], 0, 200)
        self.assertEqual(result["valid_samples"], 0)
        self.assertEqual(result["invalid_samples"], 1)
        self.assertEqual(result["health"]["missing_counts"]["cpu"], 1)

    def test_parses_optional_swap(self) -> None:
        parsed = TELEMETRY.parse_tegrastats_line(
            "SWAP 12/1024MB (cached 4MB)"
        )
        self.assertEqual(parsed.swap.used_mb, 12.0)
        self.assertEqual(parsed.swap.total_mb, 1024.0)
        self.assertEqual(parsed.swap.cached_mb, 4.0)

    def test_current_only_temperature_and_two_value_power(self) -> None:
        parsed = TELEMETRY.parse_tegrastats_line(
            "cpu@38.062C VDD_GPU 5145mW/4948mW"
        )
        self.assertEqual(parsed.temperatures_c, {"cpu": 38.062})
        self.assertEqual(parsed.power["VDD_GPU"].current_mw, 5145.0)
        self.assertEqual(
            parsed.power["VDD_GPU"].reported_extra_mw, (4948.0,)
        )

    def test_reads_mem_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "meminfo"
            path.write_text(
                "MemTotal: 1024000 kB\nMemAvailable: 524288 kB\n",
                encoding="utf-8",
            )
            self.assertEqual(TELEMETRY.read_mem_available_mb(path), 512.0)

    def test_rejects_mem_available_with_wrong_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "meminfo"
            path.write_text("MemAvailable: 524288 MB\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                TELEMETRY.read_mem_available_mb(path)


class TelemetryWriterMonitorTest(unittest.TestCase):
    def test_jsonl_preserves_raw_parsed_sample_and_marker(self) -> None:
        stream = io.StringIO()
        writer = TELEMETRY.JsonlTelemetryWriter(stream)
        monitor = TELEMETRY.TegrastatsMonitor(
            writer, clock=iter((100, 150)).__next__, mem_available_reader=lambda: 4096.0
        )
        sample = monitor.record_line(LOADED_LINE + "\n")
        marker = monitor.mark("measurement_start", {"policy": "static-mig"})

        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(sample.monotonic_ns, 100)
        self.assertEqual(marker.monotonic_ns, 150)
        self.assertEqual(records[0]["record_type"], "sample")
        self.assertEqual(records[0]["raw"], LOADED_LINE)
        self.assertEqual(records[0]["parsed"]["ram"]["used_mb"], 8353.0)
        self.assertEqual(records[0]["mem_available_mb"], 4096.0)
        self.assertEqual(records[1]["record_type"], "marker")
        self.assertEqual(records[1]["metadata"]["policy"], "static-mig")

    def test_mem_available_failure_is_recorded_not_raised(self) -> None:
        def fail() -> float:
            raise OSError("unavailable")

        stream = io.StringIO()
        monitor = TELEMETRY.TegrastatsMonitor(
            TELEMETRY.JsonlTelemetryWriter(stream),
            mem_available_reader=fail,
        )
        sample = monitor.record_line(LOADED_LINE, monotonic_ns=100)
        self.assertIsNone(sample.mem_available_mb)
        self.assertEqual(sample.collection_errors, ("mem_available:OSError",))

    def test_invalid_reader_value_is_recorded_not_serialized_as_nan(self) -> None:
        stream = io.StringIO()
        monitor = TELEMETRY.TegrastatsMonitor(
            TELEMETRY.JsonlTelemetryWriter(stream),
            mem_available_reader=lambda: math.nan,
        )
        sample = monitor.record_line(LOADED_LINE, monotonic_ns=100)
        self.assertIsNone(sample.mem_available_mb)
        self.assertEqual(sample.collection_errors, ("mem_available:ValueError",))
        json.loads(stream.getvalue())

    def test_bounded_monitor_retention_does_not_change_jsonl(self) -> None:
        stream = io.StringIO()
        monitor = TELEMETRY.TegrastatsMonitor(
            TELEMETRY.JsonlTelemetryWriter(stream),
            mem_available_reader=lambda: 1.0,
            max_samples=2,
        )
        for timestamp in (100, 200, 300):
            monitor.record_line(LOADED_LINE, monotonic_ns=timestamp)
        self.assertEqual(
            [sample.monotonic_ns for sample in monitor.samples()], [200, 300]
        )
        self.assertEqual(len(stream.getvalue().splitlines()), 3)
        result = monitor.aggregate(100, 400)
        self.assertFalse(result["health"]["healthy"])
        self.assertIn("retention_truncated", result["health"]["reasons"])
        self.assertEqual(result["retention"]["dropped_samples"], 1)
        self.assertEqual(result["retention"]["last_dropped_sample_ns"], 100)
        self.assertFalse(result["retention"]["interval_complete"])

    def test_sample_window_is_reverse_bounded_with_retention_provenance(self) -> None:
        monitor = TELEMETRY.TegrastatsMonitor(
            TELEMETRY.JsonlTelemetryWriter(io.StringIO()),
            mem_available_reader=lambda: 1.0,
            max_samples=3,
        )
        monitor.record_line(IDLE_LINE, monotonic_ns=100)
        monitor.record_line(IDLE_LINE, monotonic_ns=200)
        monitor.record_line(LOADED_LINE, monotonic_ns=200)
        monitor.record_line(LOADED_LINE, monotonic_ns=300)

        newest = monitor.sample_window(
            150, 300, reverse=True, limit=1
        )
        self.assertEqual([sample.monotonic_ns for sample in newest.samples], [200])
        self.assertEqual(newest.samples[0].raw, LOADED_LINE)
        self.assertTrue(newest.limit_reached)
        self.assertTrue(newest.interval_complete)
        inclusive = monitor.sample_window(
            150, 300, end_inclusive=True, reverse=True
        )
        self.assertEqual(
            [sample.monotonic_ns for sample in inclusive.samples],
            [300, 200, 200],
        )
        truncated = monitor.sample_window(100, 300, end_inclusive=True)
        self.assertFalse(truncated.interval_complete)
        self.assertEqual(truncated.last_dropped_sample_ns, 100)
        self.assertEqual(truncated.dropped_samples, 1)

    def test_aggregate_can_include_a_sample_at_the_exact_end_marker(self) -> None:
        monitor = TELEMETRY.TegrastatsMonitor(
            TELEMETRY.JsonlTelemetryWriter(io.StringIO()),
            mem_available_reader=lambda: 1.0,
        )
        monitor.record_line(IDLE_LINE, monotonic_ns=100)
        monitor.record_line(LOADED_LINE, monotonic_ns=200)

        half_open = monitor.aggregate(100, 200)
        inclusive = monitor.aggregate(100, 200, end_inclusive=True)
        self.assertEqual(half_open["total_samples"], 1)
        self.assertFalse(half_open["interval"]["end_inclusive"])
        self.assertEqual(inclusive["total_samples"], 2)
        self.assertTrue(inclusive["interval"]["end_inclusive"])
        self.assertEqual(inclusive["health"]["last_interval_sample_ns"], 200)

    def test_bounded_monitor_aggregate_matches_standalone_semantics(self) -> None:
        monitor = TELEMETRY.TegrastatsMonitor(
            TELEMETRY.JsonlTelemetryWriter(io.StringIO()),
            mem_available_reader=lambda: 1.0,
        )
        monitor.record_line(LOADED_LINE, monotonic_ns=90)
        monitor.record_line("Unknown command", monotonic_ns=190)
        samples = monitor.samples()
        kwargs = {
            "required_fields": ("temperature:soc012", "temperature:tj"),
            "reference_ns": 200,
            "stale_after_ns": 50,
            "end_inclusive": True,
        }
        expected = TELEMETRY.aggregate_samples(samples, 100, 200, **kwargs)
        actual = monitor.aggregate(100, 200, **kwargs)
        actual.pop("retention")
        self.assertEqual(actual, expected)

    def test_concurrent_writer_calls_remain_complete_json_lines(self) -> None:
        stream = io.StringIO()
        writer = TELEMETRY.JsonlTelemetryWriter(stream)

        def write_markers(worker: int) -> None:
            for index in range(20):
                writer.write(
                    TELEMETRY.TelemetryMarker(
                        100,
                        "concurrent",
                        {"worker": worker, "index": index},
                    )
                )

        threads = [
            threading.Thread(target=write_markers, args=(index,))
            for index in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(records), 80)
        self.assertTrue(all(record["record_type"] == "marker" for record in records))

    def test_monitor_captures_marker_timestamp_before_waiting_for_io_lock(self) -> None:
        stream = io.StringIO()
        reader_entered = threading.Event()
        release_reader = threading.Event()
        second_clock_call = threading.Event()
        clock_lock = threading.Lock()
        clock_calls = 0

        def clock() -> int:
            nonlocal clock_calls
            with clock_lock:
                clock_calls += 1
                if clock_calls == 2:
                    second_clock_call.set()
                return clock_calls * 100

        def blocked_reader() -> float:
            reader_entered.set()
            if not release_reader.wait(2.0):
                raise OSError("test timed out")
            return 4096.0

        monitor = TELEMETRY.TegrastatsMonitor(
            TELEMETRY.JsonlTelemetryWriter(stream),
            clock=clock,
            mem_available_reader=blocked_reader,
        )
        errors: list[BaseException] = []
        marker_holder = []

        def record_sample() -> None:
            try:
                monitor.record_line(LOADED_LINE)
            except BaseException as error:
                errors.append(error)

        def record_marker() -> None:
            try:
                marker_holder.append(monitor.mark("measurement_start"))
            except BaseException as error:
                errors.append(error)

        sample_thread = threading.Thread(target=record_sample)
        marker_thread = threading.Thread(target=record_marker)
        sample_thread.start()
        self.assertTrue(reader_entered.wait(1.0))
        marker_thread.start()
        try:
            self.assertTrue(second_clock_call.wait(1.0))
        finally:
            release_reader.set()
        sample_thread.join()
        marker_thread.join()
        self.assertFalse(errors)
        self.assertEqual(marker_holder[0].monotonic_ns, 200)

    def test_monitor_and_writer_reject_timestamp_regression(self) -> None:
        stream = io.StringIO()
        monitor = TELEMETRY.TegrastatsMonitor(
            TELEMETRY.JsonlTelemetryWriter(stream),
            mem_available_reader=lambda: 1.0,
        )
        monitor.record_line(LOADED_LINE, monotonic_ns=200)
        with self.assertRaises(ValueError):
            monitor.mark("regression", monotonic_ns=100)
        self.assertEqual(len(stream.getvalue().splitlines()), 1)

    def test_append_mode_is_rejected_for_unterminated_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "telemetry.jsonl"
            path.write_text('{"partial":true}', encoding="utf-8")
            with self.assertRaises(ValueError):
                TELEMETRY.JsonlTelemetryWriter(path, mode="a")
            self.assertEqual(path.read_text(encoding="utf-8"), '{"partial":true}')

    def test_close_flushes_external_stream_when_per_write_flush_is_disabled(self) -> None:
        binary = io.BytesIO()
        stream = io.TextIOWrapper(binary, encoding="utf-8")
        writer = TELEMETRY.JsonlTelemetryWriter(stream, flush=False)
        writer.write(TELEMETRY.TelemetryMarker(100, "marker", {}))
        writer.close()
        self.assertTrue(binary.getvalue().endswith(b"\n"))
        stream.detach()

    def test_buffered_writer_flushes_markers_and_periodic_sample_batches(self) -> None:
        binary = io.BytesIO()
        stream = io.TextIOWrapper(binary, encoding="utf-8")
        writer = TELEMETRY.JsonlTelemetryWriter(
            stream, flush=False, flush_every=2
        )
        writer.write(TELEMETRY.TelemetrySample(
            100, LOADED_LINE, TELEMETRY.parse_tegrastats_line(LOADED_LINE), 1.0
        ))
        self.assertEqual(binary.getvalue(), b"")
        writer.write(TELEMETRY.TelemetrySample(
            200, LOADED_LINE, TELEMETRY.parse_tegrastats_line(LOADED_LINE), 1.0
        ))
        self.assertTrue(binary.getvalue().endswith(b"\n"))
        before_marker = len(binary.getvalue())
        writer.write(TELEMETRY.TelemetryMarker(300, "boundary", {}))
        self.assertGreater(len(binary.getvalue()), before_marker)
        writer.close()
        stream.detach()

    def test_rejects_invalid_periodic_flush_interval(self) -> None:
        with self.assertRaises(ValueError):
            TELEMETRY.JsonlTelemetryWriter(io.StringIO(), flush_every=0)

    def test_invalid_line_is_retained_in_jsonl(self) -> None:
        stream = io.StringIO()
        monitor = TELEMETRY.TegrastatsMonitor(
            TELEMETRY.JsonlTelemetryWriter(stream),
            mem_available_reader=lambda: 1.0,
        )
        monitor.record_line("Unknown command: --count", monotonic_ns=100)
        record = json.loads(stream.getvalue())
        self.assertEqual(record["raw"], "Unknown command: --count")
        self.assertIsNone(record["parsed"]["ram"])


class TelemetryAggregateTest(unittest.TestCase):
    @staticmethod
    def sample(line: str, timestamp: int, available: float | None):
        return TELEMETRY.TelemetrySample(
            timestamp,
            line,
            TELEMETRY.parse_tegrastats_line(line),
            available,
        )

    def test_interval_is_half_open_and_aggregates_current_values(self) -> None:
        samples = [
            self.sample(IDLE_LINE, 100, 8000.0),
            self.sample(LOADED_LINE, 200, 7000.0),
            self.sample(LOADED_LINE, 300, 6000.0),
        ]
        result = TELEMETRY.aggregate_samples(samples, 100, 300)
        self.assertEqual(result["total_samples"], 2)
        self.assertEqual(result["valid_samples"], 2)
        self.assertTrue(result["health"]["healthy"])
        self.assertEqual(result["ram"]["used_mb"]["min"], 3429.0)
        self.assertEqual(result["ram"]["used_mb"]["max"], 8353.0)
        self.assertEqual(result["mem_available_mb"]["mean"], 7500.0)
        self.assertEqual(result["cpu"]["cores"][0]["off_samples"], 0)
        self.assertEqual(result["temperatures_c"]["tj"]["max"], 66.625)
        self.assertEqual(result["power_mw"]["VIN"]["max"], 124486.0)
        self.assertEqual(result["emc"]["utilization_pct"]["count"], 1)
        self.assertEqual(result["gr3d"]["frequencies_mhz"][0]["count"], 2)

    def test_missing_required_field_is_unhealthy_but_optional_clocks_are_not(self) -> None:
        no_clocks = LOADED_LINE.replace("EMC_FREQ 34%@4266 ", "").replace(
            "GR3D_FREQ 87%@[1574,1574,1574] ", ""
        )
        sample = self.sample(no_clocks, 100, 1000.0)
        default_result = TELEMETRY.aggregate_samples([sample], 0, 200)
        self.assertTrue(default_result["health"]["healthy"])
        required_result = TELEMETRY.aggregate_samples(
            [sample], 0, 200, required_fields=("ram", "emc_utilization")
        )
        self.assertFalse(required_result["health"]["healthy"])
        self.assertEqual(required_result["valid_samples"], 0)
        self.assertEqual(
            required_result["health"]["missing_counts"]["emc_utilization"], 1
        )

    def test_stale_health_uses_caller_supplied_cadence(self) -> None:
        sample = self.sample(LOADED_LINE, 100, 1000.0)
        result = TELEMETRY.aggregate_samples(
            [sample],
            0,
            500,
            reference_ns=500,
            stale_after_ns=200,
        )
        self.assertFalse(result["health"]["healthy"])
        self.assertTrue(result["health"]["stale"])
        self.assertIn("stale_stream", result["health"]["reasons"])

    def test_freshness_uses_sample_before_interval(self) -> None:
        sample = self.sample(LOADED_LINE, 90, 1000.0)
        result = TELEMETRY.aggregate_samples(
            [sample],
            100,
            200,
            reference_ns=200,
            stale_after_ns=150,
        )
        self.assertEqual(result["total_samples"], 0)
        self.assertEqual(result["health"]["last_sample_ns"], 90)
        self.assertFalse(result["health"]["stale"])
        self.assertIn("no_samples", result["health"]["reasons"])

    def test_reference_must_not_precede_interval_end(self) -> None:
        sample = self.sample(LOADED_LINE, 190, 1000.0)
        with self.assertRaises(ValueError):
            TELEMETRY.aggregate_samples(
                [sample],
                100,
                200,
                reference_ns=150,
                stale_after_ns=100,
            )

    def test_exact_stale_boundary_is_fresh(self) -> None:
        sample = self.sample(LOADED_LINE, 50, 1000.0)
        result = TELEMETRY.aggregate_samples(
            [sample],
            0,
            100,
            reference_ns=200,
            stale_after_ns=150,
        )
        self.assertTrue(result["health"]["healthy"])
        self.assertFalse(result["health"]["stale"])

    def test_recent_incomplete_sample_does_not_hide_stale_valid_sample(self) -> None:
        samples = [
            self.sample(LOADED_LINE, 100, 1000.0),
            self.sample("Unknown command: --count", 190, 1000.0),
        ]
        result = TELEMETRY.aggregate_samples(
            samples,
            0,
            200,
            reference_ns=200,
            stale_after_ns=50,
        )
        self.assertFalse(result["health"]["stale"])
        self.assertTrue(result["health"]["valid_sample_stale"])
        self.assertIn("stale_valid_sample", result["health"]["reasons"])

    def test_valid_sample_gap_includes_both_interval_boundaries(self) -> None:
        samples = [
            self.sample(LOADED_LINE, 100, 1000.0),
            self.sample(LOADED_LINE, 300, 1000.0),
        ]
        result = TELEMETRY.aggregate_samples(
            samples,
            0,
            400,
            maximum_valid_gap_ns=200,
        )
        self.assertTrue(result["health"]["healthy"])
        self.assertEqual(result["health"]["maximum_valid_gap_ns"], 200)
        self.assertEqual(
            result["health"]["observed_maximum_valid_gap_ns"], 200
        )
        self.assertFalse(result["health"]["valid_gap_exceeded"])

        internal_gap = TELEMETRY.aggregate_samples(
            [
                self.sample(LOADED_LINE, 100, 1000.0),
                self.sample(LOADED_LINE, 301, 1000.0),
            ],
            0,
            400,
            maximum_valid_gap_ns=200,
        )
        self.assertFalse(internal_gap["health"]["healthy"])
        self.assertTrue(internal_gap["health"]["valid_gap_exceeded"])
        self.assertIn(
            "valid_sample_gap_exceeded", internal_gap["health"]["reasons"]
        )

        boundary_gap = TELEMETRY.aggregate_samples(
            [self.sample(LOADED_LINE, 201, 1000.0)],
            0,
            400,
            maximum_valid_gap_ns=200,
        )
        self.assertFalse(boundary_gap["health"]["healthy"])
        self.assertEqual(
            boundary_gap["health"]["observed_maximum_valid_gap_ns"], 201
        )

    def test_valid_sample_gap_bound_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            TELEMETRY.aggregate_samples(
                [], 0, 100, maximum_valid_gap_ns=0
            )

    def test_incomplete_rows_are_tolerated_unless_strictly_requested(self) -> None:
        samples = [
            self.sample(LOADED_LINE, 100, 1000.0),
            self.sample("Unknown command: --count", 150, 1000.0),
        ]
        tolerant = TELEMETRY.aggregate_samples(
            samples, 0, 200, minimum_valid_samples=1
        )
        self.assertTrue(tolerant["health"]["healthy"])
        self.assertEqual(tolerant["invalid_samples"], 1)
        self.assertEqual(tolerant["health"]["missing_counts"]["ram"], 1)

        strict = TELEMETRY.aggregate_samples(
            samples,
            0,
            200,
            minimum_valid_samples=1,
            require_all_samples_valid=True,
        )
        self.assertFalse(strict["health"]["healthy"])
        self.assertIn("incomplete_samples", strict["health"]["reasons"])

    def test_no_samples_and_invalid_arguments_are_reported(self) -> None:
        result = TELEMETRY.aggregate_samples([], 100, 200)
        self.assertFalse(result["health"]["healthy"])
        self.assertIn("no_samples", result["health"]["reasons"])
        with self.assertRaises(ValueError):
            TELEMETRY.aggregate_samples([], 200, 100)
        with self.assertRaises(ValueError):
            TELEMETRY.aggregate_samples(
                [], 100, 200, required_fields=("not-a-field",)
            )
        with self.assertRaisesRegex(ValueError, "end_inclusive must be boolean"):
            TELEMETRY.aggregate_samples([], 100, 200, end_inclusive="yes")


if __name__ == "__main__":
    unittest.main()
