#!/usr/bin/env python3
"""Parse and record Jetson R39.2 ``tegrastats`` telemetry.

The installed R39.2 utility emits a superset of the public format and can omit
activity fields while retaining clock fields.  This module therefore parses
known fields independently, preserves every raw line, and leaves unknown
tokens untouched.  It deliberately contains no admission or safety policy.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import math
import pathlib
import re
import statistics
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TextIO


SCHEMA_VERSION = 1
DEFAULT_REQUIRED_FIELDS = (
    "ram",
    "mem_available",
    "cpu",
    "temperature",
    "power",
)

_NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
_RAM_RE = re.compile(
    rf"\bRAM\s+(?P<used>{_NUMBER})/(?P<total>{_NUMBER})MB"
    rf"(?:\s+\(lfb\s+(?P<count>\d+)x(?P<size>{_NUMBER})MB\))?"
)
_SWAP_RE = re.compile(
    rf"\bSWAP\s+(?P<used>{_NUMBER})/(?P<total>{_NUMBER})MB"
    rf"(?:\s+\(cached\s+(?P<cached>{_NUMBER})MB\))?"
)
_CPU_RE = re.compile(
    rf"\bCPU\s+\[(?P<cores>[^]]*)\](?:\s*@(?P<common_freq>{_NUMBER}))?"
)
_CPU_CORE_RE = re.compile(
    rf"^(?P<util>{_NUMBER})%(?:@(?P<freq>{_NUMBER}))?$"
)
_EMC_RE = re.compile(
    rf"\bEMC_FREQ\s+(?:(?P<util>{_NUMBER})%)?(?:@(?P<freq>{_NUMBER}))?"
)
_GR3D_RE = re.compile(
    rf"\bGR3D_FREQ\s+(?:(?P<util>{_NUMBER})%)?"
    rf"(?:@\[(?P<freqs>[^]]*)\]|@(?P<single_freq>{_NUMBER}))?"
)
_TEMPERATURE_RE = re.compile(
    rf"(?<![A-Za-z0-9_.-])(?P<name>[A-Za-z][A-Za-z0-9_.-]*)"
    rf"@(?P<current>{_NUMBER})C(?:/{_NUMBER}C)?"
)
_POWER_RE = re.compile(
    rf"\b(?P<name>(?:VDD|VIN)[A-Z0-9_]*)\s+"
    rf"(?P<current>{_NUMBER})mW"
    rf"(?:/(?P<extra1>{_NUMBER})mW)?"
    rf"(?:/(?P<extra2>{_NUMBER})mW)?"
)


def _number(text: str | None, *, minimum: float | None = None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        return None
    return value


def _utilization(text: str | None) -> float | None:
    value = _number(text, minimum=0.0)
    if value is None or value > 100.0:
        return None
    return value


def _validated_monotonic_ns(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("monotonic_ns must be a non-negative integer")
    return value


def _validated_mem_available_mb(value: object) -> float:
    try:
        available = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("MemAvailable must be numeric") from error
    if not math.isfinite(available) or available < 0.0:
        raise ValueError("MemAvailable must be finite and non-negative")
    return available


@dataclass(frozen=True)
class CpuCore:
    """One CPU entry from a ``tegrastats`` line."""

    utilization_pct: float | None
    frequency_mhz: float | None
    online: bool | None


@dataclass(frozen=True)
class RamSample:
    """Global system RAM counters reported by ``tegrastats``."""

    used_mb: float
    total_mb: float
    largest_free_block_count: int | None
    largest_free_block_mb: float | None


@dataclass(frozen=True)
class SwapSample:
    """Optional swap counters reported by ``tegrastats``."""

    used_mb: float
    total_mb: float
    cached_mb: float | None


@dataclass(frozen=True)
class ClockActivity:
    """Optional utilization and one or more physical clock frequencies."""

    utilization_pct: float | None
    frequencies_mhz: tuple[float, ...]


@dataclass(frozen=True)
class PowerSample:
    """A power rail's documented current value and opaque extra values."""

    current_mw: float
    reported_extra_mw: tuple[float, ...]


@dataclass(frozen=True)
class ParsedTegrastats:
    """Known fields parsed independently from one raw output line."""

    ram: RamSample | None
    swap: SwapSample | None
    cpu: tuple[CpuCore, ...]
    temperatures_c: Mapping[str, float]
    power: Mapping[str, PowerSample]
    emc: ClockActivity | None
    gr3d: ClockActivity | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class TelemetrySample:
    """A parsed line timestamped at receipt with a monotonic clock."""

    monotonic_ns: int
    raw: str
    parsed: ParsedTegrastats
    mem_available_mb: float | None
    collection_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validated_monotonic_ns(self.monotonic_ns)
        if self.mem_available_mb is not None:
            _validated_mem_available_mb(self.mem_available_mb)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "sample",
            "monotonic_ns": self.monotonic_ns,
            "raw": self.raw,
            "parsed": self.parsed.to_dict(),
            "mem_available_mb": self.mem_available_mb,
            "collection_errors": list(self.collection_errors),
        }


@dataclass(frozen=True)
class TelemetryMarker:
    """A named boundary in the same monotonic time domain as samples."""

    monotonic_ns: int
    name: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validated_monotonic_ns(self.monotonic_ns)
        if not self.name or self.name.isspace():
            raise ValueError("marker name must be non-empty")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "marker",
            "monotonic_ns": self.monotonic_ns,
            "name": self.name,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TelemetrySampleWindow:
    """A bounded sample query and the retention provenance for its interval."""

    samples: tuple[TelemetrySample, ...]
    start_ns: int
    end_ns: int
    end_inclusive: bool
    reverse: bool
    limit: int | None
    limit_reached: bool
    bounded_retention: bool
    max_samples: int | None
    dropped_samples: int
    last_dropped_sample_ns: int | None
    earliest_retained_sample_ns: int | None
    interval_complete: bool

    def retention_record(self) -> dict[str, Any]:
        """Return the stable JSON representation used by aggregate evidence."""

        return {
            "bounded": self.bounded_retention,
            "max_samples": self.max_samples,
            "dropped_samples": self.dropped_samples,
            "last_dropped_sample_ns": self.last_dropped_sample_ns,
            "earliest_retained_sample_ns": self.earliest_retained_sample_ns,
            "interval_complete": self.interval_complete,
        }


def _parse_cpu(match: re.Match[str] | None) -> tuple[CpuCore, ...]:
    if match is None:
        return ()
    common_frequency = _number(match.group("common_freq"), minimum=0.0)
    cores: list[CpuCore] = []
    for raw_core in match.group("cores").split(","):
        core = raw_core.strip()
        if not core:
            continue
        if core.casefold() == "off":
            cores.append(CpuCore(None, None, False))
            continue
        core_match = _CPU_CORE_RE.fullmatch(core)
        if core_match is None:
            cores.append(CpuCore(None, None, None))
            continue
        utilization = _utilization(core_match.group("util"))
        frequency = _number(core_match.group("freq"), minimum=0.0)
        if frequency is None:
            frequency = common_frequency
        online = True if utilization is not None else None
        cores.append(CpuCore(utilization, frequency, online))
    return tuple(cores)


def _parse_clock(
    match: re.Match[str] | None, *, frequency_group: str = "freq"
) -> ClockActivity | None:
    if match is None:
        return None
    utilization = _utilization(match.group("util"))
    frequency = _number(match.group(frequency_group), minimum=0.0)
    frequencies = () if frequency is None else (frequency,)
    if utilization is None and not frequencies:
        return None
    return ClockActivity(utilization, frequencies)


def _parse_gr3d(match: re.Match[str] | None) -> ClockActivity | None:
    if match is None:
        return None
    frequencies: list[float] = []
    raw_frequencies = match.group("freqs")
    if raw_frequencies is not None:
        for raw_frequency in raw_frequencies.split(","):
            frequency = _number(raw_frequency.strip(), minimum=0.0)
            if frequency is not None:
                frequencies.append(frequency)
    else:
        frequency = _number(match.group("single_freq"), minimum=0.0)
        if frequency is not None:
            frequencies.append(frequency)
    utilization = _utilization(match.group("util"))
    if utilization is None and not frequencies:
        return None
    return ClockActivity(utilization, tuple(frequencies))


def parse_tegrastats_line(line: str) -> ParsedTegrastats:
    """Parse known R39.2 fields without rejecting unknown or missing tokens."""

    ram_match = _RAM_RE.search(line)
    ram: RamSample | None = None
    if ram_match is not None:
        used = _number(ram_match.group("used"), minimum=0.0)
        total = _number(ram_match.group("total"), minimum=0.0)
        block_size = _number(ram_match.group("size"), minimum=0.0)
        if used is not None and total is not None and used <= total:
            block_count_text = ram_match.group("count")
            ram = RamSample(
                used,
                total,
                int(block_count_text) if block_count_text is not None else None,
                block_size,
            )

    swap_match = _SWAP_RE.search(line)
    swap: SwapSample | None = None
    if swap_match is not None:
        used = _number(swap_match.group("used"), minimum=0.0)
        total = _number(swap_match.group("total"), minimum=0.0)
        cached = _number(swap_match.group("cached"), minimum=0.0)
        if used is not None and total is not None and used <= total:
            swap = SwapSample(used, total, cached)

    temperatures = {
        match.group("name").casefold(): float(match.group("current"))
        for match in _TEMPERATURE_RE.finditer(line)
        if _number(match.group("current")) is not None
    }
    power: dict[str, PowerSample] = {}
    for match in _POWER_RE.finditer(line):
        current = _number(match.group("current"), minimum=0.0)
        if current is None:
            continue
        extras = tuple(
            value
            for value in (
                _number(match.group("extra1"), minimum=0.0),
                _number(match.group("extra2"), minimum=0.0),
            )
            if value is not None
        )
        power[match.group("name")] = PowerSample(current, extras)

    return ParsedTegrastats(
        ram=ram,
        swap=swap,
        cpu=_parse_cpu(_CPU_RE.search(line)),
        temperatures_c=temperatures,
        power=power,
        emc=_parse_clock(_EMC_RE.search(line)),
        gr3d=_parse_gr3d(_GR3D_RE.search(line)),
    )


def read_mem_available_mb(
    path: pathlib.Path | str = "/proc/meminfo",
) -> float:
    """Read Linux ``MemAvailable`` and convert its documented kB to MiB."""

    text = pathlib.Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "MemAvailable:":
            if len(fields) < 3 or fields[2] != "kB":
                break
            value_kb = _number(fields[1], minimum=0.0)
            if value_kb is None:
                break
            return value_kb / 1024.0
    raise ValueError(f"MemAvailable is missing or invalid in {path}")


class JsonlTelemetryWriter:
    """Thread-safe JSONL writer for samples and markers."""

    def __init__(
        self,
        destination: pathlib.Path | str | TextIO,
        *,
        mode: str = "w",
        flush: bool = True,
        flush_every: int | None = None,
    ) -> None:
        if mode not in {"w", "x"}:
            raise ValueError("mode must be 'w' or 'x'; JSONL append is unsafe")
        if flush_every is not None and (
            isinstance(flush_every, bool)
            or not isinstance(flush_every, int)
            or flush_every <= 0
        ):
            raise ValueError("flush_every must be a positive integer")
        self._owns_stream = not hasattr(destination, "write")
        if self._owns_stream:
            path = pathlib.Path(destination)  # type: ignore[arg-type]
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream: TextIO = path.open(mode, encoding="utf-8")
        else:
            self._stream = destination  # type: ignore[assignment]
        self._flush = flush
        self._flush_every = flush_every
        self._records_since_flush = 0
        self._closed = False
        self._last_monotonic_ns: int | None = None
        self._lock = threading.Lock()

    def write(self, record: TelemetrySample | TelemetryMarker) -> None:
        payload = record.to_record()
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("telemetry writer is closed")
            timestamp = _validated_monotonic_ns(record.monotonic_ns)
            if (
                self._last_monotonic_ns is not None
                and timestamp < self._last_monotonic_ns
            ):
                raise ValueError("telemetry timestamps must not regress")
            self._stream.write(encoded + "\n")
            self._records_since_flush += 1
            marker_boundary = isinstance(record, TelemetryMarker)
            periodic_flush = (
                self._flush_every is not None
                and self._records_since_flush >= self._flush_every
            )
            if self._flush or marker_boundary or periodic_flush:
                self._stream.flush()
                self._records_since_flush = 0
            self._last_monotonic_ns = timestamp

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._stream.flush()
            if self._owns_stream:
                self._stream.close()
            self._closed = True

    def __enter__(self) -> JsonlTelemetryWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _field_present(sample: TelemetrySample, field: str) -> bool:
    parsed = sample.parsed
    if field == "ram":
        return parsed.ram is not None
    if field == "mem_available":
        return sample.mem_available_mb is not None
    if field == "cpu":
        return any(core.utilization_pct is not None for core in parsed.cpu)
    if field == "temperature":
        return bool(parsed.temperatures_c)
    if field.startswith("temperature:"):
        return field.partition(":")[2].casefold() in parsed.temperatures_c
    if field == "power":
        return bool(parsed.power)
    if field.startswith("power:"):
        return field.partition(":")[2].upper() in parsed.power
    if field == "emc":
        return parsed.emc is not None
    if field == "emc_utilization":
        return parsed.emc is not None and parsed.emc.utilization_pct is not None
    if field == "emc_frequency":
        return parsed.emc is not None and bool(parsed.emc.frequencies_mhz)
    if field == "gr3d":
        return parsed.gr3d is not None
    if field == "gr3d_utilization":
        return parsed.gr3d is not None and parsed.gr3d.utilization_pct is not None
    if field == "gr3d_frequency":
        return parsed.gr3d is not None and bool(parsed.gr3d.frequencies_mhz)
    raise ValueError(f"unknown required telemetry field: {field}")


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int] | None:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _per_cpu_aggregate(samples: Sequence[TelemetrySample]) -> dict[str, Any]:
    core_count = max((len(sample.parsed.cpu) for sample in samples), default=0)
    cores: list[dict[str, Any]] = []
    for index in range(core_count):
        entries = [
            sample.parsed.cpu[index]
            for sample in samples
            if index < len(sample.parsed.cpu)
        ]
        cores.append(
            {
                "core": index,
                "utilization_pct": _numeric_summary(
                    entry.utilization_pct
                    for entry in entries
                    if entry.utilization_pct is not None
                ),
                "frequency_mhz": _numeric_summary(
                    entry.frequency_mhz
                    for entry in entries
                    if entry.frequency_mhz is not None
                ),
                "online_samples": sum(entry.online is True for entry in entries),
                "off_samples": sum(entry.online is False for entry in entries),
                "unknown_samples": sum(entry.online is None for entry in entries),
            }
        )
    sample_means = []
    for sample in samples:
        values = [
            core.utilization_pct
            for core in sample.parsed.cpu
            if core.utilization_pct is not None
        ]
        if values:
            sample_means.append(statistics.fmean(values))
    return {
        "mean_across_online_cores_pct": _numeric_summary(sample_means),
        "cores": cores,
    }


def _mapping_aggregate(
    samples: Sequence[TelemetrySample],
    getter: Callable[[TelemetrySample], Mapping[str, float]],
) -> dict[str, Any]:
    names = sorted({name for sample in samples for name in getter(sample)})
    return {
        name: _numeric_summary(
            getter(sample)[name] for sample in samples if name in getter(sample)
        )
        for name in names
    }


def _clock_aggregate(
    samples: Sequence[TelemetrySample],
    getter: Callable[[TelemetrySample], ClockActivity | None],
) -> dict[str, Any] | None:
    clocks = [clock for sample in samples if (clock := getter(sample)) is not None]
    if not clocks:
        return None
    frequency_count = max((len(clock.frequencies_mhz) for clock in clocks), default=0)
    return {
        "utilization_pct": _numeric_summary(
            clock.utilization_pct
            for clock in clocks
            if clock.utilization_pct is not None
        ),
        "frequencies_mhz": [
            _numeric_summary(
                clock.frequencies_mhz[index]
                for clock in clocks
                if index < len(clock.frequencies_mhz)
            )
            for index in range(frequency_count)
        ],
    }


def aggregate_samples(
    samples: Sequence[TelemetrySample],
    start_ns: int,
    end_ns: int,
    *,
    required_fields: Sequence[str] = DEFAULT_REQUIRED_FIELDS,
    minimum_valid_samples: int = 1,
    require_all_samples_valid: bool = False,
    reference_ns: int | None = None,
    stale_after_ns: int | None = None,
    maximum_valid_gap_ns: int | None = None,
    end_inclusive: bool = False,
) -> dict[str, Any]:
    """Aggregate a timestamp interval and assess stream health.

    ``required_fields`` controls structural completeness.  By default,
    ``minimum_valid_samples`` tolerates incomplete rows while reporting their
    field counts; set ``require_all_samples_valid`` for strict completeness.
    ``stale_after_ns`` and ``maximum_valid_gap_ns`` are caller-supplied because
    cadence is a deployment decision, not a parser property.  Freshness uses
    all samples at or before ``reference_ns`` while numeric metrics use only
    the requested interval.  The default interval is ``[start_ns, end_ns)``;
    set ``end_inclusive`` for exact marker replay.  The gap bound includes both
    interval boundaries.
    """

    start_ns = _validated_monotonic_ns(start_ns)
    end_ns = _validated_monotonic_ns(end_ns)
    if end_ns <= start_ns:
        raise ValueError("the telemetry interval must be positive")
    if not isinstance(end_inclusive, bool):
        raise ValueError("end_inclusive must be boolean")
    if (
        isinstance(minimum_valid_samples, bool)
        or not isinstance(minimum_valid_samples, int)
        or minimum_valid_samples <= 0
    ):
        raise ValueError("minimum_valid_samples must be positive")
    if stale_after_ns is not None:
        stale_after_ns = _validated_monotonic_ns(stale_after_ns)
    if maximum_valid_gap_ns is not None:
        maximum_valid_gap_ns = _validated_monotonic_ns(maximum_valid_gap_ns)
        if maximum_valid_gap_ns == 0:
            raise ValueError("maximum_valid_gap_ns must be positive")
    reference = (
        end_ns
        if reference_ns is None
        else _validated_monotonic_ns(reference_ns)
    )
    if reference < end_ns:
        raise ValueError("reference_ns must be at or after the interval end")
    required = tuple(dict.fromkeys(required_fields))
    for field in required:
        _field_present(
            TelemetrySample(0, "", ParsedTegrastats(None, None, (), {}, {}, None, None), None),
            field,
        )

    selected = sorted(
        (
            sample
            for sample in samples
            if start_ns <= sample.monotonic_ns
            and (
                sample.monotonic_ns < end_ns
                or (end_inclusive and sample.monotonic_ns == end_ns)
            )
        ),
        key=lambda sample: sample.monotonic_ns,
    )
    missing_counts = {
        field: sum(not _field_present(sample, field) for sample in selected)
        for field in required
    }
    valid = [
        sample
        for sample in selected
        if all(_field_present(sample, field) for field in required)
    ]
    observed = sorted(
        (sample for sample in samples if sample.monotonic_ns <= reference),
        key=lambda sample: sample.monotonic_ns,
    )
    observed_valid = [
        sample
        for sample in observed
        if all(_field_present(sample, field) for field in required)
    ]
    latest_interval_ns = selected[-1].monotonic_ns if selected else None
    latest_ns = observed[-1].monotonic_ns if observed else None
    latest_valid_ns = observed_valid[-1].monotonic_ns if observed_valid else None
    stale = False
    valid_stale = False
    if stale_after_ns is not None:
        stale = latest_ns is None or reference - latest_ns > stale_after_ns
        valid_stale = (
            latest_valid_ns is None or reference - latest_valid_ns > stale_after_ns
        )
    observed_maximum_valid_gap_ns: int | None = None
    valid_gap_exceeded = False
    if maximum_valid_gap_ns is not None:
        if valid:
            valid_timestamps = [sample.monotonic_ns for sample in valid]
            valid_gaps = [
                valid_timestamps[0] - start_ns,
                end_ns - valid_timestamps[-1],
            ]
            valid_gaps.extend(
                current - previous
                for previous, current in zip(
                    valid_timestamps, valid_timestamps[1:], strict=False
                )
            )
            observed_maximum_valid_gap_ns = max(valid_gaps)
        else:
            observed_maximum_valid_gap_ns = end_ns - start_ns
        valid_gap_exceeded = (
            observed_maximum_valid_gap_ns > maximum_valid_gap_ns
        )

    reasons: list[str] = []
    if not selected:
        reasons.append("no_samples")
    if len(valid) < minimum_valid_samples:
        reasons.append("insufficient_valid_samples")
    if require_all_samples_valid and len(valid) != len(selected):
        reasons.append("incomplete_samples")
    if stale:
        reasons.append("stale_stream")
    if valid_stale and not stale:
        reasons.append("stale_valid_sample")
    if valid_gap_exceeded:
        reasons.append("valid_sample_gap_exceeded")

    ram_samples = [sample.parsed.ram for sample in valid if sample.parsed.ram]
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "interval": {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "end_inclusive": end_inclusive,
        },
        "total_samples": len(selected),
        "valid_samples": len(valid),
        "invalid_samples": len(selected) - len(valid),
        "health": {
            "healthy": not reasons,
            "reasons": reasons,
            "required_fields": list(required),
            "require_all_samples_valid": require_all_samples_valid,
            "missing_counts": missing_counts,
            "incomplete_samples": len(selected) - len(valid),
            "last_interval_sample_ns": latest_interval_ns,
            "last_sample_ns": latest_ns,
            "last_valid_sample_ns": latest_valid_ns,
            "stale": stale,
            "valid_sample_stale": valid_stale,
            "maximum_valid_gap_ns": maximum_valid_gap_ns,
            "observed_maximum_valid_gap_ns": observed_maximum_valid_gap_ns,
            "valid_gap_exceeded": valid_gap_exceeded,
            "collection_error_samples": sum(
                bool(sample.collection_errors) for sample in selected
            ),
        },
        "ram": {
            "used_mb": _numeric_summary(ram.used_mb for ram in ram_samples),
            "total_mb": _numeric_summary(ram.total_mb for ram in ram_samples),
            "headroom_mb": _numeric_summary(
                ram.total_mb - ram.used_mb for ram in ram_samples
            ),
            "largest_free_block_count": _numeric_summary(
                float(ram.largest_free_block_count)
                for ram in ram_samples
                if ram.largest_free_block_count is not None
            ),
            "largest_free_block_mb": _numeric_summary(
                ram.largest_free_block_mb
                for ram in ram_samples
                if ram.largest_free_block_mb is not None
            ),
        },
        "mem_available_mb": _numeric_summary(
            sample.mem_available_mb
            for sample in valid
            if sample.mem_available_mb is not None
        ),
        "cpu": _per_cpu_aggregate(valid),
        "temperatures_c": _mapping_aggregate(
            valid, lambda sample: sample.parsed.temperatures_c
        ),
        "power_mw": _mapping_aggregate(
            valid,
            lambda sample: {
                name: power.current_mw for name, power in sample.parsed.power.items()
            },
        ),
        "emc": _clock_aggregate(valid, lambda sample: sample.parsed.emc),
        "gr3d": _clock_aggregate(valid, lambda sample: sample.parsed.gr3d),
    }
    return aggregate


_MEM_AVAILABLE_UNSET = object()


class TegrastatsMonitor:
    """Timestamp, persist, retain, and aggregate lines supplied by a collector."""

    def __init__(
        self,
        writer: JsonlTelemetryWriter,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        mem_available_reader: Callable[[], float] = read_mem_available_mb,
        max_samples: int | None = None,
    ) -> None:
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._writer = writer
        self._clock = clock
        self._mem_available_reader = mem_available_reader
        self._samples: collections.deque[TelemetrySample] = collections.deque(
            maxlen=max_samples
        )
        self._markers: list[TelemetryMarker] = []
        self._entry_lock = threading.Lock()
        self._lock = threading.Lock()
        self._closed = False
        self._last_monotonic_ns: int | None = None
        self._dropped_samples = 0
        self._last_dropped_sample_ns: int | None = None

    def _timestamp(self, explicit: int | None) -> int:
        value = self._clock() if explicit is None else explicit
        return _validated_monotonic_ns(value)

    def _check_timestamp_locked(self, timestamp: int) -> None:
        if (
            self._last_monotonic_ns is not None
            and timestamp < self._last_monotonic_ns
        ):
            raise ValueError("telemetry timestamps must not regress")

    def _acquire_timestamped_lock(self, explicit: int | None) -> int:
        # Serialize entry so a later caller cannot write before an earlier
        # timestamp.  The timestamp is still captured before waiting for any
        # in-progress meminfo read, parse, or JSONL write under ``_lock``.
        with self._entry_lock:
            timestamp = self._timestamp(explicit)
            self._lock.acquire()
        return timestamp

    def record_line(
        self,
        line: str,
        *,
        monotonic_ns: int | None = None,
        mem_available_mb: float | None | object = _MEM_AVAILABLE_UNSET,
    ) -> TelemetrySample:
        """Record one line; by default sample ``/proc/meminfo`` alongside it."""

        timestamp = self._acquire_timestamped_lock(monotonic_ns)
        try:
            if self._closed:
                raise RuntimeError("telemetry monitor is closed")
            self._check_timestamp_locked(timestamp)
            errors: list[str] = []
            available: float | None
            if mem_available_mb is _MEM_AVAILABLE_UNSET:
                try:
                    available = _validated_mem_available_mb(
                        self._mem_available_reader()
                    )
                except (OSError, TypeError, ValueError) as error:
                    available = None
                    errors.append(f"mem_available:{type(error).__name__}")
            elif mem_available_mb is None:
                available = None
            else:
                available = _validated_mem_available_mb(mem_available_mb)
            sample = TelemetrySample(
                monotonic_ns=timestamp,
                raw=line.rstrip("\r\n"),
                parsed=parse_tegrastats_line(line),
                mem_available_mb=available,
                collection_errors=tuple(errors),
            )
            self._writer.write(sample)
            if (
                self._samples.maxlen is not None
                and len(self._samples) == self._samples.maxlen
            ):
                dropped = self._samples[0]
                self._dropped_samples += 1
                self._last_dropped_sample_ns = dropped.monotonic_ns
            self._samples.append(sample)
            self._last_monotonic_ns = timestamp
            return sample
        finally:
            self._lock.release()

    def record_stream(self, lines: Iterable[str]) -> int:
        """Consume a foreground stream until EOF and return its line count."""

        count = 0
        for line in lines:
            self.record_line(line)
            count += 1
        return count

    def mark(
        self,
        name: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        monotonic_ns: int | None = None,
    ) -> TelemetryMarker:
        """Persist a named phase boundary in the sample clock domain."""

        if not name or name.isspace():
            raise ValueError("marker name must be non-empty")
        timestamp = self._acquire_timestamped_lock(monotonic_ns)
        try:
            if self._closed:
                raise RuntimeError("telemetry monitor is closed")
            self._check_timestamp_locked(timestamp)
            marker = TelemetryMarker(timestamp, name, dict(metadata or {}))
            self._writer.write(marker)
            self._markers.append(marker)
            self._last_monotonic_ns = timestamp
            return marker
        finally:
            self._lock.release()

    def samples(self) -> tuple[TelemetrySample, ...]:
        with self._lock:
            return tuple(self._samples)

    def _sample_window_locked(
        self,
        start_ns: int,
        end_ns: int,
        *,
        end_inclusive: bool,
        reverse: bool,
        limit: int | None,
    ) -> TelemetrySampleWindow:
        selected_reverse: list[TelemetrySample] = []
        limit_reached = False
        for sample in reversed(self._samples):
            timestamp = sample.monotonic_ns
            if timestamp > end_ns or (timestamp == end_ns and not end_inclusive):
                continue
            if timestamp < start_ns:
                break
            if limit is not None and len(selected_reverse) == limit:
                limit_reached = True
                break
            selected_reverse.append(sample)
        selected = (
            tuple(selected_reverse)
            if reverse
            else tuple(reversed(selected_reverse))
        )
        last_dropped_ns = self._last_dropped_sample_ns
        truncated = last_dropped_ns is not None and start_ns <= last_dropped_ns
        return TelemetrySampleWindow(
            samples=selected,
            start_ns=start_ns,
            end_ns=end_ns,
            end_inclusive=end_inclusive,
            reverse=reverse,
            limit=limit,
            limit_reached=limit_reached,
            bounded_retention=self._samples.maxlen is not None,
            max_samples=self._samples.maxlen,
            dropped_samples=self._dropped_samples,
            last_dropped_sample_ns=last_dropped_ns,
            earliest_retained_sample_ns=(
                self._samples[0].monotonic_ns if self._samples else None
            ),
            interval_complete=not truncated,
        )

    def sample_window(
        self,
        start_ns: int,
        end_ns: int,
        *,
        end_inclusive: bool = False,
        reverse: bool = False,
        limit: int | None = None,
    ) -> TelemetrySampleWindow:
        """Query ``[start_ns, end_ns)`` without copying older campaign samples.

        Set ``end_inclusive`` for exact marker replay and ``reverse`` for newest
        first traversal.  ``limit_reached`` distinguishes a complete short
        result from a caller-imposed truncation; retention truncation is
        reported independently by ``interval_complete``.
        """

        start = _validated_monotonic_ns(start_ns)
        end = _validated_monotonic_ns(end_ns)
        if end < start:
            raise ValueError("the telemetry query interval must not regress")
        if (
            limit is not None
            and (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or limit <= 0
            )
        ):
            raise ValueError("sample query limit must be positive")
        with self._lock:
            return self._sample_window_locked(
                start,
                end,
                end_inclusive=end_inclusive,
                reverse=reverse,
                limit=limit,
            )

    def markers(self) -> tuple[TelemetryMarker, ...]:
        with self._lock:
            return tuple(self._markers)

    def aggregate(self, start_ns: int, end_ns: int, **kwargs: Any) -> dict[str, Any]:
        start = _validated_monotonic_ns(start_ns)
        end = _validated_monotonic_ns(end_ns)
        aggregate_kwargs = dict(kwargs)
        end_inclusive = aggregate_kwargs.pop("end_inclusive", False)
        if not isinstance(end_inclusive, bool):
            raise ValueError("end_inclusive must be boolean")
        reference = _validated_monotonic_ns(
            aggregate_kwargs.get("reference_ns", end)
        )
        required = tuple(
            dict.fromkeys(
                aggregate_kwargs.get("required_fields", DEFAULT_REQUIRED_FIELDS)
            )
        )
        with self._lock:
            window = self._sample_window_locked(
                start,
                end,
                end_inclusive=end_inclusive,
                reverse=False,
                limit=None,
            )
            freshness: list[TelemetrySample] = []
            latest_observed: TelemetrySample | None = None
            latest_valid: TelemetrySample | None = None
            for sample in reversed(self._samples):
                if sample.monotonic_ns > reference:
                    continue
                if latest_observed is None:
                    latest_observed = sample
                if latest_valid is None and all(
                    _field_present(sample, field) for field in required
                ):
                    latest_valid = sample
                if latest_observed is not None and latest_valid is not None:
                    break
            retained_ids = {id(sample) for sample in window.samples}
            for sample in (latest_observed, latest_valid):
                if sample is not None and id(sample) not in retained_ids:
                    freshness.append(sample)
                    retained_ids.add(id(sample))
        aggregate = aggregate_samples(
            (*window.samples, *freshness),
            start,
            end,
            end_inclusive=end_inclusive,
            **aggregate_kwargs,
        )
        aggregate["retention"] = window.retention_record()
        if not window.interval_complete:
            aggregate["health"]["healthy"] = False
            aggregate["health"]["reasons"].append("retention_truncated")
        return aggregate

    def close(self) -> None:
        with self._entry_lock:
            with self._lock:
                if self._closed:
                    return
                self._writer.close()
                self._closed = True

    def __enter__(self) -> TegrastatsMonitor:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
