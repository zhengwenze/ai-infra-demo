"""Shared telemetry samples and client-capacity measurements."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import psutil


def utc_now_iso() -> str:
    """Return an RFC 3339-compatible UTC wall-clock timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """One timestamped scalar measurement.

    Missing facts are represented by an absent sample, never by a synthetic
    value of zero. ``monotonic_ns`` supports local duration alignment while
    ``wall_time_utc`` supports cross-process correlation.
    """

    wall_time_utc: str
    monotonic_ns: int
    source: str
    name: str
    value: float
    unit: str | None = None
    labels: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.monotonic_ns < 0:
            raise ValueError("monotonic_ns must be non-negative")
        if not self.source or not self.name:
            raise ValueError("source and name must not be empty")
        if not math.isfinite(self.value):
            raise ValueError("telemetry value must be finite")


def _client_timestamp() -> tuple[str, int]:
    return utc_now_iso(), time.perf_counter_ns()


def sample_client_process(process: psutil.Process | None = None) -> tuple[TelemetrySample, ...]:
    """Sample client CPU and resident memory without blocking the event loop.

    ``cpu_percent(interval=None)`` reports utilization since the previous call.
    Callers should therefore retain the same ``psutil.Process`` instance across
    samples. A first-call value of zero is an observed psutil baseline, not a
    stand-in for unavailable data.
    """

    client_process = process if process is not None else psutil.Process()
    wall_time_utc, monotonic_ns = _client_timestamp()
    cpu_percent = float(client_process.cpu_percent(interval=None))
    resident_memory_bytes = float(client_process.memory_info().rss)
    return (
        TelemetrySample(
            wall_time_utc=wall_time_utc,
            monotonic_ns=monotonic_ns,
            source="client",
            name="process_cpu_percent",
            value=cpu_percent,
            unit="percent",
        ),
        TelemetrySample(
            wall_time_utc=wall_time_utc,
            monotonic_ns=monotonic_ns,
            source="client",
            name="resident_memory_bytes",
            value=resident_memory_bytes,
            unit="bytes",
        ),
    )


async def measure_event_loop_lag(
    interval_seconds: float,
    *,
    clock: Callable[[], float] | None = None,
) -> TelemetrySample:
    """Measure scheduler delay beyond one requested async sleep interval."""

    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be finite and greater than zero")
    loop = asyncio.get_running_loop()
    clock_source = clock if clock is not None else loop.time
    expected = clock_source() + interval_seconds
    await asyncio.sleep(interval_seconds)
    lag_seconds = max(0.0, clock_source() - expected)
    wall_time_utc, monotonic_ns = _client_timestamp()
    return TelemetrySample(
        wall_time_utc=wall_time_utc,
        monotonic_ns=monotonic_ns,
        source="client",
        name="event_loop_lag_seconds",
        value=lag_seconds,
        unit="seconds",
    )


class ClientTelemetrySource:
    """Collect process capacity and event-loop lag using a stable process handle."""

    def __init__(self, process: psutil.Process | None = None) -> None:
        self._process = process if process is not None else psutil.Process()
        self._process.cpu_percent(interval=None)

    async def collect(self, *, lag_interval_seconds: float) -> tuple[TelemetrySample, ...]:
        """Collect one client snapshot after measuring event-loop lag."""

        lag = await measure_event_loop_lag(lag_interval_seconds)
        return (*sample_client_process(self._process), lag)
