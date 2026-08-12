"""Pure request-level latency calculations."""

from __future__ import annotations

from dataclasses import dataclass

from inferscope.metrics.definitions import NANOSECONDS_PER_MILLISECOND
from inferscope.models import RequestSample


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    """Derived request durations, all expressed in milliseconds."""

    request_id: str
    ttft_ms: float | None
    e2e_ms: float | None
    tpot_ms: float | None
    chunk_interarrival_ms: tuple[float, ...]


def calculate_ttft_ms(sample: RequestSample) -> float | None:
    """Return start-to-first-content latency, or ``None`` when no content arrived."""
    if sample.first_content_at_ns is None:
        return None
    return (sample.first_content_at_ns - sample.started_at_ns) / NANOSECONDS_PER_MILLISECOND


def calculate_e2e_ms(sample: RequestSample) -> float | None:
    """Return start-to-finish latency, or ``None`` for an unfinished request."""
    if sample.finished_at_ns is None:
        return None
    return (sample.finished_at_ns - sample.started_at_ns) / NANOSECONDS_PER_MILLISECOND


def calculate_tpot_ms(sample: RequestSample) -> float | None:
    """Return TPOT from final token count; one or fewer tokens yields ``None``."""
    ttft_ms = calculate_ttft_ms(sample)
    e2e_ms = calculate_e2e_ms(sample)
    if (
        ttft_ms is None
        or e2e_ms is None
        or sample.output_tokens is None
        or sample.output_tokens <= 1
    ):
        return None
    return (e2e_ms - ttft_ms) / (sample.output_tokens - 1)


def calculate_chunk_interarrival_ms(sample: RequestSample) -> tuple[float, ...]:
    """Return content-chunk intervals as a proxy, never as token-level ITL."""
    return tuple(
        (current - previous) / NANOSECONDS_PER_MILLISECOND
        for previous, current in zip(sample.chunk_times_ns, sample.chunk_times_ns[1:], strict=False)
    )


def calculate_request_metrics(sample: RequestSample) -> RequestMetrics:
    """Derive all supported client-side timings for one raw sample."""
    return RequestMetrics(
        request_id=sample.request_id,
        ttft_ms=calculate_ttft_ms(sample),
        e2e_ms=calculate_e2e_ms(sample),
        tpot_ms=calculate_tpot_ms(sample),
        chunk_interarrival_ms=calculate_chunk_interarrival_ms(sample),
    )
