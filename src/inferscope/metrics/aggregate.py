"""Aggregate metrics with an explicit measured wall-clock denominator."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from inferscope.metrics.definitions import P99_SMALL_SAMPLE_THRESHOLD, PERCENTILE_METHOD
from inferscope.metrics.request import calculate_e2e_ms, calculate_tpot_ms, calculate_ttft_ms
from inferscope.models import (
    AggregateReport,
    LatencySummary,
    MetricDistribution,
    RequestCounts,
    RequestSample,
    RequestStatus,
    ThroughputSummary,
    ValidationStatus,
)


def summarize_distribution(values: Iterable[float | None]) -> MetricDistribution:
    """Summarize finite values with NumPy's linear percentile convention."""
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return MetricDistribution(
            count=0,
            mean=None,
            p50=None,
            p95=None,
            p99=None,
            minimum=None,
            maximum=None,
            p99_small_sample_warning=False,
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("metric distributions cannot contain NaN or infinity")
    percentiles = np.quantile(array, [0.50, 0.95, 0.99], method=PERCENTILE_METHOD)
    return MetricDistribution(
        count=int(array.size),
        mean=float(np.mean(array)),
        p50=float(percentiles[0]),
        p95=float(percentiles[1]),
        p99=float(percentiles[2]),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        p99_small_sample_warning=array.size < P99_SMALL_SAMPLE_THRESHOLD,
    )


def _sum_known_tokens(
    samples: Sequence[RequestSample], extractor: Callable[[RequestSample], int | None]
) -> int | None:
    values = [extractor(sample) for sample in samples]
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def aggregate_request_metrics(
    samples: Sequence[RequestSample],
    *,
    run_id: str,
    measured_seconds: float,
    validation_status: ValidationStatus,
    scheduled_count: int | None = None,
    valid_request_ids: set[str] | None = None,
    goodput_requests_per_second: float | None = None,
) -> AggregateReport:
    """Aggregate samples using measured wall time, never summed request latency.

    ``valid_request_ids`` represents samples accepted by validation gates. When omitted,
    every successful sample is considered valid. Token throughput requires a known token
    count for every valid sample; otherwise it remains unavailable instead of pretending
    missing counts are zero.
    """
    if measured_seconds <= 0:
        raise ValueError("measured_seconds must be greater than zero")
    successful = [sample for sample in samples if sample.status is RequestStatus.SUCCESS]
    valid = [
        sample
        for sample in successful
        if valid_request_ids is None or sample.request_id in valid_request_ids
    ]
    scheduled = len(samples) if scheduled_count is None else scheduled_count
    if scheduled < len(samples):
        raise ValueError("scheduled_count cannot be smaller than the number of samples")

    input_tokens = _sum_known_tokens(valid, lambda sample: sample.input_tokens)
    output_tokens = _sum_known_tokens(valid, lambda sample: sample.output_tokens)
    return AggregateReport(
        run_id=run_id,
        validation_status=validation_status,
        measured_seconds=measured_seconds,
        requests=RequestCounts(
            scheduled=scheduled,
            successful=len(successful),
            valid=len(valid),
        ),
        throughput=ThroughputSummary(
            requests_per_second=len(successful) / measured_seconds,
            input_tokens_per_second=(
                None if input_tokens is None else input_tokens / measured_seconds
            ),
            output_tokens_per_second=(
                None if output_tokens is None else output_tokens / measured_seconds
            ),
            goodput_requests_per_second=goodput_requests_per_second,
        ),
        latency_ms=LatencySummary(
            ttft=summarize_distribution(calculate_ttft_ms(sample) for sample in valid),
            tpot=summarize_distribution(calculate_tpot_ms(sample) for sample in valid),
            e2e=summarize_distribution(calculate_e2e_ms(sample) for sample in valid),
        ),
    )
