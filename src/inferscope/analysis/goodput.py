"""SLO-aware goodput calculations."""

from __future__ import annotations

from collections.abc import Sequence

from inferscope.metrics.request import calculate_tpot_ms, calculate_ttft_ms
from inferscope.models import GoodputResult, RequestSample, RequestStatus, SloThresholds


def request_meets_latency_slo(sample: RequestSample, slo: SloThresholds) -> bool:
    """Return whether one successful, measurable request meets both latency limits."""
    if sample.status is not RequestStatus.SUCCESS:
        return False
    ttft_ms = calculate_ttft_ms(sample)
    tpot_ms = calculate_tpot_ms(sample)
    if ttft_ms is None or tpot_ms is None:
        return False
    return ttft_ms <= slo.ttft_p95_ms and tpot_ms <= slo.tpot_p95_ms


def calculate_goodput(
    samples: Sequence[RequestSample],
    *,
    measured_seconds: float,
    slo: SloThresholds,
    scheduled_count: int | None = None,
) -> GoodputResult:
    """Calculate rate of requests meeting latency SLOs and the run success SLO.

    A run that misses its whole-run success-rate SLO has zero goodput. The qualifying
    count is still returned so reports can explain the failure without deleting data.
    """
    if measured_seconds <= 0:
        raise ValueError("measured_seconds must be greater than zero")
    scheduled = len(samples) if scheduled_count is None else scheduled_count
    if scheduled <= 0:
        raise ValueError("scheduled_count must be greater than zero")
    if scheduled < len(samples):
        raise ValueError("scheduled_count cannot be smaller than the number of samples")

    successful = [sample for sample in samples if sample.status is RequestStatus.SUCCESS]
    eligible = [
        sample
        for sample in successful
        if calculate_ttft_ms(sample) is not None and calculate_tpot_ms(sample) is not None
    ]
    qualifying = sum(request_meets_latency_slo(sample, slo) for sample in eligible)
    success_rate = len(successful) / scheduled
    success_rate_slo_met = success_rate >= slo.success_rate_min
    goodput = qualifying / measured_seconds if success_rate_slo_met else 0.0
    return GoodputResult(
        scheduled_requests=scheduled,
        successful_requests=len(successful),
        eligible_requests=len(eligible),
        qualifying_requests=qualifying,
        measured_seconds=measured_seconds,
        success_rate=success_rate,
        success_rate_slo_met=success_rate_slo_met,
        requests_per_second=goodput,
    )
