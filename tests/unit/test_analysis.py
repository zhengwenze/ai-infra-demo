"""Tests for SLO goodput, Pareto, and repeat stability analysis."""

from __future__ import annotations

import pytest

from inferscope.analysis import analyze_stability, calculate_goodput, pareto_frontier
from inferscope.models import (
    ParetoPoint,
    RequestSample,
    RequestStatus,
    SloThresholds,
    TokenCountSource,
    ValidationStatus,
)


def make_sample(
    request_id: str,
    *,
    ttft_ms: int = 100,
    tpot_ms: int = 20,
    status: RequestStatus = RequestStatus.SUCCESS,
) -> RequestSample:
    """Build a three-token sample with a controlled TPOT."""
    finish_ms = ttft_ms + 2 * tpot_ms
    return RequestSample(
        run_id="run-1",
        request_id=request_id,
        sequence=0,
        scheduled_at_ns=0,
        started_at_ns=0,
        first_content_at_ns=ttft_ms * 1_000_000,
        finished_at_ns=finish_ms * 1_000_000,
        input_tokens=8,
        output_tokens=3,
        token_count_source=TokenCountSource.SERVER_USAGE,
        status=status,
    )


def slo() -> SloThresholds:
    """Return the unit-test SLO."""
    return SloThresholds(ttft_p95_ms=200.0, tpot_p95_ms=50.0, success_rate_min=0.75)


def test_goodput_counts_only_requests_meeting_both_latency_slos() -> None:
    result = calculate_goodput(
        [
            make_sample("fast"),
            make_sample("slow-ttft", ttft_ms=201),
            make_sample("slow-tpot", tpot_ms=51),
            make_sample("failed", status=RequestStatus.ERROR),
        ],
        measured_seconds=2.0,
        slo=slo(),
    )

    assert result.success_rate == pytest.approx(0.75)
    assert result.success_rate_slo_met is True
    assert result.eligible_requests == 3
    assert result.qualifying_requests == 1
    assert result.requests_per_second == pytest.approx(0.5)


def test_goodput_is_zero_when_whole_run_success_rate_slo_fails() -> None:
    result = calculate_goodput(
        [make_sample("fast"), make_sample("failed", status=RequestStatus.ERROR)],
        measured_seconds=1.0,
        slo=slo(),
    )

    assert result.qualifying_requests == 1
    assert result.success_rate_slo_met is False
    assert result.requests_per_second == 0.0


def point(
    observation_id: str,
    goodput: float,
    ttft: float,
    tpot: float,
    memory: int,
    status: ValidationStatus = ValidationStatus.VALID,
) -> ParetoPoint:
    """Create one complete Pareto observation."""
    return ParetoPoint(
        observation_id=observation_id,
        validation_status=status,
        goodput_requests_per_second=goodput,
        ttft_p95_ms=ttft,
        tpot_p95_ms=tpot,
        memory_peak_bytes=memory,
    )


def test_pareto_excludes_dominated_invalid_and_incomplete_points() -> None:
    balanced = point("balanced", 10.0, 100.0, 20.0, 1000)
    high_goodput = point("high-goodput", 12.0, 120.0, 25.0, 1100)
    dominated = point("dominated", 9.0, 130.0, 30.0, 1200)
    invalid = point("invalid", 20.0, 50.0, 10.0, 900, ValidationStatus.INVALID)
    incomplete = ParetoPoint(
        observation_id="incomplete",
        validation_status=ValidationStatus.VALID,
        goodput_requests_per_second=30.0,
    )

    frontier = pareto_frontier([balanced, high_goodput, dominated, invalid, incomplete])

    assert [item.observation_id for item in frontier] == ["balanced", "high-goodput"]


def test_stability_is_valid_with_three_low_variance_repeats() -> None:
    result = analyze_stability([10.0, 10.2, 9.8])

    assert result.status is ValidationStatus.VALID
    assert result.mean_requests_per_second == pytest.approx(10.0)
    assert result.coefficient_of_variation == pytest.approx(0.0163299316)


def test_stability_is_inconclusive_for_too_few_or_unstable_repeats() -> None:
    too_few = analyze_stability([10.0, 10.0])
    unstable = analyze_stability([5.0, 10.0, 15.0])

    assert too_few.status is ValidationStatus.INCONCLUSIVE
    assert "at least 3" in too_few.message
    assert unstable.status is ValidationStatus.INCONCLUSIVE
    assert unstable.coefficient_of_variation is not None
    assert unstable.coefficient_of_variation > 0.10


def test_stability_rejects_non_finite_or_negative_values() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        analyze_stability([1.0, float("nan"), 2.0])
    with pytest.raises(ValueError, match="finite and non-negative"):
        analyze_stability([1.0, -1.0, 2.0])
