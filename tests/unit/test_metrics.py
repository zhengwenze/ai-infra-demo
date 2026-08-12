"""Tests for request and aggregate metric contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferscope.metrics import (
    aggregate_request_metrics,
    calculate_chunk_interarrival_ms,
    calculate_e2e_ms,
    calculate_tpot_ms,
    calculate_ttft_ms,
    summarize_distribution,
)
from inferscope.models import (
    RequestSample,
    RequestStatus,
    TokenCountSource,
    ValidationStatus,
)


def make_sample(
    request_id: str = "request-1",
    *,
    start_ms: int = 0,
    first_ms: int | None = 100,
    finish_ms: int | None = 300,
    input_tokens: int | None = 10,
    output_tokens: int | None = 3,
    status: RequestStatus = RequestStatus.SUCCESS,
) -> RequestSample:
    """Build a request sample with timestamps expressed conveniently in ms."""
    to_ns = 1_000_000
    return RequestSample(
        run_id="run-1",
        request_id=request_id,
        sequence=0,
        scheduled_at_ns=start_ms * to_ns,
        started_at_ns=start_ms * to_ns,
        first_content_at_ns=None if first_ms is None else first_ms * to_ns,
        finished_at_ns=None if finish_ms is None else finish_ms * to_ns,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_count_source=TokenCountSource.SERVER_USAGE,
        chunk_times_ns=(100 * to_ns, 125 * to_ns, 165 * to_ns),
        status=status,
        http_status=200,
    )


def test_request_metrics_follow_ttft_e2e_and_tpot_contract() -> None:
    sample = make_sample()

    assert calculate_ttft_ms(sample) == pytest.approx(100.0)
    assert calculate_e2e_ms(sample) == pytest.approx(300.0)
    assert calculate_tpot_ms(sample) == pytest.approx(100.0)
    assert calculate_chunk_interarrival_ms(sample) == pytest.approx((25.0, 40.0))


def test_tpot_is_none_when_only_one_output_token_or_timing_is_missing() -> None:
    assert calculate_tpot_ms(make_sample(output_tokens=1)) is None
    assert calculate_tpot_ms(make_sample(output_tokens=None)) is None
    assert calculate_tpot_ms(make_sample(first_ms=None)) is None
    assert calculate_tpot_ms(make_sample(finish_ms=None)) is None


def test_request_sample_rejects_non_monotonic_timestamps() -> None:
    with pytest.raises(ValidationError, match="must not precede"):
        make_sample(first_ms=301, finish_ms=300)


def test_percentiles_use_linear_quantiles_and_warn_for_small_p99_sample() -> None:
    summary = summarize_distribution([1.0, 2.0, None, 3.0, 4.0])

    assert summary.count == 4
    assert summary.mean == pytest.approx(2.5)
    assert summary.p50 == pytest.approx(2.5)
    assert summary.p95 == pytest.approx(3.85)
    assert summary.p99 == pytest.approx(3.97)
    assert summary.minimum == pytest.approx(1.0)
    assert summary.maximum == pytest.approx(4.0)
    assert summary.p99_small_sample_warning is True


def test_empty_distribution_uses_none_not_fake_zero() -> None:
    summary = summarize_distribution([])

    assert summary.count == 0
    assert summary.mean is None
    assert summary.p99 is None
    assert summary.p99_small_sample_warning is False


def test_aggregate_throughput_uses_wall_time_and_separates_valid_samples() -> None:
    first = make_sample("request-1")
    second = make_sample("request-2", input_tokens=20, output_tokens=5)
    failed = make_sample("request-3", status=RequestStatus.ERROR)

    report = aggregate_request_metrics(
        [first, second, failed],
        run_id="run-1",
        measured_seconds=2.0,
        validation_status=ValidationStatus.VALID,
        scheduled_count=4,
        valid_request_ids={"request-2"},
    )

    assert report.requests.scheduled == 4
    assert report.requests.successful == 2
    assert report.requests.valid == 1
    assert report.throughput.requests_per_second == pytest.approx(1.0)
    assert report.throughput.input_tokens_per_second == pytest.approx(10.0)
    assert report.throughput.output_tokens_per_second == pytest.approx(2.5)
    assert report.latency_ms.ttft.count == 1


def test_aggregate_token_throughput_is_none_if_any_valid_count_is_unknown() -> None:
    report = aggregate_request_metrics(
        [make_sample("known"), make_sample("unknown", input_tokens=None)],
        run_id="run-1",
        measured_seconds=1.0,
        validation_status=ValidationStatus.VALID,
    )

    assert report.throughput.input_tokens_per_second is None
    assert report.throughput.output_tokens_per_second == pytest.approx(6.0)


def test_aggregate_rejects_invalid_measurement_window() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        aggregate_request_metrics(
            [],
            run_id="run-1",
            measured_seconds=0,
            validation_status=ValidationStatus.INVALID,
        )
