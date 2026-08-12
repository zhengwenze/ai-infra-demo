"""Unit tests for reproducible workload descriptions and schedules."""

from __future__ import annotations

import random

import pytest

from inferscope.workloads import (
    ArrivalMode,
    build_concurrency_plan,
    build_fixed_rate_plan,
    build_poisson_plan,
    build_shared_prefix_requests,
    build_synthetic_requests,
)


def test_concurrency_plan_defers_pacing_to_max_concurrency() -> None:
    plan = build_concurrency_plan(num_requests=5, max_concurrency=2)

    assert plan.mode is ArrivalMode.CONCURRENCY
    assert plan.max_concurrency == 2
    assert [arrival.sequence for arrival in plan.arrivals] == [0, 1, 2, 3, 4]
    assert {arrival.offset_ns for arrival in plan.arrivals} == {0}


def test_fixed_rate_plan_uses_absolute_even_offsets() -> None:
    plan = build_fixed_rate_plan(num_requests=4, requests_per_second=2.0)

    assert plan.mode is ArrivalMode.FIXED_RATE
    assert [arrival.offset_ns for arrival in plan.arrivals] == [
        0,
        500_000_000,
        1_000_000_000,
        1_500_000_000,
    ]


def test_poisson_plan_is_reproducible_and_monotonic() -> None:
    first = build_poisson_plan(8, 3.5, seed=20260812)
    second = build_poisson_plan(8, 3.5, seed=20260812)
    different = build_poisson_plan(8, 3.5, seed=7)

    assert first == second
    assert first != different
    offsets = [arrival.offset_ns for arrival in first.arrivals]
    assert offsets[0] == 0
    assert offsets == sorted(offsets)


def test_poisson_plan_does_not_mutate_global_random_state() -> None:
    random.seed(42)
    expected = random.random()
    random.seed(42)

    build_poisson_plan(5, 1.0, seed=99)

    assert random.random() == expected


@pytest.mark.parametrize("num_requests", [0, -1])
def test_arrival_plans_reject_non_positive_request_counts(num_requests: int) -> None:
    with pytest.raises(ValueError, match="num_requests"):
        build_concurrency_plan(num_requests, 1)


@pytest.mark.parametrize("rate", [0.0, -1.0, float("inf"), float("nan")])
def test_rate_plans_reject_invalid_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="requests_per_second"):
        build_fixed_rate_plan(2, rate)


def test_synthetic_requests_are_seeded_and_have_stable_ids() -> None:
    first = build_synthetic_requests(3, 8, 4, seed=12)
    second = build_synthetic_requests(3, 8, 4, seed=12)

    assert first == second
    assert len({request.request_id for request in first}) == 3
    assert all(len(request.prompt.split()) == 8 for request in first)
    assert all(len(request.prompt_sha256) == 64 for request in first)


def test_shared_prefix_requests_reuse_exact_prefix_with_unique_descriptions() -> None:
    requests = build_shared_prefix_requests(
        num_requests=4,
        shared_prefix_tokens=6,
        unique_suffix_tokens=3,
        output_tokens=5,
        seed=88,
    )

    prefixes = {" ".join(request.prompt.split()[:6]) for request in requests}
    assert len(prefixes) == 1
    assert len({request.shared_prefix_id for request in requests}) == 1
    assert all(request.target_input_tokens == 9 for request in requests)
    assert all(request.workload_name == "shared_prefix" for request in requests)


def test_synthetic_builder_rejects_empty_vocabulary() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        build_synthetic_requests(1, 2, 3, seed=1, vocabulary=())
