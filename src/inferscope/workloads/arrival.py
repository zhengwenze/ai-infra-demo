"""Pure, reproducible workload-arrival plan builders."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum


class ArrivalMode(StrEnum):
    """Supported request arrival processes."""

    CONCURRENCY = "concurrency"
    FIXED_RATE = "fixed_rate"
    POISSON = "poisson"


@dataclass(frozen=True, slots=True)
class ScheduledArrival:
    """A request's intended offset from the measurement-window start."""

    sequence: int
    offset_ns: int

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.offset_ns < 0:
            raise ValueError("offset_ns must be non-negative")


@dataclass(frozen=True, slots=True)
class ArrivalPlan:
    """An immutable arrival plan consumed by the experiment runner.

    For ``concurrency`` mode every request has offset zero. The runner uses
    ``max_concurrency`` to dispatch the next request whenever an active request
    completes. Rate-based modes dispatch according to each absolute offset.
    """

    mode: ArrivalMode
    arrivals: tuple[ScheduledArrival, ...]
    max_concurrency: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_concurrency is not None and self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than zero")
        if self.mode is ArrivalMode.CONCURRENCY and self.max_concurrency is None:
            raise ValueError("concurrency plans require max_concurrency")


def _validate_num_requests(num_requests: int) -> None:
    if num_requests <= 0:
        raise ValueError("num_requests must be greater than zero")


def _validate_rate(requests_per_second: float) -> None:
    if not math.isfinite(requests_per_second) or requests_per_second <= 0:
        raise ValueError("requests_per_second must be finite and greater than zero")


def build_concurrency_plan(num_requests: int, max_concurrency: int) -> ArrivalPlan:
    """Build a closed-loop plan that maintains at most ``max_concurrency`` requests."""

    _validate_num_requests(num_requests)
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than zero")
    arrivals = tuple(ScheduledArrival(sequence=index, offset_ns=0) for index in range(num_requests))
    return ArrivalPlan(
        mode=ArrivalMode.CONCURRENCY,
        arrivals=arrivals,
        max_concurrency=max_concurrency,
    )


def build_fixed_rate_plan(num_requests: int, requests_per_second: float) -> ArrivalPlan:
    """Build an open-loop plan with evenly spaced request start offsets.

    The first request is scheduled at offset zero. Integer nanosecond offsets
    are calculated from the absolute sequence number so rounding error does not
    accumulate between requests.
    """

    _validate_num_requests(num_requests)
    _validate_rate(requests_per_second)
    period_ns = 1_000_000_000 / requests_per_second
    arrivals = tuple(
        ScheduledArrival(sequence=index, offset_ns=round(index * period_ns))
        for index in range(num_requests)
    )
    return ArrivalPlan(mode=ArrivalMode.FIXED_RATE, arrivals=arrivals)


def build_poisson_plan(
    num_requests: int,
    requests_per_second: float,
    *,
    seed: int,
) -> ArrivalPlan:
    """Build a seeded Poisson arrival plan using exponential inter-arrivals.

    The first request is scheduled at offset zero. A private ``random.Random``
    instance prevents workload generation from mutating global random state.
    """

    _validate_num_requests(num_requests)
    _validate_rate(requests_per_second)
    random_source = random.Random(seed)
    offset_seconds = 0.0
    offsets_ns = [0]
    for _ in range(1, num_requests):
        offset_seconds += random_source.expovariate(requests_per_second)
        offsets_ns.append(round(offset_seconds * 1_000_000_000))
    arrivals = tuple(
        ScheduledArrival(sequence=index, offset_ns=offset_ns)
        for index, offset_ns in enumerate(offsets_ns)
    )
    return ArrivalPlan(mode=ArrivalMode.POISSON, arrivals=arrivals, seed=seed)
