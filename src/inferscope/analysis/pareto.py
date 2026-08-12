"""Multi-objective Pareto analysis for validated observations."""

from __future__ import annotations

from collections.abc import Sequence

from inferscope.models import ParetoPoint, ValidationStatus


def _has_all_objectives(point: ParetoPoint) -> bool:
    return (
        point.ttft_p95_ms is not None
        and point.tpot_p95_ms is not None
        and point.memory_peak_bytes is not None
    )


def _objective_values(point: ParetoPoint) -> tuple[float, float, float, int] | None:
    if point.ttft_p95_ms is None or point.tpot_p95_ms is None or point.memory_peak_bytes is None:
        return None
    return (
        point.goodput_requests_per_second,
        -point.ttft_p95_ms,
        -point.tpot_p95_ms,
        -point.memory_peak_bytes,
    )


def dominates(candidate: ParetoPoint, other: ParetoPoint) -> bool:
    """Return whether ``candidate`` is no worse in all and better in one objective."""
    candidate_values = _objective_values(candidate)
    other_values = _objective_values(other)
    if candidate_values is None or other_values is None:
        return False
    return all(
        left >= right for left, right in zip(candidate_values, other_values, strict=True)
    ) and any(left > right for left, right in zip(candidate_values, other_values, strict=True))


def pareto_frontier(points: Sequence[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    """Return validated, complete, non-dominated observations in input order."""
    eligible = [
        point
        for point in points
        if point.validation_status is ValidationStatus.VALID and _has_all_objectives(point)
    ]
    return tuple(
        point
        for point in eligible
        if not any(dominates(other, point) for other in eligible if other is not point)
    )
