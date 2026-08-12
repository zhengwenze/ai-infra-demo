"""Repeat-run throughput stability analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from inferscope.models import StabilityResult, ValidationStatus


def analyze_stability(
    requests_per_second: Sequence[float],
    *,
    target_coefficient_of_variation: float = 0.10,
    minimum_repeats: int = 3,
) -> StabilityResult:
    """Classify repeat stability using population standard deviation divided by mean."""
    if target_coefficient_of_variation < 0:
        raise ValueError("target_coefficient_of_variation must not be negative")
    if minimum_repeats < 2:
        raise ValueError("minimum_repeats must be at least two")
    array = np.asarray(requests_per_second, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("throughput values must be finite and non-negative")
    if array.size == 0:
        return StabilityResult(
            repeat_count=0,
            target_coefficient_of_variation=target_coefficient_of_variation,
            status=ValidationStatus.INCONCLUSIVE,
            message="no repeat results are available",
        )

    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=0))
    coefficient = 0.0 if mean == 0 else standard_deviation / mean
    if array.size < minimum_repeats:
        status = ValidationStatus.INCONCLUSIVE
        message = f"at least {minimum_repeats} repeats are required"
    elif coefficient > target_coefficient_of_variation:
        status = ValidationStatus.INCONCLUSIVE
        message = "throughput variation exceeds the configured stability target"
    else:
        status = ValidationStatus.VALID
        message = "throughput variation is within the configured stability target"
    return StabilityResult(
        repeat_count=int(array.size),
        mean_requests_per_second=mean,
        standard_deviation=standard_deviation,
        coefficient_of_variation=coefficient,
        target_coefficient_of_variation=target_coefficient_of_variation,
        status=status,
        message=message,
    )
