"""Metric unit constants and shared definitions."""

from typing import Final, Literal

NANOSECONDS_PER_MILLISECOND = 1_000_000

PERCENTILE_METHOD: Final[Literal["linear"]] = "linear"
P99_SMALL_SAMPLE_THRESHOLD = 100
