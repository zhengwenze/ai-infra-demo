"""Prometheus exposition parsing and version-tolerant vLLM metric mapping."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import httpx
from prometheus_client.parser import text_string_to_metric_families

from inferscope.telemetry.sampler import utc_now_iso


@dataclass(frozen=True, slots=True)
class PrometheusSample:
    """One parsed Prometheus sample, retaining all label dimensions."""

    name: str
    value: float
    labels: tuple[tuple[str, str], ...] = ()
    timestamp_ms: int | float | None = None


@dataclass(frozen=True, slots=True)
class LogicalMetric:
    """A logical vLLM metric mapped to one or more raw time series."""

    logical_name: str
    prometheus_name: str
    samples: tuple[PrometheusSample, ...]


@dataclass(frozen=True, slots=True)
class VLLMMetricsSnapshot:
    """A time-aligned vLLM metrics scrape with explicit capability gaps."""

    wall_time_utc: str
    monotonic_ns: int
    raw_samples: tuple[PrometheusSample, ...]
    logical_metrics: tuple[LogicalMetric, ...]
    missing_metrics: tuple[str, ...]

    def get(self, logical_name: str) -> LogicalMetric | None:
        """Return a discovered logical metric, or ``None`` when unavailable."""

        return next(
            (metric for metric in self.logical_metrics if metric.logical_name == logical_name),
            None,
        )


VLLM_METRIC_ALIASES: Final = MappingProxyType(
    {
        "running_requests": ("vllm:num_requests_running", "vllm_num_requests_running"),
        "waiting_requests": ("vllm:num_requests_waiting", "vllm_num_requests_waiting"),
        "kv_cache_usage_ratio": (
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
            "vllm_kv_cache_usage_perc",
            "vllm_gpu_cache_usage_perc",
        ),
        "prefix_cache_queries": (
            "vllm:prefix_cache_queries",
            "vllm:prefix_cache_queries_total",
            "vllm_prefix_cache_queries_total",
        ),
        "prefix_cache_hits": (
            "vllm:prefix_cache_hits",
            "vllm:prefix_cache_hits_total",
            "vllm_prefix_cache_hits_total",
        ),
        "prompt_tokens": ("vllm:prompt_tokens_total", "vllm_prompt_tokens_total"),
        "generation_tokens": (
            "vllm:generation_tokens_total",
            "vllm_generation_tokens_total",
        ),
        "ttft_seconds": (
            "vllm:time_to_first_token_seconds",
            "vllm_time_to_first_token_seconds",
        ),
        "itl_seconds": (
            "vllm:inter_token_latency_seconds",
            "vllm_inter_token_latency_seconds",
        ),
        "e2e_seconds": (
            "vllm:e2e_request_latency_seconds",
            "vllm_e2e_request_latency_seconds",
        ),
    }
)


def parse_prometheus_text(text: str) -> tuple[PrometheusSample, ...]:
    """Parse Prometheus text while preserving names, labels, and timestamps."""

    parsed: list[PrometheusSample] = []
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            value = float(sample.value)
            if not math.isfinite(value):
                continue
            parsed.append(
                PrometheusSample(
                    name=sample.name,
                    value=value,
                    labels=tuple(
                        sorted((str(key), str(item)) for key, item in sample.labels.items())
                    ),
                    timestamp_ms=(
                        float(sample.timestamp) if sample.timestamp is not None else None
                    ),
                )
            )
    return tuple(parsed)


_HISTOGRAM_LOGICAL_NAMES = frozenset({"ttft_seconds", "itl_seconds", "e2e_seconds"})
_HISTOGRAM_SUFFIXES = ("_bucket", "_count", "_sum")


def _matches_metric_name(sample_name: str, candidate_name: str, *, histogram: bool) -> bool:
    if sample_name == candidate_name:
        return True
    return histogram and sample_name in {
        f"{candidate_name}{suffix}" for suffix in _HISTOGRAM_SUFFIXES
    }


def map_vllm_metrics(samples: tuple[PrometheusSample, ...]) -> tuple[LogicalMetric, ...]:
    """Map available raw series to stable logical names without inventing values."""

    logical_metrics: list[LogicalMetric] = []
    for logical_name, aliases in VLLM_METRIC_ALIASES.items():
        for alias in aliases:
            matching = tuple(
                sample
                for sample in samples
                if _matches_metric_name(
                    sample.name,
                    alias,
                    histogram=logical_name in _HISTOGRAM_LOGICAL_NAMES,
                )
            )
            if matching:
                logical_metrics.append(
                    LogicalMetric(
                        logical_name=logical_name,
                        prometheus_name=alias,
                        samples=matching,
                    )
                )
                break
    return tuple(logical_metrics)


def build_vllm_snapshot(text: str) -> VLLMMetricsSnapshot:
    """Create a timestamped, capability-aware snapshot from exposition text."""

    wall_time_utc = utc_now_iso()
    monotonic_ns = time.perf_counter_ns()
    raw_samples = parse_prometheus_text(text)
    logical_metrics = map_vllm_metrics(raw_samples)
    present = {metric.logical_name for metric in logical_metrics}
    missing = tuple(name for name in VLLM_METRIC_ALIASES if name not in present)
    return VLLMMetricsSnapshot(
        wall_time_utc=wall_time_utc,
        monotonic_ns=monotonic_ns,
        raw_samples=raw_samples,
        logical_metrics=logical_metrics,
        missing_metrics=missing,
    )


class VLLMMetricsCollector:
    """Fetch vLLM Prometheus metrics over HTTP and parse one snapshot."""

    def __init__(
        self,
        metrics_url: str,
        *,
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self._metrics_url = metrics_url
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def collect(self) -> VLLMMetricsSnapshot:
        """Fetch a scrape and raise on HTTP or exposition-format errors."""

        if self._client is not None:
            response = await self._client.get(self._metrics_url, timeout=self._timeout_seconds)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(self._metrics_url, timeout=self._timeout_seconds)
        response.raise_for_status()
        return build_vllm_snapshot(response.text)
