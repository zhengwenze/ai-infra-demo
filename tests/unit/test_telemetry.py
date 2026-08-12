"""Unit tests for client, Prometheus, and optional GPU telemetry."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from inferscope.errors import ErrorCode
from inferscope.telemetry import (
    ClientTelemetrySource,
    NvmlGpuTelemetry,
    VLLMMetricsCollector,
    measure_event_loop_lag,
    parse_prometheus_text,
    sample_client_process,
)
from inferscope.telemetry import gpu as gpu_module
from inferscope.telemetry.vllm_metrics import build_vllm_snapshot

PROMETHEUS_TEXT = """\
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="demo"} 2
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="demo"} 120
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1"} 3
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 4
vllm:time_to_first_token_seconds_sum 0.5
vllm:time_to_first_token_seconds_count 4
"""


class _FakeProcess:
    def __init__(self) -> None:
        self.calls = 0

    def cpu_percent(self, interval: None = None) -> float:
        assert interval is None
        self.calls += 1
        return 37.5

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=123_456)


def test_prometheus_parser_preserves_labels_and_histogram_series() -> None:
    samples = parse_prometheus_text(PROMETHEUS_TEXT)

    running = next(sample for sample in samples if sample.name == "vllm:num_requests_running")
    assert running.value == 2.0
    assert running.labels == (("model_name", "demo"),)
    assert any(sample.name.endswith("_bucket") for sample in samples)


def test_vllm_mapping_reports_missing_metrics_instead_of_zero() -> None:
    snapshot = build_vllm_snapshot(PROMETHEUS_TEXT)

    running = snapshot.get("running_requests")
    assert running is not None
    assert running.samples[0].value == 2.0
    assert snapshot.get("waiting_requests") is None
    assert "waiting_requests" in snapshot.missing_metrics
    assert snapshot.get("ttft_seconds") is not None


def test_vllm_counter_mapping_does_not_mix_created_timestamp_into_counter() -> None:
    snapshot = build_vllm_snapshot(
        """\
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total 10
vllm:prefix_cache_queries_created 1
"""
    )

    queries = snapshot.get("prefix_cache_queries")
    assert queries is not None
    assert [sample.name for sample in queries.samples] == ["vllm:prefix_cache_queries_total"]


@pytest.mark.asyncio
async def test_vllm_collector_fetches_metrics_with_injected_client() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/metrics"
        return httpx.Response(200, text=PROMETHEUS_TEXT)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        collector = VLLMMetricsCollector("http://metrics.test/metrics", client=client)
        snapshot = await collector.collect()

    assert snapshot.get("prompt_tokens") is not None


def test_client_process_samples_cpu_and_memory_with_shared_timestamp() -> None:
    samples = sample_client_process(_FakeProcess())  # type: ignore[arg-type]

    assert {sample.name for sample in samples} == {
        "process_cpu_percent",
        "resident_memory_bytes",
    }
    assert samples[0].wall_time_utc == samples[1].wall_time_utc
    assert samples[0].monotonic_ns == samples[1].monotonic_ns
    assert samples[0].value == 37.5
    assert samples[1].value == 123_456.0


@pytest.mark.asyncio
async def test_event_loop_lag_is_a_non_negative_observation() -> None:
    sample = await measure_event_loop_lag(0.001)

    assert sample.name == "event_loop_lag_seconds"
    assert sample.value >= 0
    assert sample.unit == "seconds"


@pytest.mark.asyncio
async def test_client_source_primes_cpu_counter_before_collection() -> None:
    process = _FakeProcess()
    source = ClientTelemetrySource(process)  # type: ignore[arg-type]

    samples = await source.collect(lag_interval_seconds=0.001)

    assert process.calls == 2
    assert {sample.name for sample in samples} == {
        "process_cpu_percent",
        "resident_memory_bytes",
        "event_loop_lag_seconds",
    }


def test_gpu_telemetry_is_explicitly_unavailable_without_nvml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import() -> None:
        raise ImportError("pynvml is not installed")

    monkeypatch.setattr(gpu_module, "_load_pynvml", fail_import)
    telemetry = NvmlGpuTelemetry(gpu_index=0)

    result = telemetry.collect()

    assert result.available is False
    assert result.samples == ()
    assert result.reason is not None
    assert result.error_code == ErrorCode.GPU_TELEMETRY_UNAVAILABLE


def test_gpu_telemetry_collects_supported_nvml_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_nvml = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0,
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda index: f"gpu-{index}",
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=80),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=4_000),
        nvmlDeviceGetPowerUsage=lambda handle: 125_000,
        nvmlDeviceGetTemperature=lambda handle, sensor: 61,
    )
    monkeypatch.setattr(gpu_module, "_load_pynvml", lambda: fake_nvml)

    with NvmlGpuTelemetry(gpu_index=0) as telemetry:
        result = telemetry.collect()

    assert result.available is True
    assert {sample.name for sample in result.samples} == {
        "utilization_percent",
        "memory_used_bytes",
        "power_watts",
        "temperature_celsius",
    }
    assert next(sample for sample in result.samples if sample.name == "power_watts").value == 125.0
    assert calls == ["init", "shutdown"]
