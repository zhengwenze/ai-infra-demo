"""Stable tabular exports for processed benchmark summaries."""

from __future__ import annotations

import csv
from pathlib import Path

from inferscope.models import AggregateReport


def _flatten(report: AggregateReport) -> dict[str, str | int | float | None]:
    return {
        "run_id": report.run_id,
        "validation_status": report.validation_status.value,
        "measured_seconds": report.measured_seconds,
        "requests_scheduled": report.requests.scheduled,
        "requests_successful": report.requests.successful,
        "requests_valid": report.requests.valid,
        "requests_per_second": report.throughput.requests_per_second,
        "input_tokens_per_second": report.throughput.input_tokens_per_second,
        "output_tokens_per_second": report.throughput.output_tokens_per_second,
        "goodput_requests_per_second": report.throughput.goodput_requests_per_second,
        "ttft_p50_ms": report.latency_ms.ttft.p50,
        "ttft_p95_ms": report.latency_ms.ttft.p95,
        "ttft_p99_ms": report.latency_ms.ttft.p99,
        "tpot_p50_ms": report.latency_ms.tpot.p50,
        "tpot_p95_ms": report.latency_ms.tpot.p95,
        "tpot_p99_ms": report.latency_ms.tpot.p99,
        "e2e_p50_ms": report.latency_ms.e2e.p50,
        "e2e_p95_ms": report.latency_ms.e2e.p95,
        "e2e_p99_ms": report.latency_ms.e2e.p99,
        "gpu_utilization_mean": report.gpu.utilization_mean,
        "gpu_memory_peak_bytes": report.gpu.memory_peak_bytes,
        "gpu_power_mean_watts": report.gpu.power_mean_watts,
    }


def write_summary_csv(report: AggregateReport, destination: str | Path) -> Path:
    """Write one aggregate row with a stable, explicit header order."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = _flatten(report)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(path)
    return path
