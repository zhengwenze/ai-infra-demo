from __future__ import annotations

import csv
from pathlib import Path

from inferscope.config import ExperimentConfig
from inferscope.models import (
    AggregateReport,
    CheckStatus,
    LatencySummary,
    MetricDistribution,
    RequestCounts,
    ThroughputSummary,
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from inferscope.reporting import (
    build_markdown_report,
    write_summary_csv,
    write_tradeoff_svg,
)


def make_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "report-test",
            "seed": 7,
            "target": {
                "backend": "vllm",
                "base_url": "http://127.0.0.1:8000",
                "model": "test-model",
                "timeout_seconds": 30.0,
            },
            "generation": {"max_output_tokens": 10},
            "workload": {
                "type": "synthetic",
                "prompt_tokens": 8,
                "output_tokens": 10,
                "num_requests": 3,
                "arrival": {"mode": "concurrency", "values": (1,)},
            },
            "slo": {
                "ttft_p95_ms": 500.0,
                "tpot_p95_ms": 50.0,
                "success_rate_min": 0.99,
            },
        },
        strict=True,
    )


def distribution(value: float) -> MetricDistribution:
    return MetricDistribution(
        count=3,
        mean=value,
        p50=value,
        p95=value,
        p99=value,
        minimum=value,
        maximum=value,
        p99_small_sample_warning=True,
    )


def aggregate() -> AggregateReport:
    return AggregateReport(
        run_id="run-1",
        validation_status=ValidationStatus.INCONCLUSIVE,
        measured_seconds=2.0,
        requests=RequestCounts(scheduled=3, successful=3, valid=3),
        throughput=ThroughputSummary(
            requests_per_second=1.5,
            input_tokens_per_second=12.0,
            output_tokens_per_second=15.0,
            goodput_requests_per_second=1.0,
        ),
        latency_ms=LatencySummary(
            ttft=distribution(25.0),
            tpot=distribution(4.0),
            e2e=distribution(70.0),
        ),
    )


def validation() -> ValidationReport:
    return ValidationReport(
        run_id="run-1",
        status=ValidationStatus.INCONCLUSIVE,
        checks=(
            ValidationCheck(
                name="client_capacity",
                status=CheckStatus.WARN,
                message="client event-loop lag was not captured",
            ),
        ),
        warnings=("client capacity could not be verified",),
    )


def test_markdown_keeps_warning_and_metric_definitions() -> None:
    rendered = build_markdown_report(make_config(), aggregate(), validation())
    assert "INCONCLUSIVE" in rendered
    assert "client capacity could not be verified" in rendered
    assert "SSE 内容块不等同于 token" in rendered


def test_csv_has_stable_machine_readable_columns(tmp_path: Path) -> None:
    path = write_summary_csv(aggregate(), tmp_path / "summary.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["run_id"] == "run-1"
    assert row["ttft_p95_ms"] == "25.0"
    assert row["validation_status"] == "INCONCLUSIVE"


def test_svg_chart_is_dependency_light_and_escapes_title(tmp_path: Path) -> None:
    path = write_tradeoff_svg((aggregate(),), tmp_path / "tradeoff.svg", title="baseline < 4060")
    assert path is not None
    rendered = path.read_text(encoding="utf-8")
    assert rendered.startswith("<svg")
    assert "baseline &lt; 4060" in rendered
    assert "TTFT P95" in rendered
