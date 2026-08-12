from __future__ import annotations

from inferscope.config import ExperimentConfig
from inferscope.models import RequestSample, RequestStatus, TokenCountSource, ValidationStatus
from inferscope.validators import validate_experiment


def make_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": "1.0",
            "name": "unit-test",
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
                "num_requests": 1,
                "arrival": {"mode": "concurrency", "values": (1,)},
            },
            "validation": {
                "min_success_rate": 0.99,
                "output_token_tolerance_ratio": 0.1,
                "token_count_mismatch_ratio": 0.02,
                "max_client_loop_lag_ms": 20.0,
                "require_clean_gpu": True,
            },
            "slo": {
                "ttft_p95_ms": 500.0,
                "tpot_p95_ms": 50.0,
                "success_rate_min": 0.99,
            },
        },
        strict=True,
    )


def make_sample(**overrides: object) -> RequestSample:
    values: dict[str, object] = {
        "run_id": "run-1",
        "request_id": "request-1",
        "sequence": 0,
        "scheduled_at_ns": 100,
        "started_at_ns": 110,
        "first_content_at_ns": 150,
        "finished_at_ns": 300,
        "input_tokens": 8,
        "output_tokens": 10,
        "token_count_source": TokenCountSource.SERVER_USAGE,
        "chunk_times_ns": (150, 200),
        "status": RequestStatus.SUCCESS,
        "http_status": 200,
    }
    values.update(overrides)
    return RequestSample.model_validate(values, strict=True)


def test_valid_experiment_passes_all_gates() -> None:
    report = validate_experiment(
        "run-1",
        [make_sample()],
        make_config(),
        warmup_completed=True,
        max_loop_lag_ms=2.0,
        gpu_is_clean=True,
    )
    assert report.status is ValidationStatus.VALID
    assert report.warnings == ()


def test_hard_gate_failure_makes_run_invalid() -> None:
    report = validate_experiment(
        "run-1",
        [make_sample(output_tokens=2)],
        make_config(),
        warmup_completed=True,
        max_loop_lag_ms=2.0,
        gpu_is_clean=True,
    )
    assert report.status is ValidationStatus.INVALID


def test_missing_capacity_evidence_makes_run_inconclusive() -> None:
    report = validate_experiment(
        "run-1",
        [make_sample()],
        make_config(),
        warmup_completed=True,
        max_loop_lag_ms=None,
        gpu_is_clean=None,
    )
    assert report.status is ValidationStatus.INCONCLUSIVE
    assert len(report.warnings) == 2
