"""Tests for strict experiment configuration parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from inferscope.config import (
    ExperimentConfig,
    TargetConfig,
    load_config,
    validate_results_directory,
)
from inferscope.errors import ConfigurationError


def valid_config_data() -> dict[str, object]:
    """Return a complete, small configuration fixture."""
    return {
        "schema_version": "1.0",
        "name": "unit-baseline",
        "seed": 20260812,
        "target": {
            "backend": "vllm",
            "base_url": "http://127.0.0.1:8000/",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "api_key_env": "INFERSCOPE_API_KEY",
            "request_type": "chat_completions",
            "timeout_seconds": 120.0,
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 16,
            "ignore_eos": False,
        },
        "workload": {
            "type": "synthetic",
            "prompt_tokens": 32,
            "output_tokens": 16,
            "num_requests": 4,
            "arrival": {"mode": "concurrency", "values": [1, 2]},
        },
        "warmup": {"requests": 1, "include_in_metrics": False},
        "telemetry": {
            "vllm_metrics_url": "http://127.0.0.1:8000/metrics",
            "gpu_index": 0,
            "interval_ms": 500,
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
        "execution": {
            "repeats": 3,
            "cooldown_seconds": 0.0,
            "max_matrix_combinations": 4,
        },
        "output": {
            "save_prompts": False,
            "save_responses": False,
            "formats": ["json", "csv"],
        },
    }


def test_config_expands_defaults_normalizes_urls_and_hashes_stably() -> None:
    config = ExperimentConfig.model_validate(valid_config_data())
    same_config = ExperimentConfig.model_validate(valid_config_data())

    assert config.target.base_url == "http://127.0.0.1:8000"
    assert config.workload.arrival.values == (1, 2)
    assert len(config.sha256()) == 64
    assert config.sha256() == same_config.sha256()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("workload", "num_requests"), 0),
        (("telemetry", "interval_ms"), 99),
        (("validation", "min_success_rate"), 1.01),
        (("execution", "repeats"), 0),
        (("target", "timeout_seconds"), "120"),
    ],
)
def test_config_rejects_out_of_range_or_coerced_values(
    path: tuple[str, str], value: object
) -> None:
    raw = valid_config_data()
    section = raw[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_config_rejects_unknown_fields_and_plaintext_api_key() -> None:
    raw = valid_config_data()
    raw["unknown"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExperimentConfig.model_validate(raw)

    with pytest.raises(ValidationError, match="environment variable"):
        TargetConfig(
            backend="vllm",
            base_url="http://127.0.0.1:8000",
            model="model",
            api_key_env="sk-secret-value",
            timeout_seconds=1.0,
        )


def test_config_rejects_oversized_matrix_and_fractional_concurrency() -> None:
    raw = valid_config_data()
    workload = raw["workload"]
    assert isinstance(workload, dict)
    arrival = workload["arrival"]
    assert isinstance(arrival, dict)
    arrival["values"] = [1, 2, 4, 8, 16]
    with pytest.raises(ValidationError, match="more combinations"):
        ExperimentConfig.model_validate(raw)

    arrival["values"] = [1.5]
    with pytest.raises(ValidationError, match="must be integers"):
        ExperimentConfig.model_validate(raw)


def test_load_config_wraps_yaml_and_validation_failures(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("name: only-one-field\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="invalid experiment config"):
        load_config(invalid)
    with pytest.raises(ConfigurationError, match="cannot load config"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_accepts_yaml_sequences_without_scalar_coercion(tmp_path: Path) -> None:
    config_file = tmp_path / "benchmark.yaml"
    config_file.write_text(
        """\
name: yaml-list-test
seed: 7
target:
  backend: vllm
  base_url: http://127.0.0.1:8000
  model: test-model
  timeout_seconds: 30
generation:
  max_output_tokens: 10
workload:
  type: synthetic
  prompt_tokens: 8
  output_tokens: 10
  num_requests: 2
  arrival:
    mode: concurrency
    values: [1, 2]
slo:
  ttft_p95_ms: 500
  tpot_p95_ms: 50
  success_rate_min: 0.99
output:
  formats: [json, csv]
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.workload.arrival.values == (1, 2)
    assert config.output.formats == ("json", "csv")


def test_results_directory_rejects_broad_destructive_targets(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="unsafe results directory"):
        validate_results_directory(tmp_path, tmp_path)
    with pytest.raises(ConfigurationError, match="unsafe results directory"):
        validate_results_directory(Path.home(), tmp_path)

    assert validate_results_directory(tmp_path / "results", tmp_path) == tmp_path / "results"
