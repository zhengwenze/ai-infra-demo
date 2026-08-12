"""Strict YAML configuration models and reproducible hashing."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from pathlib import Path
from typing import IO, Literal, Protocol, Self, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from inferscope.errors import ConfigurationError

_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class _YamlModule(Protocol):
    """Typed subset of PyYAML used by the config loader."""

    YAMLError: type[Exception]

    def safe_load(self, stream: IO[str]) -> object:
        """Parse one YAML stream without constructing arbitrary Python objects."""


yaml = cast(_YamlModule, importlib.import_module("yaml"))


class StrictConfigModel(BaseModel):
    """Base class for configuration that rejects coercion and unknown keys."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _validate_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL must not embed credentials")
    return value.rstrip("/")


class TargetConfig(StrictConfigModel):
    """Inference endpoint and model identity."""

    backend: Literal["hf", "vllm"]
    base_url: str
    model: str = Field(min_length=1)
    model_revision: str | None = None
    api_key_env: str | None = None
    request_type: Literal["chat_completions", "completions"] = "chat_completions"
    timeout_seconds: float = Field(gt=0)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Require an HTTP URL without embedded credentials."""
        return _validate_http_url(value)

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_reference(cls, value: str | None) -> str | None:
        """Accept only an environment variable name, never a plaintext secret."""
        if value is not None and _ENV_NAME.fullmatch(value) is None:
            raise ValueError("api_key_env must be an uppercase environment variable name")
        return value


class GenerationConfig(StrictConfigModel):
    """Deterministic generation controls used by benchmark requests."""

    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    max_output_tokens: int = Field(gt=0)
    ignore_eos: bool = False


class ArrivalConfig(StrictConfigModel):
    """Request arrival mode and sweep values."""

    mode: Literal["concurrency", "fixed_rate", "poisson"]
    values: tuple[float | int, ...] = Field(min_length=1)

    @field_validator("values", mode="before")
    @classmethod
    def freeze_values(cls, value: object) -> object:
        """Normalize YAML sequences to an immutable tuple without scalar coercion."""
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values_for_mode(self) -> Self:
        """Require positive rates and positive integer concurrency levels."""
        for value in self.values:
            if isinstance(value, bool) or value <= 0:
                raise ValueError("arrival values must be positive numbers")
            if self.mode == "concurrency" and not isinstance(value, int):
                raise ValueError("concurrency arrival values must be integers")
        return self


class WorkloadConfig(StrictConfigModel):
    """Synthetic workload size and arrival plan."""

    type: Literal["synthetic", "shared_prefix", "mixed"]
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    num_requests: int = Field(gt=0)
    common_prefix_tokens: int | None = Field(default=None, gt=0)
    arrival: ArrivalConfig

    @model_validator(mode="after")
    def validate_shared_prefix(self) -> Self:
        """Require and bound the common prefix for shared-prefix workloads."""
        if self.type == "shared_prefix" and self.common_prefix_tokens is None:
            raise ValueError("shared_prefix workload requires common_prefix_tokens")
        if (
            self.common_prefix_tokens is not None
            and self.common_prefix_tokens >= self.prompt_tokens
        ):
            raise ValueError("common_prefix_tokens must be smaller than prompt_tokens")
        return self


class WarmupConfig(StrictConfigModel):
    """Warm-up requests that remain separate from formal measurements."""

    requests: int = Field(default=0, ge=0)
    include_in_metrics: Literal[False] = False


class TelemetryConfig(StrictConfigModel):
    """Service and GPU sampling settings."""

    vllm_metrics_url: str | None = None
    gpu_index: int = Field(default=0, ge=0)
    interval_ms: int = Field(default=500, ge=100)

    @field_validator("vllm_metrics_url")
    @classmethod
    def validate_metrics_url(cls, value: str | None) -> str | None:
        """Validate the optional metrics endpoint."""
        return None if value is None else _validate_http_url(value)


class ValidationConfig(StrictConfigModel):
    """Benchmark validity thresholds."""

    min_success_rate: float = Field(default=0.99, ge=0, le=1)
    output_token_tolerance_ratio: float = Field(default=0.1, ge=0, le=1)
    token_count_mismatch_ratio: float = Field(default=0.02, ge=0, le=1)
    max_client_loop_lag_ms: float = Field(default=20.0, gt=0)
    require_clean_gpu: bool = True


class SloConfig(StrictConfigModel):
    """Latency and success-rate service-level objectives."""

    ttft_p95_ms: float = Field(gt=0)
    tpot_p95_ms: float = Field(gt=0)
    success_rate_min: float = Field(ge=0, le=1)


class ExecutionConfig(StrictConfigModel):
    """Repeat, cooling, and matrix safety limits."""

    repeats: int = Field(default=3, ge=1)
    cooldown_seconds: float = Field(default=10.0, ge=0)
    max_matrix_combinations: int = Field(default=32, ge=1)


class OutputConfig(StrictConfigModel):
    """Privacy-preserving artifact output options."""

    save_prompts: bool = False
    save_responses: bool = False
    formats: tuple[Literal["json", "csv", "markdown", "png"], ...] = (
        "json",
        "csv",
        "markdown",
        "png",
    )

    @field_validator("formats", mode="before")
    @classmethod
    def freeze_formats(cls, value: object) -> object:
        """Normalize a YAML format sequence to an immutable tuple."""
        return tuple(value) if isinstance(value, list) else value

    @field_validator("formats")
    @classmethod
    def validate_unique_formats(
        cls,
        value: tuple[Literal["json", "csv", "markdown", "png"], ...],
    ) -> tuple[Literal["json", "csv", "markdown", "png"], ...]:
        """Reject empty or duplicate format lists."""
        if not value:
            raise ValueError("at least one output format is required")
        if len(set(value)) != len(value):
            raise ValueError("output formats must be unique")
        return value


class ExperimentConfig(StrictConfigModel):
    """Complete resolved experiment configuration."""

    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0)
    target: TargetConfig
    generation: GenerationConfig
    workload: WorkloadConfig
    warmup: WarmupConfig = WarmupConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    validation: ValidationConfig = ValidationConfig()
    slo: SloConfig
    execution: ExecutionConfig = ExecutionConfig()
    output: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def validate_matrix_size(self) -> Self:
        """Bound the explicit arrival sweep before any work is scheduled."""
        combinations = len(self.workload.arrival.values)
        if combinations > self.execution.max_matrix_combinations:
            raise ValueError("arrival matrix has more combinations than max_matrix_combinations")
        return self

    def sha256(self) -> str:
        """Return a stable hash of the fully resolved, secret-free config."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a strict YAML config and normalize errors to ``ConfigurationError``."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot load config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("experiment config must be a YAML mapping")
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid experiment config: {exc}") from exc


def read_api_key(config: TargetConfig) -> str | None:
    """Resolve the optional API key at runtime without persisting it."""
    if config.api_key_env is None:
        return None
    return os.environ.get(config.api_key_env)


def validate_results_directory(path: str | Path, project_root: str | Path) -> Path:
    """Reject destructive output targets such as root, home, or project root."""
    candidate = Path(path).expanduser().resolve()
    resolved_project_root = Path(project_root).expanduser().resolve()
    forbidden = {Path(candidate.anchor), Path.home().resolve(), resolved_project_root}
    if candidate in forbidden:
        raise ConfigurationError(f"unsafe results directory: {candidate}")
    return candidate
