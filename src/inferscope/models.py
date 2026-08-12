"""Shared immutable data models for experiment artifacts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and implicit type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Backend(StrEnum):
    """Supported inference backends."""

    HF = "hf"
    VLLM = "vllm"


class RequestStatus(StrEnum):
    """Terminal request states persisted in ``requests.jsonl``."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"


class TokenCountSource(StrEnum):
    """Source used for the final token counts."""

    SERVER_USAGE = "server_usage"
    LOCAL_TOKENIZER = "local_tokenizer"
    UNAVAILABLE = "unavailable"


class ValidationStatus(StrEnum):
    """Whether an experiment can support a performance conclusion."""

    VALID = "VALID"
    INVALID = "INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"
    ABORTED = "ABORTED"


class CheckStatus(StrEnum):
    """Outcome of one validation gate."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


class HardwareInfo(StrictModel):
    """Hardware fingerprint captured for an experiment."""

    cpu_model: str | None = None
    cpu_count: int | None = Field(default=None, ge=1)
    gpu_name: str | None = None
    gpu_uuid: str | None = None
    gpu_count: int | None = Field(default=None, ge=0)
    gpu_memory_total_bytes: int | None = Field(default=None, ge=0)
    driver_version: str | None = None


class SoftwareInfo(StrictModel):
    """Software fingerprint captured for an experiment."""

    python_version: str
    cuda_version: str | None = None
    pytorch_version: str | None = None
    vllm_version: str | None = None
    transformers_version: str | None = None
    numpy_version: str | None = None


class ExperimentManifest(StrictModel):
    """Identity and reproducibility metadata for one immutable run."""

    schema_version: str = "1.0"
    run_id: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime | None = None
    git_commit: str | None = None
    dirty_worktree: bool
    model_id: str = Field(min_length=1)
    model_revision: str | None = None
    backend: Backend
    hardware: HardwareInfo
    software: SoftwareInfo
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_wall_clock_order(self) -> Self:
        """Reject a finish timestamp earlier than the start timestamp."""
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if self.finished_at is not None:
            if self.finished_at.tzinfo is None:
                raise ValueError("finished_at must be timezone-aware")
            if self.finished_at < self.started_at:
                raise ValueError("finished_at must not precede started_at")
        return self


class RequestSample(StrictModel):
    """Raw monotonic timing and outcome for one measured request."""

    schema_version: str = "1.0"
    run_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    scheduled_at_ns: int = Field(ge=0)
    started_at_ns: int = Field(ge=0)
    first_content_at_ns: int | None = Field(default=None, ge=0)
    finished_at_ns: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    token_count_source: TokenCountSource = TokenCountSource.UNAVAILABLE
    chunk_times_ns: tuple[int, ...] = ()
    status: RequestStatus
    http_status: int | None = Field(default=None, ge=100, le=599)
    finish_reason: str | None = None
    error_code: str | None = None
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_monotonic_timestamps(self) -> Self:
        """Ensure request-local timestamps form a valid monotonic sequence."""
        if self.started_at_ns < self.scheduled_at_ns:
            raise ValueError("started_at_ns must not precede scheduled_at_ns")
        if self.first_content_at_ns is not None and self.first_content_at_ns < self.started_at_ns:
            raise ValueError("first_content_at_ns must not precede started_at_ns")
        if self.finished_at_ns is not None:
            if self.finished_at_ns < self.started_at_ns:
                raise ValueError("finished_at_ns must not precede started_at_ns")
            if (
                self.first_content_at_ns is not None
                and self.finished_at_ns < self.first_content_at_ns
            ):
                raise ValueError("finished_at_ns must not precede first_content_at_ns")
        previous = self.started_at_ns
        for timestamp in self.chunk_times_ns:
            if timestamp < previous:
                raise ValueError("chunk_times_ns must be monotonic and follow request start")
            if self.finished_at_ns is not None and timestamp > self.finished_at_ns:
                raise ValueError("chunk timestamp must not follow request finish")
            previous = timestamp
        return self


class ValidationCheck(StrictModel):
    """One auditable benchmark validity check."""

    name: str = Field(min_length=1)
    status: CheckStatus
    message: str = Field(min_length=1)
    value: float | int | None = None
    threshold: float | int | None = None


class ValidationReport(StrictModel):
    """Persisted result of all benchmark validity gates."""

    schema_version: str = "1.0"
    run_id: str = Field(min_length=1)
    status: ValidationStatus
    checks: tuple[ValidationCheck, ...]
    warnings: tuple[str, ...] = ()


class MetricDistribution(StrictModel):
    """Descriptive statistics using the project-wide linear quantile rule."""

    count: int = Field(ge=0)
    mean: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    minimum: float | None
    maximum: float | None
    p99_small_sample_warning: bool


class RequestCounts(StrictModel):
    """Scheduled, successful, and validation-eligible request counts."""

    scheduled: int = Field(ge=0)
    successful: int = Field(ge=0)
    valid: int = Field(ge=0)


class ThroughputSummary(StrictModel):
    """Rates whose denominator is the measured wall-clock window."""

    requests_per_second: float = Field(ge=0)
    input_tokens_per_second: float | None = Field(default=None, ge=0)
    output_tokens_per_second: float | None = Field(default=None, ge=0)
    goodput_requests_per_second: float | None = Field(default=None, ge=0)


class LatencySummary(StrictModel):
    """Aggregate request latency distributions in milliseconds."""

    ttft: MetricDistribution
    tpot: MetricDistribution
    e2e: MetricDistribution


class GpuSummary(StrictModel):
    """Optional aggregate GPU telemetry."""

    utilization_mean: float | None = Field(default=None, ge=0, le=100)
    memory_peak_bytes: int | None = Field(default=None, ge=0)
    power_mean_watts: float | None = Field(default=None, ge=0)


class AggregateReport(StrictModel):
    """Processed summary derived without modifying raw request samples."""

    schema_version: str = "1.0"
    run_id: str = Field(min_length=1)
    validation_status: ValidationStatus
    measured_seconds: float = Field(gt=0)
    requests: RequestCounts
    throughput: ThroughputSummary
    latency_ms: LatencySummary
    gpu: GpuSummary = GpuSummary()


class SloThresholds(StrictModel):
    """Latency and whole-run success-rate requirements."""

    ttft_p95_ms: float = Field(gt=0)
    tpot_p95_ms: float = Field(gt=0)
    success_rate_min: float = Field(ge=0, le=1)


class GoodputResult(StrictModel):
    """SLO-qualified request rate and its supporting counts."""

    scheduled_requests: int = Field(ge=0)
    successful_requests: int = Field(ge=0)
    eligible_requests: int = Field(ge=0)
    qualifying_requests: int = Field(ge=0)
    measured_seconds: float = Field(gt=0)
    success_rate: float = Field(ge=0, le=1)
    success_rate_slo_met: bool
    requests_per_second: float = Field(ge=0)


class ParetoPoint(StrictModel):
    """One validated configuration observation used for Pareto analysis."""

    observation_id: str = Field(min_length=1)
    validation_status: ValidationStatus
    goodput_requests_per_second: float = Field(ge=0)
    ttft_p95_ms: float | None = Field(default=None, ge=0)
    tpot_p95_ms: float | None = Field(default=None, ge=0)
    memory_peak_bytes: int | None = Field(default=None, ge=0)


class StabilityResult(StrictModel):
    """Repeat-run stability decision based on throughput CV."""

    repeat_count: int = Field(ge=0)
    mean_requests_per_second: float | None = Field(default=None, ge=0)
    standard_deviation: float | None = Field(default=None, ge=0)
    coefficient_of_variation: float | None = Field(default=None, ge=0)
    target_coefficient_of_variation: float = Field(ge=0)
    status: ValidationStatus
    message: str = Field(min_length=1)
