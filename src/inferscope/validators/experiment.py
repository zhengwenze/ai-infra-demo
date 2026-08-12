"""Deterministic validity gates for persisted benchmark samples."""

from __future__ import annotations

from collections.abc import Sequence

from inferscope.config import ExperimentConfig
from inferscope.models import (
    CheckStatus,
    RequestSample,
    RequestStatus,
    TokenCountSource,
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)


def _check(
    name: str,
    status: CheckStatus,
    message: str,
    *,
    value: float | int | None = None,
    threshold: float | int | None = None,
) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        status=status,
        message=message,
        value=value,
        threshold=threshold,
    )


def validate_experiment(
    run_id: str,
    samples: Sequence[RequestSample],
    config: ExperimentConfig,
    *,
    warmup_completed: bool,
    max_loop_lag_ms: float | None,
    gpu_is_clean: bool | None,
) -> ValidationReport:
    """Evaluate benchmark gates without network access or artifact mutation."""
    checks: list[ValidationCheck] = []
    warnings: list[str] = []

    checks.append(
        _check(
            "warmup_excluded",
            CheckStatus.PASS if warmup_completed else CheckStatus.FAIL,
            (
                f"{config.warmup.requests} warmup requests completed and were excluded"
                if warmup_completed
                else "warmup requests did not complete successfully"
            ),
        )
    )

    scheduled = len(samples)
    successful = [sample for sample in samples if sample.status is RequestStatus.SUCCESS]
    success_rate = len(successful) / scheduled if scheduled else 0.0
    success_ok = scheduled > 0 and success_rate >= config.validation.min_success_rate
    checks.append(
        _check(
            "success_rate",
            CheckStatus.PASS if success_ok else CheckStatus.FAIL,
            f"{len(successful)}/{scheduled} measured requests succeeded",
            value=success_rate,
            threshold=config.validation.min_success_rate,
        )
    )

    invalid_timing = [
        sample
        for sample in successful
        if sample.finished_at_ns is None or sample.first_content_at_ns is None
    ]
    checks.append(
        _check(
            "timing_complete",
            CheckStatus.PASS if not invalid_timing else CheckStatus.FAIL,
            (
                "all successful requests contain first-content and finish timestamps"
                if not invalid_timing
                else f"{len(invalid_timing)} successful requests have incomplete timing"
            ),
            value=len(invalid_timing),
            threshold=0,
        )
    )

    expected_tokens = config.workload.output_tokens
    tolerance = config.validation.output_token_tolerance_ratio
    lower_bound = expected_tokens * (1.0 - tolerance)
    upper_bound = expected_tokens * (1.0 + tolerance)
    invalid_lengths = [
        sample
        for sample in successful
        if sample.output_tokens is None or not lower_bound <= sample.output_tokens <= upper_bound
    ]
    checks.append(
        _check(
            "output_length",
            CheckStatus.PASS if not invalid_lengths else CheckStatus.FAIL,
            (
                "successful outputs are within the configured token tolerance"
                if not invalid_lengths
                else f"{len(invalid_lengths)} successful outputs are missing or out of range"
            ),
            value=len(invalid_lengths),
            threshold=0,
        )
    )

    unknown_token_counts = sum(
        sample.token_count_source is TokenCountSource.UNAVAILABLE for sample in successful
    )
    token_status = CheckStatus.PASS if unknown_token_counts == 0 else CheckStatus.WARN
    checks.append(
        _check(
            "token_count_available",
            token_status,
            (
                "all successful requests have an identified token-count source"
                if unknown_token_counts == 0
                else f"{unknown_token_counts} successful requests lack trustworthy token counts"
            ),
            value=unknown_token_counts,
            threshold=0,
        )
    )
    if unknown_token_counts:
        warnings.append("TPOT and token throughput are inconclusive without token counts")

    lag_threshold = config.validation.max_client_loop_lag_ms
    if max_loop_lag_ms is None:
        lag_status = CheckStatus.WARN
        lag_message = "client event-loop lag was not captured"
        warnings.append("client capacity could not be verified")
    elif max_loop_lag_ms <= lag_threshold:
        lag_status = CheckStatus.PASS
        lag_message = "client event-loop lag stayed within the configured threshold"
    else:
        lag_status = CheckStatus.WARN
        lag_message = "client event-loop lag exceeded the configured threshold"
        warnings.append("the load generator may be the bottleneck")
    checks.append(
        _check(
            "client_capacity",
            lag_status,
            lag_message,
            value=max_loop_lag_ms,
            threshold=lag_threshold,
        )
    )

    if not config.validation.require_clean_gpu:
        gpu_status = CheckStatus.PASS
        gpu_message = "clean-GPU isolation was not required by this exploratory config"
    elif gpu_is_clean is True:
        gpu_status = CheckStatus.PASS
        gpu_message = "no unknown GPU-sharing process was observed"
    else:
        gpu_status = CheckStatus.WARN
        gpu_message = (
            "GPU process isolation could not be verified"
            if gpu_is_clean is None
            else "another process shared the benchmark GPU"
        )
        warnings.append("GPU isolation is not proven; treat comparisons as exploratory")
    checks.append(_check("gpu_isolation", gpu_status, gpu_message))

    if any(check.status is CheckStatus.FAIL for check in checks):
        status = ValidationStatus.INVALID
    elif any(check.status is CheckStatus.WARN for check in checks):
        status = ValidationStatus.INCONCLUSIVE
    else:
        status = ValidationStatus.VALID
    return ValidationReport(
        run_id=run_id,
        status=status,
        checks=tuple(checks),
        warnings=tuple(warnings),
    )
