"""Domain errors and stable error codes used across InferScope."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable machine-readable error codes persisted in experiment artifacts."""

    CONFIG_INVALID = "IS_CONFIG_INVALID"
    TARGET_UNAVAILABLE = "IS_TARGET_UNAVAILABLE"
    REQUEST_TIMEOUT = "IS_REQUEST_TIMEOUT"
    STREAM_MALFORMED = "IS_STREAM_MALFORMED"
    TOKEN_COUNT_MISMATCH = "IS_TOKEN_COUNT_MISMATCH"
    OUTPUT_LENGTH_INVALID = "IS_OUTPUT_LENGTH_INVALID"
    WARMUP_FAILED = "IS_WARMUP_FAILED"
    CACHE_STATE_UNKNOWN = "IS_CACHE_STATE_UNKNOWN"
    CLIENT_SATURATED = "IS_CLIENT_SATURATED"
    GPU_TELEMETRY_UNAVAILABLE = "IS_GPU_TELEMETRY_UNAVAILABLE"
    RESOURCE_EXHAUSTED = "IS_RESOURCE_EXHAUSTED"
    EXPERIMENT_INCONCLUSIVE = "IS_EXPERIMENT_INCONCLUSIVE"


class InferScopeError(Exception):
    """Base exception carrying a stable error code."""

    def __init__(self, message: str, code: ErrorCode) -> None:
        super().__init__(message)
        self.code = code


class ConfigurationError(InferScopeError):
    """Raised when an experiment configuration is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.CONFIG_INVALID)


class TargetUnavailableError(InferScopeError):
    """Raised when an inference endpoint does not become ready."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.TARGET_UNAVAILABLE)


class MalformedStreamError(InferScopeError):
    """Raised when a streaming response violates the supported SSE contract."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCode.STREAM_MALFORMED)
