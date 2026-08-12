"""Deterministic synthetic request descriptions."""

from __future__ import annotations

import hashlib
import random
import uuid
from dataclasses import dataclass

_REQUEST_NAMESPACE = uuid.UUID("84f8331f-a4dd-4c45-ad17-fdb79fb17111")
_DEFAULT_VOCABULARY = (
    "attention",
    "batch",
    "cache",
    "decode",
    "gpu",
    "inference",
    "latency",
    "prefill",
    "request",
    "throughput",
)


@dataclass(frozen=True, slots=True)
class SyntheticRequest:
    """A transport-neutral generated request and its intended token budget.

    ``target_input_tokens`` describes the generator target, not a verified
    tokenizer count. The runner must still tokenize with the target model when
    exact token counts are required.
    """

    request_id: str
    sequence: int
    prompt: str
    target_input_tokens: int
    target_output_tokens: int
    workload_name: str = "synthetic"
    shared_prefix_id: str | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.target_input_tokens <= 0:
            raise ValueError("target_input_tokens must be greater than zero")
        if self.target_output_tokens <= 0:
            raise ValueError("target_output_tokens must be greater than zero")
        if not self.prompt:
            raise ValueError("prompt must not be empty")

    @property
    def prompt_sha256(self) -> str:
        """Return a content fingerprint suitable for persisted artifacts."""

        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


def _validate_request_parameters(
    num_requests: int,
    prompt_tokens: int,
    output_tokens: int,
    vocabulary: tuple[str, ...],
) -> None:
    if num_requests <= 0:
        raise ValueError("num_requests must be greater than zero")
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be greater than zero")
    if output_tokens <= 0:
        raise ValueError("output_tokens must be greater than zero")
    if not vocabulary or any(not token.strip() for token in vocabulary):
        raise ValueError("vocabulary must contain only non-empty strings")


def _deterministic_request_id(seed: int, workload_name: str, sequence: int) -> str:
    identity = f"{seed}:{workload_name}:{sequence}"
    return str(uuid.uuid5(_REQUEST_NAMESPACE, identity))


def _generate_words(random_source: random.Random, count: int, vocabulary: tuple[str, ...]) -> str:
    return " ".join(random_source.choice(vocabulary) for _ in range(count))


def build_synthetic_requests(
    num_requests: int,
    prompt_tokens: int,
    output_tokens: int,
    *,
    seed: int,
    vocabulary: tuple[str, ...] = _DEFAULT_VOCABULARY,
    workload_name: str = "synthetic",
) -> tuple[SyntheticRequest, ...]:
    """Generate reproducible synthetic prompts without changing global RNG state."""

    _validate_request_parameters(num_requests, prompt_tokens, output_tokens, vocabulary)
    if not workload_name:
        raise ValueError("workload_name must not be empty")
    random_source = random.Random(seed)
    return tuple(
        SyntheticRequest(
            request_id=_deterministic_request_id(seed, workload_name, sequence),
            sequence=sequence,
            prompt=_generate_words(random_source, prompt_tokens, vocabulary),
            target_input_tokens=prompt_tokens,
            target_output_tokens=output_tokens,
            workload_name=workload_name,
        )
        for sequence in range(num_requests)
    )
