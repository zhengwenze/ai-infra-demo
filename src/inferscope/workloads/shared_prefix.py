"""Shared-prefix workload construction for prefix-cache experiments."""

from __future__ import annotations

import hashlib
import random

from inferscope.workloads.synthetic import (
    _DEFAULT_VOCABULARY,
    SyntheticRequest,
    _deterministic_request_id,
    _generate_words,
    _validate_request_parameters,
)


def build_shared_prefix_requests(
    num_requests: int,
    shared_prefix_tokens: int,
    unique_suffix_tokens: int,
    output_tokens: int,
    *,
    seed: int,
    vocabulary: tuple[str, ...] = _DEFAULT_VOCABULARY,
    workload_name: str = "shared_prefix",
) -> tuple[SyntheticRequest, ...]:
    """Generate requests with one byte-identical prefix and unique suffixes.

    Token values are generation targets until verified by the model tokenizer.
    The prefix fingerprint lets artifacts prove which requests shared a prefix
    without persisting the prefix itself.
    """

    _validate_request_parameters(
        num_requests,
        shared_prefix_tokens + unique_suffix_tokens,
        output_tokens,
        vocabulary,
    )
    if shared_prefix_tokens <= 0:
        raise ValueError("shared_prefix_tokens must be greater than zero")
    if unique_suffix_tokens <= 0:
        raise ValueError("unique_suffix_tokens must be greater than zero")
    if not workload_name:
        raise ValueError("workload_name must not be empty")

    random_source = random.Random(seed)
    prefix = _generate_words(random_source, shared_prefix_tokens, vocabulary)
    prefix_id = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    target_input_tokens = shared_prefix_tokens + unique_suffix_tokens
    requests: list[SyntheticRequest] = []
    for sequence in range(num_requests):
        suffix = _generate_words(random_source, unique_suffix_tokens, vocabulary)
        requests.append(
            SyntheticRequest(
                request_id=_deterministic_request_id(seed, workload_name, sequence),
                sequence=sequence,
                prompt=f"{prefix} {suffix}",
                target_input_tokens=target_input_tokens,
                target_output_tokens=output_tokens,
                workload_name=workload_name,
                shared_prefix_id=prefix_id,
            )
        )
    return tuple(requests)
