from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from inferscope.config import ExperimentConfig
from inferscope.runner import ExperimentRunner
from inferscope.transport import OpenAIStreamingClient


def config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "runner-test",
            "seed": 11,
            "target": {
                "backend": "vllm",
                "base_url": "http://test",
                "model": "fake-model",
                "timeout_seconds": 5.0,
            },
            "generation": {"max_output_tokens": 3},
            "workload": {
                "type": "synthetic",
                "prompt_tokens": 8,
                "output_tokens": 3,
                "num_requests": 3,
                "arrival": {"mode": "concurrency", "values": (2,)},
            },
            "warmup": {"requests": 1, "include_in_metrics": False},
            "validation": {
                "min_success_rate": 1.0,
                "output_token_tolerance_ratio": 0.0,
                "token_count_mismatch_ratio": 0.02,
                "max_client_loop_lag_ms": 20.0,
                "require_clean_gpu": True,
            },
            "slo": {
                "ttft_p95_ms": 500.0,
                "tpot_p95_ms": 500.0,
                "success_rate_min": 1.0,
            },
            "execution": {
                "repeats": 1,
                "cooldown_seconds": 0.0,
                "max_matrix_combinations": 4,
            },
            "output": {
                "save_prompts": False,
                "save_responses": False,
                "formats": ("json", "csv", "markdown"),
            },
        },
        strict=True,
    )


@pytest.mark.integration
async def test_runner_persists_raw_processed_and_report_artifacts(tmp_path: Path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers.get("authorization") is None
        body = (
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" world"},'
            '"finish_reason":"length"}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":8,'
            '"completion_tokens":3,"total_tokens":11}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    async_client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    transport = OpenAIStreamingClient(base_url="http://test", client=async_client)
    runner = ExperimentRunner(
        config(),
        results_dir=tmp_path / "results",
        project_dir=tmp_path,
        client=transport,
    )
    outcomes = await runner.run()
    await async_client.aclose()

    assert calls == 4  # one warmup and three measured requests
    assert len(outcomes) == 1
    outcome = outcomes[0]
    request_lines = (outcome.run_dir / "requests.jsonl").read_text().splitlines()
    assert len(request_lines) == 3
    assert all("prompt" not in json.loads(line) for line in request_lines)
    assert (outcome.run_dir / "manifest.json").exists()
    assert (outcome.run_dir / "validation.json").exists()
    assert (tmp_path / "results" / "processed" / outcome.run_id / "aggregate.json").exists()
    assert outcome.report_path.exists()
    assert "INCONCLUSIVE" in outcome.report_path.read_text(encoding="utf-8")
