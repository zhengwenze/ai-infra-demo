"""Contract tests for the OpenAI-compatible asynchronous stream client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Sequence

import httpx
import pytest

from inferscope.errors import ErrorCode
from inferscope.transport.openai_client import (
    ChatCompletionRequest,
    ChatMessage,
    OpenAIStreamingClient,
    StreamStatus,
    redact_bearer_tokens,
)


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: Sequence[bytes],
        *,
        terminal_error: BaseException | None = None,
    ) -> None:
        self._chunks = chunks
        self._terminal_error = terminal_error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._terminal_error is not None:
            raise self._terminal_error


class _Clock:
    def __init__(self, values: Sequence[int]) -> None:
        self._values: Iterator[int] = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        request_id="request-7",
        model="Qwen/test-model",
        messages=(ChatMessage(role="user", content="Explain KV cache."),),
        max_tokens=8,
        seed=17,
    )


def test_request_emits_vllm_ignore_eos_only_when_enabled() -> None:
    default_payload = _request().to_payload()
    enabled = ChatCompletionRequest(
        request_id="request-ignore-eos",
        model="demo-model",
        messages=(ChatMessage(role="user", content="hello"),),
        max_tokens=8,
        ignore_eos=True,
    ).to_payload()
    assert "ignore_eos" not in default_payload
    assert enabled["ignore_eos"] is True


def _response(
    chunks: Sequence[bytes],
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    terminal_error: BaseException | None = None,
) -> httpx.Response:
    response_headers = {"content-type": "text/event-stream"}
    if headers:
        response_headers.update(headers)
    return httpx.Response(
        status_code,
        headers=response_headers,
        stream=_ChunkStream(chunks, terminal_error=terminal_error),
    )


@pytest.mark.contract
async def test_stream_chat_records_only_nonempty_content_as_ttft() -> None:
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return _response(
            [
                b'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
                b'data: {"choices":[{"delta":{"content":""},"finish_reason":null}]}\n\n',
                (
                    'data: {"choices":[{"delta":{"content":"缓存"},"finish_reason":null}]}\n\n'
                ).encode(),
                b'data: {"choices":[{"delta":{"content":" works"},"finish_reason":null}]}\n\n',
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n',
                b"data: [DONE]\n\n",
            ]
        )

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as http_client:
        client = OpenAIStreamingClient(
            base_url="http://test",
            api_key="top-secret",
            client=http_client,
            clock_ns=_Clock([100, 200, 300, 400, 500, 600, 700, 800]),
        )
        result = await client.stream_chat(_request())

    assert result.status is StreamStatus.SUCCESS
    assert result.started_ns == 100
    assert result.first_content_ns == 400
    assert result.chunk_times_ns == (400, 500)
    assert result.finished_ns == 800
    assert result.text == "缓存 works"
    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert result.finish_reason == "stop"
    assert seen_request is not None
    assert seen_request.headers["x-request-id"] == "request-7"
    assert seen_request.headers["authorization"] == "Bearer top-secret"
    payload = json.loads(seen_request.content)
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


@pytest.mark.contract
async def test_stream_chat_handles_multiple_events_and_utf8_across_fragments() -> None:
    wire = (
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        + 'data: {"choices":[{"delta":{"content":"推理"}}]}\n\n'.encode()
        + b"data: [DONE]\n\n"
    )
    chinese_start = wire.index("推".encode())
    chunks = [
        wire[: chinese_start + 1],
        wire[chinese_start + 1 : chinese_start + 2],
        wire[chinese_start + 2 :],
    ]

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(lambda _: _response(chunks))
    ) as http_client:
        client = OpenAIStreamingClient(
            base_url="http://test",
            client=http_client,
            clock_ns=_Clock([10, 20, 30, 40, 50]),
        )
        result = await client.stream_chat(_request())

    assert result.status is StreamStatus.SUCCESS
    assert result.text == "推理"
    assert result.first_content_ns == 40


@pytest.mark.contract
async def test_stream_chat_accepts_eof_after_valid_finish_reason() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n',
    ]
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(lambda _: _response(chunks))
    ) as http_client:
        client = OpenAIStreamingClient(
            base_url="http://test", client=http_client, clock_ns=_Clock([1, 2, 3, 4])
        )
        result = await client.stream_chat(_request())

    assert result.status is StreamStatus.SUCCESS
    assert result.finish_reason == "length"


@pytest.mark.contract
async def test_stream_chat_standardizes_interrupted_stream_and_preserves_partial_timing() -> None:
    chunks = [b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n']
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(lambda _: _response(chunks))
    ) as http_client:
        client = OpenAIStreamingClient(
            base_url="http://test", client=http_client, clock_ns=_Clock([10, 20, 30])
        )
        result = await client.stream_chat(_request())

    assert result.status is StreamStatus.ERROR
    assert result.error_code is ErrorCode.STREAM_MALFORMED
    assert result.first_content_ns == 20
    assert result.text == "partial"
    assert result.finished_ns == 30


@pytest.mark.contract
async def test_stream_chat_standardizes_timeout() -> None:
    timeout = httpx.ReadTimeout("Bearer leaked-timeout-token")
    response = _response([], terminal_error=timeout)
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(lambda _: response)
    ) as http_client:
        client = OpenAIStreamingClient(
            base_url="http://test",
            api_key="leaked-timeout-token",
            client=http_client,
            clock_ns=_Clock([10, 20]),
        )
        result = await client.stream_chat(_request())

    assert result.status is StreamStatus.TIMEOUT
    assert result.error_code is ErrorCode.REQUEST_TIMEOUT
    assert result.error_message is not None
    assert "leaked-timeout-token" not in result.error_message


@pytest.mark.contract
@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [
        (400, ErrorCode.CONFIG_INVALID),
        (429, ErrorCode.RESOURCE_EXHAUSTED),
        (500, ErrorCode.TARGET_UNAVAILABLE),
        (504, ErrorCode.REQUEST_TIMEOUT),
    ],
)
async def test_stream_chat_standardizes_http_errors_and_redacts_body(
    status_code: int, error_code: ErrorCode
) -> None:
    secret = "sensitive-token"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            text=f"Authorization: Bearer {secret}; detail={secret}",
        )

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as http_client:
        client = OpenAIStreamingClient(
            base_url="http://test",
            api_key=secret,
            client=http_client,
            clock_ns=_Clock([1, 2]),
        )
        result = await client.stream_chat(_request())

    assert result.status is StreamStatus.ERROR
    assert result.http_status == status_code
    assert result.error_code is error_code
    assert result.error_body_excerpt is not None
    assert secret not in result.error_body_excerpt


@pytest.mark.contract
async def test_stream_chat_truncates_large_http_error_body() -> None:
    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(lambda _: httpx.Response(500, content=b"x" * 5_000)),
    ) as http_client:
        client = OpenAIStreamingClient(
            base_url="http://test", client=http_client, clock_ns=_Clock([1, 2])
        )
        result = await client.stream_chat(_request())

    assert result.error_body_excerpt is not None
    assert result.error_body_excerpt.endswith("...[truncated]")
    assert len(result.error_body_excerpt) == 4_096 + len("...[truncated]")


@pytest.mark.contract
async def test_stream_chat_rejects_malformed_json() -> None:
    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(lambda _: _response([b"data: {bad-json}\n\n"])),
    ) as http_client:
        client = OpenAIStreamingClient(
            base_url="http://test", client=http_client, clock_ns=_Clock([1, 2, 3])
        )
        result = await client.stream_chat(_request())

    assert result.status is StreamStatus.ERROR
    assert result.error_code is ErrorCode.STREAM_MALFORMED


@pytest.mark.contract
async def test_stream_chat_propagates_cancellation() -> None:
    response = _response([], terminal_error=asyncio.CancelledError())
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(lambda _: response)
    ) as http_client:
        client = OpenAIStreamingClient(base_url="http://test", client=http_client)
        with pytest.raises(asyncio.CancelledError):
            await client.stream_chat(_request())


def test_redact_bearer_tokens_removes_credentials_from_free_text() -> None:
    message = "Authorization: Bearer abc123\nupstream said bearer second-secret"

    redacted = redact_bearer_tokens(message)

    assert "abc123" not in redacted
    assert "second-secret" not in redacted
    assert redacted.count("[REDACTED]") == 2
