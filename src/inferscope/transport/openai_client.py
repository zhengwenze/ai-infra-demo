"""Async OpenAI-compatible streaming client with request-level timing."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import httpx

from inferscope.errors import ErrorCode, MalformedStreamError
from inferscope.transport.sse import SSEDecoder, SSEEvent

_ERROR_BODY_LIMIT_BYTES: Final = 4_096
_BEARER_PATTERN: Final = re.compile(r"(?i)(bearer\s+)[^\s,;\"']+")
_AUTH_HEADER_PATTERN: Final = re.compile(r"(?im)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+")


class StreamStatus(StrEnum):
    """Terminal status of one streaming request."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Text-only message accepted by the supported OpenAI API subset."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatCompletionRequest:
    """Input for one OpenAI-compatible streaming chat request."""

    request_id: str
    model: str
    messages: Sequence[ChatMessage]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    include_usage: bool = True
    ignore_eos: bool = False

    def to_payload(self) -> dict[str, object]:
        """Return the JSON payload without credentials or transport metadata."""
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in self.messages
            ],
            "stream": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream_options": {"include_usage": self.include_usage},
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.ignore_eos:
            payload["ignore_eos"] = True
        return payload


@dataclass(frozen=True, slots=True)
class Usage:
    """Server-reported token counts, when supplied by the backend."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class StreamResult:
    """Raw transport result for one request; no aggregate metrics are computed."""

    request_id: str
    started_ns: int
    first_content_ns: int | None
    finished_ns: int
    chunk_times_ns: tuple[int, ...]
    status: StreamStatus
    http_status: int | None
    finish_reason: str | None
    usage: Usage
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    error_body_excerpt: str | None = None

    @property
    def started_at_ns(self) -> int:
        """Compatibility alias matching the persisted request schema."""
        return self.started_ns

    @property
    def first_content_at_ns(self) -> int | None:
        """Compatibility alias matching the persisted request schema."""
        return self.first_content_ns

    @property
    def finished_at_ns(self) -> int:
        """Compatibility alias matching the persisted request schema."""
        return self.finished_ns

    @property
    def content(self) -> str:
        """Compatibility alias for callers using OpenAI's content terminology."""
        return self.text


@dataclass(slots=True)
class _StreamState:
    content_parts: list[str] = field(default_factory=list)
    chunk_times_ns: list[int] = field(default_factory=list)
    first_content_at_ns: int | None = None
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    done: bool = False


class OpenAIStreamingClient:
    """Send and time OpenAI-compatible chat-completion streams."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        connect_timeout_seconds: float = 10.0,
        max_event_bytes: int = 1_048_576,
        client: httpx.AsyncClient | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if timeout_seconds <= 0 or connect_timeout_seconds <= 0:
            msg = "timeout values must be greater than zero"
            raise ValueError(msg)
        self._api_key = api_key
        self._max_event_bytes = max_event_bytes
        self._clock_ns = clock_ns
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds),
        )

    async def __aenter__(self) -> OpenAIStreamingClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the internally-created HTTP client."""
        if self._owns_client:
            await self._client.aclose()

    async def stream_chat(self, request: ChatCompletionRequest) -> StreamResult:
        """Execute one stream and preserve protocol/timing evidence on failure."""
        started_at_ns = self._clock_ns()
        state = _StreamState()
        http_status: int | None = None
        try:
            headers = self._headers(request.request_id)
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=request.to_payload(),
                headers=headers,
            ) as response:
                http_status = response.status_code
                if response.is_error:
                    body = await _read_limited_body(response, max_bytes=_ERROR_BODY_LIMIT_BYTES)
                    return self._http_error_result(
                        request_id=request.request_id,
                        started_at_ns=started_at_ns,
                        http_status=response.status_code,
                        body=body,
                    )
                self._validate_content_type(response)
                await self._consume_stream(response=response, state=state)
            if not state.done and state.finish_reason is None:
                raise MalformedStreamError("stream ended before [DONE] or a valid finish_reason")
            return self._success_result(
                request_id=request.request_id,
                started_at_ns=started_at_ns,
                http_status=http_status,
                state=state,
            )
        except asyncio.CancelledError:
            raise
        except (httpx.TimeoutException, TimeoutError) as exc:
            return self._failure_result(
                request_id=request.request_id,
                started_at_ns=started_at_ns,
                http_status=http_status,
                state=state,
                status=StreamStatus.TIMEOUT,
                error_code=ErrorCode.REQUEST_TIMEOUT,
                error_message=self._sanitize(str(exc) or "request timed out"),
            )
        except MalformedStreamError as exc:
            return self._failure_result(
                request_id=request.request_id,
                started_at_ns=started_at_ns,
                http_status=http_status,
                state=state,
                status=StreamStatus.ERROR,
                error_code=ErrorCode.STREAM_MALFORMED,
                error_message=self._sanitize(str(exc)),
            )
        except httpx.RequestError as exc:
            return self._failure_result(
                request_id=request.request_id,
                started_at_ns=started_at_ns,
                http_status=http_status,
                state=state,
                status=StreamStatus.ERROR,
                error_code=ErrorCode.TARGET_UNAVAILABLE,
                error_message=self._sanitize(str(exc)),
            )

    def _headers(self, request_id: str) -> dict[str, str]:
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _consume_stream(self, *, response: httpx.Response, state: _StreamState) -> None:
        decoder = SSEDecoder(max_event_bytes=self._max_event_bytes)
        async for chunk in response.aiter_bytes():
            received_at_ns = self._clock_ns()
            for event in decoder.feed(chunk, received_at_ns=received_at_ns):
                self._consume_event(event=event, state=state)
                if state.done:
                    return
        decoder.finalize()

    @staticmethod
    def _consume_event(*, event: SSEEvent, state: _StreamState) -> None:
        if event.data.strip() == "[DONE]":
            state.done = True
            return
        try:
            payload = json.loads(event.data)
        except json.JSONDecodeError as exc:
            raise MalformedStreamError("SSE data is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise MalformedStreamError("SSE JSON data must be an object")

        _consume_usage(payload, state)
        choices = payload.get("choices")
        if choices is None:
            return
        if not isinstance(choices, list):
            raise MalformedStreamError("SSE choices must be a list")
        for choice in choices:
            if not isinstance(choice, dict):
                raise MalformedStreamError("SSE choice must be an object")
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                state.finish_reason = finish_reason
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if not isinstance(content, str) or not content:
                continue
            state.content_parts.append(content)
            state.chunk_times_ns.append(event.received_at_ns)
            if state.first_content_at_ns is None:
                state.first_content_at_ns = event.received_at_ns

    @staticmethod
    def _validate_content_type(response: httpx.Response) -> None:
        content_type = response.headers.get("content-type")
        if content_type and "text/event-stream" not in content_type.lower():
            raise MalformedStreamError(
                f"expected text/event-stream response, received {content_type!r}"
            )

    def _success_result(
        self,
        *,
        request_id: str,
        started_at_ns: int,
        http_status: int,
        state: _StreamState,
    ) -> StreamResult:
        return StreamResult(
            request_id=request_id,
            started_ns=started_at_ns,
            first_content_ns=state.first_content_at_ns,
            finished_ns=self._clock_ns(),
            chunk_times_ns=tuple(state.chunk_times_ns),
            status=StreamStatus.SUCCESS,
            http_status=http_status,
            finish_reason=state.finish_reason,
            usage=state.usage,
            text="".join(state.content_parts),
            input_tokens=state.usage.prompt_tokens,
            output_tokens=state.usage.completion_tokens,
        )

    def _failure_result(
        self,
        *,
        request_id: str,
        started_at_ns: int,
        http_status: int | None,
        state: _StreamState,
        status: StreamStatus,
        error_code: ErrorCode,
        error_message: str,
        error_body_excerpt: str | None = None,
    ) -> StreamResult:
        return StreamResult(
            request_id=request_id,
            started_ns=started_at_ns,
            first_content_ns=state.first_content_at_ns,
            finished_ns=self._clock_ns(),
            chunk_times_ns=tuple(state.chunk_times_ns),
            status=status,
            http_status=http_status,
            finish_reason=state.finish_reason,
            usage=state.usage,
            text="".join(state.content_parts),
            input_tokens=state.usage.prompt_tokens,
            output_tokens=state.usage.completion_tokens,
            error_code=error_code,
            error_message=error_message,
            error_body_excerpt=error_body_excerpt,
        )

    def _http_error_result(
        self,
        *,
        request_id: str,
        started_at_ns: int,
        http_status: int,
        body: bytes,
    ) -> StreamResult:
        code = _http_error_code(http_status)
        excerpt = self._sanitize(_decode_body_excerpt(body, _ERROR_BODY_LIMIT_BYTES))
        return self._failure_result(
            request_id=request_id,
            started_at_ns=started_at_ns,
            http_status=http_status,
            state=_StreamState(),
            status=StreamStatus.ERROR,
            error_code=code,
            error_message=f"HTTP {http_status} from inference target",
            error_body_excerpt=excerpt,
        )

    def _sanitize(self, value: str) -> str:
        sanitized = redact_bearer_tokens(value)
        if self._api_key:
            sanitized = sanitized.replace(self._api_key, "[REDACTED]")
        return sanitized


def _consume_usage(payload: Mapping[str, object], state: _StreamState) -> None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return
    state.usage = Usage(
        prompt_tokens=_optional_nonnegative_int(usage.get("prompt_tokens")),
        completion_tokens=_optional_nonnegative_int(usage.get("completion_tokens")),
        total_tokens=_optional_nonnegative_int(usage.get("total_tokens")),
    )


def _optional_nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _http_error_code(status_code: int) -> ErrorCode:
    if status_code in {408, 504}:
        return ErrorCode.REQUEST_TIMEOUT
    if status_code == 429:
        return ErrorCode.RESOURCE_EXHAUSTED
    if 400 <= status_code < 500:
        return ErrorCode.CONFIG_INVALID
    return ErrorCode.TARGET_UNAVAILABLE


async def _read_limited_body(response: httpx.Response, *, max_bytes: int) -> bytes:
    collected = bytearray()
    async for chunk in response.aiter_bytes():
        remaining = max_bytes + 1 - len(collected)
        if remaining <= 0:
            break
        collected.extend(chunk[:remaining])
        if len(collected) > max_bytes:
            break
    return bytes(collected)


def _decode_body_excerpt(body: bytes, max_bytes: int) -> str:
    excerpt = body[:max_bytes]
    text = excerpt.decode("utf-8", errors="replace")
    if len(body) > max_bytes:
        text += "...[truncated]"
    return text


def redact_bearer_tokens(value: str) -> str:
    """Redact Bearer credentials and Authorization header values from text."""
    value = _AUTH_HEADER_PATTERN.sub(r"\1[REDACTED]", value)
    return _BEARER_PATTERN.sub(r"\1[REDACTED]", value)
