"""Contract tests for byte-oriented SSE framing."""

from __future__ import annotations

import pytest

from inferscope.errors import MalformedStreamError
from inferscope.transport.sse import SSEDecoder


def test_decoder_handles_every_utf8_byte_as_a_separate_network_fragment() -> None:
    decoder = SSEDecoder()
    wire = 'data: {"content":"缓存"}\n\n'.encode()

    events = []
    for timestamp, byte in enumerate(wire, start=1):
        events.extend(decoder.feed(bytes([byte]), received_at_ns=timestamp))

    decoder.finalize()
    assert [event.data for event in events] == ['{"content":"缓存"}']
    assert events[0].received_at_ns == len(wire)


def test_decoder_returns_multiple_events_from_one_network_fragment() -> None:
    decoder = SSEDecoder()

    events = decoder.feed(b"data:first\n\ndata: second\n\ndata: [DONE]\n\n", received_at_ns=42)

    assert [event.data for event in events] == ["first", "second", "[DONE]"]
    assert [event.received_at_ns for event in events] == [42, 42, 42]


def test_decoder_uses_empty_lines_not_network_fragments_as_event_boundaries() -> None:
    decoder = SSEDecoder()

    assert decoder.feed(b"data: hel", received_at_ns=1) == []
    assert decoder.feed(b"lo\n", received_at_ns=2) == []
    events = decoder.feed(b"\n", received_at_ns=3)

    assert len(events) == 1
    assert events[0].data == "hello"
    assert events[0].received_at_ns == 3


def test_decoder_supports_crlf_split_across_network_fragments() -> None:
    decoder = SSEDecoder()

    assert decoder.feed(b"data: hello\r", received_at_ns=1) == []
    assert decoder.feed(b"\n\r", received_at_ns=2) == []
    events = decoder.feed(b"\n", received_at_ns=3)

    assert [event.data for event in events] == ["hello"]


def test_decoder_ignores_comments_keepalives_and_unknown_fields() -> None:
    decoder = SSEDecoder()

    events = decoder.feed(
        b": keepalive\n\n"
        b"event: message\n"
        b"id: 7\n"
        b"retry: 250\n"
        b"unknown: ignored\n"
        b"data: first\n"
        b"data:second\n\n",
        received_at_ns=10,
    )

    assert len(events) == 1
    assert events[0].data == "first\nsecond"
    assert events[0].event == "message"
    assert events[0].event_id == "7"
    assert events[0].retry_ms == 250


def test_decoder_rejects_invalid_utf8_only_after_event_is_complete() -> None:
    decoder = SSEDecoder()

    assert decoder.feed(b"data: \xe4", received_at_ns=1) == []
    with pytest.raises(MalformedStreamError, match="invalid UTF-8"):
        decoder.feed(b"\n\n", received_at_ns=2)


def test_decoder_rejects_oversized_event_before_unbounded_buffering() -> None:
    decoder = SSEDecoder(max_event_bytes=8)

    with pytest.raises(MalformedStreamError, match="exceeds 8 byte"):
        decoder.feed(b"data: 123456", received_at_ns=1)


def test_decoder_rejects_eof_inside_an_event() -> None:
    decoder = SSEDecoder()
    decoder.feed(b"data: unfinished\n", received_at_ns=1)

    with pytest.raises(MalformedStreamError, match="ended inside an event"):
        decoder.finalize()
