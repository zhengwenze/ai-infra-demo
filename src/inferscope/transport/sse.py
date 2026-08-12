"""Incremental Server-Sent Events parsing for arbitrary network fragments."""

from __future__ import annotations

from dataclasses import dataclass

from inferscope.errors import MalformedStreamError


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """One complete SSE event.

    ``received_at_ns`` is the monotonic timestamp of the network fragment that
    completed the event boundary. It is intentionally not a wall-clock value.
    """

    data: str
    received_at_ns: int
    event: str | None = None
    event_id: str | None = None
    retry_ms: int | None = None


class SSEDecoder:
    """Decode SSE events incrementally without assuming TCP chunk boundaries.

    Input remains bytes until a complete SSE line is available, so a UTF-8
    codepoint may be split at any byte boundary. Empty events (including pure
    comments and keep-alives) are ignored.
    """

    def __init__(self, *, max_event_bytes: int = 1_048_576) -> None:
        if max_event_bytes <= 0:
            msg = "max_event_bytes must be greater than zero"
            raise ValueError(msg)
        self._max_event_bytes = max_event_bytes
        self._buffer = bytearray()
        self._event_lines: list[bytes] = []
        self._event_bytes = 0

    @property
    def has_pending_data(self) -> bool:
        """Return whether the stream ended with an incomplete line or event."""
        return bool(self._buffer or self._event_lines)

    def feed(self, chunk: bytes, *, received_at_ns: int) -> list[SSEEvent]:
        """Consume one network fragment and return every completed event.

        Args:
            chunk: Raw response bytes. Empty fragments are accepted.
            received_at_ns: Monotonic time at which this fragment was observed.

        Raises:
            MalformedStreamError: If an event is too large or contains invalid
                UTF-8/field values.
        """
        if not chunk:
            return []

        self._buffer.extend(chunk)
        events: list[SSEEvent] = []
        while (line_end := self._find_line_end()) is not None:
            line, delimiter_size = line_end
            del self._buffer[: len(line) + delimiter_size]
            if line:
                self._append_line(line)
                continue
            event = self._dispatch(received_at_ns=received_at_ns)
            if event is not None:
                events.append(event)

        self._check_size(len(self._buffer))
        return events

    def finalize(self) -> None:
        """Validate that EOF did not split an SSE line or event.

        SSE events are dispatched only by an empty line. Silently accepting a
        trailing partial event would move TTFT to an invalid protocol boundary.
        """
        if self.has_pending_data:
            raise MalformedStreamError("SSE stream ended inside an event")

    def _find_line_end(self) -> tuple[bytes, int] | None:
        for index, value in enumerate(self._buffer):
            if value == 0x0A:  # LF, including CRLF
                if index > 0 and self._buffer[index - 1] == 0x0D:
                    return bytes(self._buffer[: index - 1]), 2
                return bytes(self._buffer[:index]), 1
            if value == 0x0D:  # Bare CR. Wait if it may become CRLF.
                if index + 1 == len(self._buffer):
                    return None
                delimiter_size = 2 if self._buffer[index + 1] == 0x0A else 1
                return bytes(self._buffer[:index]), delimiter_size
        return None

    def _append_line(self, line: bytes) -> None:
        self._check_size(len(line) + 1)
        self._event_lines.append(line)
        self._event_bytes += len(line) + 1

    def _check_size(self, additional_bytes: int) -> None:
        if self._event_bytes + additional_bytes > self._max_event_bytes:
            raise MalformedStreamError(f"SSE event exceeds {self._max_event_bytes} byte limit")

    def _dispatch(self, *, received_at_ns: int) -> SSEEvent | None:
        lines = self._event_lines
        self._event_lines = []
        self._event_bytes = 0
        if not lines:
            return None

        data_lines: list[str] = []
        event_type: str | None = None
        event_id: str | None = None
        retry_ms: int | None = None
        for raw_line in lines:
            line = self._decode_line(raw_line)
            if line.startswith(":"):
                continue
            field, value = _split_field(line)
            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event_type = value
            elif field == "id" and "\x00" not in value:
                event_id = value
            elif field == "retry":
                retry_ms = _parse_retry(value)

        if not data_lines:
            return None
        return SSEEvent(
            data="\n".join(data_lines),
            received_at_ns=received_at_ns,
            event=event_type,
            event_id=event_id,
            retry_ms=retry_ms,
        )

    @staticmethod
    def _decode_line(raw_line: bytes) -> str:
        try:
            return raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MalformedStreamError("SSE event contains invalid UTF-8") from exc


def _split_field(line: str) -> tuple[str, str]:
    field, separator, value = line.partition(":")
    if separator and value.startswith(" "):
        value = value[1:]
    return field, value


def _parse_retry(value: str) -> int | None:
    if not value or not value.isascii() or not value.isdecimal():
        return None
    return int(value)
