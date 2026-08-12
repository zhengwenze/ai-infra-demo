"""Tiny OpenAI-compatible SSE server for CPU smoke tests and demos."""

from __future__ import annotations

import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar


class FakeInferenceHandler(BaseHTTPRequestHandler):
    """Serve deterministic model metadata, metrics, and streaming completions."""

    server_version = "InferScopeFake/0.1"
    protocol_version = "HTTP/1.0"
    model_id: ClassVar[str] = "inferscope/fake-model"
    first_token_delay_seconds: ClassVar[float] = 0.01
    token_delay_seconds: ClassVar[float] = 0.005

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({"status": "ok"})
        elif self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": self.model_id}]})
        elif self.path == "/metrics":
            body = b"vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("content-length", "0"))
        try:
            payload = json.loads(self.rfile.read(length))
            max_tokens = int(payload.get("max_tokens", 8))
        except (ValueError, json.JSONDecodeError):
            self._json({"error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
            return

        completion_tokens = max(1, min(max_tokens, 32))
        events: list[dict[str, object]] = [
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        ]
        generated_events: list[dict[str, object]] = []
        for index in range(completion_tokens):
            generated_events.append(
                {
                    "choices": [
                        {
                            "delta": {"content": f"token-{index} "},
                            "finish_reason": ("length" if index == completion_tokens - 1 else None),
                        }
                    ]
                }
            )
        events.extend(generated_events)
        events.append(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": completion_tokens,
                    "total_tokens": 8 + completion_tokens,
                },
            }
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for index, event in enumerate(events):
                if index == 1:
                    time.sleep(self.first_token_delay_seconds)
                elif index > 1:
                    time.sleep(self.token_delay_seconds)
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def serve(host: str = "127.0.0.1", port: int = 18000) -> None:
    """Run the fake server until interrupted."""
    server = ThreadingHTTPServer((host, port), FakeInferenceHandler)
    print(f"InferScope fake server listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    arguments = parser.parse_args()
    serve(arguments.host, arguments.port)


if __name__ == "__main__":
    main()
