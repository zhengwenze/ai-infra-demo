#!/usr/bin/env bash
set -euo pipefail

exec uv run python -m inferscope.fake_server --host 127.0.0.1 --port 18000
