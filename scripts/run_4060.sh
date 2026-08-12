#!/usr/bin/env bash
set -euo pipefail

uv run inferscope server check --base-url http://127.0.0.1:8000
uv run inferscope benchmark run \
  --config configs/rtx4060_qwen05b.yaml \
  --results-dir results
