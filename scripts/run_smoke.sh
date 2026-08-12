#!/usr/bin/env bash
set -euo pipefail

uv run inferscope server check --base-url http://127.0.0.1:18000
uv run inferscope benchmark run --config configs/smoke.yaml --results-dir results
