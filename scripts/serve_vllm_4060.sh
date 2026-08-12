#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "vLLM/CUDA benchmark requires Linux; use scripts/serve_fake.sh for CPU smoke tests." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; install a compatible NVIDIA driver first." >&2
  exit 2
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm executable not found; install the GPU environment documented in docs/RTX4060_GUIDE.md." >&2
  exit 2
fi

model_id="${INFERSCOPE_MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
exec vllm serve "$model_id" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 8 \
  --enforce-eager \
  --disable-log-requests
