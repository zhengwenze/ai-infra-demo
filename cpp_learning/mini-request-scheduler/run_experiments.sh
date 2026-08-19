#!/bin/bash
# ============================================================
# mini-request-scheduler 对照实验脚本
# 分别改变：Consumer (1/2/4)、Batch (1/2/4/8)、Queue (8/16/64)
# 每次跑 200 请求，汇总成 CSV 便于对比
# ============================================================

set -e

PROJECT_DIR="/Users/zhengwenze/Desktop/codex/ai-infra-demo/cpp_learning/mini-request-scheduler"
BIN="$PROJECT_DIR/build/mini-scheduler"
RESULTS_DIR="$PROJECT_DIR/build/results"
RESULTS_CSV="$RESULTS_DIR/experiments.csv"

mkdir -p "$RESULTS_DIR"

# CSV 表头
echo "consumers,batch,queue,req_s,tok_s,p50_lat,p90_lat,p99_lat,p50_qw,p99_qw,p50_inf,p99_inf" > "$RESULTS_CSV"

run_one() {
    local consumers=$1
    local batch=$2
    local queue=$3
    local label="C${consumers}_B${batch}_Q${queue}"
    echo "------------------------------------------------------------"
    echo " Running: consumers=$consumers batch=$batch queue=$queue"
    echo "------------------------------------------------------------"
    # 跑实验，从输出里抽取 CSV 行（以 "  consumers,..." 开头那行）
    "$BIN" \
        --consumers "$consumers" \
        --batch "$batch" \
        --queue "$queue" \
        --requests 200 \
        --producers 2 \
        --gap 2 \
        2>&1 | tee "$RESULTS_DIR/run_${label}.log" | grep -E "^  [0-9]+," | head -1 >> "$RESULTS_CSV"
}

echo "============================================================"
echo " Experiment 1: vary Consumer  (batch=4, queue=16)"
echo "============================================================"
for c in 1 2 4; do
    run_one "$c" 4 16
done

echo ""
echo "============================================================"
echo " Experiment 2: vary Batch      (consumers=2, queue=16)"
echo "============================================================"
for b in 1 2 4 8; do
    run_one 2 "$b" 16
done

echo ""
echo "============================================================"
echo " Experiment 3: vary Queue       (consumers=2, batch=4)"
echo "============================================================"
for q in 8 16 64; do
    run_one 2 4 "$q"
done

echo ""
echo "============================================================"
echo " All experiments done. Summary CSV:"
echo "============================================================"
column -t -s',' "$RESULTS_CSV" || cat "$RESULTS_CSV"
echo ""
echo " Raw CSV file: $RESULTS_CSV"
