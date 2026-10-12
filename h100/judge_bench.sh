#!/bin/bash
# Serve a candidate judge model on H100 GPU 3 and score it on the fixed
# TurtleBench 150-case sample (see runner/turtle_judge_eval.py for why that
# benchmark and what the reference numbers are).
#
#   judge_bench.sh <local-model-dir> <served-name> [extra vllm args...]
set -uo pipefail
MODEL_DIR=${1:?usage: judge_bench.sh <model-dir> <served-name> [vllm args]}
NAME=${2:?}
shift 2

CODE=$HOME/code/ar_harness
VENV=$CODE/venv_cu128
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/usr/lib/cuda
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0

L=$HOME/logs; mkdir -p "$L"
TAG=$(basename "$MODEL_DIR")
echo "host=$(hostname) model=$TAG start=$(date -Is)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 3

PORT=$((19000 + RANDOM % 900))
"$VENV/bin/vllm" serve "$MODEL_DIR" --served-model-name "$NAME" \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --enable-prefix-caching \
  --port "$PORT" "$@" > "$L/vllm_$TAG.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT

echo "waiting for vLLM on :$PORT"
for i in $(seq 1 150); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "up after ${i}0s"; break; }
  kill -0 $PID 2>/dev/null || { echo "vLLM died:"; tail -40 "$L/vllm_$TAG.log"; exit 1; }
  sleep 10
done

"$VENV/bin/python" "$CODE/runner/turtle_judge_eval.py" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$NAME" \
  --turtlebench "$HOME/TurtleBench" --concurrency 16 \
  --out "$L/judge_${TAG}.json"
echo "end=$(date -Is)"
