#!/bin/bash
# SP gate: bare + belief_injected, concurrent, via sp_runner_async.py.
set -uo pipefail
CODE=$HOME/code/ar_harness
VENV=$CODE/venv_cu128
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/usr/lib/cuda
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0

RESULTS=$HOME/results/ar_harness
mkdir -p "$RESULTS"
echo "host=$(hostname) start=$(date -Is)"

PORT=18902
"$VENV/bin/vllm" serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.55 \
  --enable-prefix-caching \
  --port "$PORT" > "$RESULTS/vllm_sp_gate_async.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

echo "waiting for vLLM on :$PORT (pid $VLLM_PID)"
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "up after ${i}0s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died -- see vllm_sp_gate_async.log"; tail -40 "$RESULTS/vllm_sp_gate_async.log"; exit 1; }
  sleep 10
done

T0=$(date +%s)
"$VENV/bin/python" "$CODE/runner/sp_runner_async.py" \
  --concurrency 32 \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --out "$RESULTS/gate_sp.jsonl" \
  --conditions bare,belief_injected \
  --n-instances 100 --seeds 1 --max-turn 25 --temperature 0.7
echo "runner exit=$? elapsed=$(( $(date +%s) - T0 ))s end=$(date -Is)"

"$VENV/bin/python" "$CODE/runner/sp_summarize.py" "$RESULTS/gate_sp.jsonl"
