#!/bin/bash
# First-ever SP (Situation Puzzle / 海龟汤) baseline. Uses AR-Bench's own
# evaluator verbatim (arbench.reasoner.sp.sp_evaluator, method=zero_shot --
# the single-call-per-turn path, _run_traditional_evaluation -- no tree
# search/branching), the same "trust the official convention" choice already
# made for GN. Sequential by construction (one instance at a time, no
# concurrency in the upstream script), so this takes a while; left running
# under setsid.
set -uo pipefail
CODE=$HOME/code/ar_harness
VENV=$CODE/venv_cu128
AR=$CODE/AR-Bench
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/usr/lib/cuda
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0

RESULTS=$HOME/results/ar_harness
mkdir -p "$RESULTS" "$AR/logs/sp"
echo "host=$(hostname) start=$(date -Is)"

PORT=18901
"$VENV/bin/vllm" serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.6 \
  --enable-prefix-caching \
  --port "$PORT" > "$RESULTS/vllm_sp_gate.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

echo "waiting for vLLM on :$PORT (pid $VLLM_PID)"
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "up after ${i}0s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died -- see vllm_sp_gate.log"; tail -40 "$RESULTS/vllm_sp_gate.log"; exit 1; }
  sleep 10
done

export POLICY_API_KEY=EMPTY
export POLICY_BASE_URL="http://127.0.0.1:$PORT/v1"
export RESPONSE_API_KEY=EMPTY
export RESPONSE_BASE_URL="http://127.0.0.1:$PORT/v1"

T0=$(date +%s)
cd "$AR"
"$VENV/bin/python" -m arbench.reasoner.sp.sp_evaluator \
  --method=zero_shot \
  --data_path=data/sp/test.json \
  --output_path=logs/sp/log_zero_shot_qwen7b.json \
  --policy_model="Qwen/Qwen2.5-7B-Instruct" \
  --response_model="Qwen/Qwen2.5-7B-Instruct" \
  --max_turn=25 \
  --policy_temperature=0.7 --policy_top_p=0.7 \
  --response_temperature=0.7 --response_top_p=0.7
echo "evaluator exit=$? elapsed=$(( $(date +%s) - T0 ))s end=$(date -Is)"

"$VENV/bin/python" - <<'EOF'
import json, statistics as st
logs = json.load(open("logs/sp/log_zero_shot_qwen7b.json"))
print(f"n={len(logs)}")
print(f"f1_char mean={st.mean(l['f1_score_char'] for l in logs):.3f}")
print(f"f1_word mean={st.mean(l['f1_score_word'] for l in logs):.3f}")
print(f"tokens/ep in={st.mean(l['input_token'] for l in logs):.0f} out={st.mean(l['output_token'] for l in logs):.0f}")
EOF
