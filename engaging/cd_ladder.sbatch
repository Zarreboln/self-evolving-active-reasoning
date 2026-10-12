#!/bin/bash
# Actionability ladder + seed harnesses on Qwen2.5-7B-Instruct, all 300
# CircuitDecoding instances x 2 seeds, one job: serve vLLM on one L40S, then
# run the seven harnesses in meta/cd_evolve/agents/ back to back.
#   bare / enum15 (= old belief_injected) / symbolic / facts (a) / splits (c) /
#   oracle_hint (d) / t3_gate
# 7 x 600 = 4200 episodes; the 2026-10-07 gate job did 600 in 1484 s at
# concurrency 32 -> about 3 h here at concurrency 48.
#
#   sbatch ~/code/ar_harness/engaging/cd_ladder.sbatch
#
#SBATCH --job-name=cd_ladder
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:45:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
CODE=$HOME/code/ar_harness
POOL=$HOME/orcd/pool/ar_harness
export HF_HOME=$POOL/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
PY=$CODE/conda_env/bin/python
VLLM=$CODE/conda_env/bin/vllm
export CUDA_HOME=$CODE/conda_env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CODE/conda_env/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
RESULTS=$HOME/results/ar_harness/${LADDER_DIR:-cd_ladder}; mkdir -p "$RESULTS"   # --export=ALL,LADDER_DIR=...,HARNESSES="a b c"
PORT=$((8000 + RANDOM % 1000))
echo "host=$(hostname) start=$(date -Is) port=$PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$VLLM" serve Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 --max-model-len 16384 --max-num-seqs 48 \
  --gpu-memory-utilization 0.88 --enable-prefix-caching \
  --port "$PORT" > "$RESULTS/vllm_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "up after ${i}0s"; break; }
  kill -0 $PID 2>/dev/null || { echo "vLLM died"; tail -30 "$RESULTS/vllm_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 10
done
for h in ${HARNESSES:-bare enum15 symbolic facts splits oracle_hint t3_gate}; do
  T0=$(date +%s)
  echo "=== $h start=$(date -Is)"
  "$PY" "$CODE/runner/cd_harness_run.py" --harness "$CODE/meta/cd_evolve/agents/$h.py" \
    --split all --seeds 2 --concurrency 48 --base-url "http://127.0.0.1:$PORT/v1" \
    --out "$RESULTS/$h.jsonl"
  echo "=== $h exit=$? elapsed=$(( $(date +%s) - T0 ))s end=$(date -Is)"
done
"$PY" "$CODE/runner/cd_score.py" "$RESULTS"/*.jsonl --split all
echo "ALL DONE end=$(date -Is)"
