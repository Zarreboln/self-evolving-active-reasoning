#!/bin/bash
# Model-size grid for CircuitDecoding: serve one task model, run a list of
# harnesses on all 300 instances x 2 seeds, score. One job per model.
#
#   sbatch --gres=gpu:l40s:1 engaging/cd_grid.sbatch Qwen/Qwen2.5-7B-Instruct      1 "bare enum15 oracle_hint"
#   sbatch --gres=gpu:l40s:1 engaging/cd_grid.sbatch Qwen/Qwen2.5-32B-Instruct-AWQ 1 "bare enum15 oracle_hint"
#   sbatch --gres=gpu:l40s:2 engaging/cd_grid.sbatch Qwen/Qwen2.5-72B-Instruct-AWQ 2 "bare enum15 oracle_hint"
#
# args: <hf model id> <tensor parallel> "<harness names>"   (served name = model id)
# Mixed precision across sizes (7B bf16, 32B/72B AWQ) is a confound; it is
# recorded in each episode's meta.model and must be stated with the table.
#
#SBATCH --job-name=cd_grid
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:55:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
MODEL=${1:?model}; TP=${2:-1}; HARNESSES=${3:-"bare enum15 oracle_hint"}
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
TAG=$(basename "$MODEL")
RESULTS=$HOME/results/ar_harness/cd_grid/$TAG; mkdir -p "$RESULTS"
PORT=$((8000 + RANDOM % 1000))
echo "host=$(hostname) start=$(date -Is) model=$MODEL tp=$TP port=$PORT harnesses=[$HARNESSES]"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$VLLM" serve "$MODEL" --served-model-name "$MODEL" --tensor-parallel-size "$TP" \
  --max-model-len 16384 --max-num-seqs 48 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --port "$PORT" > "$RESULTS/vllm_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "up after ${i}0s"; break; }
  kill -0 $PID 2>/dev/null || { echo "vLLM died"; tail -40 "$RESULTS/vllm_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 10
done
for h in $HARNESSES; do
  T0=$(date +%s); echo "=== $h start=$(date -Is)"
  "$PY" "$CODE/runner/cd_harness_run.py" --harness "$CODE/meta/cd_evolve/agents/$h.py" \
    --model "$MODEL" --split all --seeds 2 --concurrency 48 --base-url "http://127.0.0.1:$PORT/v1" \
    --out "$RESULTS/$h.jsonl"
  echo "=== $h exit=$? elapsed=$(( $(date +%s) - T0 ))s end=$(date -Is)"
done
"$PY" "$CODE/runner/cd_score.py" "$RESULTS"/*.jsonl --split all --out "$RESULTS/scores.json"
echo "ALL DONE end=$(date -Is)"
