#!/bin/bash
# Difficulty screen of the whole suite: Qwen3.5-9B (policy and every LLM-played role), the react
# seed, zero-shot, 20 search instances per subtask. Two GPUs, one server each. lowprio: if the job
# is preempted it is requeued under the same id and mb.screen resumes from the episodes on disk.
#   sbatch ~/code/multibench/mb/screen.sbatch
#SBATCH --job-name=mb_screen
#SBATCH --partition=h100_1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --qos=lowprio
#SBATCH --nodelist=ml-16-node-001
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
set -uo pipefail
REPO=Qwen/Qwen3.5-9B
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=$HOME/hf_cache VLLM_LOGGING_LEVEL=WARNING
echo "host=$(hostname) start=$(date -Is) restart=${SLURM_RESTART_COUNT:-0}"
BASE=$(( 9000 + RANDOM % 800 ))
SERVERS=(); URLS=()
for g in 0 1; do
  PORT=$((BASE + g))
  CUDA_VISIBLE_DEVICES=$g $HOME/venv/bin/vllm serve "$REPO" --port "$PORT" --max-model-len 65536 \
    --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "$HOME/logs/vllm_mb_screen_${SLURM_JOB_ID}_g$g.log" 2>&1 &
  SERVERS+=($!); URLS+=("http://127.0.0.1:$PORT/v1")
done
for u in "${URLS[@]}"; do
  for i in $(seq 1 120); do curl -sf -m 5 "${u%/v1}/health" >/dev/null && break; sleep 10; done
  curl -sf -m 5 "${u%/v1}/health" >/dev/null || { echo "serve failed: $u"; kill "${SERVERS[@]}"; exit 1; }
done
echo "servers up after ${SECONDS}s"

cd "$HOME/code/multibench" && source mb/env.sh
export POLICY_MODEL="$REPO" POLICY_BASE_URLS=$(IFS=,; echo "${URLS[*]}")
OUT=$HOME/results/mb/screen_v1
python -m mb.screen --out "$OUT" --n 20 --workers 64
python -m mb.screen --out "$OUT" --n 20 --merge
echo "end=$(date -Is)"
kill "${SERVERS[@]}"
