#!/bin/bash
# Experiment B: score one candidate harness on one split of one cell (mb/evolve/broker.py), then
# export what the proposer may read. One GPU, one vLLM server; lowprio + requeue, and the broker
# resumes from the episodes on disk.
#   CAND=dir OUT=dir DEST=dir [CELL=wordle:basic5 SPLIT=search N=30 SMOKE=1 NO_NAMES=1] sbatch eval_cand.sbatch
#   several cells (one shared harness): CELL=gn:all,wordle:basic5,... sbatch --gres=gpu:2 -t 3:00:00 ...
#SBATCH --job-name=mb_evo_b
#SBATCH --partition=h100_1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --qos=lowprio
#SBATCH --nodelist=ml-16-node-001
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
set -uo pipefail
REPO=Qwen/Qwen3.5-9B
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=$HOME/hf_cache VLLM_LOGGING_LEVEL=WARNING
echo "host=$(hostname) start=$(date -Is) restart=${SLURM_RESTART_COUNT:-0} cand=$CAND split=${SPLIT:-search}"
BASE=$(( 9000 + RANDOM % 800 ))
SERVERS=(); URLS=()
for g in $(seq 0 $(( ${SLURM_GPUS_ON_NODE:-1} - 1 ))); do
  PORT=$((BASE + g))
  CUDA_VISIBLE_DEVICES=$g $HOME/venv/bin/vllm serve "$REPO" --port "$PORT" --max-model-len 65536 \
    --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "$HOME/logs/vllm_mb_evo_b_${SLURM_JOB_ID}_g$g.log" 2>&1 &
  SERVERS+=($!); URLS+=("http://127.0.0.1:$PORT/v1")
done
for u in "${URLS[@]}"; do
  for i in $(seq 1 120); do curl -sf -m 5 "${u%/v1}/health" >/dev/null && break; sleep 10; done
  curl -sf -m 5 "${u%/v1}/health" >/dev/null || { echo "serve failed: $u"; kill "${SERVERS[@]}"; exit 1; }
done
echo "servers up after ${SECONDS}s"

cd "$HOME/code/multibench" && source mb/env.sh
export POLICY_MODEL="$REPO" POLICY_BASE_URLS=$(IFS=,; echo "${URLS[*]}")
python -m mb.evolve.broker run --cand "$CAND" --cell "${CELL:-wordle:basic5}" --split "${SPLIT:-search}" \
  --n "${N:-30}" --out "$OUT" ${SMOKE:+--smoke} ${NO_NAMES:+--no-names} --workers 64
[ -n "${DEST:-}" ] && python -m mb.evolve.broker export --out "$OUT" --dest "$DEST"
echo "end=$(date -Is)"
touch "$OUT/DONE"
kill "${SERVERS[@]}"
