#!/bin/bash
# Experiment B: score one candidate harness on one split of one cell (mb/evolve/broker.py), then
# export what the proposer may read. One GPU, one vLLM server; lowprio + requeue, and the broker
# resumes from the episodes on disk. Partition, QOS and GPU type come from the caller
# (remote.SBATCH_ARGS); cluster paths from ~/.mb_site.sh if present.
#   CAND=dir OUT=dir DEST=dir [CELL=wordle:basic5 SPLIT=search N=30 SMOKE=1] sbatch eval_cand.sbatch
#SBATCH --job-name=mb_evo_b
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
set -uo pipefail
[ -f "$HOME/.mb_site.sh" ] && source "$HOME/.mb_site.sh"
REPO=Qwen/Qwen3.5-9B
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-$HOME/hf_cache} VLLM_LOGGING_LEVEL=WARNING
echo "host=$(hostname) start=$(date -Is) restart=${SLURM_RESTART_COUNT:-0} cand=$CAND split=${SPLIT:-search}"
PORT=$(( 9000 + RANDOM % 800 ))
${VLLM_BIN:-$HOME/venv/bin/vllm} serve "$REPO" --port "$PORT" --max-model-len 65536 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > "$HOME/logs/vllm_mb_evo_b_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
for i in $(seq 1 120); do curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 10; done
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "serve failed"; kill $SERVER; exit 1; }
echo "server up after ${SECONDS}s"

cd "$HOME/code/multibench" && source mb/env.sh
export POLICY_MODEL="$REPO" POLICY_BASE_URLS="http://127.0.0.1:$PORT/v1"
python -m mb.evolve.broker run --cand "$CAND" --cell "${CELL:-wordle:basic5}" --split "${SPLIT:-search}" \
  --n "${N:-30}" --out "$OUT" ${SMOKE:+--smoke} --workers 32
[ -n "${DEST:-}" ] && python -m mb.evolve.broker export --out "$OUT" --dest "$DEST"
echo "end=$(date -Is)"
touch "$OUT/DONE"
kill $SERVER
