#!/bin/bash
# The long-lived episode player (mb/evolve/worker.py): one GPU, one vLLM server, serving play requests
# from the mailbox until shortly before its time limit, then handing over to its next copy.
#   MAILBOX=~/mailbox_worker RUNS=~/results/mb/play sbatch -p mit_normal_gpu --gres=gpu:l40s:1 -c 16 --mem=64G -t 05:59:00 worker.sbatch
#SBATCH --job-name=mb_worker
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
set -uo pipefail
[ -f "$HOME/.mb_site.sh" ] && source "$HOME/.mb_site.sh"
[ -e "$HOME/mailbox_STOP" ] && { echo "STOP file present"; exit 0; }
REPO=Qwen/Qwen3.5-9B
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-$HOME/hf_cache} VLLM_LOGGING_LEVEL=WARNING
NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID --partition="$SLURM_JOB_PARTITION" \
       --gres=gpu:l40s:1 -c "$SLURM_CPUS_PER_TASK" --mem=64G -t 05:59:00 --export=ALL "$HOME/code/multibench/mb/evolve/worker.sbatch")
PORT=$(( 9000 + RANDOM % 800 ))
${VLLM_BIN:-$HOME/venv/bin/vllm} serve "$REPO" --port "$PORT" --max-model-len 65536 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}' > "$HOME/logs/vllm_worker_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
for i in $(seq 1 120); do curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 10; done
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "serve failed"; kill $SERVER; exit 1; }
echo "host=$(hostname) server up after ${SECONDS}s next=$NEXT"
cd "$HOME/code/multibench" && source mb/env.sh
export POLICY_MODEL="$REPO" POLICY_BASE_URLS="http://127.0.0.1:$PORT/v1"
python -m mb.evolve.worker --mailbox "$MAILBOX" --runs "$RUNS" --until $(( $(date +%s) + 5*3600 + 30*60 ))
[ -e "$HOME/mailbox_STOP" ] && scancel "$NEXT"
kill $SERVER
