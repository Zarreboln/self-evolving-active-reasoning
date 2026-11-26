#!/bin/bash
# The long-lived episode player (mb/evolve/worker.py): one GPU, one vLLM server, serving play requests
# from the mailbox until shortly before its time limit, then handing over to its next copy.
#   source race.sh; MAILBOX=~/mailbox_worker RUNS=~/results/mb/play \
#     race_submit mb_worker "--requeue -c 16 --mem=64G -t 05:59:00 --export=ALL" worker.sbatch
# (one copy per GPU type; the first to start serves, the others are cancelled)
#SBATCH --job-name=mb_worker
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
set -uo pipefail
[ -f "$HOME/.mb_site.sh" ] && source "$HOME/.mb_site.sh"
[ -e "$HOME/mailbox_STOP" ] && { echo "STOP file present"; exit 0; }
REPO=Qwen/Qwen3.5-9B
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-$HOME/hf_cache} VLLM_LOGGING_LEVEL=WARNING
source "$HOME/code/multibench/mb/evolve/race.sh"
mkdir -p "$RUNS"
race_claim "$RUNS" || { echo "another worker is running; exiting"; exit 0; }
NEXT=$(race_submit mb_worker "--dependency=afterany:$SLURM_JOB_ID --requeue -c $SLURM_CPUS_PER_TASK --mem=64G -t 05:59:00 --export=ALL" \
       "$HOME/code/multibench/mb/evolve/worker.sbatch")
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
[ -e "$HOME/mailbox_STOP" ] && scancel ${NEXT//,/ }
kill $SERVER
