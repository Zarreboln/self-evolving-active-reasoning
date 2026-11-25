#!/bin/bash
# LLM smoke test of the whole suite on ONE GPU: one Qwen3.5-9B server (policy and every
# LLM-played role), the react harness on one search instance of the first two subtasks of each
# bench. Checks the loop end to end; the scores mean nothing at n=1. The job ends itself when the
# smoke run finishes, so the GPU bills only for the run.
#   sbatch ~/code/multibench/mb/smoke_llm.sbatch
#SBATCH --job-name=mb_smoke_llm
#SBATCH --partition=h100_1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --qos=lowprio
#SBATCH --nodelist=ml-16-node-001
#SBATCH --output=%x_%j.out
set -uo pipefail
REPO=Qwen/Qwen3.5-9B
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=$HOME/hf_cache VLLM_LOGGING_LEVEL=WARNING
PORT=$(( 9000 + RANDOM % 800 ))
echo "host=$(hostname) start=$(date -Is)"
$HOME/venv/bin/vllm serve "$REPO" --port "$PORT" --max-model-len 65536 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > "$HOME/logs/vllm_mb_smoke_$SLURM_JOB_ID.log" 2>&1 &
SERVE=$!
for i in $(seq 1 120); do curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 10; done
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "serve failed"; tail -30 "$HOME/logs/vllm_mb_smoke_$SLURM_JOB_ID.log"; exit 1; }
echo "server up after ${SECONDS}s"

cd "$HOME/code/multibench" && source mb/env.sh
export POLICY_BASE_URL="http://127.0.0.1:$PORT/v1" POLICY_MODEL="$REPO"   # every LLM role too
OUT=$HOME/results/mb/smoke_llm_$SLURM_JOB_ID
mkdir -p "$OUT"
# one process per bench: faster, and ScienceWorld's JVMs must not share a process
BENCHES=$(python -c "from mb.benches import BENCHES; print(' '.join(BENCHES))")
PIDS=()
for b in $BENCHES; do
  python -m mb.smoke --mode llm --n 1 --benches "$b" --out "$OUT/$b.json" > "$OUT/$b.log" 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"      # not a bare `wait`: that would also wait on the server and hold the GPU to timeout
echo "end=$(date -Is)"
grep -h -E "^\{|FAIL|episodes" "$OUT"/*.log
kill $SERVE
