#!/bin/bash
# One vLLM server per GPU (Qwen3.5-9B, thinking off), then run CMD with POLICY_BASE_URLS / POLICY_MODEL
# set. lowprio + requeue: CMD must resume from what it has written (every mb.official runner does).
#   CMD="python -m mb.official.run_textarena --out /data/team-bath/tmp_l/official/textarena --agent thought" \
#     sbatch --job-name=off_textarena ~/code/multibench/mb/official/serve.sbatch
# Partition/QOS come from the caller: bath -p h100_1 --qos=lowprio; ORCD -p mit_preemptable --gres=gpu:l40s:1.
#SBATCH --job-name=off_run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
set -uo pipefail
[ -f "$HOME/.mb_site.sh" ] && source "$HOME/.mb_site.sh"
REPO=Qwen/Qwen3.5-9B
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-$HOME/hf_cache} VLLM_LOGGING_LEVEL=WARNING
echo "host=$(hostname) start=$(date -Is) restart=${SLURM_RESTART_COUNT:-0} cmd=$CMD"
BASE=$(( 9000 + RANDOM % 800 )); SERVERS=(); URLS=()
for g in $(seq 0 $(( ${SLURM_GPUS_ON_NODE:-4} - 1 ))); do
  PORT=$((BASE + g))
  CUDA_VISIBLE_DEVICES=$g ${VLLM_BIN:-$HOME/venv/bin/vllm} serve "$REPO" --port "$PORT" --max-model-len 65536 \
    --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "$HOME/logs/vllm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}_g$g.log" 2>&1 &
  SERVERS+=($!); URLS+=("http://127.0.0.1:$PORT/v1")
done
for u in "${URLS[@]}"; do
  for i in $(seq 1 120); do curl -sf -m 5 "${u%/v1}/health" >/dev/null && break; sleep 10; done
  curl -sf -m 5 "${u%/v1}/health" >/dev/null || { echo "serve failed: $u"; kill "${SERVERS[@]}"; exit 1; }
done
echo "servers up after ${SECONDS}s"
cd "$HOME/code/multibench" && source mb/env.sh
export POLICY_MODEL="$REPO" POLICY_BASE_URLS=$(IFS=,; echo "${URLS[*]}")
eval "$CMD"
echo "cmd exit=$? end=$(date -Is)"
kill "${SERVERS[@]}"
