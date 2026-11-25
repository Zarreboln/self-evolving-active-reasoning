#!/bin/bash
# Step 1a: AR-Bench's own evaluators, unchanged, on our split, against Qwen3.5-9B (thinking off).
# GN and DC with proactive_cot (the official reasoning agent). DC's suspects are the same model, as in
# AR-Bench's own runs. Each task's items are sharded (mb/official/prep_arbench.py) and every shard is a
# separate evaluator process; AR-Bench resumes each from its own log, so a requeue after preemption
# continues where it stopped. 4 GPUs, one vLLM server each, shards spread round-robin.
#   OUT=/data/team-bath/tmp_l/official/arbench SPLIT=test sbatch ~/code/multibench/mb/official/arbench.sbatch
#SBATCH --job-name=off_arbench
#SBATCH --partition=h100_1
#SBATCH --gres=gpu:4
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
OUT=${OUT:-/data/team-bath/tmp_l/official/arbench}; SPLIT=${SPLIT:-test}; SHARDS=${SHARDS:-10}
METHODS=${METHODS:-proactive_cot}; TASKS=${TASKS:-gn,dc}
echo "host=$(hostname) start=$(date -Is) restart=${SLURM_RESTART_COUNT:-0} split=$SPLIT methods=$METHODS"
BASE=$(( 9000 + RANDOM % 800 )); SERVERS=(); URLS=()
for g in $(seq 0 $(( ${SLURM_GPUS_ON_NODE:-4} - 1 ))); do
  PORT=$((BASE + g))
  CUDA_VISIBLE_DEVICES=$g $HOME/venv/bin/vllm serve "$REPO" --port "$PORT" --max-model-len 65536 \
    --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "$HOME/logs/vllm_off_arbench_${SLURM_JOB_ID}_g$g.log" 2>&1 &
  SERVERS+=($!); URLS+=("http://127.0.0.1:$PORT/v1")
done
for u in "${URLS[@]}"; do
  for i in $(seq 1 120); do curl -sf -m 5 "${u%/v1}/health" >/dev/null && break; sleep 10; done
  curl -sf -m 5 "${u%/v1}/health" >/dev/null || { echo "serve failed: $u"; kill "${SERVERS[@]}"; exit 1; }
done
echo "servers up after ${SECONDS}s"

cd "$HOME/code/multibench" && source mb/env.sh
python -m mb.official.prep_arbench --out "$OUT/data" --split "$SPLIT" --tasks "$TASKS" --shards "$SHARDS" --n "${N:-0}"
cd "$HOME/code/AR-Bench"
export POLICY_API_KEY=EMPTY RESPONSE_API_KEY=EMPTY
PIDS=(); k=0
for task in ${TASKS//,/ }; do
  for m in ${METHODS//,/ }; do
    mkdir -p "$OUT/logs/$task/$m/$SPLIT"
    for s in $(seq 0 $((SHARDS - 1))); do
      URL=${URLS[$((k % ${#URLS[@]}))]}; k=$((k + 1))
      DATA="$OUT/data/$task/$SPLIT/shard$s.json"; LOG="$OUT/logs/$task/$m/$SPLIT/shard$s.json"
      if [ "$task" = gn ]; then
        POLICY_BASE_URL=$URL python -m arbench.reasoner.gn.gn_evaluator --model "$REPO" --method "$m" \
          --data_path "$DATA" --output_path "$LOG" > "$LOG.stdout" 2>&1 &
      else
        POLICY_BASE_URL=$URL RESPONSE_BASE_URL=$URL python -m arbench.reasoner.dc.dc_evaluator --method "$m" \
          --data_path "$DATA" --output_path "$LOG" --policy_model "$REPO" --response_model "$REPO" \
          > "$LOG.stdout" 2>&1 &
      fi
      PIDS+=($!)
    done
  done
done
fail=0; for p in "${PIDS[@]}"; do wait $p || fail=$((fail + 1)); done
echo "evaluators done, $fail failed; end=$(date -Is)"
cd "$HOME/code/multibench" && python -m mb.official.merge_arbench --out "$OUT" --split "$SPLIT" --tasks "$TASKS" --methods "$METHODS"
kill "${SERVERS[@]}"
