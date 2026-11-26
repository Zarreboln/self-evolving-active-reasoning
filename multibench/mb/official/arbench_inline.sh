#!/bin/bash
# AR-Bench's own evaluators on our split, inside a job whose vLLM servers are already up
# (POLICY_BASE_URLS, POLICY_MODEL set by serve.sbatch). Same steps as arbench.sbatch.
#   OUT=dir SPLIT=search TASKS=gn,dc SHARDS=5 METHODS=zero_shot bash arbench_inline.sh
set -u
IFS=, read -ra URLS <<< "$POLICY_BASE_URLS"
cd "$HOME/code/multibench" && python -m mb.official.prep_arbench --out "$OUT/data" --split "$SPLIT" --tasks "$TASKS" --shards "$SHARDS" --n "${N:-20}"
cd "$HOME/code/AR-Bench"
export POLICY_API_KEY=EMPTY RESPONSE_API_KEY=EMPTY
k=0; PIDS=()
for task in ${TASKS//,/ }; do for m in ${METHODS//,/ }; do
  mkdir -p "$OUT/logs/$task/$m/$SPLIT"
  for s in $(seq 0 $((SHARDS - 1))); do
    URL=${URLS[$((k % ${#URLS[@]}))]}; k=$((k + 1))
    DATA="$OUT/data/$task/$SPLIT/shard$s.json"; LOG="$OUT/logs/$task/$m/$SPLIT/shard$s.json"
    if [ "$task" = gn ]; then
      POLICY_BASE_URL=$URL python -m arbench.reasoner.gn.gn_evaluator --model "$POLICY_MODEL" --method "$m" --data_path "$DATA" --output_path "$LOG" > "$LOG.stdout" 2>&1 &
    else
      POLICY_BASE_URL=$URL RESPONSE_BASE_URL=$URL python -m arbench.reasoner.dc.dc_evaluator --method "$m" --data_path "$DATA" --output_path "$LOG" --policy_model "$POLICY_MODEL" --response_model "$POLICY_MODEL" > "$LOG.stdout" 2>&1 &
    fi
    PIDS+=($!)
  done
done; done
for p in "${PIDS[@]}"; do wait $p; done
cd "$HOME/code/multibench" && python -m mb.official.merge_arbench --out "$OUT" --split "$SPLIT" --tasks "$TASKS" --methods "$METHODS"
