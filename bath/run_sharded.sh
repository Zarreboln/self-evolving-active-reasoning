#!/bin/bash
# One AR-Bench baseline, run as N concurrent shards against one served model.
#
# AR-Bench's evaluator is sequential (no pool anywhere in the package), which leaves a 7B model on
# an H100 with one request in flight -- gn/proactive_cot measured 121 s per instance, 3h20m for a
# 100-instance run. Sharding the test set is the way to use the card without editing the benchmark:
# every worker runs the official evaluator unmodified, on its own contiguous slice, writing its own
# log. shard_tool.py merges them back into exactly the file a single process would have written.
#
#   TASK=sp METHOD=few_shot NSHARD=16 bash run_sharded.sh
set -uo pipefail
TASK=${TASK:?sp|dc|gn}; METHOD=${METHOD:?few_shot|few_shot_inst|proactive_cot|uot|tot|zero_shot}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
MAX_TURN=${MAX_TURN:-25}
NSHARD=${NSHARD:-16}
PY=$HOME/venv/bin/python
OUT=$HOME/results/${TASK}_${METHOD}.json
WORK=$HOME/results/shards/${TASK}_${METHOD}

DONE=$($PY -c "import json,sys
try: print(len(json.load(open('$OUT'))))
except Exception: print(0)")
if [ "$DONE" -ge 100 ]; then echo "[skip] $TASK/$METHOD already has $DONE"; exit 0; fi

# Take whichever published endpoint is alive, so a server replaced at its time limit does not
# strand the queue. All shards share it: vLLM batches them, which is the point.
EP=""
for f in "$HOME"/results/serve_qwen25_7b*.endpoint; do
  [ -f "$f" ] || continue
  C=$(cat "$f")
  if curl -sf -m 8 "http://$C/health" >/dev/null 2>&1; then EP="$C"; break; fi
done
[ -n "$EP" ] || { echo "no live policy endpoint"; exit 1; }
if [ "$TASK" != gn ]; then
  curl -sf -m 8 "http://127.0.0.1:8099/v1/models" >/dev/null || { echo "jev referee shim is down"; exit 1; }
fi

cd "$HOME/code/AR-Bench" || exit 1
export POLICY_BASE_URL="http://$EP/v1"   POLICY_API_KEY=x
export RESPONSE_BASE_URL="http://127.0.0.1:8099/v1" RESPONSE_API_KEY=x

mkdir -p "$WORK"
[ -f "$WORK/blocks.json" ] || $PY "$HOME/code/shard_tool.py" split "data/$TASK/test.json" "$WORK" "$NSHARD" >/dev/null
N=$($PY -c "import json;print(len(json.load(open('$WORK/blocks.json'))['blocks']))")
echo "task=$TASK method=$METHOD model=$MODEL policy=$EP shards=$N start=$(date -Is)"

PIDS=()
for j in $(seq 0 $((N - 1))); do
  JJ=$(printf "%02d" "$j")
  $PY -m arbench.reasoner."$TASK"."${TASK}_evaluator" \
      --model "$MODEL" --method "$METHOD" \
      --data_path "$WORK/shard_$JJ.json" --output_path "$WORK/out_$JJ.json" \
      --max_turn "$MAX_TURN" --temperature 0.7 --top_p 0.7 \
      > "$WORK/log_$JJ.out" 2>&1 &
  PIDS+=($!)
done
echo "launched ${#PIDS[@]} shard workers: ${PIDS[*]}"
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=$((FAIL + 1)); done
echo "shards finished, $FAIL non-zero exits $(date -Is)"
$PY "$HOME/code/shard_tool.py" merge "$WORK" "$OUT"
