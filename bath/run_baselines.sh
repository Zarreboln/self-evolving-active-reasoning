#!/bin/bash
# Reproduce AR-Bench's official baselines with our reasoner and Jev as the referee.
# The reasoner (POLICY_*) is the served model under test; the referee (RESPONSE_*) is the Jev shim,
# so no second GPU is needed. Nothing in AR-Bench's code is modified.
#   TASK=sp METHOD=few_shot bash run_baselines.sh
set -uo pipefail
TASK=${TASK:?sp|dc|gn}; METHOD=${METHOD:?few_shot|few_shot_inst|proactive_cot|uot|tot|zero_shot}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
MAX_TURN=${MAX_TURN:-25}

# Take whichever published endpoint is actually alive, so a server that hits its time limit and is
# replaced by the next job does not strand the queue.
EP=""
for f in "$HOME"/results/serve_qwen25_7b*.endpoint; do
  [ -f "$f" ] || continue
  C=$(cat "$f")
  if curl -sf -m 8 "http://$C/health" >/dev/null 2>&1; then EP="$C"; break; fi
done
[ -n "$EP" ] || { echo "no live policy endpoint"; exit 1; }
curl -sf -m 8 "http://127.0.0.1:8099/v1/models" >/dev/null || { echo "jev referee shim is down"; exit 1; }

cd "$HOME/code/AR-Bench" || exit 1
export POLICY_BASE_URL="http://$EP/v1"   POLICY_API_KEY=x
export RESPONSE_BASE_URL="http://127.0.0.1:8099/v1" RESPONSE_API_KEY=x
OUT="$HOME/results/${TASK}_${METHOD}.json"

# The three evaluators do not share a CLI: GN takes one model and one temperature, SP and DC
# separate reasoner from referee and take four sampling arguments. Fire rejects the wrong set.
case "$TASK" in
  gn) ARGS=(--model "$MODEL" --temperature 0.7 --top_p 0.7) ;;
  sp|dc) ARGS=(--policy_model "$MODEL" --response_model jev-latest
               --policy_temperature 0.7 --policy_top_p 0.7
               --response_temperature 0.7 --response_top_p 0.7) ;;
  *) echo "unknown task $TASK"; exit 1 ;;
esac

echo "task=$TASK method=$METHOD model=$MODEL policy=$EP referee=jev start=$(date -Is)"
"$HOME/venv/bin/python" -m arbench.reasoner."$TASK"."${TASK}_evaluator" \
  --method "$METHOD" \
  --data_path "data/$TASK/test.json" --output_path "$OUT" \
  --max_turn "$MAX_TURN" "${ARGS[@]}"
echo "done=$(date -Is) -> $OUT"
