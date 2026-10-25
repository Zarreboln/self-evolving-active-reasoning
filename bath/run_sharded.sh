#!/bin/bash
# One active-reasoning baseline, run as N concurrent shards against one served model.
#
# AR-Bench's evaluator is sequential (no pool anywhere in the package), which leaves a 7B model on
# an H100 with one request in flight -- gn/proactive_cot measured 121 s per instance, 3h20m for a
# 100-instance run. Sharding the test set is the way to use the card without editing the benchmark:
# every worker runs the official evaluator unmodified, on its own contiguous slice, writing its own
# log. shard_tool.py merges them back into exactly the file a single process would have written.
#
#   TASK=sp METHOD=few_shot NSHARD=16 bash run_sharded.sh
set -uo pipefail
TASK=${TASK:?gn|sp|dc|md}; METHOD=${METHOD:?few_shot|few_shot_inst|proactive_cot|uot|tot|zero_shot}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
MAX_TURN=${MAX_TURN:-25}
NSHARD=${NSHARD:-16}
# TAG keeps a second protocol's results beside the first instead of overwriting them: the
# BALAR-aligned iCraft-MD run (10 questions, temperature 0.1, the reasoner allowed to stop) is a
# different protocol from the 25-turn fixed-budget one, not a better version of it.
TAG=${TAG:-}
POLICY_TEMP=${POLICY_TEMP:-0.7}
POLICY_TOPP=${POLICY_TOPP:-0.7}
PY=$HOME/venv/bin/python
OUT=$HOME/results/${TASK}_${METHOD}${TAG}.json
WORK=$HOME/results/shards/${TASK}_${METHOD}${TAG}
DATA=$HOME/code/AR-Bench/data/$TASK/test.json

# AR-Bench's three test sets hold 100 instances; iCraft-MD holds 140, so the target comes from the
# data rather than a constant.
TOTAL=$($PY -c "import json;print(len(json.load(open('$DATA'))))") || exit 1
DONE=$($PY -c "import json,sys
try: print(len(json.load(open('$OUT'))))
except Exception: print(0)")
if [ "$DONE" -ge "$TOTAL" ]; then echo "[skip] $TASK/$METHOD already has $DONE/$TOTAL"; exit 0; fi

# Take whichever published endpoint is alive, so a server replaced at its time limit does not
# strand the queue. All shards share it: vLLM batches them, which is the point.
EP=""
for f in "$HOME"/results/serve_qwen25_7b*.endpoint; do
  [ -f "$f" ] || continue
  C=$(cat "$f")
  if curl -sf -m 8 "http://$C/health" >/dev/null 2>&1; then EP="$C"; break; fi
done
[ -n "$EP" ] || { echo "no live policy endpoint"; exit 1; }

cd "$HOME/code/AR-Bench" || exit 1
export POLICY_BASE_URL="http://$EP/v1" POLICY_API_KEY=x

# The four tasks differ in both their CLI and their second role, so each gets its own block.
#
#   gn  no second model at all -- the feedback is a pure function of the guess. One temperature.
#   sp  the referee is a three-way typed decision, Yes / No / Unknown, which is what Jev is for.
#       --response_model is only a label: requests go to RESPONSE_BASE_URL and the shim answers as
#       itself whatever name is sent.
#   dc  the second role plays a suspect and answers in a sentence (`respond_template`), and md's
#       plays a patient. Both are free-text generation, which the Jev shim refuses by design, so
#       they go to Kimi. Kimi is OpenAI-compatible, so no shim is needed -- but kimi-k3 accepts
#       only temperature 1 and top_p 0.95 and rejects anything else with a 400, so the sampling
#       arguments cannot be left at AR-Bench's 0.7.
#
# The key is read from a 0600 file into the environment rather than passed as an argument, so it
# never appears in another user's `ps`.
KIMI_MODEL=${KIMI_MODEL:-kimi-k3}
KIMI_BASE=${KIMI_BASE:-https://api.moonshot.ai/v1}

kimi_responder() {
  [ -r "$HOME/.moonshot_token" ] || { echo "no ~/.moonshot_token"; exit 1; }
  export RESPONSE_BASE_URL="$KIMI_BASE"
  RESPONSE_API_KEY=$(cat "$HOME/.moonshot_token"); export RESPONSE_API_KEY
  curl -sf -m 15 "$KIMI_BASE/models" -H "Authorization: Bearer $RESPONSE_API_KEY" >/dev/null \
    || { echo "kimi api unreachable"; exit 1; }
}

case "$TASK" in
  gn)
    ARGS=(--model "$MODEL" --temperature "$POLICY_TEMP" --top_p "$POLICY_TOPP")
    RUN=("$PY" -m "arbench.reasoner.$TASK.${TASK}_evaluator")
    ;;
  sp)
    curl -sf -m 8 "http://127.0.0.1:8099/v1/models" >/dev/null || { echo "jev referee shim is down"; exit 1; }
    export RESPONSE_BASE_URL="http://127.0.0.1:8099/v1" RESPONSE_API_KEY=x
    ARGS=(--policy_model "$MODEL" --response_model jev-latest
          --policy_temperature "$POLICY_TEMP" --policy_top_p "$POLICY_TOPP"
          --response_temperature 0.7 --response_top_p 0.7)
    RUN=("$PY" -m "arbench.reasoner.$TASK.${TASK}_evaluator")
    ;;
  dc)
    kimi_responder
    ARGS=(--policy_model "$MODEL" --response_model "$KIMI_MODEL"
          --policy_temperature "$POLICY_TEMP" --policy_top_p "$POLICY_TOPP"
          --response_temperature 1 --response_top_p 0.95)
    RUN=("$PY" -m "arbench.reasoner.$TASK.${TASK}_evaluator")
    ;;
  md)
    # BALAR runs the agent and the user simulator on the SAME model, in both of its configurations
    # (gpt-5-nano for both, or one Qwen2.5-32B-Instruct per role). So the served policy answering as
    # the patient is BALAR's design, not a shortcut -- it is Kimi that makes the responder stronger
    # than the reasoner. Local is also ~0.3 s per call against Kimi's 3-8 s, and free.
    if [ "${MD_RESPONDER:-kimi}" = local ]; then
      export RESPONSE_BASE_URL="http://$EP/v1" RESPONSE_API_KEY=x
      KIMI_MODEL="$MODEL"
    else
      kimi_responder
    fi
    if [ "${MD_RESPONDER:-kimi}" = local ]; then
      RTEMP=0.7; RTOPP=0.7       # a local server has no fixed-sampling restriction
    else
      RTEMP=1; RTOPP=0.95        # kimi-k3 rejects anything else with a 400
    fi
    ARGS=(--policy_model "$MODEL" --response_model "$KIMI_MODEL"
          --policy_temperature "$POLICY_TEMP" --policy_top_p "$POLICY_TOPP"
          --response_temperature "$RTEMP" --response_top_p "$RTOPP")
    [ "${MD_ALLOW_STOP:-0}" = 1 ] && ARGS+=(--allow_stop)
    export PYTHONPATH="$HOME/code/AR-Bench"
    RUN=("$PY" "$HOME/code/md_evaluator.py")
    ;;
  *) echo "unknown task $TASK"; exit 1 ;;
esac

mkdir -p "$WORK"
[ -f "$WORK/blocks.json" ] || $PY "$HOME/code/shard_tool.py" split "$DATA" "$WORK" "$NSHARD" >/dev/null
N=$($PY -c "import json;print(len(json.load(open('$WORK/blocks.json'))['blocks']))")
echo "task=$TASK method=$METHOD${TAG} model=$MODEL policy=$EP total=$TOTAL shards=$N"
echo "  max_turn=$MAX_TURN policy_temp=$POLICY_TEMP policy_top_p=$POLICY_TOPP allow_stop=${MD_ALLOW_STOP:-0} start=$(date -Is)"

PIDS=()
for j in $(seq 0 $((N - 1))); do
  JJ=$(printf "%02d" "$j")
  "${RUN[@]}" \
      --method "$METHOD" \
      --data_path "$WORK/shard_$JJ.json" --output_path "$WORK/out_$JJ.json" \
      --max_turn "$MAX_TURN" "${ARGS[@]}" \
      > "$WORK/log_$JJ.out" 2>&1 &
  PIDS+=($!)
done
echo "launched ${#PIDS[@]} shard workers: ${PIDS[*]}"
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=$((FAIL + 1)); done
echo "shards finished, $FAIL non-zero exits $(date -Is)"
$PY "$HOME/code/shard_tool.py" merge "$WORK" "$OUT"
