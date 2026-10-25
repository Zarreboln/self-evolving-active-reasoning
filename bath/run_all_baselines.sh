#!/bin/bash
# Every baseline BALAR compares against, on AR-Bench, with our reasoner and Jev as the referee.
#
# BALAR (arXiv 2605.05386) evaluates four on AR-Bench-DC and AR-Bench-SP: Few-Shot (the best
# method reported by AR-Bench), UoT, ToT and Proactive CoT, all at 25 turns with AR-Bench's best
# hyper-parameters. AR-Bench ships two few-shot variants, so both are run. Guess Number is ours:
# BALAR does not cover it, and its feedback is a pure function, so no referee is involved there.
#
# AR-Bench's evaluator resumes from whatever is already in the output file, so an interrupted run
# is re-entered simply by running this again. Order is GN (no referee) then SP then DC, so that a
# referee problem cannot silently spoil the cheapest results.
set -uo pipefail
cd "$HOME/code" || exit 1
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
METHODS=${METHODS:-"few_shot few_shot_inst proactive_cot uot tot"}
TASKS=${TASKS:-"gn sp dc"}
for TASK in $TASKS; do
  for METHOD in $METHODS; do
    OUT="$HOME/results/${TASK}_${METHOD}.json"
    DONE=$($HOME/venv/bin/python - "$OUT" <<'PY'
import json, sys
try:
    print(len(json.load(open(sys.argv[1]))))
except Exception:
    print(0)
PY
)
    if [ "$DONE" -ge 100 ]; then echo "[skip] $TASK/$METHOD already has $DONE"; continue; fi
    echo "===== $TASK / $METHOD  (resuming from $DONE) $(date -Is)"
    TASK=$TASK METHOD=$METHOD MODEL=$MODEL bash run_baselines.sh >> "$HOME/logs/${TASK}_${METHOD}.out" 2>&1
    AFTER=$($HOME/venv/bin/python - "$OUT" <<'PY'
import json, sys
try:
    print(len(json.load(open(sys.argv[1]))))
except Exception:
    print(0)
PY
)
    echo "      -> $AFTER / 100 $(date -Is)"
  done
done
echo "ALL BASELINES DONE $(date -Is)"
