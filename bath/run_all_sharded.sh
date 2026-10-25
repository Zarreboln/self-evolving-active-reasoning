#!/bin/bash
# Every baseline BALAR compares against, on AR-Bench, sharded across the card.
#
# Same combos and same canonical output paths as run_all_baselines.sh -- only the inner runner
# changes, from one sequential evaluator to NSHARD concurrent ones over contiguous slices. Combos
# already at 100 are skipped, and each shard resumes from its own log, so re-running this is how an
# interrupted pass is repaired. Order is GN (no referee) then SP then DC, so a referee problem
# cannot silently spoil the cheapest results.
set -uo pipefail
cd "$HOME/code" || exit 1
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
METHODS=${METHODS:-"few_shot few_shot_inst proactive_cot uot tot"}
TASKS=${TASKS:-"gn sp dc"}
NSHARD=${NSHARD:-16}
PY=$HOME/venv/bin/python
count() { $PY -c "import json,sys
try: print(len(json.load(open(sys.argv[1]))))
except Exception: print(0)" "$1"; }
for TASK in $TASKS; do
  for METHOD in $METHODS; do
    OUT="$HOME/results/${TASK}_${METHOD}.json"
    DONE=$(count "$OUT")
    # AR-Bench's test sets hold 100 instances, iCraft-MD holds 140, so the target comes from data.
    TOTAL=$($PY -c "import json;print(len(json.load(open('$HOME/code/AR-Bench/data/$TASK/test.json'))))")
    if [ "$DONE" -ge "$TOTAL" ]; then echo "[skip] $TASK/$METHOD already has $DONE/$TOTAL"; continue; fi
    echo "===== $TASK / $METHOD  (have $DONE/$TOTAL) $(date -Is)"
    TASK=$TASK METHOD=$METHOD MODEL=$MODEL NSHARD=$NSHARD bash run_sharded.sh \
        >> "$HOME/logs/${TASK}_${METHOD}.sharded.out" 2>&1
    echo "      -> $(count "$OUT") / $TOTAL $(date -Is)"
  done
done
echo "ALL BASELINES DONE $(date -Is)"
