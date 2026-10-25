#!/bin/bash
# Re-enter the baseline queue until every combo has its 100 episodes.
#
# run_all_sharded.sh is a single pass: a combo that dies mid-way (a served model hitting its
# Slurm time limit, a referee hiccup) leaves a partial file and the pass moves on. AR-Bench's
# evaluator resumes from the output file, so the fix is simply to run the pass again. This waits
# for the first pass to announce itself done, then sweeps until nothing is left incomplete.
set -uo pipefail
LOG=$HOME/logs/all_baselines.out
while ! grep -q "ALL BASELINES DONE" "$LOG" 2>/dev/null; do sleep 120; done
for pass in 2 3 4; do
  MISSING=$($HOME/venv/bin/python $HOME/code/count_incomplete.py)
  echo "===== sweep pass $pass: $MISSING incomplete $(date -Is)"
  if [ "$MISSING" -eq 0 ]; then echo "SWEEP CLEAN $(date -Is)"; exit 0; fi
  bash $HOME/code/run_all_sharded.sh >> "$LOG" 2>&1
done
echo "SWEEP GAVE UP AFTER 3 EXTRA PASSES $(date -Is)"
