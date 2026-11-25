#!/bin/bash
# Steps 1b-1c in one 4-GPU job, each benchmark's default agent (TextArena standard, BALROG naive, ReAct, SwiftSage ReAct) (run by serve.sbatch, which has the servers up and POLICY_* set):
# first a smoke pass (2 episodes per runner); if any runner errors, stop there. Otherwise the full test
# split for all four runners at once, sharing the four servers. Every runner resumes from disk.
#   CMD="bash ~/code/multibench/mb/official/chain_1bc.sh" sbatch --job-name=off_1bc ~/code/multibench/mb/official/serve.sbatch
set -u
O=/data/team-bath/tmp_l/official
BAL="env PYTHONPATH=$HOME/code/multibench $HOME/venvs/balrog/bin/python"
run_all() {   # $1 = suffix of the output dir, $2 = n
  python -m mb.official.run_textarena --out $O/textarena$1 --agent standard --split test --n $2 &
  $BAL -m mb.official.run_balrog --out $O/balrog$1 --agent naive --level GoToLocal --split test --n $2 &
  python -m mb.official.run_react_alfworld --out $O/alfworld$1 --agent react --split test ${3:+--n $2} &
  python -m mb.official.run_swiftsage --out $O/sciworld$1 --agent react --split test --n $2 &
  wait
}
echo "== smoke $(date -Is)"
run_all _smoke 2 limit
bad=0
for f in $O/textarena_smoke/summary_*.json $O/balrog_smoke/summary_*.json $O/alfworld_smoke/summary_*.json $O/sciworld_smoke/summary_*.json; do
  [ -f "$f" ] || { echo "missing summary: $f"; bad=1; continue; }
  python - "$f" <<'EOF' || bad=1
import json, sys
rows = json.load(open(sys.argv[1]))
rows = rows if isinstance(rows, list) else [rows]
for r in rows:
    print(json.dumps(r))
    if r.get("errors") or r.get("mean_score") is None:
        sys.exit(1)
EOF
done
if [ $bad = 1 ]; then echo "SMOKE FAILED: stopping before the full run"; exit 1; fi
echo "== full $(date -Is)"
run_all "" 100000
echo "== done $(date -Is)"
