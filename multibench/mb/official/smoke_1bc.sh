#!/bin/bash
# Smoke pass for steps 1b-1c (one GPU, via serve.sbatch): 2 episodes per runner, each benchmark's default
# agent. Exits non-zero if any runner errors, so the full array (afterok) does not start.
set -u
O=/data/team-bath/tmp_l/official
BAL="env PYTHONPATH=$HOME/code/multibench $HOME/venvs/balrog/bin/python"
python -m mb.official.run_textarena --out $O/textarena_smoke --agent standard --split test --n 2 &
$BAL -m mb.official.run_balrog --out $O/balrog_smoke --agent naive --level GoToLocal --split test --n 2 &
python -m mb.official.run_react_alfworld --out $O/alfworld_smoke --agent react --split test --n 2 &
python -m mb.official.run_swiftsage --out $O/sciworld_smoke --agent react --split test --n 2 &
wait
bad=0
for f in $O/textarena_smoke/summary_*.json $O/balrog_smoke/summary_*.json $O/alfworld_smoke/summary_*.json $O/sciworld_smoke/summary_*.json; do
  [ -f "$f" ] || { echo "missing summary: $f"; bad=1; continue; }
  python - "$f" <<'PY' || bad=1
import json, sys
rows = json.load(open(sys.argv[1]))
rows = rows if isinstance(rows, list) else [rows]
for r in rows:
    print(json.dumps(r))
    if r.get("errors") or r.get("mean_score") is None:
        sys.exit(1)
PY
done
[ $bad = 1 ] && { echo "SMOKE FAILED"; exit 1; }
echo "SMOKE OK"
