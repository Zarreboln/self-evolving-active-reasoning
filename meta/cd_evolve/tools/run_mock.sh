#!/bin/bash
# Smoke-test a candidate harness on the search instances with a scripted
# task model (no GPU, seconds). Catches crashes and shows what the harness
# actually sends. Policies: greedy (oracle) | sequential | random | follow | flaky
#   tools/run_mock.sh agents/<name>.py [policy] [n_instances]
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
H=${1:?agents/<name>.py}; POL=${2:-sequential}; N=${3:-10}
OUT=$HERE/logs/mock/$(basename "$H" .py)__$POL.jsonl
rm -f "$OUT"
python3 "$ROOT/runner/cd_harness_run.py" --harness "$HERE/$H" --mock "$POL" --split search \
  --limit "$N" --seeds 1 --concurrency 16 --out "$OUT" >/dev/null
python3 "$ROOT/runner/cd_score.py" "$OUT" --split search
echo "trace: python3 $HERE/tools/show_trace.py $OUT --key <k>"
