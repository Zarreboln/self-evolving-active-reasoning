#!/bin/bash
# Keep the evolution loop going across crashes, dropped sessions and usage limits.
#
# Two things make this more than a retry wrapper.
#
# One iteration per launch. The loop advances its own counter even when the proposer returns
# nothing, so a usage limit hit at iteration 1 would burn all five in minutes -- that is what the
# first run did, twice, over a different failure. Launching with --iterations 1 means a failed
# proposer costs one launch, writes no line to evolution_summary.jsonl, and the next attempt
# retries the same iteration.
#
# A probe before each launch. A forty-minute proposer session started against an exhausted limit
# is forty minutes lost, so the limit is checked with one cheap call first and the supervisor waits
# instead. The endpoint is re-read every attempt too: the serve job is replaced at its time limit
# and returns on a new port.
#
#   nohup setsid bash supervise.sh <run-name> <iterations> &
set -uo pipefail
RUN=${1:-pilot1}
WANT=${2:-5}
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=$HERE/logs/$RUN/supervisor.log
mkdir -p "$(dirname "$LOG")"

export PATH="$HOME/.local/bin:$PATH"
export MH_HOST=""
export MH_PYTHON="$HOME/venv/bin/python"

say () { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

done_iters () {
  local f=$HERE/logs/$RUN/evolution_summary.jsonl
  [ -f "$f" ] || { echo 0; return; }
  "$MH_PYTHON" - "$f" <<'PY'
import json, sys
m = 0
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        m = max(m, json.loads(line).get("iteration", 0))
    except json.JSONDecodeError:
        pass
print(m)
PY
}

stall=0
while :; do
  n=$(done_iters)
  if [ "$n" -ge "$WANT" ]; then
    say "$RUN finished: $n of $WANT iterations recorded"
    exit 0
  fi

  # Is the proposer usable right now?
  probe=$(echo "Reply with exactly: PONG" | timeout 120 claude -p --model opus 2>&1 | head -2)
  if ! printf '%s' "$probe" | grep -q PONG; then
    say "proposer unavailable ($(printf '%s' "$probe" | tr '\n' ' ' | cut -c1-110)); waiting 20m"
    sleep 1200
    continue
  fi

  # Is the policy model up?
  EPF=$(ls -t "$HOME"/results/serve_qwen25_7b*.endpoint 2>/dev/null | head -1)
  EP=$(cat "$EPF" 2>/dev/null)
  if ! curl -sf -m 8 "http://$EP/health" >/dev/null 2>&1; then
    say "policy endpoint ${EP:-none} not answering; waiting 3m"
    sleep 180
    continue
  fi
  export POLICY_BASE_URL="http://$EP/v1" POLICY_API_KEY=x
  export RESPONSE_BASE_URL="http://127.0.0.1:8099/v1" RESPONSE_API_KEY=x

  say "iteration $((n + 1))/$WANT, endpoint $EP"
  uv run python meta_harness.py --iterations 1 --run-name "$RUN" \
    >> "$HERE/logs/$RUN/loop.out" 2>&1
  code=$?

  after=$(done_iters)
  if [ "$after" -gt "$n" ]; then
    say "iteration $after recorded (loop exit $code)"
    stall=0
  else
    stall=$((stall + 1))
    say "no iteration recorded (loop exit $code), stall=$stall"
    if [ "$stall" -ge 3 ]; then
      say "three launches with no progress; waiting 30m before trying again"
      sleep 1800
      stall=0
    else
      sleep 30
    fi
  fi
done
