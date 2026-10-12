#!/bin/bash
# Open (or refresh) a local port to the vLLM server published by
# engaging/serve_7b.sbatch, through the existing `engaging` ControlMaster.
#   tunnel.sh            -> prints the local base URL once the server answers
#   tunnel.sh --wait     -> blocks until an endpoint exists and is healthy
set -u
LOCAL=${LOCAL_PORT:-18000}
WAIT=${1:-}
while true; do
  EP=$(ssh -o BatchMode=yes -o ConnectTimeout=15 engaging 'cat ~/results/ar_harness/serve_7b.endpoint 2>/dev/null')
  if [ -n "$EP" ]; then
    NODE=${EP%%:*}; PORT=${EP##*:}
    ssh -O cancel -L "$LOCAL:$NODE:$PORT" engaging >/dev/null 2>&1
    ssh -O forward -L "$LOCAL:$NODE:$PORT" engaging >/dev/null 2>&1 || ssh -f -N -L "$LOCAL:$NODE:$PORT" engaging
    if curl -sf -m 10 "http://127.0.0.1:$LOCAL/health" >/dev/null; then
      echo "http://127.0.0.1:$LOCAL/v1"; exit 0
    fi
    echo "endpoint $EP published but not healthy yet" >&2
  else
    echo "no endpoint yet (is serve7b running?  ssh engaging squeue -u \$USER)" >&2
  fi
  [ "$WAIT" = "--wait" ] || exit 1
  sleep 30
done
