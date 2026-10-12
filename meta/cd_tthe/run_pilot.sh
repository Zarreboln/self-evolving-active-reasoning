#!/bin/bash
# One-shot launcher for the Engaging pilot of the TTHE loop (run on the mac, after `ssh engaging`
# has passed Duo so the ControlMaster socket is alive):
#   meta/cd_tthe/run_pilot.sh            # submits a 7B server (6 h), opens the tunnel, starts strict + labeled pilots
# Resume a stopped run: python3 meta/cd_tthe/optimize.py --run-name <name> --start-batch <k> ... (same args)
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
LOCAL=${LOCAL_PORT:-18000}
EP_NAME=serve_7b_tthe.endpoint
BATCHES=${BATCHES:-3}          # pilot: 3 batches x 10 instances per arm
B=${B:-10}; G=${G:-3}; RNDS=${RNDS:-2}

echo "[1/4] server job"
ssh -o BatchMode=yes engaging "cd ~/code/ar_harness && squeue -u \$USER -h -n serve7b -o '%i %T %M' ; \
  if ! squeue -u \$USER -h -n serve7b -o %T | grep -q RUNNING; then sbatch --export=ALL,EP_NAME=$EP_NAME engaging/serve_7b.sbatch; fi" || { echo "ssh failed (Duo?)"; exit 1; }

echo "[2/4] waiting for the endpoint file ($EP_NAME)"
for i in $(seq 1 90); do
  EP=$(ssh -o BatchMode=yes engaging "cat ~/results/ar_harness/$EP_NAME 2>/dev/null")
  [ -n "$EP" ] && break; sleep 20
done
[ -n "${EP:-}" ] || { echo "no endpoint after 30 min"; exit 1; }
NODE=${EP%%:*}; PORT=$(echo "$EP" | cut -d: -f2); echo "endpoint $NODE:$PORT"

echo "[3/4] tunnel 127.0.0.1:$LOCAL -> $NODE:$PORT"
ssh -O cancel -L "$LOCAL:$NODE:$PORT" engaging >/dev/null 2>&1
ssh -O forward -L "$LOCAL:$NODE:$PORT" engaging || ssh -f -N -L "$LOCAL:$NODE:$PORT" engaging
curl -sf -m 10 "http://127.0.0.1:$LOCAL/health" >/dev/null && echo "health ok" || { echo "tunnel/health failed"; exit 1; }

echo "[4/4] pilots (strict, labeled) -> meta/cd_tthe/logs/"
cd "$R/meta/cd_tthe"
LIMIT=$((BATCHES * B))
for ARM in strict labeled; do
  NAME=pilot_${ARM}_b${B}g${G}r${RNDS}
  env -u CLAUDECODE -u ANTHROPIC_API_KEY nohup python3 optimize.py --run-name "$NAME" --signals "$ARM" --arm scratch \
    --proposer-backend claude --proposer-model opus --effort high \
    --solver-base-url "http://127.0.0.1:$LOCAL/v1" --solver-model Qwen/Qwen2.5-7B-Instruct \
    --split heldout --stream-seed 0 --limit "$LIMIT" --batch-size "$B" --group "$G" --rounds "$RNDS" \
    --propose-timeout 900 --judge-timeout 600 --solve-timeout 900 --workers 10 > "logs/${NAME}_nohup.out" 2>&1 &
  echo "  launched $NAME (pid $!)"
  sleep 60     # stagger the two arms' proposer sessions
done
echo "done. watch: tail -f meta/cd_tthe/logs/pilot_*_nohup.out ; trajectory: python3 meta/cd_tthe/trajectory.py meta/cd_tthe/logs/<run>"
