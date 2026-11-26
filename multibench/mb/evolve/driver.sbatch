#!/bin/bash
# The mailbox driver (mb/evolve/driver.py) as a CPU job that hands over to its next copy before its time
# limit, so it keeps running without anyone logged in. Stop the chain with `touch ~/mailbox_STOP`.
#   CLUSTER=orcd MAILBOX=~/mailbox RUNS=~/results/mb/evo_b sbatch --partition=mit_normal driver.sbatch
#SBATCH --job-name=mb_driver
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=11:50:00
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
set -uo pipefail
[ -f "$HOME/.mb_site.sh" ] && source "$HOME/.mb_site.sh"
cd "$HOME/code/multibench" && source mb/env.sh
[ -e "$HOME/mailbox_STOP" ] && { echo "STOP file present, not starting"; exit 0; }
# The next copy waits for this one to end; it is cancelled below if this one was told to stop.
NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID --partition="$SLURM_JOB_PARTITION" \
       --export=ALL "$HOME/code/multibench/mb/evolve/driver.sbatch")
echo "host=$(hostname) start=$(date -Is) next=$NEXT"
python -m mb.evolve.driver --mailbox "$MAILBOX" --cluster "$CLUSTER" --runs "$RUNS" \
  --max-jobs "${MAX_JOBS:-4}" --until $(( $(date +%s) + 11*3600 + 30*60 ))
[ -e "$HOME/mailbox_STOP" ] && scancel "$NEXT"
echo "end=$(date -Is)"
