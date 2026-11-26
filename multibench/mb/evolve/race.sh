# Sourced by the GPU job scripts. ORCD rewrites an untyped GPU request to L40S, so "whichever GPU is
# free" is done by submitting one copy per GPU type under the same job name; the first copy to start
# claims the work and cancels its pending siblings; a copy that starts later sees the claim and exits.
GPU_TYPES=${GPU_TYPES:-"l40s a100 h100 h200 rtx_a6000"}   # Blackwell, L4 and MIG left out (see notes)
GPU_PARTS=mit_normal_gpu,mit_preemptable

# race_claim DIR: 0 = this job owns DIR's work (siblings cancelled), 1 = another live copy owns it
race_claim() {
  local d=$1/claim me=$SLURM_JOB_ID owner
  if ! mkdir "$d" 2>/dev/null; then
    owner=$(cat "$d/job" 2>/dev/null)
    if [ -n "$owner" ] && [ "$owner" != "$me" ] && squeue -h -j "$owner" 2>/dev/null | grep -q .; then
      return 1
    fi
  fi
  echo "$me" > "$d/job"
  squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING -o %i | grep -vx "$me" | xargs -r scancel
  return 0
}

# race_submit NAME "SBATCH ARGS" SCRIPT: one copy per GPU type; prints the job ids, comma-separated
race_submit() {
  local name=$1 args=$2 script=$3 ids=() t j
  for t in $GPU_TYPES; do
    j=$(sbatch --parsable -J "$name" --partition=$GPU_PARTS --gres=gpu:$t:1 $args "$script" 2>/dev/null) && ids+=("$j")
  done
  (IFS=,; echo "${ids[*]}")
}
