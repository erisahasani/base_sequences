#!/bin/bash
# launch_range.sh — submit ONE 1-node search job per tuple over a range,
# all using the single multi_node_test.slurm (no per-tuple .slurm copies).
#
# Usage:
#   ./launch_range.sh <N> <start>-<end>      e.g.  ./launch_range.sh 44 4-9
#   ./launch_range.sh <N> <start> <end>      e.g.  ./launch_range.sh 44 4 9
#
# Each job runs multi_node_test.slurm with N and TUPLE passed via --export, and is
# named n<N>_t<tuple>. Because TUPLE comes from the loop variable, the name and the
# tuple it actually searches can never disagree (no t8-running-t0 bugs).
#
# MAX_WINDOWS=1 is passed so multi_node_test.slurm does NOT self-chain — the launcher
# is the single control point. To advance a window, just RE-RUN this same command:
# the guard below skips anything still queued/running and resubmits anything that
# stopped, which resumes from its checkpoint.

set -u
SLURM=${SLURM_FILE:-$HOME/base_sequences/multi_node_test.slurm}
N=${1:-}; RANGE=${2:-}; END3=${3:-}

if [ -z "$N" ] || [ -z "$RANGE" ]; then
  echo "Usage: $0 <N> <start>-<end>     e.g.  $0 44 4-9"; exit 1
fi

# parse the range: accept "4-9" (one arg) or "4 9" (two args)
if [[ "$RANGE" == *-* ]]; then
  START=${RANGE%-*}; END=${RANGE#*-}
else
  START=$RANGE; END=${END3:-$RANGE}
fi

if ! [[ "$START" =~ ^[0-9]+$ && "$END" =~ ^[0-9]+$ ]] || [ "$START" -gt "$END" ]; then
  echo "Bad range '$RANGE $END3' — need start<=end, e.g. 4-9"; exit 1
fi
[ -f "$SLURM" ] || { echo "Cannot find slurm file: $SLURM  (set SLURM_FILE=/abs/path)"; exit 1; }

echo "Launching n=$N, tuples $START..$END as individual 1-node jobs via $SLURM"
for t in $(seq "$START" "$END"); do
  # guard: never double-launch a tuple already queued/running (that would put two
  # writers on one checkpoint file). Matches n<N>_t<t> exactly, not n<N>_t<t>0.
  if squeue -u "$USER" -h -o "%j" | grep -qE "^n${N}_t${t}(_|\$)"; then
    echo "  SKIP tuple $t — a job matching n${N}_t${t} is already in the queue"
    continue
  fi
  jid=$(sbatch --parsable --job-name="n${N}_t${t}" \
        --export=ALL,N=$N,TUPLE=$t,MAX_WINDOWS=1 "$SLURM")
  rc=$?
  if [ $rc -eq 0 ] && [[ "$jid" =~ ^[0-9] ]]; then
    echo "  tuple $t -> job ${jid%%;*}  (n${N}_t${t})"
  else
    echo "  tuple $t -> SUBMIT FAILED (sbatch rc=$rc)"
  fi
done
