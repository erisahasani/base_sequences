#!/usr/bin/env bash
# Sweep every canonical tuple for a given n, each with a per-tuple wall-clock
# timeout. Records one BS_{n+1}_{n}_V7Parallel_tuple{K}_*s.txt per tuple —
# either a solution file or a TIMEOUT file.
#
# Usage: ./sweep_tuples.sh N TIMEOUT_SECS [BIN]
#   N             target n
#   TIMEOUT_SECS  per-tuple wall-clock timeout, e.g. 600
#   BIN           optional path to binary (default: ./target/release/find_bs_v7_parallel)

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 N TIMEOUT_SECS [BIN]" >&2
  exit 1
fi

N="$1"
TIMEOUT="$2"
BIN="${3:-./target/release/find_bs_v7_parallel}"

if [ ! -x "$BIN" ]; then
  echo "Binary not found or not executable: $BIN" >&2
  exit 1
fi

# Count canonical tuples.
NUM_TUPLES=$("$BIN" "$N" --list-tuples 2>/dev/null | grep -cE '^[0-9]+ ')
echo "# Sweeping n=$N over $NUM_TUPLES tuples, timeout ${TIMEOUT}s per tuple"

for ((k=0; k<NUM_TUPLES; k++)); do
  echo ""
  echo "### Tuple $k/$NUM_TUPLES (n=$N)  timeout=${TIMEOUT}s"
  if "$BIN" "$N" --tuple "$k" --timeout "$TIMEOUT" | tee /dev/stderr | grep "SUCCESS!" > /dev/null; then
    echo ""
    echo "# Solution found at tuple $k for n=$N — skipping remaining tuples"
    break
  fi
done

echo ""
echo "# Sweep complete for n=$N. Output files: BS_$((N+1))_${N}_V7Parallel_tuple*.txt"
