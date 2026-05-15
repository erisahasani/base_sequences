#!/usr/bin/env bash
# Sweep all canonical tuples for a range of n, each with a per-tuple wall-clock
# timeout. Output files for each n are saved into sweeps/n{N}/.
#
# Usage: ./sweep_all.sh N_START N_END TIMEOUT_SECS
#   N_START       first n (inclusive)
#   N_END         last n (inclusive)
#   TIMEOUT_SECS  per-tuple wall-clock timeout, e.g. 600

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 N_START N_END TIMEOUT_SECS" >&2
  echo "Example: $0 20 35 600" >&2
  exit 1
fi

N_START="$1"
N_END="$2"
TIMEOUT="$3"

# Resolve absolute paths so the script works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/target/release/find_bs_v7_parallel"
SWEEP="$SCRIPT_DIR/sweep_tuples.sh"

if [ ! -x "$BIN" ]; then
  echo "Binary not found: $BIN" >&2
  echo "Run: cargo build --release --bin find_bs_v7_parallel" >&2
  exit 1
fi
if [ ! -x "$SWEEP" ]; then
  echo "sweep_tuples.sh not found or not executable: $SWEEP" >&2
  exit 1
fi

echo "=================================================="
echo "  Sweeping n in [$N_START, $N_END], timeout ${TIMEOUT}s per tuple"
echo "=================================================="

for ((n=N_START; n<=N_END; n++)); do
  OUTDIR="$SCRIPT_DIR/sweeps/n${n}"
  mkdir -p "$OUTDIR"
  echo ""
  echo "##########################################"
  echo "###  n=$n  (output: $OUTDIR)"
  echo "##########################################"
  # Run the per-n sweep inside OUTDIR so the BS_*.txt files land there.
  (cd "$OUTDIR" && "$SWEEP" "$n" "$TIMEOUT" "$BIN") \
    2>&1 | tee "$OUTDIR/sweep.log"
done

echo ""
echo "=================================================="
echo "  All sweeps complete. Results in: $SCRIPT_DIR/sweeps/"
echo "=================================================="
