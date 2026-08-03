#!/bin/bash
#SBATCH -J bs_blind
#SBATCH -o find_bs_multi_node.%j.out
#SBATCH -e find_bs_multi_node.%j.err
#SBATCH -p normal
#SBATCH -N 1                    # single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 48:00:00             # wall; TIMEOUT below self-stops ~1h under this
#SBATCH -A DMS25004
#SBATCH --mail-user=ehasani@utexas.edu
#SBATCH --mail-type=END,FAIL

# BLIND, RESUMABLE partition search of ONE tuple -- the setup we reuse at any n.
# CURRENT: n=40 tuple 0 on 8 nodes, 2h -- a VALIDATION run (known solution
# results_lonestar/BS_41_40_V7Parallel_tuple0_25078s.txt, ~7h single-node, so
# ~50min expected on 8 nodes). To retarget, edit N / TUPLE / NNODES / TIMEOUT
# and the #SBATCH -N / -t lines to match.
#
#   --partition R/N     provably-exact CD-tree coverage, coordination-free (no gaps)
#   --ab-limit unlimited a finite cap could SKIP the solution's CD -> never found
#   --timeout           binary self-stops CLEANLY under the wall: emits PERF stats,
#                       coverage cert, and flushes the resume log (a SLURM SIGKILL
#                       at the wall gives NONE of that -- that's what bit job 3270492)
#   --checkpoint-dir    per-rank resume log of finished CD-PREFIXES (prefix-level, so
#                       progress survives even when no whole pair completes); RESUBMIT the
#                       same script to continue instead of restarting from scratch
#   --coverage-dir      per-rank completion markers + exhaustive-coverage certificate
#   --found-flag        first rank to find it writes the sentinel; peers stop in ~5s
#   --shard-depth 4     16^4=65536 chunks/pair /45 ~= 1456 per node (cv ~0.8% at /8)

# ---- knobs ----
NNODES=1
N=${N:-42}                        # override per job: sbatch --export=ALL,N=42,TUPLE=95 ...
TUPLE=${TUPLE:-0}
DEPTH=6
TIMEOUT=169200                  # 47h: MUST be < the 48h -t wall (leaves ~1h to
                                # write the cert, flush resume log, roll up summary)

# ---- auto-chain: keep re-queueing 47h windows until the tuple is settled ----
# One 48h wall is not enough for a hard tuple, so each job queues its OWN successor
# (see the CHAIN block below) with --dependency=afterany, then resumes from
# CHECKPOINT_DIR. Only ONE job per tuple is ever in flight (stays under the queue's
# MaxSubmitJobs), and the successor accrues 47h of AGE priority while this one runs,
# so it starts almost immediately instead of waiting behind a cold submission.
WINDOW=${WINDOW:-1}             # which window this job is (1-based; set by the chain)
MAX_WINDOWS=${MAX_WINDOWS:-8}   # hard stop: 8 x 47h ~= 15.6 days per tuple

source "$HOME/.cargo/env"
cd base_sequences
CPT=${SLURM_CPUS_PER_TASK:-128}
export SRUN_CPUS_PER_TASK=$CPT

# Persistent (NOT job-scoped) dirs so a RESUBMISSION continues instead of restarting.
# Keyed on problem + partition geometry: changing N/TUPLE/NNODES/DEPTH uses a fresh
# set (old resume markers are meaningless under new ownership/prefix encoding).
# To force a clean restart, rm -rf these three dirs.
TAG=n${N}_t${TUPLE}_N${NNODES}_d${DEPTH}
FOUND_FLAG=$SCRATCH/bs_found_${TAG}
COVERAGE_DIR=$SCRATCH/cov_${TAG}
CHECKPOINT_DIR=$SCRATCH/ckpt_${TAG}
LOG_DIR=$SCRATCH/blind_${TAG}_${SLURM_JOB_ID}   # per-attempt logs (fresh each run)
mkdir -p "$COVERAGE_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR  (reused across resubmissions => resume)"
echo "COVERAGE_DIR=$COVERAGE_DIR  FOUND_FLAG=$FOUND_FLAG"
echo "WINDOW=$WINDOW/$MAX_WINDOWS"

# A chained successor is queued with --dependency=afterany, so it also runs when the
# predecessor SUCCEEDS. Bail out immediately (costing seconds, not 48h of a node) if
# this tuple is already settled.
if [ -f "$FOUND_FLAG" ]; then
    echo "ALREADY SOLVED (found flag exists) -- nothing to do. Chain ends here."; cat "$FOUND_FLAG"; exit 0
fi
if [ "$(ls "$COVERAGE_DIR" 2>/dev/null | grep -c '\.done$')" -ge "$NNODES" ]; then
    echo "ALREADY EXHAUSTED (all $NNODES rank(s) certified) -- no solution exists for tuple $TUPLE. Chain ends here."; exit 0
fi

# Build WITH AVX2 on a compute node so target-cpu=native matches the run nodes.
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin find_bs_multi_node || exit 1
BIN=target/release/find_bs_multi_node

# ---- provenance into LOG_DIR (so analyze_perftest.sh yields a full summary) ----
{
  echo "rustc: $(rustc --version)"
  echo "rustflags_env: -C target-cpu=native"
  echo "cargo_config_target_cpu: $(grep -E '^[[:space:]]*rustflags' .cargo/config.toml 2>/dev/null | grep -o 'target-cpu=[^\"]*' | head -1 || echo none)"
  echo "avx_fma_mul_ops: $(objdump -d $BIN | grep -cE 'vfmadd|vmulpd|vaddpd')"
  echo "avx512_zmm_refs: $(objdump -d $BIN | grep -cE '%zmm')"
  echo "git_rev: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty_files: $(git status --porcelain 2>/dev/null | wc -l)"
  echo "algo: original CD backtracking (audit H1/M2/M3/L8 measured net-negative and reverted; git_rev is source of truth)"
} > "$LOG_DIR/build_info.txt"
cat "$LOG_DIR/build_info.txt"

# PRE-FLIGHT: refuse to burn the nodes on a scalar build (~2.2x slower).
AVX=$(grep -oE 'avx_fma_mul_ops: [0-9]+' "$LOG_DIR/build_info.txt" | awk '{print $2}')
if [ "${AVX:-0}" -eq 0 ]; then
  echo "FATAL: AVX2/FMA not compiled in (avx_fma_mul_ops=0) -- scalar fallback. Aborting." >&2
  exit 1
fi

{
  echo "job_id: $SLURM_JOB_ID"
  echo "nodes: $NNODES"
  echo "n: $N"
  echo "tuple: $TUPLE"
  echo "shard_depth: $DEPTH"
  echo "timeout_s: $TIMEOUT"
  echo "checkpoint_dir: $CHECKPOINT_DIR"
  echo "window: $WINDOW/$MAX_WINDOWS"
  echo "start: $(date -Is)"
} > "$LOG_DIR/run_config.txt"

# ---- CHAIN: queue the NEXT window NOW, held until this job ends ----
# Submitted here (post-build, pre-search) on purpose:
#   * after the build+AVX preflight => a broken build never spawns a chain
#   * before the 47h search         => the successor sits in the queue accruing AGE
#                                      priority the whole time, so it starts right
#                                      after this one instead of from a cold start
# afterany (not afterok) so the chain survives a timeout/failure; the successor's
# early-exit guard above handles the "already solved/exhausted" cases.
SCRIPT=${SCRIPT:-$SLURM_SUBMIT_DIR/multi_node_test.slurm}
[ -f "$SCRIPT" ] || SCRIPT=$SLURM_SUBMIT_DIR/base_sequences/multi_node_test.slurm
[ -f "$SCRIPT" ] || SCRIPT=$HOME/base_sequences/multi_node_test.slurm
if [ "$WINDOW" -ge "$MAX_WINDOWS" ]; then
  echo "CHAIN: window $WINDOW is the last (MAX_WINDOWS=$MAX_WINDOWS); not queueing a successor."
elif [ ! -f "$SCRIPT" ]; then
  echo "CHAIN: DISABLED -- cannot locate multi_node_test.slurm (tried \$SLURM_SUBMIT_DIR and \$HOME)." >&2
  echo "CHAIN: resubmit by hand after this window, or re-launch with SCRIPT=/abs/path in --export." >&2
else
  NEXT=$((WINDOW + 1))
  CHAIN_EXPORT="ALL,N=$N,TUPLE=$TUPLE,WINDOW=$NEXT,MAX_WINDOWS=$MAX_WINDOWS,SCRIPT=$SCRIPT"
  if OUT=$(sbatch --dependency=afterany:$SLURM_JOB_ID --export="$CHAIN_EXPORT" "$SCRIPT" 2>&1); then
    echo "CHAIN: queued window $NEXT/$MAX_WINDOWS (held until job $SLURM_JOB_ID ends) -> $OUT"
    echo "chain_next: $OUT" >> "$LOG_DIR/run_config.txt"
  else
    echo "CHAIN: sbatch from the compute node FAILED -- $OUT" >&2
    echo "CHAIN: resume by hand from login1 once this window ends:" >&2
    echo "  sbatch --export=ALL,N=$N,TUPLE=$TUPLE,WINDOW=$NEXT,MAX_WINDOWS=$MAX_WINDOWS $SCRIPT" >&2
  fi
fi

# Per-node machine info + the REAL core count each task sees. nproc=1 == the
# single-thread bug that made job 3267766 crawl; -c on srun is the fix.
srun -c "$CPT" bash -c 'echo "node=${SLURM_PROCID} host=$(hostname) nproc=$(nproc) avx512=$(grep -qm1 avx512f /proc/cpuinfo && echo yes || echo no) cpu=\"$(lscpu | sed -n "s/^Model name: *//p" | head -1)\"" > '"$LOG_DIR"'/machine_node${SLURM_PROCID}.txt'
echo "core counts seen by tasks (want $CPT, NOT 1):"
grep -ho 'nproc=[0-9]*' "$LOG_DIR"/machine_node*.txt | sort | uniq -c

echo "=== start $(date) : $NNODES nodes, n=$N tuple $TUPLE, partition /$NNODES depth $DEPTH, timeout ${TIMEOUT}s, ab-limit UNLIMITED ==="

# -c on srun (NOT just SRUN_CPUS_PER_TASK -- the env export alone left nproc=1 on ls6).
srun -c "$CPT" bash -c "./$BIN $N \
    --tuple $TUPLE \
    --partition \${SLURM_PROCID}/$NNODES \
    --shard-depth $DEPTH \
    --ab-limit unlimited \
    --timeout $TIMEOUT \
    --found-flag $FOUND_FLAG \
    --coverage-dir $COVERAGE_DIR \
    --checkpoint-dir $CHECKPOINT_DIR \
    --log $LOG_DIR/node\${SLURM_PROCID}.log \
    > $LOG_DIR/node\${SLURM_PROCID}.out 2>&1"

echo "end: $(date -Is)" >> "$LOG_DIR/run_config.txt"
echo "=== end $(date) ==="

# Roll up the one shareable summary (build/machine/throughput/balance/stability).
# analyze_perftest.sh lives in the repo main dir (we cd'd into base_sequences).
if [ -f analyze_perftest.sh ]; then
  bash analyze_perftest.sh "$LOG_DIR" || true
else
  echo "NOTE: analyze_perftest.sh not found in $(pwd); summary not generated. Logs in $LOG_DIR"
fi

# Bundle EVERYTHING (summary + per-node .out/.log + machine/build/run info) into one
# archive so the whole run can be pulled back with a single scp of TARBALL.
TARBALL=$SCRATCH/perftest_${TAG}_${SLURM_JOB_ID}.tar.gz
if tar -czf "$TARBALL" -C "$(dirname "$LOG_DIR")" "$(basename "$LOG_DIR")" 2>/dev/null; then
  echo "BRING-BACK TARBALL: $TARBALL  ($(du -h "$TARBALL" | cut -f1))"
  echo "  retrieve with:  scp ls6:$TARBALL ."
else
  echo "NOTE: could not create bring-back tarball; logs remain in $LOG_DIR"
fi

# ---- verdict ----
# The successor (window $((WINDOW+1))) is ALREADY queued and starts when this job ends.
# It re-runs the early-exit guard at the top, so a find/exhaustion here stops the chain
# on its own -- nothing below needs to cancel anything.
if [ -f "$FOUND_FLAG" ]; then
    echo "SOLUTION FOUND (window $WINDOW):"; cat "$FOUND_FLAG"
    ls -t BS_${N}+1_* BS_*tuple${TUPLE}* 2>/dev/null | head
    echo "CHAIN: the queued successor will start, see the found flag, and exit in seconds."
else
    markers=$(ls "$COVERAGE_DIR" 2>/dev/null | grep -c '\.done$')
    echo "No solution flag this run (window $WINDOW/$MAX_WINDOWS)."
    echo "coverage markers (ranks fully exhausted, CUMULATIVE across windows): $markers/$NNODES"
    banked=$(cat "$CHECKPOINT_DIR"/*_pfx.log 2>/dev/null | grep -vc '^#')
    echo "resume log: $banked CD-prefix(es) banked so far ($CHECKPOINT_DIR)"
    if [ "${markers:-0}" -ge "$NNODES" ]; then
      echo "ALL ranks exhausted => tuple $TUPLE FULLY searched at n=$N, NO solution exists."
      echo "CHAIN: the queued successor will start, see the coverage cert, and exit in seconds."
    elif [ "$WINDOW" -ge "$MAX_WINDOWS" ]; then
      echo "Hit the self-timeout, and this was the LAST window (MAX_WINDOWS=$MAX_WINDOWS)."
      echo "  To keep going, re-launch (it resumes from the banked prefixes above):"
      echo "  sbatch --export=ALL,N=$N,TUPLE=$TUPLE,MAX_WINDOWS=$MAX_WINDOWS $SCRIPT"
    else
      echo "Hit the self-timeout before exhausting -- EXPECTED for a hard tuple."
      echo "CHAIN: window $((WINDOW+1))/$MAX_WINDOWS is already queued and resumes from the"
      echo "  banked prefixes above. Nothing to do by hand."
    fi
fi
