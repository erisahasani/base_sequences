#!/bin/bash
#SBATCH -J n42_balchk
#SBATCH -o find_bs_multi_node.%j.out
#SBATCH -e find_bs_multi_node.%j.err
#SBATCH -p development
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 02:00:00
#SBATCH -A DMS25004
#SBATCH --mail-user=ehasani@utexas.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# PARTITION-BALANCE test: --count-only (no AB search) so each node's rate
# reflects ONLY CD-tree work. This isolates partition balance from AB-time
# variance and removes the AB stragglers. --timeout is a hard cap: a node that
# fully enumerates its partition reports the EXACT leaf count (partition size);
# one that hits the cap reports a PERF snapshot at that fixed wall time. Either
# way every node emits a parseable PERF block, so no node is censored and the
# job never gets wall-killed mid-srun (cap << SBATCH -t leaves room for analyzer).
# NOTE: requires the find_bs_multi_node.rs fix that makes --count-only honor
# --timeout (checks ABORT_SEARCH at every recurse() entry). Rebuild picks it up.

source "$HOME/.cargo/env"
cd base_sequences

# Make srun inherit the batch --cpus-per-task (else tasks pin to 1 core).
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-128}

NNODES=8
N=42
TUPLE=95
DEPTH=4
TIMEOUT=6000   # 100 min hard cap: finishers report exact size, rest snapshot at T; fits 2h wall

LOG_DIR=$SCRATCH/balance_n${N}_${SLURM_JOB_ID}
mkdir -p "$LOG_DIR"
echo "LOG_DIR=$LOG_DIR"

# Build WITH AVX2 (RUSTFLAGS on the command line, independent of .cargo/config).
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin find_bs_multi_node
BIN=target/release/find_bs_multi_node
{
  echo "rustc: $(rustc --version)"
  echo "avx_fma_mul_ops: $(objdump -d $BIN | grep -cE 'vfmadd|vmulpd|vaddpd')"
  echo "avx512_zmm_refs: $(objdump -d $BIN | grep -cE '%zmm')"
} > "$LOG_DIR/build_info.txt"
cat "$LOG_DIR/build_info.txt"

{
  echo "job_id: $SLURM_JOB_ID"; echo "nodes: $NNODES"; echo "n: $N"; echo "tuple: $TUPLE"
  echo "shard_depth: $DEPTH"; echo "timeout_s: $TIMEOUT"; echo "mode: count-only"; echo "start: $(date -Is)"
} > "$LOG_DIR/run_config.txt"

srun bash -c 'echo "node=${SLURM_PROCID} host=$(hostname) nproc=$(nproc) avx512=$(grep -qm1 avx512f /proc/cpuinfo && echo yes || echo no) cpu=\"$(lscpu | sed -n "s/^Model name: *//p" | head -1)\"" > '"$LOG_DIR"'/machine_node${SLURM_PROCID}.txt'

echo "=== start $(date) : $NNODES nodes, n=$N tuple $TUPLE, partition /$NNODES, depth $DEPTH, COUNT-ONLY, timeout ${TIMEOUT}s ==="

# --count-only: full CD enumeration, no AB search.
srun bash -c "./$BIN $N \
    --tuple $TUPLE \
    --partition \${SLURM_PROCID}/$NNODES \
    --shard-depth $DEPTH \
    --count-only \
    --timeout $TIMEOUT \
    --log $LOG_DIR/node\${SLURM_PROCID}.log \
    > $LOG_DIR/node\${SLURM_PROCID}.out 2>&1"

echo "end: $(date -Is)" >> "$LOG_DIR/run_config.txt"
echo "=== end $(date) ==="

bash analyze_perftest.sh "$LOG_DIR" || true
echo
echo "SHARE THIS FILE: $LOG_DIR/perftest_summary.txt"
