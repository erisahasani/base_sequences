#!/bin/bash
#SBATCH -J n42_perftest
#SBATCH -o find_bs_multi_node.%j.out
#SBATCH -e find_bs_multi_node.%j.err
#SBATCH -p normal
#SBATCH -N 12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 02:00:00
#SBATCH -A DMS25004
#SBATCH --mail-user=ehasani@utexas.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# PERF / SHAKEOUT test: 12 nodes partition ONE tuple (95) of n=42 for ~2h.
# It will NOT finish or certify coverage. Goal: per-node throughput, cross-node
# balance, AVX width on the real CPU, plumbing/stability. Everything needed to
# assess performance is written under $LOG_DIR; run analyze_perftest.sh after.

source "$HOME/.cargo/env"
cd base_sequences

NNODES=12
N=42
TUPLE=95
DEPTH=4
TIMEOUT=6900   # ~115 min: graceful exit before the 2h SLURM kill

LOG_DIR=$SCRATCH/perftest_n${N}_${SLURM_JOB_ID}
COV_DIR=$LOG_DIR/cov
FOUND_FLAG=$LOG_DIR/found
mkdir -p "$COV_DIR"
echo "LOG_DIR=$LOG_DIR"

# Build on a compute node so target-cpu=native matches the run nodes.
cargo build --release --bin find_bs_multi_node
BIN=target/release/find_bs_multi_node

# --- build context (critical for the AVX-512 question) ---
{
  echo "rustc: $(rustc --version)"
  echo "rustflags_env: ${RUSTFLAGS:-<unset>}"
  echo "cargo_config_target_cpu: $(grep -E '^[[:space:]]*rustflags' .cargo/config.toml 2>/dev/null | grep -o 'target-cpu=[^\"]*' | head -1 || echo none)"
  echo "avx_fma_mul_ops: $(objdump -d $BIN | grep -cE 'vfmadd|vmulpd|vaddpd')"
  echo "avx512_zmm_refs: $(objdump -d $BIN | grep -cE '%zmm')"
} > "$LOG_DIR/build_info.txt"
cat "$LOG_DIR/build_info.txt"

# --- run config ---
{
  echo "job_id: $SLURM_JOB_ID"
  echo "nodes: $NNODES"
  echo "n: $N"
  echo "tuple: $TUPLE"
  echo "shard_depth: $DEPTH"
  echo "timeout_s: $TIMEOUT"
  echo "start: $(date -Is)"
} > "$LOG_DIR/run_config.txt"

# --- per-node machine info (detects heterogeneous partitions) ---
srun bash -c 'echo "node=${SLURM_PROCID} host=$(hostname) nproc=$(nproc) avx512=$(grep -qm1 avx512f /proc/cpuinfo && echo yes || echo no) cpu=\"$(lscpu | sed -n "s/^Model name: *//p" | head -1)\"" > '"$LOG_DIR"'/machine_node${SLURM_PROCID}.txt'

echo "=== start $(date) : $NNODES nodes, n=$N tuple $TUPLE, partition /$NNODES, depth $DEPTH, ~2h ==="

# Per-node stdout captured (PERF lines + progress) for analysis.
srun bash -c "./$BIN $N \
    --tuple $TUPLE \
    --partition \${SLURM_PROCID}/$NNODES \
    --shard-depth $DEPTH \
    --coverage-dir $COV_DIR \
    --timeout $TIMEOUT \
    --ab-limit unlimited \
    --found-flag $FOUND_FLAG \
    --log $LOG_DIR/node\${SLURM_PROCID}.log \
    > $LOG_DIR/node\${SLURM_PROCID}.out 2>&1"

echo "end: $(date -Is)" >> "$LOG_DIR/run_config.txt"
echo "=== end $(date) ==="

# Build the single shareable summary file.
./rust_code/analyze_perftest.sh "$LOG_DIR" || true
echo
echo "SHARE THIS FILE: $LOG_DIR/perftest_summary.txt"
rm -f "$FOUND_FLAG"
