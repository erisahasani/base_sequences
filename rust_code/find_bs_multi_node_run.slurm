#!/bin/bash
#SBATCH -J n42_test
#SBATCH -o find_bs_multi_node.%j.out
#SBATCH -e find_bs_multi_node.%j.err
#SBATCH -p normal
#SBATCH -N 45
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 48:00:00
#SBATCH -A DMS25004
#SBATCH --mail-user=ehasani@utexas.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/.cargo/env"
cd base_sequences

NNODES=45

cargo build --release --bin find_bs_multi_node
echo "AVX ops in binary: $(objdump -d target/release/find_bs_multi_node | grep -cE 'vfmadd|vmulpd|vaddpd')"

FOUND_FLAG=$SCRATCH/bs_found_${SLURM_JOB_ID}
COV_DIR=$SCRATCH/cov_n42_t95_${SLURM_JOB_ID}
mkdir -p "$COV_DIR"

echo "=== Starting tuple 95 at $(date) (48h, $NNODES nodes, partition depth 4) ==="

srun bash -c "./target/release/find_bs_multi_node 42 \
    --tuple 95 \
    --partition \${SLURM_PROCID}/$NNODES \
    --shard-depth 4 \
    --coverage-dir $COV_DIR \
    --timeout 171000 \
    --ab-limit unlimited \
    --found-flag $FOUND_FLAG \
    --log test_42_t95_node\${SLURM_PROCID}.log"

echo "=== Tuple 95 finished at $(date) ==="
echo "Coverage markers present: $(ls "$COV_DIR" | wc -l) / $NNODES"

rm -f "$FOUND_FLAG"
