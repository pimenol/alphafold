#!/usr/bin/env bash
#SBATCH --job-name=af2-bench
#SBATCH --account=open-37-88
#SBATCH --partition=qgpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
#
# Predicts both benchmark datasets with AlphaFold on one A100.
#
# Submit with:
#   sbatch scripts/af2_predict.slurm
set -euo pipefail

REPO=/scratch/project/open-37-88/pimenol/af2ttt/alphafold
ENV_PREFIX=/scratch/project/open-37-88/pimenol/af2ttt/af2env
PARAMS_DIR=/scratch/project/open-37-88/pimenol/af2ttt/af2params

cd "${REPO}"

# AlphaFold's unified memory settings; without these a long chain can trip the
# default 75% allocator cap on a single GPU.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
export TF_CPP_MIN_LOG_LEVEL=2

nvidia-smi --query-gpu=name,memory.total --format=csv

"${ENV_PREFIX}/bin/python" scripts/run_af2_on_dataset.py \
  --dataset fails \
  --out_dir "${REPO}/predictions/fails" \
  --params_dir "${PARAMS_DIR}"

"${ENV_PREFIX}/bin/python" scripts/run_af2_on_dataset.py \
  --dataset lowconf \
  --out_dir "${REPO}/predictions/lowconf" \
  --params_dir "${PARAMS_DIR}"

echo "=== all predictions complete ==="
