#!/bin/bash
#SBATCH --job-name=af2_filter
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/project/open-35-8/pimenol1/alphafold/jobs/af2_filter/af2_filter_%A.out
#SBATCH --error=/scratch/project/open-35-8/pimenol1/alphafold/jobs/af2_filter/af2_filter_%A.err
#
# Run baseline AF2 on 100 proteins, filter by pLDDT < 70 and not dynamic.
#
# Submit:  sbatch jobs/af2_filter/run_af2_filter.sh

# ---- CUDA ----
module load CUDA/12.2.2 2>/dev/null || true
module load cuDNN/8.9.2.26-CUDA-12.2.0 2>/dev/null || true
XLA_FLAGS_EXTRA="--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer="
if [[ -n "${EBROOTCUDA:-}" ]]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${EBROOTCUDA} ${XLA_FLAGS_EXTRA}"
else
  export XLA_FLAGS="${XLA_FLAGS_EXTRA}"
fi

# ---- Conda ----
CONDA_ROOT="${CONDA_ROOT:-/scratch/project/open-35-8/pimenol1/miniconda3}"
CONDA_ENV="${CONDA_ROOT}/envs/alphafold_evottt"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate alphafold_evottt 2>/dev/null || true
export PATH="${CONDA_ENV}/bin:${PATH}"

# ---- Repo ----
REPO_ROOT="/scratch/project/open-35-8/pimenol1/alphafold"
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---- Memory ----
export PYTHONUNBUFFERED=1
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0

echo "============================================"
echo "AF2 baseline filter run"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(which python3) ($(python3 --version 2>&1))"
echo "============================================"

python3 -u scripts/run_af2_filter.py \
  --benchmark_csv /scratch/project/open-35-8/data/bfvd/bfvd_beta/sample_100_proteins.csv \
  --msa_dir /scratch/project/open-35-8/data/bfvd/bfvd_beta/input/logan \
  --data_dir /scratch/project/open-35-8/pimenol1/af2_data \
  --output_dir data/bfvd/AF2 \
  --summary_csv data/bfvd/summary.csv \
  --model_name model_1_ptm \
  --skip_existing

echo "Job finished."
