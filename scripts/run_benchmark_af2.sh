#!/bin/bash
#SBATCH --job-name=af2_bench
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/project/open-35-8/pimenol1/alphafold/jobs/af2_benchmark/af2_bench_%A.out
#SBATCH --error=/scratch/project/open-35-8/pimenol1/alphafold/jobs/af2_benchmark/af2_bench_%A.err
#
# Full AlphaFold2 on ProteinTTT benchmark CSV (one structure per protein).
#
# Prerequisites:
#   - Conda env alphafold_evottt with AlphaFold + JAX + MSA tools (jackhmmer, hhblits, hhsearch, ...)
#   - Genetic databases + params under DATA_DIR (same layout as scripts/download_all_data.sh)
#
# Submit from repo root:
#   sbatch jobs/af2_benchmark/run_benchmark_af2.sh
#
# Edit variables below, or export them before sbatch.

# ---- CUDA (match your jaxlib build; comment out if not using modules) ----
module load CUDA/12.2.2 2>/dev/null || true
module load cuDNN/8.9.2.26-CUDA-12.2.0 2>/dev/null || true

if [[ -n "${EBROOTCUDA:-}" ]]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${EBROOTCUDA}"
fi

# ---- Conda ----
CONDA_ROOT="${CONDA_ROOT:-/scratch/project/open-35-8/pimenol1/miniconda3}"
CONDA_ENV="${CONDA_ROOT}/envs/alphafold_evottt"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate alphafold_evottt 2>/dev/null || true
# Ensure conda env's bin is first in PATH (module loads above may have
# prepended a system Python via gompi-2022a dependencies).
export PATH="${CONDA_ENV}/bin:${PATH}"

# ---- Repo ----
REPO_ROOT="/scratch/project/open-35-8/pimenol1/alphafold"
cd "${REPO_ROOT}" || exit 1
mkdir -p jobs/af2_benchmark
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---- Memory ----
export PYTHONUNBUFFERED=1
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0

# ---- Paths ----
DATA_DIR="${DATA_DIR:-/scratch/project/open-35-8/pimenol1/af2_data}"
MSA_DIR="${MSA_DIR:-/scratch/project/open-35-8/data/bfvd/bfvd_beta/input/logan}"
BENCHMARK_CSV="${BENCHMARK_CSV:-${REPO_ROOT}/data/bfvd/summary.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark_af2_outputs}"
MODEL_NAME="${MODEL_NAME:-model_1_ptm}"

# Optional: run a slice (for array jobs or retries)
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"   # empty = all

echo "============================================"
echo "AlphaFold2 benchmark run"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "Python:      $(which python3) ($(python3 --version 2>&1))"
echo "DATA_DIR:    ${DATA_DIR}"
echo "MSA_DIR:     ${MSA_DIR}"
echo "OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "CSV:         ${BENCHMARK_CSV}"
echo "Model:       ${MODEL_NAME}"
echo "============================================"

CMD=(
  python3 -u scripts/run_alphafold_benchmark.py
  --benchmark_csv "${BENCHMARK_CSV}"
  --msa_dir "${MSA_DIR}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --model_name "${MODEL_NAME}"
  --start_idx "${START_IDX}"
)

if [[ -n "${END_IDX}" ]]; then
  CMD+=(--end_idx "${END_IDX}")
fi

if [[ "${SKIP_EXISTING:-}" == "true" || "${SKIP_EXISTING:-}" == "1" ]]; then
  CMD+=(--skip_existing)
fi

"${CMD[@]}" "$@"

echo "Job finished."
