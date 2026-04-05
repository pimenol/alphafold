#!/bin/bash
#SBATCH --job-name=evottt
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/project/open-35-8/pimenol1/alphafold/jobs/evottt/evottt_%A.log
#SBATCH --error=/scratch/project/open-35-8/pimenol1/alphafold/jobs/evottt/evottt_%A.log
#
# EvoTTT benchmark: baseline AF2 → TTT adaptation → adapted AF2.
#
# Submit from repo root:
#   sbatch jobs/evottt/run_evottt.sh
#
# Optional environment variables (export before sbatch):
#   TTT_STEPS, TTT_LR, LORA_RANK, LAST_N_BLOCKS, LORA_ALPHA
#   START_IDX, END_IDX, PROTEIN_IDS

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
# Ensure conda env's bin is first in PATH
export PATH="${CONDA_ENV}/bin:${PATH}"

# ---- Repo ----
REPO_ROOT="/scratch/project/open-35-8/pimenol1/alphafold"
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---- Memory ----
export PYTHONUNBUFFERED=1
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0

# ---- Load defaults from YAML config ----
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/evottt/default.yaml}"
echo "Loading defaults from: ${CONFIG_FILE}"
eval "$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG_FILE}'))
mapping = {
    'ttt_steps': 'TTT_STEPS', 'ttt_lr': 'TTT_LR', 'lora_rank': 'LORA_RANK',
    'last_n_blocks': 'LAST_N_BLOCKS', 'lora_alpha': 'LORA_ALPHA',
    'mask_fraction': 'MASK_FRACTION',
    'eval_interval': 'EVAL_INTERVAL', 'msa_dir': 'MSA_DIR',
    'benchmark_csv': 'BENCHMARK_CSV', 'data_dir': 'DATA_DIR',
    'output_dir': 'OUTPUT_DIR', 'model_name': 'MODEL_NAME',
    'start_idx': 'START_IDX', 'end_idx': 'END_IDX',
    'protein_ids': 'PROTEIN_IDS', 'seed': 'SEED',
}
for key, env in mapping.items():
    if key in cfg and cfg[key] is not None:
        print(f'{env}=\"\${{{env}:-{cfg[key]}}}\"')
")"

# ---- Fallback defaults (if not in YAML) ----
DATA_DIR="${DATA_DIR:-/scratch/project/open-35-8/pimenol1/af2_data}"
MSA_DIR="${MSA_DIR:-/scratch/project/open-35-8/antonb/bfvd/bfvd_msa}"
BENCHMARK_CSV="${BENCHMARK_CSV:-/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh/data/benchmark/summary.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evottt_outputs}"
TTT_STEPS="${TTT_STEPS:-50}"
TTT_LR="${TTT_LR:-3e-4}"
LORA_RANK="${LORA_RANK:-4}"
LAST_N_BLOCKS="${LAST_N_BLOCKS:-8}"
LORA_ALPHA="${LORA_ALPHA:-1.0}"

MASK_FRACTION="${MASK_FRACTION:-0.15}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
MODEL_NAME="${MODEL_NAME:-model_1_ptm}"
SEED="${SEED:-0}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"
PROTEIN_IDS="${PROTEIN_IDS:-}"

# ---- MSA generation (optional, for single-sequence input) ----
JACKHMMER_BIN="${JACKHMMER_BIN:-}"
SEQ_DATABASE="${SEQ_DATABASE:-}"
MSA_N_CPU="${MSA_N_CPU:-8}"

echo "============================================"
echo "EvoTTT benchmark run"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "TTT: steps=${TTT_STEPS} lr=${TTT_LR} rank=${LORA_RANK} blocks=${LAST_N_BLOCKS} alpha=${LORA_ALPHA}"
echo "Python: $(which python3) ($(python3 --version 2>&1))"
echo "============================================"

CMD=(
  python3 -u scripts/run_evottt_benchmark.py
  --benchmark_csv "${BENCHMARK_CSV}"
  --msa_dir "${MSA_DIR}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --model_name "${MODEL_NAME}"
  --ttt_steps "${TTT_STEPS}"
  --ttt_lr "${TTT_LR}"
  --lora_rank "${LORA_RANK}"
  --last_n_blocks "${LAST_N_BLOCKS}"
  --lora_alpha "${LORA_ALPHA}"

  --mask_fraction "${MASK_FRACTION}"
  --eval_interval "${EVAL_INTERVAL}"
  --seed "${SEED}"
  --start_idx "${START_IDX}"
  --skip_existing
)

if [[ -n "${END_IDX}" ]]; then
  CMD+=(--end_idx "${END_IDX}")
fi

if [[ -n "${PROTEIN_IDS}" ]]; then
  CMD+=(--protein_ids "${PROTEIN_IDS}")
fi

if [[ -n "${JACKHMMER_BIN}" && -n "${SEQ_DATABASE}" ]]; then
  CMD+=(--jackhmmer_binary_path "${JACKHMMER_BIN}"
        --seq_database_path "${SEQ_DATABASE}"
        --msa_n_cpu "${MSA_N_CPU}")
fi

"${CMD[@]}" "$@"

echo "Job finished."
