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
    'mask_fraction': 'MASK_FRACTION', 'ttt_msa_clusters': 'TTT_MSA_CLUSTERS',
    'ttt_crop_size': 'TTT_CROP_SIZE',
    'eval_interval': 'EVAL_INTERVAL', 'msa_dir': 'MSA_DIR',
    'benchmark_csv': 'BENCHMARK_CSV', 'data_dir': 'DATA_DIR',
    'output_dir': 'OUTPUT_DIR', 'model_name': 'MODEL_NAME',
    'start_idx': 'START_IDX', 'end_idx': 'END_IDX',
    'protein_ids': 'PROTEIN_IDS', 'seed': 'SEED',
    'optimizer': 'OPTIMIZER', 'grad_accum_steps': 'GRAD_ACCUM_STEPS',
    'lambda_pair': 'LAMBDA_PAIR',
    'ttt_prev_num_recycle': 'TTT_PREV_NUM_RECYCLE',
}
bool_mapping = {
    'skip_existing': 'SKIP_EXISTING',
    'skip_baseline': 'SKIP_BASELINE',
    'lora_triangle_attention': 'LORA_TRIANGLE_ATTENTION',
    'block_mask': 'BLOCK_MASK',
    'ttt_recycle_prev': 'TTT_RECYCLE_PREV',
    'distogram_consistency': 'DISTOGRAM_CONSISTENCY',
}
for key, env in mapping.items():
    if key in cfg and cfg[key] is not None:
        print(f'{env}=\"\${{{env}:-{cfg[key]}}}\"')
for key, env in bool_mapping.items():
    if cfg.get(key):
        print(f'{env}=true')
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
TTT_MSA_CLUSTERS="${TTT_MSA_CLUSTERS:-}"
TTT_CROP_SIZE="${TTT_CROP_SIZE:-}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
MODEL_NAME="${MODEL_NAME:-model_1_ptm}"
SEED="${SEED:-0}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"
PROTEIN_IDS="${PROTEIN_IDS:-}"
OPTIMIZER="${OPTIMIZER:-adam}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LORA_TRIANGLE_ATTENTION="${LORA_TRIANGLE_ATTENTION:-}"
BLOCK_MASK="${BLOCK_MASK:-}"
TTT_RECYCLE_PREV="${TTT_RECYCLE_PREV:-}"
TTT_PREV_NUM_RECYCLE="${TTT_PREV_NUM_RECYCLE:-}"
DISTOGRAM_CONSISTENCY="${DISTOGRAM_CONSISTENCY:-}"
LAMBDA_PAIR="${LAMBDA_PAIR:-0.1}"

# ---- MSA generation (optional, for single-sequence input) ----
JACKHMMER_BIN="${JACKHMMER_BIN:-}"
SEQ_DATABASE="${SEQ_DATABASE:-}"
MSA_N_CPU="${MSA_N_CPU:-8}"

# ---- Per-run output subdirectory ----
RUN_TAG="lr${TTT_LR}_r${LORA_RANK}_b${LAST_N_BLOCKS}_a${LORA_ALPHA}_s${TTT_STEPS}"
OUTPUT_DIR="${OUTPUT_DIR}/${RUN_TAG}_${SLURM_JOB_ID:-local}"

echo "============================================"
echo "EvoTTT benchmark run"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Node:            $(hostname)"
echo "Config file:     ${CONFIG_FILE}"
echo "Python:          $(which python3) ($(python3 --version 2>&1))"
echo "--------------------------------------------"
echo "Model:           ${MODEL_NAME}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "MSA dir:         ${MSA_DIR}"
echo "Benchmark CSV:   ${BENCHMARK_CSV}"
echo "Data dir:        ${DATA_DIR}"
echo "--------------------------------------------"
echo "TTT steps:       ${TTT_STEPS}"
echo "TTT lr:          ${TTT_LR}"
echo "LoRA rank:       ${LORA_RANK}"
echo "LoRA alpha:      ${LORA_ALPHA}"
echo "Last N blocks:   ${LAST_N_BLOCKS}"
echo "Optimizer:       ${OPTIMIZER}"
echo "Mask fraction:   ${MASK_FRACTION}"
echo "Block mask:      ${BLOCK_MASK:-false}"
echo "Grad accum:      ${GRAD_ACCUM_STEPS}"
echo "Distogram cons:  ${DISTOGRAM_CONSISTENCY:-false}"
echo "Lambda pair:     ${LAMBDA_PAIR}"
echo "LoRA tri attn:   ${LORA_TRIANGLE_ATTENTION:-false}"
echo "MSA clusters:    ${TTT_MSA_CLUSTERS:-none}"
echo "Crop size:       ${TTT_CROP_SIZE:-none}"
echo "Eval interval:   ${EVAL_INTERVAL}"
echo "Seed:            ${SEED}"
echo "Start idx:       ${START_IDX}"
echo "End idx:         ${END_IDX:-none}"
echo "Protein IDs:     ${PROTEIN_IDS:-none}"
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
  --optimizer "${OPTIMIZER}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
)

if [[ "${SKIP_EXISTING:-}" == "true" ]]; then
  CMD+=(--skip_existing)
fi

if [[ "${SKIP_BASELINE:-}" == "true" ]]; then
  CMD+=(--skip_baseline)
fi

if [[ -n "${TTT_MSA_CLUSTERS}" ]]; then
  CMD+=(--ttt_msa_clusters "${TTT_MSA_CLUSTERS}")
fi

if [[ -n "${TTT_CROP_SIZE}" ]]; then
  CMD+=(--ttt_crop_size "${TTT_CROP_SIZE}")
fi

if [[ -n "${END_IDX}" ]]; then
  CMD+=(--end_idx "${END_IDX}")
fi

if [[ -n "${PROTEIN_IDS}" ]]; then
  CMD+=(--protein_ids "${PROTEIN_IDS}")
fi

if [[ "${LORA_TRIANGLE_ATTENTION:-}" == "true" ]]; then
  CMD+=(--lora_triangle_attention)
fi

if [[ "${DISTOGRAM_CONSISTENCY:-}" == "true" ]]; then
  CMD+=(--distogram_consistency --lambda_pair "${LAMBDA_PAIR}")
fi

if [[ "${BLOCK_MASK:-}" == "true" ]]; then
  CMD+=(--block_mask)
fi

if [[ "${TTT_RECYCLE_PREV:-}" == "true" ]]; then
  CMD+=(--ttt_recycle_prev)
fi

if [[ -n "${TTT_PREV_NUM_RECYCLE}" ]]; then
  CMD+=(--ttt_prev_num_recycle "${TTT_PREV_NUM_RECYCLE}")
fi

if [[ -n "${JACKHMMER_BIN}" && -n "${SEQ_DATABASE}" ]]; then
  CMD+=(--jackhmmer_binary_path "${JACKHMMER_BIN}"
        --seq_database_path "${SEQ_DATABASE}"
        --msa_n_cpu "${MSA_N_CPU}")
fi

"${CMD[@]}" "$@"

echo "Job finished."
