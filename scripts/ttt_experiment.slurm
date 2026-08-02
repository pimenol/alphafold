#!/usr/bin/env bash
#SBATCH --job-name=af2-ttt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out
#
# Runs one or more TTT experiments, one per line of TTT_RUNS. Queue waits on
# this cluster dominate a short experiment, so batching several configs into a
# single allocation is much cheaper than a job each.
#
# A failing run does not abort the rest -- a bad hyperparameter should cost its
# own slot, not the whole allocation.
#
# Submit with:
#   sbatch -A open-37-88 -p qgpu -t 8:00:00 \
#          --export=ALL,TTT_RUNS="--name expA --loss violation ...
#   --name expB --loss entropy ..." scripts/ttt_experiment.slurm
set -uo pipefail

REPO=/scratch/project/open-37-88/pimenol/af2ttt/alphafold
ENV_PREFIX=/scratch/project/open-37-88/pimenol/af2ttt/af2env
PARAMS_DIR=/scratch/project/open-37-88/pimenol/af2ttt/af2params

cd "${REPO}"

export TF_CPP_MIN_LOG_LEVEL=2

# The GPU partitions on this cluster are mostly down or in maintenance while the
# CPU partitions sit idle, so the same script has to run either way.
if nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null; then
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export TF_FORCE_UNIFIED_MEMORY=1
  export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
else
  echo "no GPU visible -- running on CPU with ${SLURM_CPUS_PER_TASK:-1} cores"
  export JAX_PLATFORMS=cpu
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-1}"
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi

rc_all=0
while IFS= read -r run_args; do
  [ -z "${run_args// }" ] && continue
  echo ""
  echo "############################################################"
  echo "### $(date '+%Y-%m-%d %H:%M:%S')  ${run_args}"
  echo "############################################################"
  # shellcheck disable=SC2086
  "${ENV_PREFIX}/bin/python" -m ttt.run_ttt --params_dir "${PARAMS_DIR}" ${run_args}
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "### RUN FAILED (exit ${rc}) -- continuing with the next one"
    rc_all=$rc
  fi
done <<< "${TTT_RUNS}"

echo "=== all runs complete (last failure code ${rc_all}) ==="
exit 0
