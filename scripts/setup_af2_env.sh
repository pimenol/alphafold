#!/usr/bin/env bash
#
# Creates a conda environment able to run AlphaFold inference on a GPU and
# downloads the model parameters.
#
# Only what inference needs: the genetic-search tools (jackhmmer, hhblits) and
# their databases are deliberately not installed, because
# scripts/run_af2_on_dataset.py feeds AlphaFold precomputed MSAs instead.
# Relaxation dependencies (openmm, pdbfixer) are skipped too -- predictions are
# scored unrelaxed.
#
# Usage: bash scripts/setup_af2_env.sh /path/to/env /path/to/params
set -euo pipefail

ENV_PREFIX="${1:?usage: setup_af2_env.sh ENV_PREFIX PARAMS_DIR}"
PARAMS_DIR="${2:?usage: setup_af2_env.sh ENV_PREFIX PARAMS_DIR}"

echo "=== creating conda env at ${ENV_PREFIX} ==="
if [[ ! -d "${ENV_PREFIX}" ]]; then
  conda create -y -p "${ENV_PREFIX}" python=3.11
fi

PIP="${ENV_PREFIX}/bin/pip"

echo "=== installing inference dependencies ==="
"${PIP}" install --no-input --upgrade "pip" "setuptools<72.0.0" wheel

# Everything in one resolve. Installing jax first and the rest afterwards lets
# a later dependency silently upgrade jax/jaxlib past the pinned CUDA plugins,
# which breaks with "jax_cuda12_plugin._triton has no attribute
# get_arch_details". jaxlib is pinned explicitly for the same reason.
"${PIP}" install --no-input \
  "jax[cuda12]==0.4.26" \
  "jaxlib==0.4.26" \
  "absl-py==1.0.0" \
  "dm-haiku==0.0.12" \
  "ml-collections==0.1.0" \
  "numpy==1.24.3" \
  "scipy==1.11.4" \
  "tensorflow-cpu==2.16.1" \
  "biopython==1.83"

# Guard against a silent re-resolve: these three must agree.
"${ENV_PREFIX}/bin/python" - <<'PY'
import importlib.metadata as md
versions = {p: md.version(p) for p in
            ('jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda12-pjrt')}
print(versions)
if len(set(versions.values())) != 1:
  raise SystemExit(f'jax/jaxlib/plugin versions disagree: {versions}')
PY

echo "=== verifying imports ==="
# Forced onto CPU: this usually runs on a login node with no GPU, and asking
# for the CUDA backend there raises rather than reporting an empty device list.
# The real GPU check happens inside the batch job.
JAX_PLATFORMS=cpu "${ENV_PREFIX}/bin/python" -c "
import jax, haiku, ml_collections, tensorflow, Bio, numpy
print('jax', jax.__version__, '| haiku', haiku.__version__)
print('numpy', numpy.__version__, '| tf', tensorflow.__version__, '| biopython', Bio.__version__)
"

echo "=== downloading AlphaFold parameters into ${PARAMS_DIR} ==="
# scripts/download_alphafold_params.sh needs aria2c, which is not installed
# here, so fetch the same tarball with curl.
mkdir -p "${PARAMS_DIR}/params"
PARAMS_TAR="${PARAMS_DIR}/alphafold_params_2022-12-06.tar"
if [[ ! -f "${PARAMS_DIR}/params/params_model_1_ptm.npz" ]]; then
  if [[ ! -s "${PARAMS_TAR}" ]]; then
    curl -fL --retry 5 --retry-delay 10 -C - -o "${PARAMS_TAR}" \
      "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
  fi
  tar --extract --file="${PARAMS_TAR}" --directory="${PARAMS_DIR}/params"
  rm -f "${PARAMS_TAR}"
fi
ls -la "${PARAMS_DIR}/params" | head

echo "=== setup complete ==="
