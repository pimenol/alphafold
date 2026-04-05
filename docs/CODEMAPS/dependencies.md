<!-- Generated: 2026-04-05 | Files scanned: 96 | Token estimate: ~500 -->
# Dependencies

## Core ML Stack
- **JAX** — primary ML framework (forward/backward, JIT, vmap)
- **Haiku (dm-haiku)** — JAX module system (AlphaFold model definition)
- **Optax** — optimizers (Adam + warmup cosine decay for TTT)
- **TensorFlow** — feature pipeline only (data transforms, not training)
- **ML Collections** — config management

## Scientific / Data
- **NumPy** — array ops
- **SciPy** — sparse matrices (MSA features)
- **Biopython** — PDB/mmCIF I/O (via alphafold.common.protein)

## External Tools (called via subprocess)
- **JackHMMER** — MSA search against sequence databases
- **HHblits / HHsearch** — profile HMM search
- **hmmbuild / hmmsearch** — HMM construction & search
- **Kalign** — MSA realignment
- **OpenMM / Amber** — structure relaxation (alphafold/relax/)

## Infrastructure
- **SLURM** — job scheduling (sbatch, via launch_evottt.py)
- **CUDA / cuDNN** — GPU acceleration
- **Conda** — environment management (alphafold_evottt env)

## Data Sources
- **UniRef90 / UniRef30** — sequence databases for MSA
- **BFD / MGnify** — additional MSA databases
- **PDB70 / PDB mmCIF** — template databases
- **BFVD benchmark CSV** — protein IDs + sequences for benchmarking

## Internal Module Dependency Graph
```
scripts/run_evottt_benchmark.py
  -> alphafold.evottt.ttt (make_ttt_apply, run_ttt)
  -> alphafold.model (RunModel, config)
  -> alphafold.data (pipeline, parsers, features)
  -> alphafold.common (confidence, protein)

scripts/launch_evottt.py
  -> yaml, subprocess (no ML deps)
  -> jobs/evottt/run_evottt.sh -> scripts/run_evottt_benchmark.py

alphafold.evottt.ttt
  -> alphafold.evottt.lora
  -> alphafold.model.modules (AlphaFold, softmax_cross_entropy)
  -> jax, optax, haiku

alphafold.evottt.lora
  -> jax, numpy (pure functional, no external ML deps)
```
