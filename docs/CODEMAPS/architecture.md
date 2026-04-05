<!-- Generated: 2026-04-05 | Files scanned: 96 | Token estimate: ~900 -->
# Architecture

## Project Type
AlphaFold2 fork with **EvoTTT** (Evolutionary Test-Time Training) extension.
Adapts Evoformer weights at inference time via LoRA + masked-language-modeling on the input MSA.

## System Diagram

```
Input MSA (A3M)
    |
    v
Feature Pipeline (TensorFlow)
    |--- parsers.parse_a3m
    |--- pipeline.make_sequence_features + make_msa_features
    |--- af_features.np_example_to_features (padding, masking, clustering)
    v
Baseline AF2 Prediction (optional)
    |--- RunModel (model.py) with num_recycle=3
    |--- Output: baseline pLDDT + baseline.pdb
    v
TTT Adaptation Loop (JAX, no recycling)
    |--- For N steps:
    |       remask_msa_jax (15% BERT-style) 
    |       -> Evoformer forward (masked_msa head only)
    |       -> MLM loss (softmax_cross_entropy)
    |       -> Backward on LoRA A/B params (jax.checkpoint for memory)
    |       -> Adam step (warmup + cosine decay)
    |--- Merge best LoRA deltas into base params
    v
Adapted AF2 Prediction (full model)
    |--- RunModel with adapted params, num_recycle=3
    |--- Output: adapted pLDDT + adapted.pdb
    v
Results
    |--- evottt_result.json (per-protein)
    |--- ttt_log.csv (per-step)
    |--- evottt_summary.csv (across proteins)
```

## Directory Layout

```
alphafold/                  # Core AlphaFold2 library (upstream)
  common/                   # Shared: confidence, protein, residue_constants
  data/                     # Feature pipeline, MSA parsing, templates
    tools/                  # External tool wrappers (JackHMMER, HHblits, etc.)
  evottt/                   # *** EvoTTT extension ***
    ttt.py                  # TTT loop, masking, loss (398 lines)
    lora.py                 # LoRA init, merge, pytree mgmt (211 lines)
  model/                    # JAX model: modules, config, geometry
    tf/                     # TensorFlow data transforms
  relax/                    # Amber energy minimization
  notebooks/                # Colab utilities
configs/evottt/             # YAML experiment configs
scripts/                    # Entry points & utilities
  run_evottt_benchmark.py   # End-to-end benchmark (527 lines)
  launch_evottt.py          # YAML -> SLURM/local launcher (248 lines)
  run_af2_filter.py         # Baseline filtering (pLDDT < 70)
  check_loop_fractions.py   # Secondary structure analysis
jobs/evottt/                # SLURM job scripts
evottt_outputs/             # Benchmark results (PDBs, JSONs, CSVs)
data/                       # MSA databases, benchmark CSVs
```

## Key Design Patterns

- **LoRA as JAX PyTree**: trainable A/B matrices extracted as flat pytree for optax
- **JIT + Checkpoint**: `jax.checkpoint` recomputes Evoformer blocks to save GPU memory
- **Config flow**: YAML -> Python -> shell env vars -> sbatch
- **Per-step eval**: optional callback tracks best pLDDT during TTT loop
