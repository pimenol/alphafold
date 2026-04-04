# EvoTTT: Test-Time Training on AlphaFold2's Evoformer

EvoTTT adapts AlphaFold2's Evoformer weights at inference time using a
masked-language-modeling (MLM) objective on the input MSA, with LoRA
(Low-Rank Adaptation) for parameter-efficient fine-tuning.

No ground-truth structure is used — only the input MSA.

## Installation

EvoTTT requires one additional dependency on top of the standard AlphaFold2
environment:

```bash
pip install optax
```

Ensure the `alphafold_evottt` conda environment is active and `PYTHONPATH`
includes the repo root.

## Quick Start

### Single protein

```python
from alphafold.model import config as af_config, data as af_data
from alphafold.evottt.ttt import make_ttt_apply, run_ttt
from scripts.run_evottt_benchmark import load_features_from_a3m

# 1. Load model
model_name = 'model_1_ptm'
config = af_config.model_config(model_name)
params = af_data.get_model_haiku_params(model_name, '/path/to/af2_data')

# 2. Load features from precomputed MSA
features = load_features_from_a3m(
    'protein.a3m', 'MKTL...', 'my_protein', config
)

# 3. Create TTT apply function
ttt_config = af_config.model_config(model_name)
with ttt_config.unlocked():
    ttt_config.model.num_recycle = 0
    ttt_config.data.common.num_recycle = 0
ttt_apply = make_ttt_apply(ttt_config.model)

# 4. Run TTT
adapted_params, losses, eval_logs = run_ttt(
    apply_fn=ttt_apply,
    base_params=params,
    model_config=ttt_config.model,
    batch=features,
    num_steps=50,
    learning_rate=1e-4,
    rank=4,
)

# 5. Run full inference with adapted params
from alphafold.model.model import RunModel
runner = RunModel(config, params=adapted_params)
result = runner.predict(features, random_seed=0)
print(f"pLDDT: {result['plddt'].mean():.2f}")
```

### Benchmark

```bash
python scripts/run_evottt_benchmark.py \
    --benchmark_csv /path/to/summary.csv \
    --msa_dir /path/to/bfvd_msa \
    --data_dir /path/to/af2_data \
    --output_dir evottt_outputs \
    --model_name model_1_ptm \
    --ttt_steps 50 \
    --ttt_lr 3e-4 \
    --lora_rank 4 \
    --last_n_blocks 8 \
    --eval_interval 1
```

### YAML config (recommended)

```bash
# Submit as SLURM job from a YAML config:
python scripts/launch_evottt.py configs/evottt/default.yaml

# Override any hparam from CLI:
python scripts/launch_evottt.py configs/evottt/default.yaml --ttt_lr 1e-3 --ttt_steps 100

# Run locally (no SLURM):
python scripts/launch_evottt.py configs/evottt/default.yaml --local

# Dry-run (print the sbatch script without submitting):
python scripts/launch_evottt.py configs/evottt/default.yaml --dry_run
```

### SLURM (direct)

```bash
sbatch jobs/evottt/run_evottt.sh
```

Override defaults via environment variables:

```bash
TTT_STEPS=100 TTT_LR=5e-5 LORA_RANK=8 PROTEIN_IDS=A0A6J5N0Y1,A5A3S1 \
    sbatch jobs/evottt/run_evottt.sh
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ttt_steps` | 50 | Number of gradient steps |
| `ttt_lr` | 3e-4 | Peak Adam learning rate (warmup + cosine decay) |
| `lora_rank` | 4 | LoRA rank (r) |
| `last_n_blocks` | 8 | Number of Evoformer blocks to adapt (from end) |
| `lora_alpha` | 1.0 | LoRA scaling factor |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `mask_fraction` | 0.15 | MSA masking fraction per step |
| `eval_interval` | 1 | Evaluate pLDDT every N TTT steps (0 to disable) |

## Architecture

```
                    ┌────────────────────┐
 Input MSA ───────▶│  Feature Pipeline   │
                    │  (TF, runs once)    │
                    └────────┬───────────┘
                             │ processed features
                    ┌────────▼───────────┐
                    │   TTT Loop (JAX)    │◄── LoRA params (A, B)
                    │                     │
                    │  for step in steps: │
                    │   1. Re-mask MSA    │
                    │   2. Evoformer fwd  │
                    │   3. MLM loss       │
                    │   4. ∇LoRA params   │
                    │   5. Adam update    │
                    └────────┬───────────┘
                             │ adapted params
                    ┌────────▼───────────┐
                    │ Full AF2 Inference  │
                    │ (Evoformer+IPA+     │
                    │  recycling)         │
                    └────────┬───────────┘
                             │
                    Structure + pLDDT + pTM
```

## Outputs

Per-protein: `evottt_outputs/{protein_id}/evottt_result.json`

Summary: `evottt_outputs/evottt_summary.csv` with columns:
`id, length, nmsa, baseline_plddt, adapted_plddt, delta_plddt,
ttt_loss_start, ttt_loss_end, ttt_time_s`

### Per-step logging

When `eval_interval > 0`, each TTT step logs pLDDT to the Python `evottt`
logger:

```
2026-04-03 12:00:01,234 | INFO | step: 0, loss: None, ttt_step_time: 0.00000, eval_step_time: 1.75260, plddt: 81.13863
2026-04-03 12:00:04,567 | INFO | step: 1, loss: 2.62891, ttt_step_time: 2.09960, eval_step_time: 0.92640, plddt: 81.53083
```

Step 0 is the baseline (before any training). The per-step `eval_logs` are
also saved in each protein's `evottt_result.json`.

## Code Structure

| File | Description |
|------|-------------|
| `alphafold/evottt/lora.py` | LoRA init, merge, target discovery |
| `alphafold/evottt/ttt.py` | TTT forward function, MSA re-masking, training loop |
| `scripts/run_evottt_benchmark.py` | End-to-end benchmark script |
| `scripts/launch_evottt.py` | Launch experiments from YAML configs |
| `configs/evottt/default.yaml` | Default experiment config |
| `jobs/evottt/run_evottt.sh` | SLURM job submission script |
