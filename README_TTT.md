# EvoTTT: Test-Time Training on AlphaFold2's Evoformer

EvoTTT adapts AlphaFold2's Evoformer weights at inference time on a single
protein, using only its input MSA — no ground-truth structure.

The original objective is masked-language-modeling (MLM) over the MSA. The
codebase has grown into a small experiment harness with several optional
loss terms, two parameterisation strategies (LoRA and full fine-tune),
optional frozen-prev conditioning from a recycled baseline forward pass,
and an always-on diagnostic logger.

This document covers everything that is currently implemented.

## Installation

On top of the standard AlphaFold2 environment:

```bash
pip install optax
```

Activate `alphafold_evottt` and ensure the repo root is on `PYTHONPATH`.

## What gets adapted

Only parameters in the **main Evoformer stack** (the `evoformer_iteration`
scope) — never the extra-MSA stack, never the structure module, never the
heads. Two strategies:

* **LoRA** (default `full_finetune=false`) — low-rank A/B matrices added
  to attention `query_w / key_w / value_w / output_w` projections in the
  last `last_n_blocks` blocks. Optional triangle-attention LoRA via
  `lora_triangle_attention=true`. Discovered by `find_lora_targets`
  ([alphafold/evottt/lora.py:22-57](alphafold/evottt/lora.py#L22)).
* **Full fine-tune** (`full_finetune=true`) — direct weight slicing of
  every parameter (attention, MLP/transition, triangle multiplication,
  triangle attention, outer-product-mean, layer norms) in the last
  `last_n_blocks` blocks. Discovered by `find_finetune_targets`
  ([alphafold/evottt/lora.py:60-79](alphafold/evottt/lora.py#L60)).

Block selection is "the last N out of 48". Earlier blocks stay frozen.

The `predicted_lddt` head, structure module, and all output heads are
**always frozen** regardless of strategy.

## Loss functions

All four are masked-MSA cross-entropy variants implemented in
[alphafold/evottt/ttt.py](alphafold/evottt/ttt.py); the variant used is
selected by config flags.

### 1. MLM (default)

The original AF2 masked-MSA cross-entropy (re-implementation of
`MaskedMsaHead.loss`). Each TTT step samples a fresh BERT-style mask in
JAX (`remask_msa_jax`) and predicts masked residues from the rest of the
MSA + pair representation.

```
L_mlm = mean over masked positions of CE(true_msa, masked_msa.logits)
```

* `ttt_loss_fn` — LoRA variant, [ttt.py:279](alphafold/evottt/ttt.py#L279).
* `ttt_finetune_loss_fn` — full FT variant, [ttt.py:469](alphafold/evottt/ttt.py#L469).

### 2. MLM + distogram consistency (`distogram_consistency=true`)

Two forward passes with differently-masked MSAs; symmetric KL divergence
between their distogram predictions, plus average MLM:

```
L = 0.5*(L_mlm_1 + L_mlm_2) + lambda_pair * sym_KL(distogram_1, distogram_2)
```

Provides pair-level gradient signal without ground truth, useful when
LoRA-ing triangle attention. `ttt_consistency_loss_fn`
([ttt.py:336](alphafold/evottt/ttt.py#L336)) and
`ttt_finetune_consistency_loss_fn`
([ttt.py:546](alphafold/evottt/ttt.py#L546)).

### 3. Joint MLM + pLDDT (`plddt_loss_weight > 0`, full-FT only) — diagnostic

Adds a structure-aware term that pulls `E[pLDDT]` upward:

```
L = w_mlm * L_mlm  -  w_plddt * (E[pLDDT] / 100)
```

with `E[pLDDT] = mean over residues of sum_i softmax(predicted_lddt.logits)_i * c_i * 100`,
bin centers `c_i = (i + 0.5)/50`.

Activating this requires the structure module and `predicted_lddt` head
to be live in the TTT forward, which is automatic when
`plddt_loss_weight > 0` (see `make_ttt_apply(..., keep_structure=True)`,
[ttt.py:40](alphafold/evottt/ttt.py#L40)). Implementation:
`ttt_finetune_plddt_loss_fn` and `_expected_plddt`
([ttt.py:336-368](alphafold/evottt/ttt.py#L336)).

**Caveat — confidence gaming.** This loss optimises the model's own
trainable confidence head, which can rise without the underlying
structure improving. Use `w_mlm = 1.0, w_plddt ≤ 0.1` to keep the head
anchored, and validate any pLDDT gain against TM-score between
`baseline.pdb` and `adapted.pdb`. With `w_mlm = 0` the model trivially
sharpens the head.

### Optimizer & schedule

`optax.warmup_cosine_decay_schedule` with `warmup_steps = max(1, steps//10)`,
`init_value = 0`, `peak = ttt_lr`, `end = ttt_lr * 0.01`. **Step 1 of the
loop runs at LR=0** because optax counts updates from 0 — the diagnostic
logger prints `LR schedule (0..5)` at startup so you can see this.

Optimisers: `adam` (default), `adamw` (weight decay 1e-3), `sgd`. Set via
`optimizer:` in YAML or `--optimizer`.

## Frozen prev conditioning

`ttt_recycle_prev=true` runs a full AF2 inference with recycling on the
base model **before** TTT, captures `prev_pos / prev_msa_first_row /
prev_pair`, stops gradients, and injects them into every TTT batch.
The Evoformer then runs with a realistic geometric prior instead of zeros
(`compute_prev_features`, [ttt.py:84](alphafold/evottt/ttt.py#L84);
injection, [ttt.py:618-628](alphafold/evottt/ttt.py#L618)).

Number of recycles for the prev pass: `ttt_prev_num_recycle` (default
`null` = model default = 3).

When `ttt_recycle_prev=false`, the log explicitly prints
`No prev conditioning: Evoformer will see zero prev_*`.

## MSA handling

* `mask_fraction` — fraction of MSA positions masked per step
  (default 0.15, matches AF2 BERT masking).
* `block_mask=true` — mask whole residue columns (Exp 8) instead of
  per-position masking; forces pair representation to carry the prediction.
* `ttt_msa_clusters` — number of MSA rows kept per step (random subsample;
  query row 0 is always retained). Each step re-samples; `null` = use all.
* `ttt_crop_size` — residue crop applied during TTT only (`null` = no crop).
* `grad_accum_steps` — gradient accumulation across N differently-masked
  subsamples per optimizer update.

## Always-on diagnostic logging

Every TTT run now prints:

* **LR schedule preview** at startup
  ```
  LR schedule (step_index -> lr): 0->0, 1->0.1, 2->0.097, ... (warmup_steps=1, peak=0.1)
  ```
* **prev-injection state** (`Injecting frozen prev conditioning ...` or
  `No prev conditioning: ...`).
* **Loss recipe** when joint loss is active:
  ```
  TTT loss = 1.000*MLM - 0.100*(E[pLDDT]/100) (requires keep_structure apply_fn)
  ```
* **Per-step trace**:
  ```
  step: K, loss: ..., ttt_step_time: ..., eval_step_time: ..., plddt: ...
  ```
* **Best-step selection** at the end (`Using best pLDDT params from step K`).
* **Weight delta** for two snapshots — `[final]` (post-last-step) and
  `[best]` (the snapshot returned). Both report `total ||Δ||`,
  `||Δ||/||W||`, and the top-10 most-moved targets:
  ```
  Weight delta [final]: total ||Δ||=0.871, ||Δ||/||W||=5.835e-04 across 93 targets
    top Δ [final] alphafold/.../msa_transition/input_layer_norm//offset  ||Δ||=3.27e-01  rel=4.05e-02
    ...
  ```

These diagnostics are not opt-in — they always run.

## Hyperparameters (full list)

| YAML key | CLI flag | Default | Description |
|---|---|---|---|
| `model_name` | `--model_name` | `model_1_ptm` | AF2 model variant |
| `ttt_steps` | `--ttt_steps` | 50 | Number of gradient steps |
| `ttt_lr` | `--ttt_lr` | 3e-4 | Peak LR (warmup + cosine decay) |
| `optimizer` | `--optimizer` | `adam` | `adam` / `adamw` / `sgd` |
| `lora_rank` | `--lora_rank` | 4 | LoRA rank (LoRA mode only) |
| `lora_alpha` | `--lora_alpha` | 1.0 | LoRA scaling (LoRA mode only) |
| `last_n_blocks` | `--last_n_blocks` | 8 | Trailing Evoformer blocks to adapt |
| `full_finetune` | `--full_finetune` | false | Full FT instead of LoRA |
| `lora_triangle_attention` | `--lora_triangle_attention` | false | Also LoRA triangle attention |
| `distogram_consistency` | `--distogram_consistency` | false | Add symmetric-KL distogram term |
| `lambda_pair` | `--lambda_pair` | 0.1 | Weight for distogram consistency |
| `plddt_loss_weight` | `--plddt_loss_weight` | 0.0 | Weight for −E[pLDDT]/100 (full FT only) |
| `mlm_loss_weight` | `--mlm_loss_weight` | 1.0 | Weight for MLM term |
| `mask_fraction` | `--mask_fraction` | 0.15 | MSA mask fraction |
| `block_mask` | `--block_mask` | false | Mask whole columns (Exp 8) |
| `ttt_msa_clusters` | `--ttt_msa_clusters` | null | MSA rows kept per step |
| `ttt_crop_size` | `--ttt_crop_size` | null | Residue crop during TTT |
| `grad_accum_steps` | `--grad_accum_steps` | 1 | Gradient accumulation |
| `ttt_recycle_prev` | `--ttt_recycle_prev` | false | Inject frozen prev features |
| `ttt_prev_num_recycle` | `--ttt_prev_num_recycle` | null | Recycles for prev pass |
| `eval_interval` | `--eval_interval` | 1 | Eval pLDDT every N steps |
| `seed` | `--seed` | 0 | Master RNG seed |
| `protein_ids` | `--protein_ids` | — | Comma-separated subset |
| `start_idx` / `end_idx` | `--start_idx` / `--end_idx` | 0 / null | CSV row range |
| `skip_existing` | `--skip_existing` | false | Skip proteins already done |
| `skip_baseline` | `--skip_baseline` | false | Skip baseline prediction |

## Pipeline

```
                ┌────────────────────┐
 Input MSA ────▶│  Feature Pipeline   │   (TF, runs once)
                └────────┬───────────┘
                         │ processed features
                ┌────────▼───────────┐
                │ Baseline Inference  │   (full AF2 with recycling)
                └────────┬───────────┘
                         │ baseline_plddt, baseline.pdb
                ┌────────▼───────────┐
                │ (optional) Recycled │   if ttt_recycle_prev=true
                │  prev features      │   → prev_pos, prev_pair, prev_msa_first_row
                └────────┬───────────┘
                         │
                ┌────────▼───────────┐
                │   TTT Loop (JAX)    │◄── trainable params (LoRA A/B
                │                     │     OR full-FT slices)
                │  for step in steps: │
                │   1. (sub)sample MSA│
                │   2. Re-mask MSA    │
                │   3. Evoformer fwd  │   ± structure module (if pLDDT loss)
                │   4. Loss           │   MLM ± distogram-KL ± −E[pLDDT]/100
                │   5. ∇ trainable    │
                │   6. Optimizer step │   adam / adamw / sgd
                │   7. Eval pLDDT     │   full heads, no recycling
                └────────┬───────────┘
                         │ adapted params (best-pLDDT snapshot)
                ┌────────▼───────────┐
                │ Adapted Inference   │   (full AF2 with recycling)
                └────────┬───────────┘
                         │ adapted_plddt, adapted.pdb
                Δ pLDDT, ttt_log.csv, evottt_result.json
```

## Quick start

### Single protein (Python)

```python
from alphafold.model import config as af_config, data as af_data
from alphafold.evottt.ttt import make_ttt_apply, run_ttt
from scripts.run_evottt_benchmark import load_features_from_a3m

model_name = 'model_1_ptm'
config = af_config.model_config(model_name)
params = af_data.get_model_haiku_params(model_name, '/path/to/af2_data')

ttt_config = af_config.model_config(model_name)
with ttt_config.unlocked():
    ttt_config.model.num_recycle = 0
    ttt_config.data.common.num_recycle = 0
features = load_features_from_a3m('protein.a3m', 'MKTL...', 'my_id', ttt_config)
ttt_apply = make_ttt_apply(ttt_config.model)

adapted_params, losses, eval_logs, best_step = run_ttt(
    apply_fn=ttt_apply,
    base_params=params,
    model_config=ttt_config.model,
    batch=features,
    num_steps=30,
    learning_rate=3e-4,
    rank=4,
    last_n_blocks=8,
    optimizer_name='adam',
)

from alphafold.model.model import RunModel
runner = RunModel(config, params=adapted_params)
result = runner.predict(features, random_seed=0)
```

### Benchmark (CLI)

```bash
python scripts/run_evottt_benchmark.py \
    --benchmark_csv data/bfvd/summary.csv \
    --msa_dir /path/to/bfvd_msa \
    --data_dir /path/to/af2_data \
    --output_dir evottt_outputs \
    --ttt_steps 30 --ttt_lr 1e-1 \
    --full_finetune --last_n_blocks 24 \
    --optimizer sgd --grad_accum_steps 1 \
    --ttt_recycle_prev --ttt_msa_clusters 4 --ttt_crop_size 1024
```

### SLURM (YAML-driven)

```bash
CONFIG_FILE=$PWD/configs/evottt/default.yaml sbatch jobs/evottt/run_evottt.sh
```

The launcher reads the YAML and exports each key as an `${ENV_VAR:-default}`,
then assembles the CLI command.

## Diagnostic experiments shipped

Five YAML configs in `configs/evottt/` mirror the diagnostic runs used to
debug "loss decreases but pLDDT doesn't move":

| File | Purpose |
|---|---|
| `diag_baseline.yaml` | Same hparams as the working full-FT default but Q98576 only, 10 steps. Sanity baseline. |
| `diag_noprev.yaml` | Same as above with `ttt_recycle_prev: false`. Tests whether frozen prev pins pLDDT. |
| `diag_plddt.yaml` | `plddt_loss_weight: 1.0, mlm_loss_weight: 0.0`. Gradient-path probe via −E[pLDDT]/100. |
| `diag_joint.yaml` | `plddt_loss_weight: 0.1, mlm_loss_weight: 1.0`. Joint anchor-and-nudge. |
| `diag_lowlr.yaml` | MLM only at `ttt_lr: 1e-3`. Tests whether MLM hurts pLDDT only at high lr. |

Submit with e.g. `CONFIG_FILE=$PWD/configs/evottt/diag_plddt.yaml sbatch jobs/evottt/run_evottt.sh`.

### Findings (Q98576, 10 steps)

| Exp | Loss | lr | end loss | best_step | Adapted pLDDT | Δ vs base 46.27 | \|\|Δ\|\|/\|\|W\|\| |
|---|---|---|---|---|---|---|---|
| baseline | MLM | 1e-1 | 0.0014 | 0  | 46.27 | +0.00 | 0 (best=base) |
| no-prev | MLM | 1e-1 | 0.0015 | 0  | 46.27 | +0.00 | 0 (best=base) |
| pLDDT  | −E[pLDDT]/100 | 1e-1 | −0.804 | 10 | 72.16 | **+25.90** | 1.8e-3 |
| joint  | MLM + 0.1·pLDDT | 1e-1 | −0.058 | 10 | 50.43 | **+4.16** | 5.8e-4 |
| low-lr | MLM | 1e-3 | 0.160 | 7  | 46.24 | −0.03 | 2.5e-5 |

Conclusion: MLM-only TTT does not raise pLDDT at any LR. The gradient
path through structure module → `predicted_lddt` is healthy (proven by
the pLDDT-only run). Frozen prev is not the pin. The remaining concern
is confidence gaming in the pLDDT-targeted runs — verify with TM-score
on the saved PDBs before treating these Δ values as quality gains.

## Outputs

* Per protein:
  * `evottt_outputs/<run>/<protein_id>/baseline.pdb`
  * `evottt_outputs/<run>/<protein_id>/adapted.pdb`
  * `evottt_outputs/<run>/<protein_id>/steps/eval_NNNN.pdb` (one per eval call)
  * `evottt_outputs/<run>/<protein_id>/evottt_result.json`
  * `evottt_outputs/<run>/<protein_id>/ttt_log.csv` — `step_num, loss, plddt, pdb`
* Per run:
  * `evottt_outputs/<run>/config.json`
  * `evottt_outputs/<run>/evottt_summary.csv`

## Code structure

| File | Description |
|---|---|
| [alphafold/evottt/lora.py](alphafold/evottt/lora.py) | LoRA target discovery, A/B init, merge; full-FT target discovery |
| [alphafold/evottt/ttt.py](alphafold/evottt/ttt.py) | TTT forward (`make_ttt_apply`), MSA re-masking + subsampling, prev-feature computation, all four loss variants, training loop with diagnostic logging |
| [scripts/run_evottt_benchmark.py](scripts/run_evottt_benchmark.py) | End-to-end benchmark: baseline → TTT → adapted, plus per-step PDB dumps |
| [configs/evottt/default.yaml](configs/evottt/default.yaml) | Default experiment config |
| [configs/evottt/diag_*.yaml](configs/evottt/) | Diagnostic configs (baseline / noprev / plddt / joint / lowlr) |
| [jobs/evottt/run_evottt.sh](jobs/evottt/run_evottt.sh) | SLURM launcher; reads YAML, exports env vars, builds CLI |

## Limitations & known caveats

* **Step 1 has LR=0** because of the warmup schedule (`init_value=0`,
  `warmup_steps ≥ 1`). Step-0 and step-1 pLDDT will always match. To get
  a real first update, lower `warmup_steps` or set a non-zero
  `init_value` (requires editing `run_ttt`).
* **Best-step selection uses the eval config (no recycling, full MSA),
  not the final inference config (recycling, full MSA)**. The
  `best_step` snapshot is the one with highest *eval* pLDDT, which is a
  proxy for the recycled inference pLDDT, not equal to it.
* **`predicted_lddt` head is always frozen by find-target rules**; only
  Evoformer params change. The structure module is also always frozen.
  This is intentional but means the head can't be retrained to recalibrate.
* **MLM-only adaptation does not improve pLDDT** in the regime tested
  (full FT, last 24 blocks, lr=1e-1 SGD or lr=1e-3). Diagnostic data
  above suggests the MLM objective is fundamentally misaligned with
  structural quality — at high LR it actively degrades it.
* **The pLDDT-loss path optimises a trainable confidence head** and can
  produce confidence gains that don't correspond to structural quality.
