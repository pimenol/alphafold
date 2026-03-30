# EvoTTT Implementation Plan

## Context

Implement test-time training (TTT) on AlphaFold2's Evoformer using masked language modeling (MLM) with LoRA. At inference time, before making the final structure prediction, adapt the Evoformer weights using the self-supervised MLM objective on the input MSA. No ground-truth structures are used — only MSA reconstruction loss.

## Approach: Zero-modification to AF2 code

LoRA deltas are merged into the AF2 params dict before calling the existing model forward pass. No AF2 source files are modified. Two forward functions are used:
1. **TTT forward**: `AlphaFold` with modified config (zero all head weights except `masked_msa`, `num_recycle=0`) → runs only Evoformer + MaskedMsaHead
2. **Inference forward**: Standard `RunModel.predict()` with adapted params (full model with recycling)

## Files to Create

### 1. `alphafold/evottt/__init__.py` (empty)
### 2. `alphafold/evottt/lora.py` — LoRA parameter management
### 3. `alphafold/evottt/ttt.py` — TTT forward, MSA re-masking, training loop
### 4. `scripts/run_evottt_benchmark.py` — End-to-end benchmark script
### 5. `jobs/evottt/run_evottt.sh` — SLURM job script
### 6. `README_TTT.md` — Documentation

## Detailed Design

### A. LoRA Parameter Management (`alphafold/evottt/lora.py`)

**Targets**: Q, K, V, O projections in MSA row attention and MSA column attention.

Parameter paths (with layer_stack, shapes include leading dim of 48 blocks):
- `evoformer_iteration/msa_row_attention_with_pair_bias/attention`: query_w(48,256,8,32), key_w, value_w, output_w(48,8,32,256)
- `evoformer_iteration/msa_column_attention/attention`: same shapes

Implementation auto-discovers targets by scanning param tree for keys matching `evoformer_iteration` + `attention` + target param names (robust to exact naming).

**LoRA decomposition** for weight W of shape `(48, d1, d2, ...)`:
- Reshape non-batch dims to 2D: `(48, d_in, d_out)`
- For `query_w/key_w/value_w`: d_in=256, d_out=256 (8×32)
- For `output_w`: d_in=256 (8×32), d_out=256
- A: `(48, d_in, rank)` — small random for last N blocks, zero for others
- B: `(48, rank, d_out)` — zero (delta starts at zero)
- Delta: `(alpha/rank) * A @ B` reshaped to original shape

**Functions**: `find_lora_targets`, `init_lora_params`, `merge_lora_into_params`, `lora_trainable_pytree`

### B. TTT Forward & Training (`alphafold/evottt/ttt.py`)

**TTT forward**: Uses `AlphaFold` with modified config:
- `num_recycle=0` → no recycling, single pass
- All head weights zeroed except `masked_msa` → structure module never instantiated (line 216 of modules.py: `if not head_config.weight: continue`)
- Call via `hk.transform(_forward).apply(merged_params, rng, batch)` directly (not RunModel.predict)

**MSA re-masking in JAX** (replicates `make_masked_msa` from `data_transforms.py:417-448`):
- Input: `batch['true_msa']` (unmasked MSA from TF pipeline)
- Each TTT step: new random mask, BERT-style replacement (70% mask token 22, 10% same, 10% profile, 10% random AA)
- Update `batch['bert_mask']` and first 23 channels of `batch['msa_feat']`
- `msa_feat` layout: [one_hot_msa(23), has_deletion(1), deletion_value(1), cluster_profile(23), cluster_deletion_mean(1)]

**Loss**: `softmax_cross_entropy(one_hot(true_msa, 23), logits) * bert_mask / sum(bert_mask)` — same as MaskedMsaHead.loss

**Training loop**: optax Adam + gradient clipping, JIT-compiled step, N steps per protein

### C. Feature Loading

```python
msa = parsers.parse_a3m(a3m_string)
raw = {**pipeline.make_sequence_features(seq, pid, n), **pipeline.make_msa_features([msa])}
processed = features.np_example_to_features(raw, config, seed)
```

### D. Benchmark Script (`scripts/run_evottt_benchmark.py`)

Per protein: load A3M → process features → baseline predict → TTT adapt → adapted predict → compare pLDDT

### E. SLURM + README

Based on existing `jobs/af2_benchmark/run_benchmark_af2.sh` patterns.

## Key Existing Code Reused (no modifications)

- `alphafold/model/modules.py` — AlphaFold, MaskedMsaHead, Attention, EmbeddingsAndEvoformer
- `alphafold/model/model.py` — RunModel
- `alphafold/model/config.py` — model_config()
- `alphafold/model/data.py` — get_model_haiku_params
- `alphafold/model/features.py` — np_example_to_features
- `alphafold/data/parsers.py` — parse_a3m
- `alphafold/data/pipeline.py` — make_sequence_features, make_msa_features
- `alphafold/common/confidence.py` — compute_plddt

## Verification

1. Verify find_lora_targets returns 8 targets
2. Zero LoRA (B=0) produces identical output to base model
3. Single TTT step: loss computed, gradients non-zero
4. 50 steps on one protein: loss decreases
5. Full benchmark: compare baseline vs adapted pLDDT
6. After each created feature, commit changes
7. After you done everything, run benchmark and log the results
