<!-- Generated: 2026-04-05 | Files scanned: 96 | Token estimate: ~800 -->
# Backend / Core Logic

## EvoTTT Module (alphafold/evottt/)

### ttt.py — TTT Loop & Loss
```
make_ttt_apply(model_config) -> apply_fn
  Sets num_recycle=0, zeros all heads except masked_msa
  Returns haiku apply_fn(params, rng, batch) -> output_dict

remask_msa_jax(batch, rng, replace_fraction=0.15) -> masked_batch
  BERT-style re-masking: uniform AAs / MSA profile / MASK token

ttt_loss_fn(trainable, base_params, lora_meta, alpha, rank, ...) -> (loss, logits)
  Softmax cross-entropy on masked positions only

run_ttt(apply_fn, base_params, model_config, batch, ...) -> (adapted_params, losses, eval_logs, best_step)
  Adam optimizer with warmup (10%) + cosine decay
  Tracks best pLDDT via optional eval_fn callback
  Merges best LoRA deltas at end
```

### lora.py — LoRA Parameter Management
```
find_lora_targets(params, target_param_names) -> List[LoRATarget]
  Discovers evoformer attention weights (query_w, key_w, value_w, output_w)
  Excludes triangle attention & extra_msa_stack

init_lora_params(base_params, targets, rank=4, last_n_blocks=48, alpha=1.0) -> dict
  A: zeros, B: Kaiming uniform (bound = sqrt(1/rank))
  Zeros blocks outside last_n_blocks

merge_lora_into_params(base_params, lora, alpha, rank) -> params
  delta = (alpha / rank) * A @ B, added to base weights

trainable_from_lora(lora) -> flat pytree {key: {A, B}}
update_lora_from_trainable(lora, trainable) -> updated lora
```

## AlphaFold2 Model (alphafold/model/)

### model.py — RunModel Container
```
RunModel.__init__(config, params) -> self
  Creates _forward_fn via haiku.transform for monomer/multimer

RunModel.predict(feat, random_seed) -> prediction_result
  JIT-compiled forward pass

get_confidence_metrics(prediction_result) -> {plddt, ranking_confidence, ...}
```

### Key Config Mutations for TTT
```
num_recycle = 0           (no recycling during TTT)
max_msa_clusters = 32     (vs 128 normal)
max_extra_msa = 128       (vs 1152 normal)
use_remat = True          (gradient checkpointing)
```

## Feature Pipeline (alphafold/data/)

```
pipeline.py         -> DataPipeline: MSA search + template search + feature assembly
parsers.py          -> parse_a3m, convert_stockholm_to_a3m, parse_fasta
feature_processing.py -> np_example_to_features (crop, pad, cluster)
mmcif_parsing.py    -> PDB/mmCIF structure parsing
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| alphafold/evottt/ttt.py | 398 | TTT loop, masking, loss |
| alphafold/evottt/lora.py | 211 | LoRA init/merge |
| alphafold/model/model.py | 184 | JAX model container |
| alphafold/model/modules.py | ~2500 | AlphaFold modules (Evoformer, heads) |
| alphafold/model/config.py | ~300 | Model configurations |
| alphafold/data/pipeline.py | ~300 | Feature pipeline |
| alphafold/data/parsers.py | ~400 | MSA format parsers |
| alphafold/common/confidence.py | ~200 | pLDDT, pTM, PAE metrics |
| alphafold/common/protein.py | ~300 | Protein data structures, PDB I/O |
