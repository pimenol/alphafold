<!-- Generated: 2026-04-05 | Files scanned: 96 | Token estimate: ~500 -->
# Data Flow & Formats

## Input Data

### MSA Files (A3M format)
- Located in: `data/bfvd_msa/{protein_id}.a3m`
- Parsed by: `alphafold.data.parsers.parse_a3m`
- Alternative: generated on-the-fly via JackHMMER

### Benchmark CSV
- Located in: `data/bfvd/summary_filtered.csv`
- Columns: protein_id, sequence, (additional metadata)

## Feature Pipeline (TF-based, runs once per protein)

```
A3M -> parse_a3m -> Msa(sequences, descriptions)
    -> make_sequence_features (aatype, residue_index, seq_length)
    + make_msa_features (msa, deletion_matrix, num_alignments)
    + empty_template_features
    -> np_example_to_features (crop, pad, cluster, ensemble)
    -> JAX arrays: {msa_feat, msa_mask, bert_mask, ...}
```

Key feature shapes (for TTT):
- `msa_feat`: [N_seq, N_res, 49] — MSA one-hot + features
- `bert_mask`: [N_seq, N_res] — masking indicator
- `msa_mask`: [N_seq, N_res] — sequence presence mask

## Output Data

### Per-Protein (evottt_outputs/{protein_id}/)
| File | Format | Contents |
|------|--------|----------|
| baseline.pdb | PDB | Baseline AF2 structure |
| adapted.pdb | PDB | Post-TTT adapted structure |
| evottt_result.json | JSON | {id, baseline_plddt, adapted_plddt, delta_plddt, losses, eval_logs, best_step, config} |
| ttt_log.csv | CSV | step, loss, plddt per TTT step |
| steps/eval_NNNN.pdb | PDB | Per-step evaluation structures |

### Summary (evottt_outputs/)
| File | Format | Contents |
|------|--------|----------|
| evottt_summary.csv | CSV | protein_id, baseline_plddt, adapted_plddt, delta_plddt, best_step |

## Experiment Configs (configs/evottt/*.yaml)
```yaml
# Key hyperparameters
ttt_steps: 20          # TTT optimization steps
ttt_lr: 3.0e-4         # Learning rate
lora_rank: 8           # LoRA rank
last_n_blocks: 8       # Evoformer blocks to adapt
lora_alpha: 16         # LoRA scaling factor
mask_fraction: 0.15    # BERT-style mask ratio
ttt_crop_size: 1024    # Max sequence length for TTT
eval_interval: 1       # Evaluate pLDDT every N steps
```
