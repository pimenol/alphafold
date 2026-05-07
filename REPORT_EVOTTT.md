# EvoTTT diagnostic report

## Goal

Adapt AlphaFold2's Evoformer at inference time on a single protein using
masked-MSA loss (MLM TTT), in the spirit of ProteinTTT (which reports gains
on ESMFold). Specifically: full fine-tune of the last 24 of 48 Evoformer
blocks (~44M params), various masking and loss strategies.

## Setup

- Test protein: **Q98576** (len=146, baseline AF2 pLDDT = **46.27** with
  full recycling, **48.56** with no recycling).
- 10 TTT steps, SGD, full fine-tune, `ttt_msa_clusters=4` (subsample 4 MSA
  rows per step), `ttt_recycle_prev=true` unless noted, evaluation each
  step under no-recycle config; final adapted prediction with full recycling.

## Experiments (Q98576, 10 steps)

| Job | Loss | Mask mode | lr | Loss start→end | Eval pLDDT (step 0→best) | Adapted pLDDT (recycle) | Δ vs base | \|\|Δ\|\|/\|\|W\|\| | Cα-RMSD vs base (eval) |
|---|---|---|---|---|---|---|---|---|---|
| A | MLM | per-cell | 1e-1 | 0.398 → 0.001 | 48.56 → 48.56 | 46.27 (best=0) | +0.00 | (best=base) | 1.77 Å |
| B | MLM | per-cell | 1e-1 | 0.188 → 0.002 | 48.56 → 48.56 | 46.27 (best=0) | +0.00 | (best=base) | 0.97 Å |
| E | MLM | per-cell | 1e-3 | 0.398 → 0.161 | 48.56 → 48.58 | 46.24 | −0.03 | 2.5e-5 | 0.085 Å |
| F | MLM | query-only | 1e-3 | **0.0002 → 0.0002** | 48.56 → 48.56 | 46.28 | +0.01 | 1.2e-8 | ≈ 0 |
| G | MLM | block | 1e-3 | 0.399 → 0.148 | 48.56 → 48.58 | 46.24 | −0.02 | 2.5e-5 | small |
| H | MLM | block | 1e-1 | 0.399 → 0.003 | 48.56 → 48.43 | 46.27 (best=0) | +0.00 | 5.2e-4 | ~1.5 Å |
| C | pLDDT-only | per-cell | 1e-1 | −0.555 → −0.804 | 48.56 → **74.07** | 72.16 | **+25.90** | 1.8e-3 | **11.4 Å** |
| D | MLM + 0.1·pLDDT | per-cell | 1e-1 | 0.343 → −0.058 | 48.56 → **52.49** | 50.43 | **+4.16** | 5.8e-4 | 8.7 Å |

(Cα-RMSD computed via Kabsch alignment between baseline.pdb and per-step
eval PDBs. Job B has no prev conditioning; all others use frozen prev.)

## What we observed (facts only)

**1. Pure MLM-loss collapses to ≈ 0 within 1–3 steps with multi-row MSA**
(jobs A, B at lr=1e-1, F at lr=1e-3). Per-cell masking goes 0.40 → 0.001;
query-only is 0.00018 from the start.

**2. The collapse is driven by an MSA shortcut, not by good
generalisation.** With `block_mask` (entire columns masked across all
rows), loss falls only to 0.15 (Job G, lr=1e-3) — there is real signal
when no homolog row at the same column carries the answer. At lr=1e-1
even block_mask collapses to 0.003 in 3 steps (Job H), suggesting a
secondary shortcut survives — likely through row-attention from
neighbouring unmasked columns.

**3. Weights do change.** Jobs E/G (lr=1e-3): ‖Δ‖/‖W‖ ≈ 2.5e-5. Job H
(lr=1e-1): ‖Δ‖/‖W‖ = 5.2e-4. Top-moved targets across all runs are
layer-norm offsets/scales of `msa_transition`, `msa_row_attention_with_pair_bias`,
and `outer_product_mean`.

**4. Structure also changes — sometimes substantially.** Job A: a single
SGD step at lr=1e-1 moves Cα by **1.77 Å** RMSD. Job H: ~1.5 Å. Job E
(low lr): 0.085 Å. So the claim "MLM doesn't change structure" was
wrong; the claim "MLM changes structure in directions that the model's
own confidence head doesn't recognise as improvement" is the precise one.

**5. pLDDT moves orders of magnitude less than structure does.**

| | ‖Δ‖/‖W‖ | Cα-RMSD (eval) | ΔpLDDT (eval) |
|---|---|---|---|
| H (MLM, lr=1e-1) | 5.2e-4 | 1.5 Å | −0.13 |
| D (joint, lr=1e-1) | 5.8e-4 | 2.6 Å | +3.93 |
| C (pLDDT-only, lr=1e-1) | 1.8e-3 | 11.4 Å | +25.5 |

Per unit weight movement, MLM gives ≈ 250 pLDDT-units; joint gives
≈ 6800; pLDDT-only gives ≈ 14000. **The pLDDT response is
direction-sensitive, not magnitude-sensitive.** MLM gradient and
pLDDT-relevant subspace of the weight space are nearly orthogonal.

**6. When pLDDT-loss is in the objective, it games the head, not the
structure.** Job C: ΔpLDDT = +25.9 with adapted-vs-baseline Cα-RMSD =
**8.2 Å** under full recycling. Job D: ΔpLDDT = +4.16 with adapted RMSD
= 8.7 Å. The model produces a structure 8 Å away from baseline that it
"thinks" is high quality; without ground truth we cannot confirm any
real improvement.

**7. `best_step=0` for every meaningful MLM-only run.** The best-pLDDT
step on the eval trajectory is consistently the untrained baseline. The
returned adapted parameters equal base parameters → final-prediction
ΔpLDDT = 0 even though the eval trajectory shows movement. The "+0.00"
in the summary CSV reflects nothing meaningful happened along the
pLDDT-improvement axis, not that nothing happened at all.

## Why it does not work for AF2 Evoformer

(My interpretation, supported by the facts above.)

**a. No structural anchor at TTT.** During AF2 pretraining, `loss = 0.5·FAPE
+ 2.0·MLM + 0.3·distogram + ...`. FAPE is the pin — it forces every gradient
direction to be structurally valid. At test time we only have MLM. The
gradient is free to optimise MSA-prediction in directions that are
structurally arbitrary. Evidence: Job A first SGD step moves Cα by 1.77 Å
in a direction that lowers pLDDT, not raises it.

**b. Architectural shortcut.** AF2's Evoformer has both row attention
(within sequence) and column attention (across MSA rows). With even one
homolog row visible at a given column, column attention trivially
copies the answer for any masked cell at that column → MLM gradient ≈ 0.
Block-masking removes this on the column axis but leaves the row-axis
shortcut. ProteinTTT for ESMFold avoids this by training a
**single-sequence** language model (ESM2) which has no column attention
at all.

**c. The Evoformer is both the encoder and the structural backbone.**
ProteinTTT updates ESM2 (a pure LM whose representations feed a frozen
folding trunk). When MLM updates ESM2, its outputs stay on the manifold
the folding trunk was trained to read. In AF2, the same Evoformer
weights produce both `masked_msa.logits` and `pair_rep / single_rep`.
MLM updates them to be better at MSA-prediction at the cost of moving
the structural representation off the manifold the frozen
`structure_module` and `predicted_lddt` head can read. Evidence: Jobs
A/H show structure moving 1–2 Å while pLDDT drops, indicating the new
geometry is "off-distribution" from the head's perspective.

**d. Frozen prev features and frozen `extra_msa_stack` damp the
adaptation.** `prev_pair` (146×146×128) injected from a recycled
baseline forward dominates the pair representation by magnitude;
`extra_msa_stack` (also frozen) processes 5120 MSA rows and contributes
the bulk of `pair_rep` mass. The 24 trainable Evoformer blocks
contribute a small perturbation on top of large frozen tensors — even
substantial weight changes barely move the final geometry. Removing
prev (Job B) didn't help: 0.97 Å Cα-RMSD with same ≈0 ΔpLDDT.

**e. The MLM optimum on a multi-row MSA is already attained by the
pretrained model.** With column-shortcut available, loss starts at
0.0002 (query-only) or 0.40 with collapse to 0.001 in 3 steps
(per-cell). Either there is no learning signal at all, or the model
finds the shortcut almost instantly. In neither case is the gradient
informative for structure.

**f. ProteinTTT's recipe relies on properties AF2 doesn't have.** Their
working stack is:
- ESM2: single-sequence LM, MLM is its native pretraining objective →
  TTT updates stay in-distribution.
- Folding trunk: frozen, decoupled from the LM-loss optimisation.
- Optional MSA: used only as a source of training rows, fed one at a
  time to the encoder; the trunk never sees the MSA.

AF2 has no architectural single-sequence mode, no separation between
"language model" and "structural backbone", and the Evoformer always
takes a multi-row MSA. The naive port of "do MLM TTT on the encoder"
maps onto "do MLM on the Evoformer's last 24 blocks", which (a) breaks
the structural representation and (b) doesn't give meaningful gradient
because of the MSA shortcut.

## Conclusion

For AF2 specifically, MLM TTT in the form attempted here does not
improve pLDDT. The blocker is architectural (MSA shortcut + entangled
encoder-backbone) and informational (no structural anchor at test
time). Working alternatives would need a structural objective that
doesn't require ground truth at TTT — for instance, self-distilled FAPE
against the recycled-baseline coordinates with `stop_gradient`, or
distogram-consistency between two differently-masked views of the MSA
(both already implementable in the codebase). Direct optimisation of
predicted-pLDDT (Jobs C, D) does move the metric, but the resulting
structures diverge 8–11 Å from baseline with no validation that the
divergence is in the correct direction; it is therefore confidence
gaming, not adaptation.
