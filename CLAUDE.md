# CLAUDE.md

# goal of this repo

make test-time training (TTT) improve AlphaFold2 structure quality on the **lowconf**
set — targets where AF2 is both wrong and unconfident — without ever seeing the ground
truth during TTT and without touching the pLDDT head.

## metrics — read this first

two different numbers, do not confuse them:

- **lDDT** — true accuracy against the ground truth structure, 0–100 scale.
  this is the only success criterion. computed by the existing eval script,
  after the prediction is final. never enters the TTT loop.
- **pLDDT** — AF2's own confidence, 0–100 scale. unsupervised, so it may be used
  as a selection signal (pick the best TTT step). it is not a success criterion.

the main failure mode of this project is pLDDT going up while lDDT goes down.
any run where mean pLDDT improves and mean lDDT does not is a negative result and
must be logged as such, not presented as progress.

Δ always means `TTT − baseline` on the same protein, same eval script.

**scale.** this document uses the 0–100 convention: `ΔlDDT ≥ +7` means 7 lDDT points.
`af2_lowconf.csv`, `af2_run_results.csv` and `scripts/score_af2_predictions.py` all store
lDDT as a **0–1 fraction**, so +7 there reads as **+0.07**. multiply by 100 before
comparing against any threshold in this file, and report in one scale consistently.

## acceptance criteria

on the hard set — the **18 proteins** in `subsets/lowconf18.txt`:

1. at least **6 proteins with ΔlDDT ≥ +7**
2. proteins with **ΔlDDT ≤ −7** must be fewer than the proteins from (1)
3. the winning configuration reproduces on a second seed: at least 6 proteins
   with ΔlDDT ≥ +7, same config, no re-tuning
4. `git diff --stat` against the base commit stays small and every touched
   upstream AF2 line is justified in LOG.md

always report alongside: mean and median ΔlDDT, count of ΔlDDT ≥ +7, count of
ΔlDDT ≤ −7, mean ΔpLDDT, and a per-protein Δ table. report the deep-MSA and
shallow-MSA halves separately as well — see "the set you are optimising".

## the hard set — 18 of the 25

`af2_lowconf.csv` holds 25 chains, selected on AlphaFold DB's published pLDDT (< 70)
and lDDT (< 0.7). Acceptance is judged on the subset where **our own M0 run** is still
in that regime with the MSAs TTT actually uses:

```
run_plddt < 70  AND  run_lddt < 0.70     (af2_run_results.csv, dataset == lowconf)
```

That is exactly 18 chains — both conditions select the same 18 — pinned in
`subsets/lowconf18.txt` so the list is a file, not a threshold re-evaluated per script.
Hard set: mean pLDDT **49.6**, mean lDDT **0.430**, max lDDT **0.654**, lengths 73–475.

The other 7 (`1ZW8_A 2MJM_A 2N7P_A 2WBV_A 5M86_A 6CFW_F 8Q66_B`) were selected on the
AlphaFold DB model, but a fresh MSA made AF2 confident again — 71.0 to 86.0 pLDDT. They
are no longer "wrong and knows it". They stay in the dataset and are still predicted and
reported; they simply do not gate. Do not quietly re-add them to inflate a count, and do
not delete them either — a method that helps the 18 while wrecking those 7 is worth
seeing.

## milestones

do not move to the next milestone before the current one verifies. each full-set
run costs node-hours — the point of the small subsets is to fail cheaply.

0. **M0 — baseline. done.** vanilla AF2 (`model_1_ptm`, seed 0, unrelaxed, precomputed
   MSA, no templates) over all 25, committed in `af2_run_results.csv` at `f3211e1`.
   mean lDDT **0.430** over the hard 18 (0.500 over all 25), per-protein numbers in the
   table below. every TTT Δ is measured against these rows and against nothing else.
   re-deriving the baseline is only necessary if the eval script or the runner changes.

1. **M1 — does TTT move the structure at all (5 proteins).** any TTT variant where
   predicted coordinates differ from baseline and |ΔlDDT| ≥ 1 on at least one protein.
   **moving in the wrong direction passes M1** — it proves gradients reach the
   structure module. a run where coordinates are bit-identical to baseline is a bug,
   not a negative result.
   → verify: coordinate diff is non-zero, ΔlDDT logged for all 5.

2. **M2 — TTT helps on the 5.** mean ΔlDDT > 0 across the 5-protein subset, and at
   least one protein with ΔlDDT ≥ +7.
   → verify: repeat the winning config on a second seed, mean ΔlDDT still > 0.

3. **M3 — the hard 18.** the acceptance criteria above, on `subsets/lowconf18.txt`.
   predict the excluded 7 in the same run and report them, outside the criteria.

## hard constraints

these are not preferences. violating one invalidates the experiment.

- **no ground truth inside TTT.** not in the loss, not in early stopping, not in step
  or checkpoint selection, not in hyperparameter selection on the TTT set. the TTT code
  path must not read the ground-truth files at all — add a guard that fails loudly if
  it does, and mention that guard in LOG.md. concretely: nothing under
  `data/lowconf/pdbs/` may be opened by any module the TTT loop imports.
- **do not modify the pLDDT head or its loss.** raising pLDDT by changing the head that
  predicts it is not a result.
- **minimal diff to AF2.** TTT lives in new files. changes to original AF2 code are
  limited to the minimum hooks needed to inject a loss or expose an internal
  representation. no refactoring, no renaming, no style fixes in upstream files.
  include `git diff --stat` against the base commit in every LOG.md entry.
- **10 TTT steps per target** by default. report both the step-10 result and the
  best-pLDDT-step result, so it is visible whether pLDDT selection actually helps.
- **start from AF2 paper hyperparameters** wherever the paper specifies them. change
  one thing per experiment; a sweep of two axes at once is not interpretable at this
  scale.
- **do not reimplement lDDT or TM-score.** `scripts/score_af2_predictions.py` is the
  eval script; it is calibrated against the official CASP15 values to within 0.02 on
  the sibling `af2_fails` set. a TTT number produced by a different implementation is
  not comparable to the M0 baseline.

## the set you are optimising — read before picking a loss

most of the hard set has almost no MSA:

| depth | all 25 | hard 18 | note |
| --- | ---: | ---: | --- |
| ≤ 30 sequences | 12 | **11** | Neff 1–16. effectively single-sequence. |
| 31–999 | 6 | 3 | |
| ≥ 1000 | 7 | 4 | Neff 313–4630. |

this is a property of the selection criterion, not a defect: AF2 is unconfident here
largely *because* the alignment is empty. and note what restricting to the hard 18 did —
it removed mostly deep-MSA chains, so the gating set is **61 % effectively
single-sequence**, worse than the full 25.

that has a direct consequence for method choice: **any loss that consumes MSA depth has
nothing to work with on 11 of the 18 targets.** the MSA-free losses are the ones that
can carry M3. an MSA-based method would have to clear the bar on the 4 deep chains
alone, which is not possible when the bar is 6. report the deep/shallow split separately
every time — a mean over the 18 will hide which mechanism actually fired.

## method ideas

try roughly in this order. cheap and unsupervised first. the ordering below differs
from the generic one: the MSA-free losses are promoted because they are the only ones
that can act on the shallow half of this set.

1. **structural violation loss.** AF2 already computes bond-length, angle and clash
   violation terms. physics-based, fully unsupervised, cheap, MSA-independent, and it
   directly targets the kind of failure that shows up on hard targets.
2. **distogram entropy minimisation.** sharpen the predicted distance distribution.
   very cheap, no extra forward passes, MSA-independent.
3. **recycling consistency.** minimise the change in distogram or pair representation
   between consecutive recycling iterations — TTT toward the model's own fixed point.
4. **dropout consistency.** two stochastic forward passes with dropout active,
   minimise the distance between predictions. works when MSA depth is already low,
   which is exactly this set.
5. **MLM loss on the Evoformer single representation.** mask sequence/MSA positions and
   predict them, backprop into Evoformer. closest analogue to what already works in
   ProteinTTT on ESMFold. **deep-MSA subset only** — on a 4-sequence alignment this
   degenerates to memorising the query.
6. **MSA subsampling consistency.** two different MSA subsamples of the same target,
   minimise disagreement between their predicted distograms. exploits AF2's known
   sensitivity to MSA depth. costs one extra forward pass per step.
   **deep-MSA subset only** — undefined at depth 2.
7. **combinations**, with a loss-weight sweep once each component works alone.
8. **restrict TTT to a subset of Evoformer blocks.** AF2 has 48 blocks — update only
   some of them and freeze the rest. memory is not a constraint here, so this is not
   about fitting on the GPU: fewer updated parameters means less drift from the
   pretrained weights, and it localises where TTT actually needs to act.
   start with the last 8, then 16, then 24, then all 48 as the reference point.
   worth one extra probe once a loss shows signal: last-N vs first-N vs every-k-th
   block, since it is not obvious whether the update should enter early
   (MSA and coevolution processing) or late (structure-facing representation).
   this is an axis orthogonal to the loss choice, not a separate loss idea — apply it
   to whichever loss from 1–7 works, and always report the all-48 number next to it.

if none of 1–8 produces signal by M2 after a fair attempt at each, stop and write a
negative-result summary in LOG.md rather than continuing to tune.

## data and budget

- hard set (18 proteins): `subsets/lowconf18.txt`. the dataset it is drawn from is
  `data/lowconf/` (25 chains) — natives `data/lowconf/pdbs/<id>.pdb`, models
  `data/lowconf/af2_models/<id>.pdb` (AlphaFold DB, pLDDT in B-factor), metadata
  `af2_lowconf.csv`, provenance `data/lowconf/README.md`
- MSAs: `data/lowconf/msa/<id>.a3m`, query line first. precomputed, on disk, do not
  regenerate — they are what the M0 baseline consumed
- baseline predictions: `predictions/lowconf/<id>.pdb` + `<id>_plddt.npy` (gitignored,
  regenerate with `scripts/af2_predict.slurm` if lost); scores in `af2_run_results.csv`
- fixed subset: `subsets/lowconf5.txt` — the M1/M2 five, all drawn from the hard 18.
  committed so numbers stay comparable across experiments. spans MSA depth (2 deep,
  1 mid, 2 shallow) and is short (73–169 residues) so a probe is minutes, not hours
- base commit for the diff: `f3211e1a19940012a230a8f5cc9b7d2f665f92c0`
- accounts and queues:
  - probes on the 5-protein subset: `-A OPEN-35-8 -p qgpu_exp` (express queue)
  - if a job stays PENDING longer than 20 minutes, switch to `-A OPEN-37-88 -p qgpu`
    and record the wait in LOG.md
  - full-set and long runs: `-A OPEN-37-88 -p qgpu` directly
- budget: **50 node-hours** on OPEN-37-88. for scale on an A100-40GB, one plain forward
  pass takes **471 s over the 5** and **1708 s over the 18** (2359 s over all 25). 10 TTT
  steps with a backward pass is roughly 20–30× that per target, so budget on the order of
  **2–4 GPU-hours per hard-set config** — i.e. ~15 hard-set runs total, and most
  iteration must happen on the 5
- artifacts: aggregate per-protein predictions into one archive per experiment —
  there is an inode quota on this filesystem, do not emit thousands of small pdb files

## out of scope

- changing the eval script or the lDDT implementation
- changing the MSA or template pipeline used for the baseline, except where a method
  idea above explicitly subsamples the MSA at TTT time
- training or fine-tuning AF2 on any external dataset
- any new objective that is a function of pLDDT
- the sibling `af2_fails` set (13 CASP15 targets, `af2_fails.csv`). it is the opposite
  regime — AF2 confidently wrong, pLDDT 76–94 — and is not part of any acceptance
  criterion here. leave it untouched; it is the held-out check if lowconf ever
  produces a winner

## logging

- during experiments write logs to `./logs`. start each log with a short description
  of what the experiment is and what it is testing, then one line per TTT step with
  the pLDDT at that step, so the pLDDT trajectory is visible without re-running
- after each experiment append to `LOG.md`: what was run, what was decided and why,
  the Δ table, and `git diff --stat` against the base commit. it is the record of
  steps and decisions, not just of results — negative results go in it too

## definition of done

create `DONE` containing: the per-protein Δ table, count of ΔlDDT ≥ +7, count of
ΔlDDT ≤ −7, the winning config, second-seed reproduction numbers,
`git diff --stat` against the base commit, and the exact command that reproduces
the final result.

## M0 baseline — per protein

`model_1_ptm`, seed 0, unrelaxed, precomputed MSA, no templates. lDDT and TM on the
0–1 scale as stored in `af2_run_results.csv`. sorted by lDDT. `set` marks membership:
**H** = hard 18, gating; `-` = one of the excluded 7, predicted and reported but not
gating. `*` marks the M1/M2 five.

| set | id | len | depth | Neff | pLDDT | lDDT | TM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H | 4ZKQ_A | 412 | 2 | 1 | 25.5 | 0.182 | 0.23 |
| H | 8V1J_A | 116 | 4 | 2 | 32.0 | 0.190 | 0.20 |
| H | 7AD5_A | 122 | 3 | 3 | 42.7 | 0.273 | 0.24 |
| H\* | 2CME_A | 90 | 15 | 2 | 37.2 | 0.297 | 0.22 |
| H | 7ZKD_A | 88 | 4 | 4 | 41.6 | 0.302 | 0.30 |
| H | 8FW5_H | 117 | 4 | 1 | 41.7 | 0.334 | 0.29 |
| H | 2GFP_A | 375 | 6659 | 4630 | 63.9 | 0.337 | 0.50 |
| H\* | 9SLR_A | 73 | 10 | 2 | 37.7 | 0.365 | 0.38 |
| H | 5MQC_A | 280 | 53 | 34 | 42.8 | 0.402 | 0.43 |
| H\* | 1YDU_A | 169 | 3275 | 313 | 64.0 | 0.476 | 0.46 |
| – | 2N7P_A | 103 | 5107 | 4611 | 81.5 | 0.479 | 0.54 |
| H | 6RIM_A | 475 | 27 | 16 | 51.1 | 0.499 | 0.74 |
| H | 7JZJ_A | 284 | 28 | 12 | 48.1 | 0.504 | 0.56 |
| H | 6Z5S_W | 102 | 11 | 1 | 55.9 | 0.515 | 0.64 |
| H\* | 7DDQ_X | 83 | 136 | 56 | 64.5 | 0.581 | 0.57 |
| H\* | 2MP4_A | 156 | 4935 | 1959 | 65.9 | 0.590 | 0.77 |
| H | 5A98_A | 248 | 7 | 6 | 60.0 | 0.608 | 0.70 |
| – | 2MJM_A | 96 | 306 | 34 | 71.0 | 0.611 | 0.72 |
| H | 6OEE_A | 280 | 130 | 50 | 52.0 | 0.634 | 0.38 |
| H | 8Z11_F | 206 | 6062 | 3002 | 66.9 | 0.654 | 0.75 |
| – | 1ZW8_A | 64 | 5010 | 2580 | 79.9 | 0.667 | 0.76 |
| – | 8Q66_B | 246 | 20 | 14 | 71.7 | 0.689 | 0.59 |
| – | 5M86_A | 324 | 87 | 60 | 81.6 | 0.711 | 0.90 |
| – | 6CFW_F | 148 | 6434 | 2697 | 86.0 | 0.751 | 0.89 |
| – | 2WBV_A | 188 | 81 | 47 | 84.3 | 0.858 | 0.95 |

**headroom is not what limits this.** every member of the hard 18 sits below lDDT 0.70
by construction — the worst is 0.182 and the best 0.654 — so the required +0.07 does not
push any of them past the set's own exclusion threshold, let alone a physical ceiling.
6 of 18 is a question of whether TTT works at all, not of ceiling effects. the binding
constraint is the MSA one above: 11 of the 18 are effectively single-sequence.

---

# repo facts

## environment

there are **no AlphaFold genetic databases on this cluster** (no BFD, UniRef, MGnify,
PDB70). anything that calls jackhmmer/hhblits — stock `run_alphafold.py`,
`docker/run_docker.py`, the whole install path in `README.md` — cannot run here.
inference is database-free from the precomputed a3m files.

| thing | path |
| --- | --- |
| repo | `/scratch/project/open-37-88/pimenol/af2ttt/alphafold` |
| inference env | `/scratch/project/open-37-88/pimenol/af2ttt/af2env` |
| model parameters | `/scratch/project/open-37-88/pimenol/af2ttt/af2params/params` |

`af2env` (built by `scripts/setup_af2_env.sh`) has jax 0.4.26 + `jax[cuda12]`,
dm-haiku 0.0.12, tensorflow-cpu 2.16.1, numpy 1.24.3. it deliberately omits
openmm/pdbfixer (predictions are scored unrelaxed) and matplotlib — which is why
`run_af2_on_dataset.py` inlines `empty_template_features` instead of importing
`alphafold.notebooks.notebook_utils`.

**jax version pinning is load-bearing.** jax, jaxlib, jax-cuda12-plugin and
jax-cuda12-pjrt must all be the same version. a second `pip install` that lets jax
drift ahead of the CUDA plugins fails at runtime with
`jax_cuda12_plugin._triton has no attribute get_arch_details`. `setup_af2_env.sh` does
one resolve and then asserts the four agree — keep that guard. TTT adds an optimiser
(optax); install it in the same resolve, not on top.

login nodes have no GPU. import checks need `JAX_PLATFORMS=cpu`; real runs go via SLURM.

## commands

```bash
pytest alphafold/model/lddt_test.py -k unit      # upstream tests; conftest.py parses absl flags
sbatch scripts/af2_predict.slurm                 # both sets on one A100
$ENV/bin/python scripts/run_af2_on_dataset.py --dataset lowconf \
    --out_dir predictions/lowconf --params_dir $PARAMS --only 2N7P_A
python3 scripts/score_af2_predictions.py         # -> af2_run_results.csv, af2_run_report.html
python3 scripts/build_af2_lowconf_dataset.py --stage verify   # dataset self-check
```

## inference path

`run_alphafold.py` → `alphafold/data/pipeline.py` (feature dict) →
`alphafold/model/model.py:RunModel.process_features` (TF transforms in
`alphafold/model/tf/`) → `RunModel.predict` → `alphafold/model/modules.py`
(`AlphaFold` → `EmbeddingsAndEvoformer` → `EvoformerIteration` ×48 → `folding.py`
structure module, plus `PredictedLDDTHead`, `DistogramHead`, `MaskedMsaHead`,
`PredictedAlignedErrorHead`) → `alphafold/common/protein.py:from_prediction` →
optional relaxation in `alphafold/relax/`. config in `alphafold/model/config.py`
(`MODEL_PRESETS`, `CONFIG_DIFFS`, `get_model_config`), weights via
`alphafold/model/data.py:get_model_haiku_params`. multimer code is parallel throughout
and irrelevant — this project is single-chain only.

`scripts/run_af2_on_dataset.py` bypasses `DataPipeline` and builds features directly:

```python
features = {**pipeline.make_sequence_features(seq, target, len(seq)),
            **pipeline.make_msa_features([msa]),
            **empty_template_features(len(seq))}
```

templates are always empty and that is deliberate — every one of these structures is
in the PDB today, so templates would hand the model the answer.

`alphafold/model/lddt.py` is the differentiable in-graph lDDT used by the pLDDT head.
it is **not** the eval metric; `scripts/score_af2_predictions.py` has its own numpy
all-atom lDDT. TTT losses may use the former; reported results must use the latter.

## traps that already cost a day

- **pLDDT must be passed explicitly** as `b_factors` to `protein.from_prediction`, or
  every output PDB gets a zero B-factor column. the scorer hard-fails on zero mean
  pLDDT now.
- **the ColabFold API terminates each a3m part with a NUL byte.** `parsers.parse_a3m`
  passes it through silently; `pipeline.make_msa_features` then dies with
  `KeyError('\x00')`. both dataset builders strip it and both verifiers assert no stray
  characters remain.
- HTTP outside `alphafold/` goes through **curl subprocesses, not urllib** — the conda
  Python's CA bundle rejects predictioncenter.org and zhanggroup.org.
- the dataset builders depend on **numpy only** — no pandas, no biopython. PDB parsing
  is column slicing. keep it that way.
- `predictions/`, `logs/`, `data/.cache/` and `scripts/bin/` are gitignored.
- upstream style is 2-space indent, 80 columns (`[tool.pyink]` in `pyproject.toml`).
