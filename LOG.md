# TTT experiment log

Every experiment appends here: what was run, what was decided and why, the Δ table, and
`git diff --stat` against the base commit. Negative results go in too — a run where mean
pLDDT improves and mean lDDT does not is a negative result and is recorded as one.

Base commit: `f3211e1a19940012a230a8f5cc9b7d2f665f92c0`
Success criteria, hard set and metric conventions: `CLAUDE.md`.

Δ is always `TTT − baseline` on the same protein through the same eval path, in **lDDT
points (0–100)**. The CSVs store fractions, so +7 points is +0.07 there.

---

## 2026-08-02 — setup: TTT harness, before any experiment

### What was built

| file | role |
| --- | --- |
| `ttt/core.py` | unsupervised losses, Adam, ground-truth guard, Evoformer block mask |
| `ttt/run_ttt.py` | experiment runner: N gradient steps per target, then predict |
| `ttt/evaluate.py` | scores an experiment against M0, prints the Δ table |
| `scripts/ttt_experiment.slurm` | batches several configs into one allocation |

`git diff --stat f3211e1 -- alphafold/ run_alphafold.py docker/` is **empty**: no
upstream AlphaFold source was modified. The only changes to tracked files are
`.gitignore` and `scripts/setup_af2_env.sh`, both additive and neither in the model path.

### Design decisions and why

**TTT gradient steps run at `num_recycle=0`; evaluation is untouched.**
`modules.AlphaFold.__call__` drives recycling with `hk.while_loop`
(`alphafold/model/modules.py:395`), which has no reverse-mode transpose rule — `jax.grad`
through a recycling forward pass raises. Setting `num_recycle=0` skips the loop and
leaves a single differentiable `AlphaFoldIteration`. Every *reported* prediction still
comes from the unmodified M0 forward pass (3 recycles, ensembling, seed 0), so a Δ only
ever reflects the parameters. The alternative — hooking upstream to expose a
differentiable last iteration — would have cost a diff in `modules.py` for no gain at
this stage. Cost side-effect: the TTT forward is ~4× cheaper than the eval forward.

**Losses reach the structure module without touching upstream code.**
The structure module only returns `final_atom14_positions` when built with
`compute_loss=True`, which inference never does (`folding.py:504`). atom14 is a subset of
atom37 per residue, so `ttt/core.py:_atom14_positions` recovers it with an exact,
differentiable gather through `residx_atom14_to_atom37`. The violation term is then
AlphaFold's own `find_structural_violations` + `structural_violation_loss`, not a
reimplementation.

**The pLDDT head is frozen structurally, not just incidentally.** `split_trainable`
holds back the 4 `predicted_lddt_head` modules. Neither current loss reads that head so
its gradient is already zero; freezing makes the constraint impossible to violate by
accident later.

**Ground truth cannot enter the loop.** `core.forbid_ground_truth` wraps `builtins.open`
for the duration of the TTT steps and raises `GroundTruthAccess` on any read under
`data/lowconf/pdbs/`. Scoring runs in a separate process, outside the guard.

**Optimiser is hand-rolled Adam** (b1 0.9, b2 0.999, eps 1e-6, global-norm clip 0.1 —
AF2 paper values) rather than optax, so the pinned jax/jaxlib/CUDA-plugin set in `af2env`
is not disturbed by another dependency resolve.

### Bugs found and fixed before the first run

A shape-only smoke test (`jax.eval_shape` over the loss and its gradient, CPU, seconds)
caught three defects that would each have wasted a GPU allocation:

1. **`--blocks` would have trained nothing.** I assumed the 48 Evoformer blocks were 48
   haiku modules named `evoformer_iteration`, `_1`, …. They are not: the trunk is one
   `hk.scan`-stacked module and every array under it carries a leading block axis of
   length 48. Module-name selection matched zero modules. Replaced with `block_mask`, a
   0/1 gradient mask along axis 0. Verified: `blocks=None` → 93.2M trainable,
   `blocks=24` → 49.2M, `blocks=8` → 20.0M.
2. **`stereo_chemical_props.txt` was missing.** `find_structural_violations` →
   `residue_constants.make_atom14_dists_bounds` reads it, and upstream does not vendor
   it — `docker/Dockerfile:65` downloads it at image build time. Fetched it and added the
   same download to `scripts/setup_af2_env.sh` so the env stays reproducible.
3. **`confidence.compute_plddt` cannot be traced** — it converts to numpy internally, so
   using it as an in-loop selection signal raised `TracerArrayConversionError`. Added a
   jnp bin-centre expectation in `core.plddt_from_output`, used *only* for step
   selection; every reported pLDDT still comes from the upstream function.

### Cluster note

`qgpu` is heavily loaded: 110 jobs pending, and a 12 h walltime request was scheduled
~24 h out. Shorter requests backfill far better, so the campaign uses 1–5 h allocations
that batch several configs per job rather than one job per config. A 20-minute wait on
`qgpu_exp` (`open-35-8`) triggered the documented switch to `open-37-88 -p qgpu`.

### Next

M1 — does TTT move the structure at all. Violation loss and distogram entropy, lr sweep
{1e-5, 1e-4, 1e-3}, 10 steps, on `subsets/lowconf5.txt`.
