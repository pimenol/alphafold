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

---

## 2026-08-02 — M1 PASSED (on CPU), and lr 1e-4 is far too large

**Run:** `cpudryrun` — violation loss, lr 1e-4, **1 step**, `9SLR_A` (73 res, depth 10),
CPU. Intended only as an end-to-end plumbing check while the GPU queue was blocked; it
answered M1 outright.

### M1 verdict: passed

| check | result |
| --- | --- |
| coordinates differ from baseline | yes — 3.613 Å CA RMSD superposed (14.419 Å raw) |
| \|ΔlDDT\| ≥ 1 on at least one protein | yes — **7.48** |

Gradients reach the structure module. Per the milestone definition, moving in the wrong
direction passes M1: this proves the machinery works, not that the method works.

### But the direction is wrong, and the optimiser is overshooting

| metric | baseline | after 1 step | Δ |
| --- | ---: | ---: | ---: |
| lDDT | 44.99 | 37.52 | **−7.48** |
| TM | 0.38 | 0.21 | −0.17 |
| pLDDT (eval) | 47.22 | 41.08 | −6.14 |
| violation loss | 0.01054 | 0.05638 | **+0.045** |
| residues in violation | 0.233 | 0.329 | +0.096 |

**The loss went up after a step that was supposed to minimise it.** I read that as
overshoot and shifted the sweep down two decades.

> **CORRECTION (same day, see the M2 entry).** The overshoot reading was wrong. A matched
> control — identical config, only `lr` differing — shows the violation loss *falling* at
> lr 1e-4: 0.06959 → 0.00743 → 0.00291, with pLDDT climbing faster than at any lower rate.
> This run's step-0 loss of **0.01054 does not reproduce**; every later run of the same
> target and parameters starts at **0.06959**, which is also what the independent
> forward-only padding check measured. The step-0 value here is anomalous and I could not
> attribute it to any config difference, so the loss-went-up comparison rested on a bad
> baseline and is withdrawn.
>
> What survives is the accuracy measurement, which was taken against the native and does
> not depend on that baseline: **one step at lr 1e-4 cost 7.48 lDDT points.** Combined
> with the control, that is a sharper and more useful result than "overshoot" — the
> violation loss goes *down* while lDDT goes *down with it*. The proxy and the target
> disagree. That is the project's named failure mode, arriving at the first method.
>
> The sweep was still widened rather than narrowed: {1e-6, 1e-5, 3e-5, 1e-4} all ran.

### Second finding: the pLDDT selection signal does not track eval pLDDT

The in-loop 0-recycle pLDDT *rose* over the step (39.22 → 39.90) while the 3-recycle eval
pLDDT *fell* (47.22 → 41.08). Best-pLDDT step selection is therefore selecting on a
quantity that disagrees in sign with the thing it is a proxy for. This is a direct
consequence of the num_recycle=0 train / num_recycle=3 eval split. Watch it across the
lr sweep; if it persists, either evaluate the selection signal with recycling (costly) or
report step-10 only and say so.

### Bug found and fixed: CA RMSD was measuring rotation, not conformational change

The first version reported raw coordinate RMSD. AlphaFold's output frame is arbitrary, so
that number mixes a global rotation in with any real change — it read 14.419 Å where the
actual conformational change was 3.613 Å. Added Kabsch superposition; a rigid-motion
control now gives 0.000 Å superposed against 67.5 Å raw. Both numbers are recorded.

### Cost

2510 s for 1 step + 1 eval forward on 73 residues on CPU. Fine as a one-off; everything
else needs the GPU.

`git diff --stat f3211e1 -- alphafold/ run_alphafold.py docker/`: still empty.

### Next

M2 on `subsets/lowconf5.txt`, 10 steps: violation and entropy at lr {1e-6, 1e-5}, plus
violation lr 1e-6 restricted to the last 8 Evoformer blocks (method idea 8 — worth
pulling forward given how lr-sensitive the full-parameter update turned out to be).

---

## 2026-08-02 — infrastructure: the campaign moved to CPU, and TTT got ~3× cheaper

Not an experiment. Two changes that decide whether the campaign can run at all.

### The GPU partitions are unusable, the CPU partitions are empty

`qgpu` has 72 nodes: **16 in maintenance, 18 drained, 14 down**, 23 allocated, 1
draining. Nothing free, 110 jobs pending, and the scheduler put a 12 h request ~24 h out.
All five GPU partitions map to that same pool, so there is no partition trick.

`qcpu` has **306 idle nodes at 128 cores each**, and a job starts in about ten seconds.
The whole M2 sweep therefore runs on CPU: five configs as five parallel jobs, each on its
own 128-core node, rather than queued behind each other for a GPU. GPU tickets stay
queued as a bonus. CPU hours come from `open-35-15`, so the 50 GPU node-hours on
`open-37-88` are untouched.

### Fitting the MSA padding: same objective, ~3× less work

The hard set is 11/18 effectively single-sequence, and the eval-sized feature dict pads
regardless. For `9SLR_A` (10 real sequences): `msa` is padded to 508 rows and `extra_msa`
to 5120 rows, of which **zero carry a real sequence**. The extra-MSA stack processes all
5120 on every forward *and* backward pass.

Those rows are masked, so dropping them should change nothing. Verified rather than
assumed, forward-only, both MSA regimes:

| target | depth | padding | violation | entropy | pLDDT | coords |
| --- | ---: | --- | --- | --- | --- | --- |
| 9SLR_A | 10 | 508/5120 → 14/8 | Δ 1.6e-7 | Δ 4.8e-7 | Δ 7.6e-6 | **0.0000 Å** |
| 1YDU_A | 3275 | 508/5120 → 512/2771 | Δ 2.0e-8 | Δ 1.2e-7 | Δ **0** | **0.0000 Å** |

2.7× faster on the shallow target including compile (205 s → 75 s); the steady-state gain
is larger because compile dominates that figure. `--fit_msa_padding` applies this to the
**TTT config only** — the eval forward stays byte-identical to M0, which is what keeps
every Δ attributable to the parameters.

### Three infrastructure bugs

1. **`nohup` from a tool call does not survive.** The first padding check was launched in
   the background from a shell that was then reaped; the process died with it and left an
   empty log. Everything substantial now goes through Slurm.
2. **The job script hard-coded `#SBATCH --gpus=1`**, so every CPU-partition submission
   failed with "Requested node configuration is not available". Resources now come from
   the command line.
3. **Unquoted expansion word-split the run arguments.** `--description "M2 dev5: ..."`
   reached argparse as a stray `dev5:` token and all five runs died after 8 s. `eval`
   honours the inner quotes; checked before resubmitting.
