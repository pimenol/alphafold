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

## 2026-08-02 — M1 PASSED (on CPU)

**Run:** `cpudryrun` — violation loss, lr 1e-4, **1 step**, `9SLR_A` (73 res, depth 10),
CPU. Intended only as an end-to-end plumbing check while the GPU queue was blocked; it
answered M1 outright.

### M1 verdict: passed

| check | result |
| --- | --- |
| coordinates differ from baseline | yes — 3.613 Å CA RMSD superposed (14.419 Å raw) |
| \|ΔlDDT\| ≥ 1 on at least one protein | yes — **+1.02** |

Gradients reach the structure module.

### Result, against the M0 baseline

| metric | M0 | after 1 step | Δ |
| --- | ---: | ---: | ---: |
| lDDT | 36.50 | 37.52 | **+1.02** |
| TM | 0.38 | 0.36 | −0.02 |
| pLDDT | 37.74 | 41.08 | +3.34 |
| violation loss | 0.06959 | — | falls monotonically (see M2) |

One step at lr 1e-4 moves the structure 3.6 Å and gains a point of lDDT. Not evidence the
method works — one step, one protein, and TM went slightly the wrong way — but the
direction is not hostile.

### Two errors I made reading this run, both now fixed

**1. Scored against the wrong baseline.** `af2_run_results.csv` carries two sets of
columns: `base_*` is the shipped AlphaFold DB model, `run_*` is our own M0 AlphaFold run
with these MSAs. CLAUDE.md defines the baseline as M0, i.e. `run_*`. `ttt/evaluate.py`
read `base_*`. For `9SLR_A` those differ a lot — AFDB lDDT 45.0 versus M0 36.5 — so the
first reading came out as **ΔlDDT −7.48** when the correct figure is **+1.02**. The sign
was inverted. Fixed in `ttt/evaluate.py`; the AFDB number is kept as a separate
`afdb_lddt` column so the two can never be confused again.

**2. Claimed lr 1e-4 "overshoots".** That rested on this run's step-0 violation loss of
0.01054 rising to 0.05638. The 0.01054 **does not reproduce**: every later run of the same
target with the same parameters starts at **0.06959**, as does an independent forward-only
check. A matched control (identical config, only `lr` differing) shows the loss *falling*
at lr 1e-4 — 0.06959 → 0.00743 → 0.00291 — with pLDDT climbing faster than at any lower
rate. The overshoot claim is withdrawn. I could not attribute the anomalous step-0 value
to any config difference; it came from a throwaway run and is not worth further GPU time,
but it is recorded here rather than quietly dropped.

The correction that mattered: I had written that "the violation loss goes down while lDDT
goes down with it — proxy and target disagree". With the baseline fixed, that is false.
At lr 1e-4 on this target the loss goes down, pLDDT goes up, **and lDDT goes up**. No
divergence was observed. The sweep was widened to {1e-6, 1e-5, 3e-5, 1e-4} rather than
narrowed.

### Second finding: the pLDDT selection signal may not track eval pLDDT

The in-loop 0-recycle pLDDT rose 39.22 → 39.90 while eval pLDDT rose 37.74 → 41.08 — same
direction here, but the in-loop signal moved by 0.7 where eval moved by 3.3. Worth
watching across the sweep: best-pLDDT step selection is only meaningful if the proxy
ranks steps the way the eval forward would. This is a consequence of the num_recycle=0
train / num_recycle=3 eval split.

### Bug found and fixed: CA RMSD was measuring rotation, not conformational change

The first version reported raw coordinate RMSD. AlphaFold's output frame is arbitrary, so
that number mixes a global rotation in with any real change — it read 14.419 Å where the
actual conformational change was 3.613 Å. Added Kabsch superposition; a rigid-motion
control now gives 0.000 Å superposed against 67.5 Å raw. Both numbers are recorded.
(`cpudryrun` predates the fix, so its stored summary carries only the raw figure.)

### Cost

2510 s for 1 step + 1 eval forward on 73 residues on CPU.

`git diff --stat f3211e1 -- alphafold/ run_alphafold.py docker/`: still empty.

### Next

M2 on `subsets/lowconf5.txt`, 10 steps: violation at lr {1e-6, 1e-5, 3e-5, 1e-4}, entropy
at {1e-6, 1e-5}, plus violation lr 1e-6 restricted to the last 8 Evoformer blocks.

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
