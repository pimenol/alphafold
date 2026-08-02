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

---

## 2026-08-02 — AlphaFold is not reproducible across devices: 9.4 Å, same code and seed

This invalidated the first M2 reading and changes how every future Δ must be measured.

### What happened

The M2 sweep on `9SLR_A` showed all seven configs degrading lDDT by 4.8–10.6 points. The
ordering was wrong, though: **lr 1e-6 moved the parameters ~100× less than lr 1e-4 and
did nearly as much damage** (−8.10 vs −10.62), and distogram entropy at lr 1e-6, which
shifted its own loss by 0.7 %, still produced 9 Å of structural change. Damage that barely
depends on step size is not a property of the method.

### The control

Ran the TTT runner with `--steps 0`: unchanged parameters, so its eval forward must
reproduce the M0 prediction exactly.

| comparison | superposed CA RMSD |
| --- | ---: |
| my zero-step TTT eval vs **stock `run_af2_on_dataset.py` on CPU** | **0.0000 Å** |
| stock `run_af2_on_dataset.py` on CPU vs **stored A100 M0** | **9.3935 Å** |
| my zero-step TTT eval vs stored A100 M0 | 9.3935 Å |

The TTT eval path is byte-perfect. The 9.4 Å is **AlphaFold itself**, run through the
unmodified M0 script with identical parameters, features, seed and config — only the
device differs. Not numerical noise on a set selected for low confidence: a different
fold. Measured accuracy on `9SLR_A` differs accordingly — A100 M0 lDDT **36.50**, CPU M0
lDDT **29.00**, a 7.5-point gap from hardware alone, which is larger than the +7 the
project must demonstrate.

Running the zero-step control with fitted and with full MSA padding gave *identical*
predictions (17.43369 Å raw, both), independently reconfirming that the padding fit is a
no-op and isolating the difference to the device.

### Consequences

1. **The first M2 table is void.** Those Δs were CPU predictions scored against an A100
   baseline, so they measured the device gap, not TTT. Not reported as a result.
2. **Every Δ must be device-matched.** `ttt/evaluate.py` now measures the baseline itself
   from an M0 prediction made on the same device (`--device cpu|gpu`) rather than reading
   the A100 numbers out of `af2_run_results.csv`. Verified: the zero-step control now
   scores exactly +0.00 lDDT, +0.00 TM, +0.00 pLDDT.
3. **A CPU M0 baseline is being built** for all 18 hard-set targets
   (`predictions/lowconf_cpu/`), since the campaign runs on CPU while the GPU partitions
   are down.
4. The A100 M0 in `af2_run_results.csv` stays the reference for the *dataset* and for the
   hard-set definition; it is kept as a `gpu_m0_lddt` column so the two are never mixed.

### Why this was worth the compute

Three separate wrong conclusions came out of not having this control: a sign-inverted
ΔlDDT, a withdrawn "overshoot" claim, and a uniform-degradation result that was an
artifact. A zero-step control costs one eval forward and would have caught all three. It
is now the first thing run against any new configuration.

---

## 2026-08-02 — M2 on dev5: seven configs, no signal (NEGATIVE RESULT)

**Runs:** `m2-{viol,entropy}-*` on `subsets/lowconf5.txt`, 10 steps each, scored against
the **CPU** M0 baseline (device-matched — see the previous entry). 4 of 5 targets complete
at the time of writing; the fifth (`1YDU_A`, deep MSA) is still running and cannot change
the conclusion, since every effect is an order of magnitude below the threshold.

### Δ lDDT in points, at step 10

| config | mean | median | ≥ +7 | ≤ −7 | mean ΔpLDDT |
| --- | ---: | ---: | ---: | ---: | ---: |
| violation lr 1e-5 | **+0.76** | +0.72 | 0 | 0 | −0.08 |
| violation lr 1e-6, last 8 blocks | −0.05 | +0.01 | 0 | 0 | −0.04 |
| violation lr 1e-6 | −0.13 | −0.00 | 0 | 0 | +0.21 |
| entropy lr 1e-5 | −0.16 | −0.03 | 0 | 0 | +1.34 |
| violation lr 3e-5 | −0.22 | +0.00 | 0 | 0 | +1.70 |
| entropy lr 1e-6 | −0.38 | −0.19 | 0 | 0 | +0.12 |
| violation lr 1e-4 | −0.56 | +0.00 | 0 | 0 | +0.27 |

### M2 verdict: not met

M2 needs mean ΔlDDT > 0 **and** at least one protein at ΔlDDT ≥ +7. Only violation at
lr 1e-5 clears the first half. **Nothing reaches +7 on any protein** — the largest single
gain anywhere is +2.74, the largest loss −3.39. Seven configurations spanning four
learning rates over two decades and two losses all land within ±0.8 lDDT of zero.

Nothing is catastrophic either: no regression past −7, and mean ΔpLDDT is near zero. The
earlier "every config degrades by 4.8–10.6 points" picture was entirely the device
artifact, now removed.

### What the traces show

Both losses optimise correctly — this is not an optimisation failure:

- violation lr 1e-5: 0.06959 → **0.00004** over 10 steps, residues-in-violation 0.260 →
  0.027. The physics term is essentially solved.
- entropy lr 1e-5: 3.205 → 2.919.
- lr 1e-6 on either loss is a near-no-op: `7DDQ_X` comes back **exactly ±0.00** on lDDT,
  TM and pLDDT under most configs, so the low rates change nothing at all.

**Driving the violation loss to zero moves true accuracy by less than one lDDT point.**
That is the substantive finding: AlphaFold's structural violations on these targets are
not what makes them wrong. The predictions are stereochemically fine and globally
misfolded, so a physics-based self-consistency objective has almost no purchase on the
error. Distogram sharpening likewise buys nothing.

Deep-MSA target `2MP4_A` is mildly positive under most configs (+0.04 to +0.77), with
entropy lr 1e-5 giving +4.67 pLDDT for +0.77 lDDT — the largest confidence gain in the
sweep, and still far from useful accuracy.

### Cost

Seven configs × 5 targets on 128-core CPU nodes, ~10 h wall-clock, run in parallel. Zero
GPU node-hours consumed; the `open-37-88` budget is untouched.

`git diff --stat f3211e1 -- alphafold/ run_alphafold.py docker/`: still empty.

### Next

Per CLAUDE.md, stop tuning these two and work down the method list. Queued: **dropout
consistency** (method idea 7) — two stochastic forward passes, minimise the symmetric KL
between their distograms. MSA-free, so it applies to all 18, and it needs no upstream
hook. A zero-step control runs alongside it, which is now standard for every new config.

Recycling consistency (idea 6) is blocked on the same `hk.while_loop` limitation that
forced `num_recycle=0`: comparing consecutive recycling iterations requires either a
differentiable loop or an upstream hook to feed `prev` in explicitly. That would be the
first justified change to `modules.py` if dropout consistency also comes back flat.
