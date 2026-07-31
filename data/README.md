# `af2_fails`: CASP15 targets where standard AlphaFold 2 fails despite a deep MSA

13 single-chain proteins, 189-547 residues. Every one has a deep MSA
and an experimentally determined structure, and standard AlphaFold 2 still gets
it wrong. Built by `scripts/build_af2_fails_dataset.py`; re-running it
reproduces this directory.

## Where the AlphaFold numbers come from

The baseline is CASP15 group **270, `NBIS-AF2-standard`** (Elofsson lab), which
ran a standard AlphaFold v2.2 protocol as a CASP15 server. Its GDT_TS, LDDT and
TM-score are the official values published by the Prediction Center, so the
failures are documented independently of this repository. pLDDT is the mean
B-factor of the submitted model over the assessed residues.

CASP15 targets were released to the PDB in 2022 or later, after the 2021-09-30
training cutoff of the released AlphaFold v2.3 parameters, so these are blind
targets for the model in this repository. That is why the set is CASP15 only and
not CASP14, whose targets are inside that cutoff.

## Layout

| Path | Contents |
| --- | --- |
| `pdbs/<id>.pdb` | ground-truth structure, one chain, numbered by the CASP target sequence |
| `msa/<id>.a3m` | merged ColabFold MMseqs2 MSA, query first |
| `msa/raw/<id>.*.a3m.gz` | the UniRef and environmental parts as the API returned them |
| `af2_models/<id>.pdb` | the NBIS-AF2-standard model 1, pLDDT in the B-factor column |
| `../af2_fails.csv` | one row per protein |

About 70 MB in total, dominated by the MSAs. Downloads and intermediates
live in `data/.cache/` and are gitignored; delete that directory to reclaim
space, at the cost of re-fetching on the next build.

## Selection criteria

- AF2 baseline whole-chain TM-score < 0.8
- 60 <= length <= 800 residues
- MSA Neff >= 100 and depth >= 300 sequences, so a shallow
  alignment cannot explain the failure
- our recomputed lDDT and TM-score agree with the official ones to within
  0.12, which confirms the structure we rebuilt is the one CASP assessed

## Columns

| Column | Meaning |
| --- | --- |
| `id` | CASP15 target identifier |
| `sequence`, `length` | the full target sequence |
| `plddt_af2` | mean pLDDT of the AF2 baseline over assessed residues |
| `lddt_af2`, `tmscore_af2`, `gdt_ts_af2` | official CASP15 scores for group 270 |
| `n_eval_res` | residues CASP assessed (the rest are unresolved) |
| `msa_depth`, `msa_neff`, `neff_per_res` | MSA size and effective size at 62% identity |
| `native_source`, `native_pdb_id` | where the ground truth came from |
| `lddt_af2_recomputed`, `tmscore_af2_recomputed` | our own measurement of the same model |
| `plddt_af2_dm`, `lddt_af2_dm`, `tmscore_af2_dm` | DeepMind's own CASP15 baseline, scored the same way |

Ground truth is either the CASP-posted assessment domains concatenated back into
a chain (`casp15_domain_natives`), or the matching PDB chain renumbered onto the
target sequence (`rcsb_exact` for a verbatim match, `rcsb_aligned` where an
expression tag or unresolved region required an alignment).

## What the failures look like

| id | len | Neff | pLDDT | lDDT | TM | GDT_TS | native |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `T1137s4` | 547 | 3046 | 76.2 | 0.80 | 0.34 | 29.4 | CASP |
| `T1137s1` | 409 | 1412 | 86.4 | 0.79 | 0.41 | 36.0 | CASP |
| `T1137s3` | 524 | 2170 | 80.5 | 0.84 | 0.41 | 36.3 | CASP |
| `T1137s6` | 518 | 2884 | 85.3 | 0.83 | 0.41 | 34.5 | CASP |
| `T1114s1` | 189 | 244 | 80.6 | 0.81 | 0.43 | 42.4 | 7UUS_Q |
| `T1137s2` | 343 | 5248 | 86.4 | 0.84 | 0.43 | 38.5 | CASP |
| `T1137s5` | 390 | 2481 | 85.5 | 0.83 | 0.43 | 37.7 | CASP |
| `T1173` | 204 | 7584 | 78.2 | 0.53 | 0.47 | 40.0 | 8OML_A |
| `T1174` | 338 | 6804 | 76.8 | 0.66 | 0.56 | 40.6 | 8OND_A |
| `T1157s2` | 495 | 17954 | 93.7 | 0.78 | 0.64 | 45.9 | CASP |
| `T1121` | 381 | 612 | 92.1 | 0.80 | 0.74 | 61.6 | CASP |
| `T1115` | 288 | 10539 | 90.9 | 0.79 | 0.75 | 67.8 | 9Z7U_G |
| `T1180` | 404 | 10478 | 93.7 | 0.83 | 0.77 | 68.2 | CASP |

Correlation between `plddt_af2` and `tmscore_af2` across the set: **+0.73**.
10 of 13 are confidently wrong (pLDDT >= 80 yet TM < 0.8): `T1137s1`, `T1137s3`, `T1137s6`, `T1114s1`, `T1137s2`, `T1137s5`, `T1157s2`, `T1121`, `T1115`, `T1180`. Those are the interesting cases for test-time training, because the model gives no internal signal that anything is off.

## Limitations

- The MSAs are ColabFold MMseqs2 (UniRef30 + ColabFoldDB), not AlphaFold's own
  jackhmmer/HHblits triple, because no AlphaFold sequence databases are
  installed on this machine. Depth is comparable but not identical, so
  re-running AlphaFold with these MSAs will not reproduce the CASP numbers
  exactly. The `_recomputed` columns exist to make that gap visible.
- The MSAs were built against current databases and therefore contain sequences
  deposited after CASP15. Templates are deliberately excluded for the same
  reason: a run that uses templates loses the blind-target property, since these
  structures are now in the PDB.
- `lddt_af2` is CASP's all-atom LDDT over assessed residues only, which is why
  `n_eval_res` is a column and is sometimes well below `length`.
- DeepMind's own CASP15 baseline (`*_dm` columns, vendored in `docs/casp15_predictions.zip`) used a better protocol than group 270 and recovers several of these targets. Targets DeepMind hand-tuned are left blank rather than reported as automated results.

## Dropped candidates

| id | reason |
| --- | --- |
| `T1131` | Neff 1.0 below 100; depth 2 below 300 |
| `T1130` | Neff 1.0 below 100; depth 2 below 300 |
| `T1122` | Neff 1.0 below 100; depth 2 below 300 |
| `T1106s1` | Neff 17.8 below 100; depth 105 below 300 |
| `T1155` | Neff 30.8 below 100; depth 105 below 300 |
| `T1179` | Neff 2.0 below 100; depth 6 below 300 |
| `T1113` | Neff 4.0 below 100; depth 7 below 300 |
| `T1185s2` | TM 0.8 held in the reserve tier (>= 0.8) |
| `T1104` | Neff 33.6 below 100; depth 84 below 300 |
| `T1181` | TM 0.82 held in the reserve tier (>= 0.8) |
| `T1195` | TM 0.84 held in the reserve tier (>= 0.8) |
