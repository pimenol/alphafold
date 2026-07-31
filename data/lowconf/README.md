# `af2_lowconf`: chains AlphaFold 2 gets wrong and knows it

25 single chains, 64-475 residues, every one with
**mean pLDDT < 70** and **lDDT < 0.7** against its
experimental structure. Built by `scripts/build_af2_lowconf_dataset.py`.

This is the complement of `../af2_fails.csv`, where AlphaFold is confidently
wrong (pLDDT 76-94, deep MSAs, blind CASP15 targets). Here the model's own
confidence already flags the failure, which makes the two sets useful to
contrast: one asks whether test-time training can fix errors the model cannot
see, the other whether it can fix errors the model can.

## How the set was found

CASP15 cannot supply this regime -- only two of its whole-chain targets clear
both cuts and still have a usable native -- so candidates come from the
AlphaFold Protein Structure Database:

1. SIFTS (`pdb_chain_uniprot.tsv`) gives one representative PDB chain per
   UniProt accession, keeping spans of 60-800 residues.
2. AFDB's API supplies each accession's model; those whose whole-chain mean
   pLDDT is above 75 are skipped as a cheap prefilter.
3. The model is trimmed to the chain's UniProt span, the experimental chain is
   renumbered onto the same span, and pLDDT, lDDT and TM-score are recomputed
   over the residues the two share. At least 80% of the span must be
   resolved in the crystal.
4. Chains clearing pLDDT < 70 and lDDT < 0.7 are kept, one
   per UniProt accession. The scan inspected 785 accessions to find them.

## Layout

| Path | Contents |
| --- | --- |
| `pdbs/<id>.pdb` | experimental structure, one chain, numbered 1..length over the span |
| `msa/<id>.a3m` | merged ColabFold MMseqs2 MSA, query first |
| `msa/raw/<id>.*.a3m.gz` | the UniRef and environmental parts as returned |
| `af2_models/<id>.pdb` | the AFDB model trimmed to the same span, pLDDT in B-factors |
| `../af2_lowconf.csv` | one row per chain |

`id` is `<PDB entry>_<chain>`. About 19 MB in total.

## Contents

| id | UniProt | len | MSA depth | Neff | pLDDT | lDDT | TM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `4ZKQ_A` | E9M5R0 | 412 | 3 | 1 | 25.4 | 0.14 | 0.17 |
| `8V1J_A` | A0A5B0PMB3 | 116 | 5 | 1 | 32.3 | 0.20 | 0.14 |
| `7AD5_A` | V5TFR9 | 122 | 4 | 2 | 41.5 | 0.33 | 0.36 |
| `7ZKD_A` | L7ISI9 | 88 | 5 | 3 | 41.7 | 0.28 | 0.20 |
| `2CME_A` | P59636 | 90 | 16 | 1 | 44.3 | 0.31 | 0.23 |
| `9SLR_A` | A0A679PMY0 | 73 | 10 | 1 | 47.2 | 0.45 | 0.53 |
| `2N7P_A` | Q388A3 | 103 | 5107 | 4635 | 47.8 | 0.30 | 0.26 |
| `5M86_A` | Q9HIW9 | 324 | 87 | 58 | 51.4 | 0.45 | 0.45 |
| `6Z5S_W` | Q6N1K3 | 102 | 13 | 1 | 57.4 | 0.61 | 0.79 |
| `8FW5_H` | G2TRJ4 | 117 | 6 | 1 | 58.9 | 0.35 | 0.30 |
| `8Q66_B` | Q19672 | 246 | 20 | 13 | 59.4 | 0.63 | 0.72 |
| `6OEE_A` | P97245 | 280 | 130 | 49 | 60.2 | 0.51 | 0.30 |
| `2MJM_A` | C3VPR6 | 96 | 306 | 32 | 60.5 | 0.58 | 0.69 |
| `6RIM_A` | A0A069CUU9 | 475 | 28 | 15 | 62.2 | 0.58 | 0.82 |
| `6CFW_F` | I6TXN5 | 148 | 6434 | 2638 | 62.4 | 0.67 | 0.81 |
| `7DDQ_X` | A0A2T4JIP3 | 83 | 136 | 54 | 63.4 | 0.59 | 0.48 |
| `7JZJ_A` | Q05128 | 284 | 28 | 11 | 64.8 | 0.63 | 0.79 |
| `8Z11_F` | A0A7S3X675 | 206 | 6062 | 3067 | 65.5 | 0.66 | 0.77 |
| `1YDU_A` | Q9M015 | 169 | 3275 | 311 | 66.0 | 0.47 | 0.46 |
| `1ZW8_A` | P47043 | 64 | 5011 | 2622 | 66.9 | 0.61 | 0.58 |
| `5A98_A` | Q9ELS6 | 248 | 7 | 5 | 67.3 | 0.67 | 0.73 |
| `5MQC_A` | Q9J7C2 | 280 | 53 | 32 | 67.4 | 0.62 | 0.77 |
| `2WBV_A` | Q65914 | 188 | 81 | 46 | 67.4 | 0.64 | 0.83 |
| `2GFP_A` | P31442 | 375 | 6659 | 4602 | 68.4 | 0.34 | 0.51 |
| `2MP4_A` | Q07750 | 156 | 4935 | 1976 | 69.5 | 0.60 | 0.78 |

## Important limitations

- **These predictions are not blind.** Most of these PDB entries were released
  before AlphaFold's 2021-09-30 training cutoff, so the structure was likely in
  its training set. Low accuracy here means the target is intrinsically hard --
  disorder, a conformation induced by a binding partner, a domain that is not
  independently foldable -- not that the model failed to generalise. If you need
  held-out targets, use `../af2_fails.csv`, which is CASP15-only and blind.
- **The accuracy numbers are computed here, not published by an assessor**,
  unlike the CASP15 set whose lDDT and TM-score come from the Prediction Center.
  The same lDDT and TM-score code is used for both, and it reproduces the
  official CASP15 values to within 0.02 on 13 of 13 targets there, which is the
  evidence that it is calibrated.
- **MSA depth was not filtered.** Low pLDDT often follows from a shallow
  alignment, and gating on depth would have shrunk the pool sharply. `msa_depth`
  and `msa_neff` are recorded so you can filter as needed; 18 of 25 have Neff below 100.
- The span is the region resolved in one crystal form. A chain whose fold
  depends on a partner will look like a prediction failure in isolation, which
  is a real AlphaFold limitation but not the same failure mode as a wrong fold.
