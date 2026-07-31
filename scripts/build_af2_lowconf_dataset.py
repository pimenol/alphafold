#!/usr/bin/env python3
"""Builds a benchmark of single chains where AlphaFold 2 is both wrong and
knows it: mean pLDDT below 70 and lDDT below 0.7 against the experimental
structure.

This is the complement of scripts/build_af2_fails_dataset.py, which collects
targets AlphaFold gets wrong *confidently*.  CASP15 cannot supply this regime --
only two of its whole-chain targets clear both cuts and still have a usable
native -- so candidates come from the AlphaFold Protein Structure Database
instead: every PDB chain has a UniProt mapping in SIFTS, AFDB publishes a model
for that accession with pLDDT in the B-factor column, and the PDB entry is the
ground truth.

An entry is the observed span of one chain: the UniProt subsequence covering the
residues resolved in the crystal, renumbered from 1.  Model, native and MSA all
refer to that span, so each item is self-contained.

Unlike the CASP15 set these predictions are NOT blind -- most of these PDB
entries predate AlphaFold's 2021-09-30 training cutoff -- and the accuracy
numbers are computed here rather than published by an assessor.

Usage:
  python3 scripts/build_af2_lowconf_dataset.py --stage all
  python3 scripts/build_af2_lowconf_dataset.py --stage candidates
  python3 scripts/build_af2_lowconf_dataset.py --stage msa
  python3 scripts/build_af2_lowconf_dataset.py --stage csv
  python3 scripts/build_af2_lowconf_dataset.py --stage verify
"""

import argparse
import concurrent.futures as cf
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

import build_af2_fails_dataset as base

REPO = base.REPO
DATA = os.path.join(REPO, 'data', 'lowconf')
CACHE = os.path.join(base.CACHE, 'lowconf')
PDB_DIR = os.path.join(DATA, 'pdbs')
MSA_DIR = os.path.join(DATA, 'msa')
MSA_RAW_DIR = os.path.join(MSA_DIR, 'raw')
MODEL_DIR = os.path.join(DATA, 'af2_models')
CSV_PATH = os.path.join(REPO, 'af2_lowconf.csv')

SIFTS_URL = ('https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/'
             'pdb_chain_uniprot.tsv.gz')
AFDB_API = 'https://alphafold.ebi.ac.uk/api/prediction'

# The two cuts that define this set.
MAX_PLDDT = 70.0
MAX_LDDT = 0.70

# AFDB's whole-chain mean pLDDT is only a prefilter: a protein with long
# disordered tails can score low overall while the crystallised domain is
# predicted well, and vice versa.  Anything at or below this gets the full
# region-restricted check.  Set a little above the real cut because the
# resolved span is usually the better-ordered part of the chain.
PREFILTER_PLDDT = 75.0

MIN_LEN = 60
MAX_LEN = 800
# Enough of the span must be resolved for lDDT to mean anything.
MIN_OBSERVED_FRACTION = 0.8
TARGET_SIZE = 25
MIN_SET_SIZE = 10
MAX_SET_SIZE = 30
# Accessions screened on AFDB metadata, and the cap on how many of the
# resulting shortlist get the expensive structural evaluation.
METADATA_POOL = 30000
MAX_ACCESSIONS = 3000

log = base.log


def ensure_dirs() -> None:
  for d in (DATA, CACHE, PDB_DIR, MSA_DIR, MSA_RAW_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------
# Stage 1: find chains where AlphaFold is unconfident and wrong
# --------------------------------------------------------------------------


def sifts_chains() -> list[dict]:
  """One representative PDB chain per UniProt accession, longest span first."""
  path = os.path.join(base.CACHE, 'pdb_chain_uniprot.tsv.gz')
  base.fetch(SIFTS_URL, path)
  best = {}
  with gzip.open(path, 'rt') as f:
    for line in f:
      if line.startswith('#') or line.startswith('PDB\t'):
        continue
      parts = line.rstrip('\n').split('\t')
      if len(parts) < 9:
        continue
      pdb, chain, accession = parts[0], parts[1], parts[2]
      try:
        sp_beg, sp_end = int(parts[7]), int(parts[8])
      except ValueError:
        continue
      span = sp_end - sp_beg + 1
      if not MIN_LEN <= span <= MAX_LEN:
        continue
      current = best.get(accession)
      if current is None or span > current['span']:
        best[accession] = {'accession': accession, 'pdb': pdb, 'chain': chain,
                           'sp_beg': sp_beg, 'sp_end': sp_end, 'span': span}
  # Ordered by a hash of the accession: deterministic, so the scan and the
  # dataset are reproducible, but spread across the PDB rather than walking
  # alphabetically through a run of near-identical TrEMBL entries.
  order = sorted(best, key=lambda a: hashlib.md5(a.encode()).hexdigest())
  return [best[a] for a in order]


def afdb_entry(accession: str) -> dict | None:
  """AFDB metadata for one accession, cached; None when there is no model.

  Accessions with no AFDB model answer 400, which is an expected outcome rather
  than a transient error, so this issues a single request without curl's
  --fail/retry behaviour; retrying those would dominate the scan.
  """
  path = os.path.join(CACHE, 'afdb', f'{accession}.json')
  os.makedirs(os.path.dirname(path), exist_ok=True)
  if os.path.exists(path):
    payload = open(path).read()
  else:
    proc = subprocess.run(
        ['curl', '-sSL', '--max-time', '30', f'{AFDB_API}/{accession}'],
        capture_output=True)
    payload = proc.stdout.decode('utf8', 'replace') if proc.returncode == 0 else ''
    if not payload.lstrip().startswith('['):
      payload = ''
    with open(path, 'w') as f:
      f.write(payload)
  if not payload.strip():
    return None
  try:
    records = json.loads(payload)
  except json.JSONDecodeError:
    return None
  return records[0] if records else None


def afdb_model(entry: dict) -> str | None:
  """Downloads the AFDB PDB model for an entry and returns its path."""
  url = entry.get('pdbUrl')
  if not url:
    return None
  path = os.path.join(CACHE, 'models', f'{entry["modelEntityId"]}.pdb')
  os.makedirs(os.path.dirname(path), exist_ok=True)
  try:
    base.fetch(url, path)
  except RuntimeError:
    return None
  return path


def experimental_pdb_text(pdb_id: str) -> str | None:
  """Downloads a PDB entry in PDB format, transparently ungzipping it."""
  path = os.path.join(base.CACHE, 'rcsb', f'{pdb_id}.pdb')
  os.makedirs(os.path.dirname(path), exist_ok=True)
  try:
    payload = base.fetch(f'https://files.rcsb.org/download/{pdb_id}.pdb',
                         path, retries=1, timeout=120)
  except RuntimeError:
    return None
  # RCSB sometimes serves the gzipped file for this URL.
  if payload[:2] == b'\x1f\x8b':
    try:
      payload = gzip.decompress(payload)
    except (OSError, EOFError):
      return None
  return payload.decode('utf8', 'replace')


def experimental_chain(pdb_id: str, chain_id: str):
  """Returns the residues of one chain of a PDB entry, or None."""
  text = experimental_pdb_text(pdb_id)
  if text is None:
    return None
  return base.read_pdb_chains(text).get(chain_id)


def evaluate(candidate: dict) -> dict | None:
  """Scores one candidate chain; returns a record when it clears both cuts."""
  entry = afdb_entry(candidate['accession'])
  if entry is None:
    return None
  if entry.get('globalMetricValue', 100.0) > PREFILTER_PLDDT:
    return None
  uniprot_seq = entry.get('sequence', '')
  beg, end = candidate['sp_beg'], candidate['sp_end']
  if end > len(uniprot_seq):
    return None
  span_seq = uniprot_seq[beg - 1:end]
  if not MIN_LEN <= len(span_seq) <= MAX_LEN:
    return None

  residues = experimental_chain(candidate['pdb'], candidate['chain'])
  if not residues:
    return None
  mapped, how = base.renumber_to_target(residues, span_seq)
  if mapped is None:
    return None
  observed = sorted({resnum for resnum, _ in mapped})
  if len(observed) < MIN_OBSERVED_FRACTION * len(span_seq):
    return None

  model_path = afdb_model(entry)
  if model_path is None:
    return None
  # Trim the AFDB model to the same span and renumber it from 1 so that model
  # and native share a residue numbering.
  model_lines = []
  for line in base.atom_lines(open(model_path).read(), keep_hetatm_mse=False):
    resnum = int(line[22:26])
    if not beg <= resnum <= end:
      continue
    model_lines.append(f'{line[:16]} {line[17:21]}A{resnum - beg + 1:4d} {line[27:]}')
  if not model_lines:
    return None

  paths = {
      'native': os.path.join(CACHE, 'work', f'{candidate["accession"]}_native.pdb'),
      'model': os.path.join(CACHE, 'work', f'{candidate["accession"]}_model.pdb'),
  }
  os.makedirs(os.path.dirname(paths['native']), exist_ok=True)
  with open(paths['native'], 'w') as f:
    f.write(base.write_native_from_chain(mapped))
  with open(paths['model'], 'w') as f:
    f.write('\n'.join(model_lines) + '\nTER\nEND\n')

  _, native_residues = base.load_atoms(paths['native'])
  plddt = base.mean_plddt(paths['model'], native_residues)
  score_lddt = base.lddt(paths['model'], paths['native'])
  if plddt is None or score_lddt is None:
    return None
  if plddt >= MAX_PLDDT or score_lddt >= MAX_LDDT:
    return None
  score_tm = base.tm_score(paths['model'], paths['native'], base.ensure_usalign())
  if score_tm is None:
    return None

  return {
      'id': f'{candidate["pdb"].upper()}_{candidate["chain"]}',
      'sequence': span_seq,
      'length': len(span_seq),
      'plddt_af2': round(plddt, 2),
      'lddt_af2': round(score_lddt, 3),
      'tmscore_af2': round(score_tm, 3),
      'n_observed_res': len(observed),
      'uniprot': candidate['accession'],
      'uniprot_range': f'{beg}-{end}',
      'pdb_id': candidate['pdb'].upper(),
      'pdb_chain': candidate['chain'],
      'afdb_model': entry['modelEntityId'],
      'afdb_version': entry.get('latestVersion', ''),
      'afdb_global_plddt': entry.get('globalMetricValue', ''),
      'mapping': how,
      'native_path': paths['native'],
      'model_path': paths['model'],
  }


def stage_candidates() -> list[dict]:
  chains = sifts_chains()
  log(f'{len(chains)} UniProt accessions with a PDB chain of '
      f'{MIN_LEN}-{MAX_LEN} residues')
  # AFDB metadata is cheap (hundreds of lookups a second) while a structural
  # evaluation costs two downloads, so screen a large pool on metadata first
  # and only pay for the accessions that are plausibly low-confidence.
  pool_chains = chains[:METADATA_POOL]
  log(f'fetching AFDB metadata for {len(pool_chains)} accessions ...')
  with cf.ThreadPoolExecutor(max_workers=32) as pool:
    list(pool.map(lambda c: afdb_entry(c['accession']), pool_chains))

  shortlist = []
  for candidate in pool_chains:
    entry = afdb_entry(candidate['accession'])
    if entry is None:
      continue
    if entry.get('globalMetricValue', 100.0) <= PREFILTER_PLDDT:
      shortlist.append(candidate)
  log(f'{len(shortlist)} have whole-chain pLDDT <= {PREFILTER_PLDDT:.0f}; '
      f'evaluating in hash order for diversity')

  found, inspected = [], 0
  block = 48
  for start in range(0, min(len(shortlist), MAX_ACCESSIONS), block):
    if len(found) >= TARGET_SIZE:
      break
    window = shortlist[start:start + block]
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
      list(pool.map(lambda c: experimental_pdb_text(c['pdb']), window))
      list(pool.map(lambda c: afdb_model(afdb_entry(c['accession'])), window))
    for candidate in window:
      if len(found) >= TARGET_SIZE:
        break
      inspected += 1
      try:
        record = evaluate(candidate)
      except (RuntimeError, ValueError, KeyError, OSError) as e:
        log(f'  {candidate["accession"]}: skipped ({type(e).__name__}: {e})')
        continue
      if record is None:
        continue
      found.append(record)
      log(f'[{len(found):2d}/{TARGET_SIZE}] {record["id"]} ({record["uniprot"]}) '
          f'L={record["length"]} pLDDT={record["plddt_af2"]:.1f} '
          f'lDDT={record["lddt_af2"]:.2f} TM={record["tmscore_af2"]:.2f}')
      with open(os.path.join(CACHE, 'candidates.json'), 'w') as f:
        json.dump(found, f, indent=1)
    log(f'  ... {inspected} evaluated, {len(found)} kept')
  log(f'evaluated {inspected} accessions, kept {len(found)}')
  with open(os.path.join(CACHE, 'candidates.json'), 'w') as f:
    json.dump(found, f, indent=1)
  with open(os.path.join(CACHE, 'scan_summary.json'), 'w') as f:
    json.dump({'inspected': inspected, 'kept': len(found)}, f)
  return found


def load_candidates() -> list[dict]:
  with open(os.path.join(CACHE, 'candidates.json')) as f:
    return json.load(f)


# --------------------------------------------------------------------------
# Stage 2: MSAs
# --------------------------------------------------------------------------


def stage_msa() -> dict:
  candidates = load_candidates()
  stats_path = os.path.join(CACHE, 'msa_stats.json')
  stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

  todo = [c for c in candidates
          if c['id'] not in stats
          or not os.path.exists(os.path.join(MSA_DIR, f'{c["id"]}.a3m'))]
  failures = {}
  jobs = {}
  for cand in todo:
    if os.path.exists(base.msa_cache_path(cand['id'])):
      continue
    try:
      jobs[cand['id']] = base.submit_msa_job(cand['id'], cand['sequence'])
    except RuntimeError as e:
      failures[cand['id']] = str(e)
  if jobs:
    log(f'queued {len(jobs)} MMseqs2 job(s)')
    failures.update(base.collect_msa_jobs(jobs))

  for cand in todo:
    target = cand['id']
    if target in failures or not os.path.exists(base.msa_cache_path(target)):
      failures.setdefault(target, 'no MMseqs2 result')
      continue
    parts = base.read_msa_parts(target)
    for filename, text in parts.items():
      with gzip.open(os.path.join(MSA_RAW_DIR, f'{target}.{filename}.gz'), 'wt') as f:
        f.write(text)
    merged, seqs = base.merge_a3m(target, cand['sequence'], parts)
    with open(os.path.join(MSA_DIR, f'{target}.a3m'), 'w') as f:
      f.write(merged)
    neff = base.compute_neff(seqs)
    stats[target] = {
        'msa_depth': len(seqs),
        'msa_neff': round(neff, 1),
        'neff_per_res': round(neff / cand['length'], 3),
    }
    log(f'{target}: depth={len(seqs)} neff={neff:.1f}')
    with open(stats_path, 'w') as f:
      json.dump(stats, f, indent=1)

  for target, reason in sorted(failures.items()):
    log(f'{target}: MSA UNAVAILABLE ({reason}). Rerun --stage msa to retry.')
  return stats


# --------------------------------------------------------------------------
# Stage 3: assemble the dataset
# --------------------------------------------------------------------------


COLUMNS = [
    'id', 'sequence', 'length', 'plddt_af2', 'lddt_af2', 'tmscore_af2',
    'n_observed_res', 'msa_depth', 'msa_neff', 'neff_per_res',
    'uniprot', 'uniprot_range', 'pdb_id', 'pdb_chain', 'afdb_model',
    'afdb_version', 'afdb_global_plddt', 'mapping',
]


def stage_csv() -> list[dict]:
  candidates = load_candidates()
  with open(os.path.join(CACHE, 'msa_stats.json')) as f:
    msa_stats = json.load(f)

  rows, dropped = [], []
  for cand in candidates:
    target = cand['id']
    stats = msa_stats.get(target)
    if stats is None:
      dropped.append((target, 'no MSA'))
      continue
    row = {key: cand[key] for key in COLUMNS if key in cand}
    row.update(stats)
    rows.append(row)
    for kind, directory in (('native', PDB_DIR), ('model', MODEL_DIR)):
      with open(cand[f'{kind}_path']) as src, \
           open(os.path.join(directory, f'{target}.pdb'), 'w') as dst:
        dst.write(src.read())

  rows.sort(key=lambda r: r['plddt_af2'])
  with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
  log(f'wrote {CSV_PATH} with {len(rows)} proteins')
  for target, reason in dropped:
    log(f'  dropped {target}: {reason}')

  if not MIN_SET_SIZE <= len(rows) <= MAX_SET_SIZE:
    raise SystemExit(f'dataset has {len(rows)} proteins, outside '
                     f'[{MIN_SET_SIZE},{MAX_SET_SIZE}]')

  keep = {row['id'] for row in rows}
  for directory, suffix in ((PDB_DIR, '.pdb'), (MSA_DIR, '.a3m'), (MODEL_DIR, '.pdb')):
    for name in os.listdir(directory):
      if name.endswith(suffix) and name[:-len(suffix)] not in keep:
        os.remove(os.path.join(directory, name))
  for name in os.listdir(MSA_RAW_DIR):
    if name.split('.')[0] not in keep:
      os.remove(os.path.join(MSA_RAW_DIR, name))

  write_readme(rows)
  return rows


README_TEMPLATE = """# `af2_lowconf`: chains AlphaFold 2 gets wrong and knows it

{n} single chains, {min_len}-{max_len} residues, every one with
**mean pLDDT < {max_plddt:.0f}** and **lDDT < {max_lddt}** against its
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
   UniProt accession, keeping spans of {min_len_bound}-{max_len_bound} residues.
2. AFDB's API supplies each accession's model; those whose whole-chain mean
   pLDDT is above {prefilter:.0f} are skipped as a cheap prefilter.
3. The model is trimmed to the chain's UniProt span, the experimental chain is
   renumbered onto the same span, and pLDDT, lDDT and TM-score are recomputed
   over the residues the two share. At least {observed:.0%} of the span must be
   resolved in the crystal.
4. Chains clearing pLDDT < {max_plddt:.0f} and lDDT < {max_lddt} are kept, one
   per UniProt accession. {inspected}

## Layout

| Path | Contents |
| --- | --- |
| `pdbs/<id>.pdb` | experimental structure, one chain, numbered 1..length over the span |
| `msa/<id>.a3m` | merged ColabFold MMseqs2 MSA, query first |
| `msa/raw/<id>.*.a3m.gz` | the UniRef and environmental parts as returned |
| `af2_models/<id>.pdb` | the AFDB model trimmed to the same span, pLDDT in B-factors |
| `../af2_lowconf.csv` | one row per chain |

`id` is `<PDB entry>_<chain>`. About {footprint} in total.

## Contents

{table}

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
  and `msa_neff` are recorded so you can filter as needed; {shallow}
- The span is the region resolved in one crystal form. A chain whose fold
  depends on a partner will look like a prediction failure in isolation, which
  is a real AlphaFold limitation but not the same failure mode as a wrong fold.
"""


def write_readme(rows) -> None:
  lengths = [r['length'] for r in rows]
  header = ('| id | UniProt | len | MSA depth | Neff | pLDDT | lDDT | TM |\n'
            '| --- | --- | --- | --- | --- | --- | --- | --- |\n')
  table = header + '\n'.join(
      f"| `{r['id']}` | {r['uniprot']} | {r['length']} | {r['msa_depth']} | "
      f"{r['msa_neff']:.0f} | {r['plddt_af2']:.1f} | {r['lddt_af2']:.2f} | "
      f"{r['tmscore_af2']:.2f} |" for r in rows)

  n_shallow = sum(1 for r in rows if r['msa_neff'] < 100)
  shallow = (f'{n_shallow} of {len(rows)} have Neff below 100.' if n_shallow
             else 'as it turns out, all of them have Neff of 100 or more.')

  megabytes = sum(
      os.path.getsize(os.path.join(root, name))
      for directory in (PDB_DIR, MSA_DIR, MODEL_DIR)
      for root, _, names in os.walk(directory)
      for name in names) / 1e6

  inspected_note = ''
  scan_log = os.path.join(CACHE, 'scan_summary.json')
  if os.path.exists(scan_log):
    with open(scan_log) as f:
      inspected_note = (f'The scan inspected {json.load(f)["inspected"]} '
                        f'accessions to find them.')

  text = README_TEMPLATE.format(
      n=len(rows), min_len=min(lengths), max_len=max(lengths),
      max_plddt=MAX_PLDDT, max_lddt=MAX_LDDT, min_len_bound=MIN_LEN,
      max_len_bound=MAX_LEN, prefilter=PREFILTER_PLDDT,
      observed=MIN_OBSERVED_FRACTION, inspected=inspected_note, table=table,
      shallow=shallow, footprint=f'{megabytes:.0f} MB')
  with open(os.path.join(DATA, 'README.md'), 'w') as f:
    f.write(text)
  log(f'wrote {os.path.join(DATA, "README.md")}')


# --------------------------------------------------------------------------
# Stage 4: verify
# --------------------------------------------------------------------------


def stage_verify() -> None:
  with open(CSV_PATH, newline='') as f:
    rows = list(csv.DictReader(f))
  problems = []

  def check(condition, message):
    if not condition:
      problems.append(message)

  check(MIN_SET_SIZE <= len(rows) <= MAX_SET_SIZE,
        f'{len(rows)} proteins, expected between {MIN_SET_SIZE} and {MAX_SET_SIZE}')

  for row in rows:
    target = row['id']
    native_path = os.path.join(PDB_DIR, f'{target}.pdb')
    model_path = os.path.join(MODEL_DIR, f'{target}.pdb')
    msa_path = os.path.join(MSA_DIR, f'{target}.a3m')
    for path in (native_path, model_path, msa_path):
      check(os.path.exists(path) and os.path.getsize(path) > 0,
            f'{target}: missing or empty {os.path.relpath(path, REPO)}')
    if not all(os.path.exists(p) for p in (native_path, model_path, msa_path)):
      continue

    sequence, length = row['sequence'], int(row['length'])
    check(len(sequence) == length, f'{target}: length disagrees with sequence')
    check(float(row['plddt_af2']) < MAX_PLDDT,
          f'{target}: pLDDT {row["plddt_af2"]} does not meet the < {MAX_PLDDT} cut')
    check(float(row['lddt_af2']) < MAX_LDDT,
          f'{target}: lDDT {row["lddt_af2"]} does not meet the < {MAX_LDDT} cut')

    _, seqs = base.parse_a3m(open(msa_path).read())
    check(seqs and seqs[0] == sequence,
          f'{target}: first a3m row is not the query sequence')
    allowed = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ-') | set('abcdefghijklmnopqrstuvwxyz')
    stray = {c for s in seqs for c in s} - allowed
    check(not stray, f'{target}: a3m has characters AlphaFold cannot encode: '
                     f'{sorted(repr(c) for c in stray)}')
    check(int(row['msa_depth']) == len(seqs),
          f'{target}: msa_depth disagrees with the a3m')

    _, native_residues = base.load_atoms(native_path)
    check(native_residues and min(native_residues) >= 1
          and max(native_residues) <= length,
          f'{target}: native residue numbers fall outside 1..{length}')

    # Recompute both metrics from the shipped files, not the cached workspace.
    recomputed_plddt = base.mean_plddt(model_path, native_residues)
    recomputed_lddt = base.lddt(model_path, native_path)
    check(recomputed_plddt is not None
          and abs(recomputed_plddt - float(row['plddt_af2'])) < 0.05,
          f'{target}: pLDDT does not reproduce from the shipped files')
    check(recomputed_lddt is not None
          and abs(recomputed_lddt - float(row['lddt_af2'])) < 0.005,
          f'{target}: lDDT does not reproduce from the shipped files')

  ids = [row['id'] for row in rows]
  check(len(set(ids)) == len(ids), 'duplicate ids in the CSV')
  accessions = [row['uniprot'] for row in rows]
  check(len(set(accessions)) == len(accessions),
        'the same UniProt accession appears twice')

  if REPO not in sys.path:
    sys.path.insert(0, REPO)
  try:
    from alphafold.data import parsers as af_parsers
    for row in rows:
      msa = af_parsers.parse_a3m(
          open(os.path.join(MSA_DIR, f'{row["id"]}.a3m')).read())
      check(len(msa.sequences) == int(row['msa_depth']),
            f'{row["id"]}: alphafold parser disagrees on MSA depth')
    log(f'all {len(rows)} MSAs parse with alphafold.data.parsers.parse_a3m')
  except ImportError:
    log('alphafold package not importable here; skipped the parser check')

  if problems:
    for problem in problems:
      log(f'FAIL {problem}')
    raise SystemExit(f'{len(problems)} verification failure(s)')
  log(f'verification passed: {len(rows)} proteins')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--stage', default='all',
                      choices=['all', 'candidates', 'msa', 'csv', 'verify'])
  args = parser.parse_args()
  base.ensure_dirs()
  ensure_dirs()
  if args.stage in ('all', 'candidates'):
    stage_candidates()
  if args.stage in ('all', 'msa'):
    stage_msa()
  if args.stage in ('all', 'csv'):
    stage_csv()
  if args.stage in ('all', 'verify'):
    stage_verify()


if __name__ == '__main__':
  main()
