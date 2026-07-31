#!/usr/bin/env python3
"""Builds a benchmark of single-chain proteins where stock AlphaFold 2 fails
despite having a deep MSA.

The set is drawn from CASP15, where group 270 (``NBIS-AF2-standard``, a standard
AlphaFold v2.2 protocol run by the Elofsson lab) participated as an official
server.  That gives us published GDT_TS / LDDT / TM-score per target, and the
submitted coordinates carry pLDDT in the B-factor column.  CASP15 targets were
released to the PDB after the 2021-09-30 training cutoff of the released
AlphaFold v2.3 parameters, so they are blind targets.

Nothing here runs AlphaFold; the script only assembles published predictions,
experimental structures and MSAs fetched from the ColabFold MMseqs2 API.

Usage:
  python3 scripts/build_af2_fails_dataset.py --stage all
  python3 scripts/build_af2_fails_dataset.py --stage candidates
  python3 scripts/build_af2_fails_dataset.py --stage msa
  python3 scripts/build_af2_fails_dataset.py --stage natives
  python3 scripts/build_af2_fails_dataset.py --stage models
  python3 scripts/build_af2_fails_dataset.py --stage csv
"""

import argparse
import csv
import gzip
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.parse
import zipfile
from collections.abc import Mapping, Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data')
CACHE = os.path.join(DATA, '.cache')
PDB_DIR = os.path.join(DATA, 'pdbs')
MSA_DIR = os.path.join(DATA, 'msa')
MSA_RAW_DIR = os.path.join(MSA_DIR, 'raw')
MODEL_DIR = os.path.join(DATA, 'af2_models')
BIN_DIR = os.path.join(REPO, 'scripts', 'bin')
CSV_PATH = os.path.join(REPO, 'af2_fails.csv')

PC = 'https://predictioncenter.org/download_area/CASP15'
COLABFOLD = 'https://api.colabfold.com'
RCSB_SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query'
USALIGN_SRC = 'https://zhanggroup.org/US-align/bin/module/USalign.cpp'

# CASP15 group number of the standard AlphaFold2 baseline server.
AF2_GROUP = '270'

# Selection thresholds (see data/README.md).
TM_THRESHOLD = 0.80
TM_THRESHOLD_RELAXED = 0.85
MIN_LEN = 60
MAX_LEN = 800
MIN_NEFF = 100.0
MIN_NEFF_RELAXED = 50.0
MIN_DEPTH = 300
MIN_SET_SIZE = 10
MAX_SET_SIZE = 50
# How long to wait for the MMseqs2 queue. The public API is shared and its
# queue regularly runs ~100 deep, and an individual job occasionally wedges;
# lower this to give up early and retry the stragglers on a later run.
MSA_JOB_TIMEOUT = int(os.environ.get('AF2_FAILS_MSA_TIMEOUT', 3 * 3600))
# Pause between MMseqs2 submissions; the API answers 429 to rapid bursts.
SUBMIT_INTERVAL = 5.0
# Largest tolerated disagreement between our recomputed scores and the official
# CASP ones before we conclude the native we rebuilt is the wrong structure.
MAX_SCORE_GAP = 0.12

# Targets that DeepMind hand-tuned for CASP15; see the manual_interventions.md
# inside docs/casp15_predictions.zip.  Their DeepMind baseline numbers are not
# the output of an automated pipeline.
DM_MANUAL = {'T1123', 'T1125', 'T1130', 'T1131', 'T1169'}

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}


def log(msg):
  print(msg, flush=True)


def ensure_dirs():
  for d in (DATA, CACHE, PDB_DIR, MSA_DIR, MSA_RAW_DIR, MODEL_DIR, BIN_DIR):
    os.makedirs(d, exist_ok=True)


def _curl(args: Sequence[str], timeout: int, retries: int,
          allow_empty: bool = False) -> bytes:
  """Runs curl, retrying transient failures.

  curl rather than urllib because the CA bundle shipped with some conda Python
  builds rejects predictioncenter.org and zhanggroup.org, while the system
  trust store accepts both.
  """
  last = ''
  for attempt in range(retries):
    proc = subprocess.run(
        ['curl', '-sSL', '--fail', '--max-time', str(timeout), *args],
        capture_output=True)
    if proc.returncode == 0 and (proc.stdout or allow_empty):
      return proc.stdout
    last = proc.stderr.decode('utf8', 'replace').strip() or f'exit {proc.returncode}'
    # A 429 means we are being throttled, so back off far harder than for a
    # transient network error.
    backoff = 30 * (attempt + 1) if 'error: 429' in last else 3 * (attempt + 1)
    time.sleep(backoff)
  raise RuntimeError(f'curl failed ({last}) for: {" ".join(args)}')


def fetch(url: str, dest: str | None = None, retries: int = 4,
          timeout: int = 600) -> bytes:
  """Downloads a URL, caching by destination path when one is given."""
  if dest is not None and os.path.exists(dest) and os.path.getsize(dest) > 0:
    with open(dest, 'rb') as f:
      return f.read()
  payload = _curl([url], timeout, retries)
  if dest is not None:
    with open(dest, 'wb') as f:
      f.write(payload)
  return payload


# --------------------------------------------------------------------------
# Stage 1: candidate selection from the official CASP15 result tables
# --------------------------------------------------------------------------


def parse_casp15_tables():
  """Returns {assessment_unit: {column: value}} for the AF2 baseline model 1."""
  tgz = os.path.join(CACHE, 'casp15_regular_tables.tar.gz')
  fetch(f'{PC}/results/tables/regular.tar.gz', tgz)
  rows = {}
  marker = f'TS{AF2_GROUP}_1'
  with tarfile.open(tgz) as tf:
    for member in tf.getmembers():
      if not member.name.endswith('.txt'):
        continue
      unit = os.path.basename(member.name)[:-4]
      header = None
      for raw in tf.extractfile(member).read().decode('utf8', 'replace').splitlines():
        if raw.startswith('#'):
          header = raw[1:].split()
          continue
        parts = raw.split()
        # Column 0 is the rank, column 1 the model name.
        if len(parts) < 5 or marker not in parts[1]:
          continue
        rows[unit] = dict(zip(header, parts[1:]))
  return rows


def parse_casp15_sequences():
  txt = fetch(f'{PC}/sequences/casp15.seq.txt',
              os.path.join(CACHE, 'casp15.seq.txt')).decode()
  seqs, name = {}, None
  for line in txt.splitlines():
    line = line.strip()
    if line.startswith('>'):
      name = line[1:].split()[0]
      seqs[name] = ''
    elif name:
      seqs[name] += line
  return seqs


def as_float(value):
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def stage_candidates():
  rows = parse_casp15_tables()
  seqs = parse_casp15_sequences()
  targets = sorted({unit.split('-D')[0] for unit in rows})

  candidates, rejected = [], []
  for target in targets:
    # Multi-domain targets have a whole-chain row; single-domain targets are
    # scored only as "-D1", which is the whole chain.
    unit = target if target in rows else f'{target}-D1'
    if unit not in rows:
      continue
    row = rows[unit]
    tm, lddt, gdt = (as_float(row.get('TMscore')), as_float(row.get('LDDT')),
                     as_float(row.get('GDT_TS')))
    seq = seqs.get(target, '')
    if tm is None or lddt is None or not seq:
      rejected.append((target, 'missing score or sequence'))
      continue
    entry = {
        'id': target,
        'unit': unit,
        'sequence': seq,
        'length': len(seq),
        'n_eval_res': int(row['NP']),
        'gdt_ts_af2': gdt,
        'lddt_af2': lddt,
        'tmscore_af2': tm,
        'n_domains': len([u for u in rows if u.startswith(f'{target}-D')]),
    }
    if not MIN_LEN <= len(seq) <= MAX_LEN:
      rejected.append((target, f'length {len(seq)} outside [{MIN_LEN},{MAX_LEN}]'))
      continue
    if tm >= TM_THRESHOLD_RELAXED:
      rejected.append((target, f'TM {tm:.2f} >= {TM_THRESHOLD_RELAXED}'))
      continue
    entry['tier'] = 'primary' if tm < TM_THRESHOLD else 'relaxed'
    candidates.append(entry)

  candidates.sort(key=lambda c: c['tmscore_af2'])
  out = os.path.join(CACHE, 'candidates.json')
  with open(out, 'w') as f:
    json.dump({'candidates': candidates, 'rejected': rejected}, f, indent=1)
  n_primary = sum(1 for c in candidates if c['tier'] == 'primary')
  log(f'{len(candidates)} candidates ({n_primary} at TM<{TM_THRESHOLD}, '
      f'{len(candidates) - n_primary} held in reserve at TM<{TM_THRESHOLD_RELAXED})')
  for c in candidates:
    log(f"  {c['id']:10s} tier={c['tier']:8s} L={c['length']:4d} "
        f"TM={c['tmscore_af2']:.2f} LDDT={c['lddt_af2']:.2f} GDT={c['gdt_ts_af2']:.2f}")
  return candidates


def load_candidates():
  with open(os.path.join(CACHE, 'candidates.json')) as f:
    return json.load(f)['candidates']


# --------------------------------------------------------------------------
# Stage 2: MSAs from the ColabFold MMseqs2 API
# --------------------------------------------------------------------------


def post_form(url: str, fields: Mapping[str, str]) -> bytes:
  """POSTs a urlencoded form; the MMseqs2 API rejects multipart bodies."""
  args = []
  for key, value in fields.items():
    args += ['--data-urlencode', f'{key}={value}']
  return _curl([*args, url], timeout=600, retries=4)


def msa_cache_path(target: str) -> str:
  return os.path.join(CACHE, 'msa', f'{target}.tar.gz')


def submit_msa_job(target: str, sequence: str) -> str:
  """Queues one MMseqs2 job and returns its ticket id.

  The API keys jobs by sequence, so resubmitting an already-queued query
  returns the same ticket rather than taking a new place in the queue.
  """
  ticket = json.loads(post_form(f'{COLABFOLD}/ticket/msa',
                                {'q': f'>{target}\n{sequence}', 'mode': 'env'}))
  if ticket.get('status') == 'ERROR':
    raise RuntimeError(f'{target}: MMseqs2 API rejected the query: {ticket}')
  # The API rate-limits bursts of submissions with 429s, so pace them.
  time.sleep(SUBMIT_INTERVAL)
  return ticket['id']


def collect_msa_jobs(jobs: Mapping[str, str]) -> dict[str, str]:
  """Waits on queued MMseqs2 jobs, downloading each as it finishes.

  All jobs are polled round-robin so that a single slow query does not hold up
  the ones behind it.  Returns {target: failure reason} for jobs that did not
  finish; everything else has its tarball cached on disk.
  """
  pending = dict(jobs)
  failures = {}
  deadline = time.monotonic() + MSA_JOB_TIMEOUT
  while pending and time.monotonic() < deadline:
    for target, job in list(pending.items()):
      try:
        status = json.loads(fetch(f'{COLABFOLD}/ticket/{job}'))['status']
      except RuntimeError as e:
        failures[target] = f'status check failed: {e}'
        del pending[target]
        continue
      if status == 'COMPLETE':
        fetch(f'{COLABFOLD}/result/download/{job}', msa_cache_path(target))
        del pending[target]
        log(f'  {target}: MSA ready ({len(pending)} still queued)')
      elif status in ('ERROR', 'UNKNOWN'):
        failures[target] = f'job ended as {status}'
        del pending[target]
    if pending:
      time.sleep(15)
  for target in pending:
    failures[target] = 'timed out waiting for the MMseqs2 queue'
  return failures


def read_msa_parts(target: str) -> dict[str, str]:
  """Returns {filename: a3m_text} from a downloaded MMseqs2 result tarball.

  The API terminates the last sequence of each part with a NUL byte.  It has to
  go: parsers.parse_a3m passes it through happily, but it then reaches
  pipeline.make_msa_features and raises KeyError('\\x00').
  """
  parts = {}
  with tarfile.open(msa_cache_path(target)) as tf:
    for member in tf.getmembers():
      if member.name.endswith('.a3m'):
        text = tf.extractfile(member).read().decode().replace('\x00', '')
        parts[os.path.basename(member.name)] = text
  return parts


def parse_a3m(text):
  names, seqs, name = [], [], None
  cur = []
  for line in text.splitlines():
    if line.startswith('>'):
      if name is not None:
        names.append(name)
        seqs.append(''.join(cur))
      name, cur = line[1:], []
    elif line.strip():
      cur.append(line.strip())
  if name is not None:
    names.append(name)
    seqs.append(''.join(cur))
  return names, seqs


def merge_a3m(target: str, sequence: str,
              parts: Mapping[str, str]) -> tuple[str, list[str]]:
  """Merges the API's a3m parts into one, query first, duplicates removed.

  Every part repeats the ungapped query as its first row, so seeding ``seen``
  with the query drops those copies.
  """
  merged_names, merged_seqs = [target], [sequence]
  seen = {sequence}
  # Sorted so that re-runs produce a byte-identical file.
  for filename in sorted(parts):
    names, seqs = parse_a3m(parts[filename])
    for name, seq in zip(names, seqs):
      if seq in seen:
        continue
      seen.add(seq)
      merged_names.append(name)
      merged_seqs.append(seq)
  lines = []
  for name, seq in zip(merged_names, merged_seqs):
    lines.append(f'>{name}')
    lines.append(seq)
  return '\n'.join(lines) + '\n', merged_seqs


def compute_neff(seqs: Sequence[str], identity: float = 0.62,
                 max_seqs: int = 4000) -> float:
  """Number of effective sequences: sum of 1/(cluster size) at 62% identity.

  Sequence identity is counted over the columns where at least one of the two
  sequences is not a gap.  The pairwise counts come from one matmul per amino
  acid symbol, which keeps an N^2*L comparison out of memory and hands the work
  to BLAS.
  """
  if len(seqs) <= 1:
    return float(len(seqs))
  # Strip lowercase insertion columns so every row has the query's length.
  rows = [re.sub('[a-z]', '', s) for s in seqs]
  width = len(rows[0])
  rows = [s for s in rows if len(s) == width]
  if width == 0:
    return float(len(rows))
  if len(rows) > max_seqs:
    step = len(rows) / max_seqs
    sampled = [rows[int(i * step)] for i in range(max_seqs)]
    scale = len(rows) / len(sampled)
  else:
    sampled, scale = rows, 1.0

  arr = np.frombuffer(''.join(sampled).encode('ascii'),
                      dtype=np.uint8).reshape(len(sampled), width)
  gap = ord('-')
  matches = np.zeros((len(sampled), len(sampled)), dtype=np.float32)
  for symbol in np.unique(arr):
    indicator = (arr == symbol).astype(np.float32)
    matches += indicator @ indicator.T
  gaps = (arr == gap).astype(np.float32)
  both_gap = gaps @ gaps.T
  valid = width - both_gap
  frac = np.where(valid > 0, (matches - both_gap) / np.maximum(valid, 1.0), 0.0)
  neighbours = (frac >= identity).sum(axis=1)
  return float((1.0 / np.maximum(neighbours, 1)).sum()) * scale


def stage_msa():
  candidates = load_candidates()
  stats_path = os.path.join(CACHE, 'msa_stats.json')
  stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

  todo = [c for c in candidates
          if c['id'] not in stats
          or not os.path.exists(os.path.join(MSA_DIR, f'{c["id"]}.a3m'))]
  failures = {}

  # Queue everything first, then collect. Submitting one at a time and waiting
  # for each result in turn would put every query behind the slowest one.
  jobs = {}
  for cand in todo:
    target = cand['id']
    if os.path.exists(msa_cache_path(target)):
      continue
    try:
      jobs[target] = submit_msa_job(target, cand['sequence'])
    except RuntimeError as e:
      failures[target] = str(e)
  if jobs:
    log(f'queued {len(jobs)} MMseqs2 job(s): {", ".join(sorted(jobs))}')
    failures.update(collect_msa_jobs(jobs))

  for cand in todo:
    target = cand['id']
    if target in failures or not os.path.exists(msa_cache_path(target)):
      failures.setdefault(target, 'no MMseqs2 result')
      continue
    parts = read_msa_parts(target)
    # The merged a3m is the deliverable; the parts are kept only for
    # provenance, so they go in compressed.
    for filename, text in parts.items():
      with gzip.open(os.path.join(MSA_RAW_DIR, f'{target}.{filename}.gz'), 'wt') as f:
        f.write(text)
    merged, seqs = merge_a3m(target, cand['sequence'], parts)
    with open(os.path.join(MSA_DIR, f'{target}.a3m'), 'w') as f:
      f.write(merged)
    neff = compute_neff(seqs)
    stats[target] = {
        'msa_depth': len(seqs),
        'msa_neff': round(neff, 1),
        'neff_per_res': round(neff / cand['length'], 3),
    }
    log(f'{target} (L={cand["length"]}): depth={len(seqs)} neff={neff:.1f} '
        f'neff/L={neff / cand["length"]:.2f}')
    with open(stats_path, 'w') as f:
      json.dump(stats, f, indent=1)

  for target, reason in sorted(failures.items()):
    log(f'{target}: MSA UNAVAILABLE ({reason}). Rerun --stage msa to retry.')
  return stats


# --------------------------------------------------------------------------
# Stage 3: ground-truth native structures
# --------------------------------------------------------------------------


def casp_domain_natives():
  tgz = os.path.join(CACHE, 'casp15_domain_natives.tar.gz')
  fetch(f'{PC}/targets/casp15.targets.TS-domains.public_12.20.2022.tar.gz', tgz)
  natives = {}
  with tarfile.open(tgz) as tf:
    for member in tf.getmembers():
      if member.name.endswith('.pdb'):
        unit = os.path.basename(member.name)[:-4]
        natives[unit] = tf.extractfile(member).read().decode()
  return natives


def atom_lines(text, keep_hetatm_mse=True):
  """Yields ATOM lines, dropping hydrogens and non-primary altlocs.

  Only the first model of an NMR ensemble is read, matching CASP, which scores
  against model 1.
  """
  for line in text.splitlines():
    if line.startswith('ENDMDL'):
      return
    is_atom = line.startswith('ATOM')
    is_mse = keep_hetatm_mse and line.startswith('HETATM') and line[17:20] == 'MSE'
    if not (is_atom or is_mse):
      continue
    if line[76:78].strip() == 'H' or line[12:16].strip().startswith('H'):
      continue
    if line[16] not in (' ', 'A'):
      continue
    yield line


def merge_domain_natives(target, natives):
  units = sorted(u for u in natives if u.startswith(f'{target}-D'))
  if not units:
    return None
  seen, out = set(), []
  for unit in units:
    for line in atom_lines(natives[unit]):
      key = (int(line[22:26]), line[12:16], line[26])
      if key in seen:
        continue
      seen.add(key)
      # Normalise to a single chain A with a blank altloc.
      out.append(line[:16] + ' ' + line[17:21] + 'A' + line[22:])
  out.sort(key=lambda l: (int(l[22:26]), l[12:16]))
  return '\n'.join(out) + '\nTER\nEND\n'


def rcsb_entry_for(sequence):
  query = {
      'query': {
          'type': 'terminal',
          'service': 'sequence',
          'parameters': {
              'evalue_cutoff': 1e-10,
              'identity_cutoff': 0.95,
              'sequence_type': 'protein',
              'value': sequence,
          },
      },
      'request_options': {
          'paginate': {'start': 0, 'rows': 10},
          'results_verbosity': 'compact',
      },
      'return_type': 'polymer_entity',
  }
  url = f'{RCSB_SEARCH}?{urllib.parse.urlencode({"json": json.dumps(query)})}'
  # RCSB answers 204 with an empty body when nothing matches.
  payload = _curl([url], timeout=120, retries=3, allow_empty=True)
  if not payload.strip():
    return []
  return json.loads(payload).get('result_set', [])


def read_pdb_chains(text):
  """Returns {chain_id: [(resnum, resname, [line, ...]), ...]}."""
  chains = {}
  for line in atom_lines(text):
    chain = line[21]
    resnum = int(line[22:26])
    icode = line[26]
    residues = chains.setdefault(chain, {})
    residues.setdefault((resnum, icode), (line[17:20].strip(), []))[1].append(line)
  ordered = {}
  for chain, residues in chains.items():
    ordered[chain] = [(key, name, lines)
                      for key, (name, lines) in sorted(residues.items())]
  return ordered


def chain_sequence(residues):
  return ''.join(THREE_TO_ONE.get(name, 'X') for _, name, _ in residues)


def align_free_ends(query: str, reference: str, match: int = 2,
                    mismatch: int = -2, gap: int = -1) -> list[tuple[int, int]]:
  """Needleman-Wunsch with free end gaps; returns aligned (query, ref) indices.

  End gaps are free so that expression tags on the deposited construct and
  unobserved terminal regions of the target both come out as unaligned rather
  than dragging the whole alignment out of register.
  """
  n, m = len(query), len(reference)
  if n == 0 or m == 0:
    return []
  q = np.frombuffer(query.encode('ascii'), dtype=np.uint8)
  r = np.frombuffer(reference.encode('ascii'), dtype=np.uint8)
  score = np.zeros((n + 1, m + 1), dtype=np.int32)
  # 0 = diagonal, 1 = gap in reference (consume query), 2 = gap in query.
  trace = np.zeros((n + 1, m + 1), dtype=np.uint8)
  trace[0, 1:] = 2
  trace[1:, 0] = 1
  for i in range(1, n + 1):
    diag = score[i - 1, :-1] + np.where(q[i - 1] == r, match, mismatch)
    up = score[i - 1, 1:] + gap
    # The left-neighbour dependency inside a row forces a scalar sweep.
    row = score[i]
    row_trace = trace[i]
    best_diag_up = np.where(diag >= up, diag, up)
    choice = np.where(diag >= up, 0, 1).astype(np.uint8)
    running = row[0]
    for j in range(1, m + 1):
      left = running + gap
      if left > best_diag_up[j - 1]:
        running, row_trace[j] = left, 2
      else:
        running, row_trace[j] = best_diag_up[j - 1], choice[j - 1]
      row[j] = running

  # Free end gaps: start the traceback from the best cell on the last row/col.
  last_row, last_col = score[n, :], score[:, m]
  if last_row.max() >= last_col.max():
    i, j = n, int(last_row.argmax())
  else:
    i, j = int(last_col.argmax()), m

  pairs = []
  while i > 0 and j > 0:
    step = trace[i, j]
    if step == 0:
      pairs.append((i - 1, j - 1))
      i, j = i - 1, j - 1
    elif step == 1:
      i -= 1
    else:
      j -= 1
  pairs.reverse()
  return pairs


def renumber_to_target(residues, target_seq):
  """Maps an observed chain onto target-sequence numbering.

  Returns ``(mapped, how)`` where ``mapped`` is a list of
  ``(target_resnum, atom_lines)``.  Observed residues that do not align to an
  identical amino acid in the target (tags, cloning artefacts, point
  substitutions) are dropped rather than mis-numbered.
  """
  observed = chain_sequence(residues)
  if not observed:
    return None, 'unmapped'
  # Fast path: the observed stretch appears verbatim in the target sequence.
  if observed in target_seq:
    offset = target_seq.index(observed)
    return [(i + offset + 1, lines) for i, (_, _, lines) in enumerate(residues)], 'exact'

  pairs = align_free_ends(observed, target_seq)
  mapped = [(j + 1, residues[i][2]) for i, j in pairs if observed[i] == target_seq[j]]
  if len(mapped) < 0.5 * len(observed):
    return None, 'unmapped'
  return mapped, 'aligned'


def write_native_from_chain(mapped) -> str:
  """Rewrites a mapped chain as chain A in target numbering.

  The insertion code is cleared along with the numbering, otherwise residues
  that were 100A/100B upstream would collide once renumbered.
  """
  out = []
  for resnum, lines in mapped:
    for line in lines:
      out.append(f'{line[:16]} {line[17:21]}A{resnum:4d} {line[27:]}')
  return '\n'.join(out) + '\nTER\nEND\n'


def stage_natives():
  candidates = load_candidates()
  natives = casp_domain_natives()
  provenance = {}
  for cand in candidates:
    target, seq = cand['id'], cand['sequence']
    dest = os.path.join(PDB_DIR, f'{target}.pdb')
    merged = merge_domain_natives(target, natives)
    if merged is not None:
      with open(dest, 'w') as f:
        f.write(merged)
      units = sorted(u for u in natives if u.startswith(f'{target}-D'))
      provenance[target] = {'native_source': 'casp15_domain_natives',
                            'native_pdb_id': '', 'units': units}
      log(f'{target}: native from CASP domains {units}')
      continue

    # CASP tells us exactly how many residues were assessed, so prefer the
    # deposited chain whose observed length matches that count -- it is the
    # structure the official scores were computed against.
    n_eval = cand['n_eval_res']
    best = None
    for entity in rcsb_entry_for(seq):
      entry_id = entity.split('_')[0]
      try:
        text = fetch(f'https://files.rcsb.org/download/{entry_id}.pdb',
                     os.path.join(CACHE, f'{entry_id}.pdb')).decode('utf8', 'replace')
      except RuntimeError:
        continue
      for chain, residues in read_pdb_chains(text).items():
        if len(residues) < 0.5 * n_eval:
          continue
        mapped, how = renumber_to_target(residues, seq)
        if mapped is None:
          continue
        key = (abs(len(mapped) - n_eval), -len(mapped))
        if best is None or key < best[0]:
          best = (key, entry_id, chain, mapped, how)
    if best is None:
      log(f'{target}: NO NATIVE FOUND')
      provenance[target] = {'native_source': 'missing', 'native_pdb_id': ''}
      continue
    _, entry_id, chain, mapped, how = best
    with open(dest, 'w') as f:
      f.write(write_native_from_chain(mapped))
    provenance[target] = {'native_source': f'rcsb_{how}',
                          'native_pdb_id': f'{entry_id}_{chain}'}
    log(f'{target}: native from {entry_id} chain {chain} '
        f'({len(mapped)}/{n_eval} assessed residues, {how} mapping)')

  with open(os.path.join(CACHE, 'natives.json'), 'w') as f:
    json.dump(provenance, f, indent=1)
  return provenance


# --------------------------------------------------------------------------
# Stage 4: AF2 baseline models and metrics
# --------------------------------------------------------------------------


def stage_models():
  candidates = load_candidates()
  for cand in candidates:
    target = cand['id']
    dest = os.path.join(MODEL_DIR, f'{target}.pdb')
    if os.path.exists(dest):
      continue
    tgz = os.path.join(CACHE, f'preds_{target}.tar.gz')
    fetch(f'{PC}/predictions/regular/{target}.tar.gz', tgz)
    with tarfile.open(tgz) as tf:
      member = next((m for m in tf.getmembers()
                     if m.name.endswith(f'{target}TS{AF2_GROUP}_1')), None)
      if member is None:
        log(f'{target}: no TS{AF2_GROUP}_1 submission')
        continue
      text = tf.extractfile(member).read().decode('utf8', 'replace')
    out = [line[:21] + 'A' + line[22:] for line in atom_lines(text, keep_hetatm_mse=False)]
    with open(dest, 'w') as f:
      f.write('\n'.join(out) + '\nTER\nEND\n')
    log(f'{target}: AF2 baseline model written ({len(out)} atoms)')


def deepmind_baseline_models():
  """Extracts DeepMind's own CASP15 single-chain baseline from docs/."""
  out_dir = os.path.join(CACHE, 'deepmind')
  os.makedirs(out_dir, exist_ok=True)
  zip_path = os.path.join(REPO, 'docs', 'casp15_predictions.zip')
  available = {}
  with zipfile.ZipFile(zip_path) as zf:
    for name in zf.namelist():
      if not name.startswith('casp15_predictions/single_chain/') or not name.endswith('.pdb'):
        continue
      target = os.path.basename(name)[:-4]
      dest = os.path.join(out_dir, f'{target}.pdb')
      if not os.path.exists(dest):
        text = zf.read(name).decode()
        lines = [line[:21] + 'A' + line[22:]
                 for line in atom_lines(text, keep_hetatm_mse=False)]
        with open(dest, 'w') as f:
          f.write('\n'.join(lines) + '\nTER\nEND\n')
      available[target] = dest
  return available


def ensure_usalign():
  binary = os.path.join(BIN_DIR, 'USalign')
  if os.path.exists(binary) and os.access(binary, os.X_OK):
    return binary
  src = os.path.join(CACHE, 'USalign.cpp')
  fetch(USALIGN_SRC, src)
  subprocess.run(['g++', '-O3', '-ffast-math', '-o', binary, src], check=True)
  return binary


def tm_score(model_path, native_path, binary):
  """TM-score of a model against a native, normalised by the native length.

  ``-TMscore 1`` selects the sequence-dependent superposition CASP uses, i.e.
  residues are paired by number rather than structurally re-aligned.
  """
  proc = subprocess.run([binary, '-TMscore', '1', model_path, native_path],
                        capture_output=True, text=True)
  if proc.returncode != 0:
    return None
  for line in proc.stdout.splitlines():
    if line.startswith('TM-score=') and 'Structure_2' in line:
      return float(line.split('=')[1].split()[0])
  return None


def load_atoms(path):
  """Returns {(resnum, atom_name): xyz} and the set of residue numbers."""
  coords, residues = {}, set()
  with open(path) as f:
    for line in atom_lines(f.read()):
      resnum = int(line[22:26])
      name = line[12:16].strip()
      coords[(resnum, name)] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
      residues.add(resnum)
  return coords, residues


def lddt(model_path, native_path, radius=15.0, thresholds=(0.5, 1.0, 2.0, 4.0)):
  """All-atom lDDT of a model against a native, over their shared atoms."""
  model, _ = load_atoms(model_path)
  native, _ = load_atoms(native_path)
  shared = [key for key in native if key in model]
  if len(shared) < 4:
    return None
  shared.sort()
  res_index = np.array([key[0] for key in shared])
  nat = np.array([native[key] for key in shared])
  mod = np.array([model[key] for key in shared])

  d_nat = np.linalg.norm(nat[:, None, :] - nat[None, :, :], axis=-1)
  d_mod = np.linalg.norm(mod[:, None, :] - mod[None, :, :], axis=-1)
  # Only inter-residue pairs inside the inclusion radius are scored.
  mask = (d_nat < radius) & (res_index[:, None] != res_index[None, :])
  if not mask.any():
    return None
  delta = np.abs(d_nat - d_mod)[mask]
  preserved = np.mean([(delta < t).mean() for t in thresholds])
  return float(preserved)


def mean_plddt(model_path, native_residues=None):
  """Mean B-factor (pLDDT) over model residues, optionally restricted."""
  values = {}
  with open(model_path) as f:
    for line in atom_lines(f.read()):
      resnum = int(line[22:26])
      if native_residues is not None and resnum not in native_residues:
        continue
      values[resnum] = float(line[60:66])
  if not values:
    return None
  return float(np.mean(list(values.values())))


# --------------------------------------------------------------------------
# Stage 5: assemble the CSV
# --------------------------------------------------------------------------


COLUMNS = [
    'id', 'sequence', 'length', 'plddt_af2', 'lddt_af2', 'tmscore_af2',
    'gdt_ts_af2', 'n_eval_res', 'msa_depth', 'msa_neff', 'neff_per_res',
    'native_source', 'native_pdb_id', 'lddt_af2_recomputed',
    'tmscore_af2_recomputed', 'plddt_af2_dm', 'lddt_af2_dm', 'tmscore_af2_dm',
    'notes',
]


def stage_csv():
  candidates = load_candidates()
  with open(os.path.join(CACHE, 'msa_stats.json')) as f:
    msa_stats = json.load(f)
  with open(os.path.join(CACHE, 'natives.json')) as f:
    natives = json.load(f)
  binary = ensure_usalign()
  deepmind = deepmind_baseline_models()

  # Decide eligibility from candidates.json and msa_stats.json first. Both
  # survive a rerun, so a candidate's drop reason does not change depending on
  # whether an earlier run already cleaned its structure files away.
  def msa_verdict(cand, min_neff):
    stats = msa_stats.get(cand['id'])
    if stats is None:
      return 'no MSA'
    reasons = []
    if stats['msa_neff'] < min_neff:
      reasons.append(f'Neff {stats["msa_neff"]} below {min_neff:g}')
    if stats['msa_depth'] < MIN_DEPTH:
      reasons.append(f'depth {stats["msa_depth"]} below {MIN_DEPTH}')
    return '; '.join(reasons)

  def eligible(tiers, min_neff):
    return [c for c in candidates
            if c['tier'] in tiers and not msa_verdict(c, min_neff)]

  settings = [({'primary'}, MIN_NEFF, None),
              ({'primary', 'relaxed'}, MIN_NEFF,
               f'TM threshold relaxed {TM_THRESHOLD} -> {TM_THRESHOLD_RELAXED}'),
              ({'primary', 'relaxed'}, MIN_NEFF_RELAXED,
               f'Neff threshold relaxed {MIN_NEFF:g} -> {MIN_NEFF_RELAXED:g}')]
  tiers, min_neff, relaxations = settings[0][0], settings[0][1], []
  for step, (step_tiers, step_neff, note) in enumerate(settings):
    tiers, min_neff = step_tiers, step_neff
    if note:
      relaxations.append(note)
    if len(eligible(tiers, min_neff)) >= MIN_SET_SIZE or step == len(settings) - 1:
      break

  rows, dropped = [], []
  for cand in candidates:
    target = cand['id']
    stats = msa_stats.get(target)
    prov = natives.get(target, {})
    native_path = os.path.join(PDB_DIR, f'{target}.pdb')
    model_path = os.path.join(MODEL_DIR, f'{target}.pdb')

    verdict = msa_verdict(cand, min_neff)
    if verdict:
      dropped.append((target, verdict))
      continue
    if cand['tier'] not in tiers:
      dropped.append((target, f'TM {cand["tmscore_af2"]} held in the reserve tier '
                              f'(>= {TM_THRESHOLD})'))
      continue
    if not os.path.exists(native_path):
      dropped.append((target, 'no native structure'))
      continue
    if not os.path.exists(model_path):
      dropped.append((target, 'no AF2 baseline model'))
      continue

    _, native_residues = load_atoms(native_path)
    plddt = mean_plddt(model_path, native_residues)
    if plddt is None:
      dropped.append((target, 'model and native share no residue numbering'))
      continue
    recomputed_lddt = lddt(model_path, native_path)
    recomputed_tm = tm_score(model_path, native_path, binary)
    if recomputed_lddt is None or recomputed_tm is None:
      dropped.append((target, 'could not score model against native'))
      continue
    # If our scores do not reproduce the official ones, the structure we
    # rebuilt is not the one CASP assessed -- usually a same-sequence entry of
    # a different construct or conformation.  Drop it rather than ship a
    # mislabelled ground truth.
    tm_gap = abs(recomputed_tm - cand['tmscore_af2'])
    lddt_gap = abs(recomputed_lddt - cand['lddt_af2'])
    if tm_gap > MAX_SCORE_GAP or lddt_gap > MAX_SCORE_GAP:
      dropped.append((target, f'native disagrees with official scores '
                              f'(dTM={tm_gap:.2f}, dLDDT={lddt_gap:.2f})'))
      continue

    notes = []
    if target in DM_MANUAL:
      notes.append('deepmind baseline used manual intervention')
    dm_path = deepmind.get(target)
    row = {
        'id': target,
        'sequence': cand['sequence'],
        'length': cand['length'],
        'plddt_af2': round(plddt, 2),
        'lddt_af2': cand['lddt_af2'],
        'tmscore_af2': cand['tmscore_af2'],
        'gdt_ts_af2': cand['gdt_ts_af2'],
        'n_eval_res': cand['n_eval_res'],
        'msa_depth': stats['msa_depth'],
        'msa_neff': stats['msa_neff'],
        'neff_per_res': stats['neff_per_res'],
        'native_source': prov.get('native_source', ''),
        'native_pdb_id': prov.get('native_pdb_id', ''),
        'lddt_af2_recomputed': round(recomputed_lddt, 3),
        'tmscore_af2_recomputed': round(recomputed_tm, 3),
        'plddt_af2_dm': '',
        'lddt_af2_dm': '',
        'tmscore_af2_dm': '',
        'notes': '; '.join(notes),
        '_tier': cand['tier'],
        '_neff': stats['msa_neff'],
        '_depth': stats['msa_depth'],
    }
    if dm_path and target not in DM_MANUAL:
      dm_plddt = mean_plddt(dm_path, native_residues)
      dm_lddt = lddt(dm_path, native_path)
      dm_tm = tm_score(dm_path, native_path, binary)
      row['plddt_af2_dm'] = round(dm_plddt, 2) if dm_plddt is not None else ''
      row['lddt_af2_dm'] = round(dm_lddt, 3) if dm_lddt is not None else ''
      row['tmscore_af2_dm'] = round(dm_tm, 3) if dm_tm is not None else ''
    rows.append(row)

  selection = sorted(rows, key=lambda r: r['tmscore_af2'])
  with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
    writer.writeheader()
    for row in selection:
      writer.writerow(row)

  log(f'\nwrote {CSV_PATH} with {len(selection)} proteins')
  for reason in relaxations:
    log(f'  RELAXATION APPLIED: {reason}')
  log('dropped candidates:')
  for target, reason in dropped:
    log(f'  {target:10s} {reason}')
  with open(os.path.join(CACHE, 'dropped.json'), 'w') as f:
    json.dump({'dropped': dropped, 'relaxations': relaxations}, f, indent=1)

  if not MIN_SET_SIZE <= len(selection) <= MAX_SET_SIZE:
    raise SystemExit(f'dataset has {len(selection)} proteins, outside '
                     f'[{MIN_SET_SIZE},{MAX_SET_SIZE}]')

  # Keep data/ limited to the selected proteins.
  keep = {row['id'] for row in selection}
  for directory, suffix in ((PDB_DIR, '.pdb'), (MSA_DIR, '.a3m'), (MODEL_DIR, '.pdb')):
    for name in os.listdir(directory):
      if name.endswith(suffix) and name[:-len(suffix)] not in keep:
        os.remove(os.path.join(directory, name))
  for name in os.listdir(MSA_RAW_DIR):
    if name.split('.')[0] not in keep:
      os.remove(os.path.join(MSA_RAW_DIR, name))

  write_readme(selection, dropped, relaxations)
  return selection


README_TEMPLATE = """# `af2_fails`: CASP15 targets where standard AlphaFold 2 fails despite a deep MSA

{n} single-chain proteins, {min_len}-{max_len} residues. Every one has a deep MSA
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

About {footprint} in total, dominated by the MSAs. Downloads and intermediates
live in `data/.cache/` and are gitignored; delete that directory to reclaim
space, at the cost of re-fetching on the next build.

## Selection criteria

- AF2 baseline whole-chain TM-score < {tm_threshold}
- {min_len_bound} <= length <= {max_len_bound} residues
- MSA Neff >= {min_neff:.0f} and depth >= {min_depth} sequences, so a shallow
  alignment cannot explain the failure
- our recomputed lDDT and TM-score agree with the official ones to within
  {max_gap}, which confirms the structure we rebuilt is the one CASP assessed
{relaxation_note}
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

{table}

Correlation between `plddt_af2` and `tmscore_af2` across the set: **{corr:+.2f}**.
{corr_note}

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
- {dm_note}

## Dropped candidates

{dropped_table}
"""


def write_readme(selection, dropped, relaxations) -> None:
  """Writes data/README.md describing provenance, criteria and contents."""
  lengths = [row['length'] for row in selection]
  plddt = np.array([row['plddt_af2'] for row in selection])
  tm = np.array([row['tmscore_af2'] for row in selection])
  corr = float(np.corrcoef(plddt, tm)[0, 1]) if len(selection) > 2 else float('nan')
  confident_failures = [row['id'] for row in selection if row['plddt_af2'] >= 80]

  header = ('| id | len | Neff | pLDDT | lDDT | TM | GDT_TS | native |\n'
            '| --- | --- | --- | --- | --- | --- | --- | --- |\n')
  table = header + '\n'.join(
      f"| `{r['id']}` | {r['length']} | {r['msa_neff']:.0f} | {r['plddt_af2']:.1f} | "
      f"{r['lddt_af2']:.2f} | {r['tmscore_af2']:.2f} | {r['gdt_ts_af2']:.1f} | "
      f"{r['native_pdb_id'] or 'CASP'} |" for r in selection)

  dropped_table = ('| id | reason |\n| --- | --- |\n' + '\n'.join(
      f'| `{target}` | {reason} |' for target, reason in dropped)
                   if dropped else 'None.')

  if confident_failures:
    corr_note = (
        f'{len(confident_failures)} of {len(selection)} are confidently wrong '
        f'(pLDDT >= 80 yet TM < {TM_THRESHOLD}): '
        + ', '.join(f'`{t}`' for t in confident_failures) + '. Those are the '
        'interesting cases for test-time training, because the model gives no '
        'internal signal that anything is off.')
  else:
    corr_note = ('Every failure here is accompanied by low confidence, so pLDDT '
                 'already flags them.')

  dm_note = ('DeepMind\'s own CASP15 baseline (`*_dm` columns, vendored in '
             '`docs/casp15_predictions.zip`) used a better protocol than group '
             '270 and recovers several of these targets. Targets DeepMind '
             'hand-tuned are left blank rather than reported as automated '
             'results.')

  relaxation_note = ''
  if relaxations:
    relaxation_note = ('\nThresholds relaxed to reach the minimum set size:\n'
                       + '\n'.join(f'- {r}' for r in relaxations) + '\n')

  megabytes = sum(
      os.path.getsize(os.path.join(root, name))
      for directory in (PDB_DIR, MSA_DIR, MODEL_DIR)
      for root, _, names in os.walk(directory)
      for name in names) / 1e6

  text = README_TEMPLATE.format(
      footprint=f'{megabytes:.0f} MB',
      n=len(selection), min_len=min(lengths), max_len=max(lengths),
      tm_threshold=TM_THRESHOLD, min_len_bound=MIN_LEN, max_len_bound=MAX_LEN,
      min_neff=MIN_NEFF, min_depth=MIN_DEPTH, max_gap=MAX_SCORE_GAP,
      relaxation_note=relaxation_note, table=table, corr=corr,
      corr_note=corr_note, dm_note=dm_note, dropped_table=dropped_table)
  with open(os.path.join(DATA, 'README.md'), 'w') as f:
    f.write(text)
  log(f'wrote {os.path.join(DATA, "README.md")}')


# --------------------------------------------------------------------------
# Stage 6: verify the built dataset
# --------------------------------------------------------------------------


def stage_verify() -> None:
  """Re-checks the finished dataset from disk and fails loudly on any problem."""
  with open(CSV_PATH, newline='') as f:
    rows = list(csv.DictReader(f))
  problems = []

  def check(condition, message):
    if not condition:
      problems.append(message)

  check(MIN_SET_SIZE <= len(rows) <= MAX_SET_SIZE,
        f'{len(rows)} proteins, expected between {MIN_SET_SIZE} and {MAX_SET_SIZE}')

  agreements = 0
  for row in rows:
    target = row['id']
    native_path = os.path.join(PDB_DIR, f'{target}.pdb')
    msa_path = os.path.join(MSA_DIR, f'{target}.a3m')
    model_path = os.path.join(MODEL_DIR, f'{target}.pdb')
    for path in (native_path, msa_path, model_path):
      check(os.path.exists(path) and os.path.getsize(path) > 0,
            f'{target}: missing or empty {os.path.relpath(path, REPO)}')
    if not all(os.path.exists(p) for p in (native_path, msa_path, model_path)):
      continue

    sequence, length = row['sequence'], int(row['length'])
    check(len(sequence) == length, f'{target}: length column disagrees with sequence')

    names, seqs = parse_a3m(open(msa_path).read())
    check(seqs and seqs[0] == sequence,
          f'{target}: first a3m row is not the target sequence')
    # Every residue must be something AlphaFold's feature builder accepts;
    # parse_a3m itself is happy to pass through junk like the API's NUL bytes.
    allowed = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ-') | set('abcdefghijklmnopqrstuvwxyz')
    stray = {c for s in seqs for c in s} - allowed
    check(not stray, f'{target}: a3m has characters AlphaFold cannot encode: '
                     f'{sorted(repr(c) for c in stray)}')
    check(int(row['msa_depth']) == len(seqs),
          f'{target}: msa_depth disagrees with the a3m')
    check(float(row['msa_neff']) >= MIN_NEFF,
          f'{target}: Neff {row["msa_neff"]} below the stated threshold')
    check(float(row['tmscore_af2']) < TM_THRESHOLD_RELAXED,
          f'{target}: TM-score {row["tmscore_af2"]} is not a failure')

    _, native_residues = load_atoms(native_path)
    check(native_residues and min(native_residues) >= 1
          and max(native_residues) <= length,
          f'{target}: native residue numbers fall outside 1..{length}')
    # Concatenated assessment domains can differ from the whole-chain assessed
    # count by a residue at a domain boundary; anything larger means we
    # rebuilt the wrong chain.
    check(abs(len(native_residues) - int(row['n_eval_res'])) <= 2
          or row['native_source'].startswith('rcsb'),
          f'{target}: native has {len(native_residues)} residues, '
          f'CASP assessed {row["n_eval_res"]}')

    if (abs(float(row['tmscore_af2_recomputed']) - float(row['tmscore_af2']))
        <= MAX_SCORE_GAP
        and abs(float(row['lddt_af2_recomputed']) - float(row['lddt_af2']))
        <= MAX_SCORE_GAP):
      agreements += 1

  if rows:
    fraction = agreements / len(rows)
    check(fraction >= 0.9,
          f'only {agreements}/{len(rows)} rows reproduce the official scores')
    log(f'{agreements}/{len(rows)} rows reproduce the official CASP scores '
        f'within {MAX_SCORE_GAP}')

  # The MSAs must load through the pipeline this repository actually runs.
  # sys.path[0] is scripts/ when run as a script, so add the repo root.
  if REPO not in sys.path:
    sys.path.insert(0, REPO)
  try:
    from alphafold.data import parsers as af_parsers
    for row in rows:
      msa = af_parsers.parse_a3m(open(os.path.join(MSA_DIR, f'{row["id"]}.a3m')).read())
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
                      choices=['all', 'candidates', 'msa', 'natives', 'models',
                               'csv', 'verify'])
  args = parser.parse_args()
  ensure_dirs()
  if args.stage in ('all', 'candidates'):
    stage_candidates()
  if args.stage in ('all', 'msa'):
    stage_msa()
  if args.stage in ('all', 'natives'):
    stage_natives()
  if args.stage in ('all', 'models'):
    stage_models()
  if args.stage in ('all', 'csv'):
    stage_csv()
  if args.stage in ('all', 'verify'):
    stage_verify()


if __name__ == '__main__':
  main()
