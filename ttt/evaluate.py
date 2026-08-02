#!/usr/bin/env python3
"""Scores a TTT experiment against the M0 baseline and prints the Delta table.

lDDT and TM-score come from `scripts/build_af2_fails_dataset.py`, the same
implementations `scripts/score_af2_predictions.py` uses -- the goal forbids a
second implementation, because a Delta between two different metrics is
meaningless.

Deltas are reported in lDDT *points* (0-100), the convention CLAUDE.md uses for
the +7 threshold.  The CSVs store fractions, so everything is scaled by 100 on
the way out.

Usage:
  python3 -m ttt.evaluate --name exp001-violation
"""

import argparse
import csv
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO, os.path.join(REPO, 'scripts')):
  if path not in sys.path:
    sys.path.insert(0, path)

import build_af2_fails_dataset as base  # noqa: E402  (path set above)

NATIVE_DIR = os.path.join(REPO, 'data', 'lowconf', 'pdbs')
RESULTS_CSV = os.path.join(REPO, 'af2_run_results.csv')
HARD_SET = os.path.join(REPO, 'subsets', 'lowconf18.txt')
THRESHOLD = 7.0  # lDDT points

# AlphaFold is not reproducible across devices. The same code, parameters, seed
# and features give a prediction 9.39 A (superposed CA) apart on an A100 versus
# on CPU for 9SLR_A -- verified by running the stock M0 script on both. On a set
# selected for low confidence that is not numerical noise, it is a different
# fold. So a TTT run must be scored against a baseline produced on the *same*
# device, or the device difference swamps the effect being measured.
CPU_BASELINE_DIR = os.path.join(REPO, 'predictions', 'lowconf_cpu')
GPU_BASELINE_DIR = os.path.join(REPO, 'predictions', 'lowconf')


def read_ids(path):
  return [l.split('#', 1)[0].strip() for l in open(path)
          if l.split('#', 1)[0].strip()]


def baseline_rows():
  with open(RESULTS_CSV, newline='') as handle:
    return {r['id']: r for r in csv.DictReader(handle)
            if r['dataset'] == 'lowconf'}


def score(pred_path, native_path, binary):
  """(lDDT points, TM, mean pLDDT) for one predicted structure."""
  _, native_residues = base.load_atoms(native_path)
  return (
      base.lddt(pred_path, native_path) * 100.0,
      base.tm_score(pred_path, native_path, binary),
      base.mean_plddt(pred_path, native_residues),
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--name', required=True)
  parser.add_argument('--variant', default='both',
                      choices=['step', 'bestplddt', 'both'],
                      help='which TTT checkpoint to score')
  parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                      help='device the TTT run used; picks the matching M0 '
                           'baseline, because AlphaFold predictions are not '
                           'reproducible across devices')
  args = parser.parse_args()

  out_dir = os.path.join(REPO, 'predictions', 'ttt', args.name)
  summary = json.load(open(os.path.join(out_dir, 'ttt_summary.json')))
  steps = summary['args']['steps']
  base_rows = baseline_rows()
  hard = set(read_ids(HARD_SET))
  binary = base.ensure_usalign()

  baseline_dir = CPU_BASELINE_DIR if args.device == 'cpu' else GPU_BASELINE_DIR
  rows = []
  skipped = []
  for target, info in summary['results'].items():
    native = os.path.join(NATIVE_DIR, f'{target}.pdb')
    baseline_pdb = os.path.join(baseline_dir, f'{target}.pdb')
    if not os.path.exists(baseline_pdb):
      skipped.append(target)
      continue
    b_lddt, b_tm, b_plddt = score(baseline_pdb, native, binary)
    row = {
        'id': target,
        'hard': target in hard,
        'length': info['length'],
        'msa_depth': info['msa_depth'],
        'baseline_device': args.device,
        # Baseline is measured here, from the M0 prediction on the same device,
        # rather than read from af2_run_results.csv -- those numbers are from
        # the A100 run and do not transfer. The CSV's run_* values are kept
        # alongside for reference; base_* there is the AlphaFold DB model, a
        # different prediction entirely, and conflating the two inverted the
        # sign of the first result I reported.
        'base_lddt': b_lddt,
        'base_tm': b_tm,
        'base_plddt': b_plddt,
        'gpu_m0_lddt': float(base_rows[target]['run_lddt']) * 100.0,
        'afdb_lddt': float(base_rows[target]['base_lddt']) * 100.0,
        'ca_rmsd': info['ca_rmsd_vs_baseline'],
        'best_step': info['best_step'],
    }
    for suffix, tag in (('step%d' % steps, 'step'), ('bestplddt', 'best')):
      path = os.path.join(out_dir, f'{target}_{suffix}.pdb')
      if not os.path.exists(path) and tag == 'best':
        # best step was the last step, so no separate file was written
        path = os.path.join(out_dir, f'{target}_step%d.pdb' % steps)
      if os.path.exists(path):
        lddt, tm, plddt = score(path, native, binary)
        row[f'{tag}_lddt'], row[f'{tag}_tm'], row[f'{tag}_plddt'] = lddt, tm, plddt
        row[f'{tag}_dlddt'] = lddt - row['base_lddt']
        row[f'{tag}_dtm'] = tm - row['base_tm']
        row[f'{tag}_dplddt'] = plddt - row['base_plddt']
    rows.append(row)

  if skipped:
    print(f'  ! no {args.device} M0 baseline for {len(skipped)} target(s), '
          f'skipped: {" ".join(sorted(skipped))}')
  if not rows:
    print(f'{args.name}: nothing scorable yet '
          f'(no results, or no {args.device} baseline)')
    return 1

  rows.sort(key=lambda r: -r.get('step_dlddt', 0.0))
  csv_path = os.path.join(out_dir, 'scores.csv')
  # A run killed at the walltime can leave a target without its bestplddt file,
  # so the key set is the union rather than whatever the first row happens to have.
  fields = sorted({k for r in rows for k in r})
  with open(csv_path, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, restval='')
    writer.writeheader()
    writer.writerows(rows)

  def block(tag, label, subset):
    if not subset:
      return
    d = [r[f'{tag}_dlddt'] for r in subset if f'{tag}_dlddt' in r]
    if not d:
      return
    up = sum(1 for x in d if x >= THRESHOLD)
    down = sum(1 for x in d if x <= -THRESHOLD)
    dp = [r[f'{tag}_dplddt'] for r in subset if f'{tag}_dplddt' in r]
    print(f'  {label:22} n={len(d):2}  mean {statistics.mean(d):+6.2f}  '
          f'median {statistics.median(d):+6.2f}  '
          f'>=+{THRESHOLD:.0f}: {up}  <=-{THRESHOLD:.0f}: {down}  '
          f'mean dpLDDT {statistics.mean(dp):+6.2f}')

  print(f'\n=== {args.name} ===')
  print(f'{summary["args"]["loss"]}  steps={steps}  lr={summary["args"]["lr"]}  '
        f'blocks={summary["args"]["blocks"]}')
  print('\nDelta lDDT in points (0-100 scale); threshold +/-%g' % THRESHOLD)
  for tag, label in (('step', f'at step {steps}'), ('best', 'at best-pLDDT step')):
    print(f'\n{label}:')
    block(tag, 'all scored', rows)
    block(tag, 'hard set only', [r for r in rows if r['hard']])
    block(tag, 'deep MSA (>=1000)', [r for r in rows if r['msa_depth'] >= 1000])
    block(tag, 'shallow MSA (<=30)', [r for r in rows if r['msa_depth'] <= 30])

  print('\nper protein (step %d):' % steps)
  print(f'  {"id":9} {"H":1} {"len":>4} {"depth":>6} {"lDDT":>7} {"dlDDT":>7} '
        f'{"dTM":>6} {"pLDDT":>6} {"dpLDDT":>7} {"CA rmsd":>8} {"best":>4}')
  for r in rows:
    if 'step_lddt' not in r:
      print(f'  {r["id"]:9} -- not scored --')
      continue
    rmsd = 'n/a' if r['ca_rmsd'] is None else '%.3f' % r['ca_rmsd']
    print(f'  {r["id"]:9} {"H" if r["hard"] else "-":1} {r["length"]:>4} '
          f'{r["msa_depth"]:>6} {r["step_lddt"]:>7.2f} {r["step_dlddt"]:>+7.2f} '
          f'{r["step_dtm"]:>+6.2f} {r["step_plddt"]:>6.2f} '
          f'{r["step_dplddt"]:>+7.2f} {rmsd:>8} {r["best_step"]:>4}')
  print(f'\nwrote {csv_path}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
