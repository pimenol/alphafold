#!/usr/bin/env python3
"""Scores our own AlphaFold predictions and writes an HTML report.

Reads the predicted structures produced by scripts/run_af2_on_dataset.py,
measures pLDDT, lDDT and TM-score against the same ground-truth files the
datasets ship, and compares them with the published baselines already recorded
in af2_fails.csv and af2_lowconf.csv.

Usage:
  python3 scripts/score_af2_predictions.py
"""

import csv
import html
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
  sys.path.insert(0, REPO)

sys.path.insert(0, os.path.join(REPO, 'scripts'))
import build_af2_fails_dataset as base  # noqa: E402  (path set above)

REPORT_PATH = os.path.join(REPO, 'af2_run_report.html')

DATASETS = {
    'fails': {
        'title': 'Confidently wrong, deep MSA (CASP15)',
        'csv': os.path.join(REPO, 'af2_fails.csv'),
        'native_dir': os.path.join(REPO, 'data', 'pdbs'),
        'pred_dir': os.path.join(REPO, 'predictions', 'fails'),
        'baseline_label': 'CASP15 group 270 (published)',
        'blurb': 'CASP15 targets released after the 2021-09-30 training cutoff '
                 'of the released v2.3 parameters, so they are blind. The '
                 'baseline columns are the Prediction Center&rsquo;s published '
                 'numbers for the standard AlphaFold2 server.',
    },
    'lowconf': {
        'title': 'Wrong and unconfident (AlphaFold DB &times; PDB)',
        'csv': os.path.join(REPO, 'af2_lowconf.csv'),
        'native_dir': os.path.join(REPO, 'data', 'lowconf', 'pdbs'),
        'pred_dir': os.path.join(REPO, 'predictions', 'lowconf'),
        'baseline_label': 'AlphaFold DB model',
        'blurb': 'Chains where the deposited AlphaFold DB model has pLDDT &lt; 70 '
                 'and lDDT &lt; 0.7. These are <strong>not</strong> blind: most of '
                 'these PDB entries predate the training cutoff.',
    },
}


def score_dataset(name, spec, binary):
  """Returns per-protein rows comparing our run with the shipped baseline."""
  with open(spec['csv'], newline='') as f:
    baseline = list(csv.DictReader(f))
  summary_path = os.path.join(spec['pred_dir'], 'run_summary.json')
  summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

  rows = []
  for entry in baseline:
    target = entry['id']
    native = os.path.join(spec['native_dir'], f'{target}.pdb')
    predicted = os.path.join(spec['pred_dir'], f'{target}.pdb')
    row = {
        'id': target,
        'length': int(entry['length']),
        'msa_depth': int(entry['msa_depth']),
        'msa_neff': float(entry['msa_neff']),
        'base_plddt': float(entry['plddt_af2']),
        'base_lddt': float(entry['lddt_af2']),
        'base_tm': float(entry['tmscore_af2']),
        'run_plddt': None, 'run_lddt': None, 'run_tm': None, 'run_ptm': None,
        'seconds': None, 'status': 'not run',
    }
    info = summary.get(target, {})
    if 'error' in info:
      row['status'] = f'failed: {info["error"]}'
    elif os.path.exists(predicted) and os.path.exists(native):
      _, native_residues = base.load_atoms(native)
      row['run_plddt'] = base.mean_plddt(predicted, native_residues)
      # An all-zero B-factor column means the prediction was written without
      # b_factors; report that rather than a pLDDT of 0.
      if row['run_plddt'] is not None and row['run_plddt'] == 0.0:
        raise SystemExit(
            f'{target}: predicted PDB has no pLDDT in its B-factor column. '
            f'Re-run scripts/run_af2_on_dataset.py -- it must pass b_factors '
            f'to protein.from_prediction.')
      row['run_lddt'] = base.lddt(predicted, native)
      row['run_tm'] = base.tm_score(predicted, native, binary)
      row['run_ptm'] = info.get('ptm')
      row['seconds'] = info.get('seconds')
      row['status'] = 'ok'
    rows.append(row)
  rows.sort(key=lambda r: r['base_tm'])
  return rows


def fmt(value, spec='{:.2f}'):
  return '&mdash;' if value is None else spec.format(value)


def delta_cell(new, old, spec='{:+.2f}'):
  if new is None or old is None:
    return '<td class="num">&mdash;</td>'
  diff = new - old
  cls = 'up' if diff > 0.02 else ('down' if diff < -0.02 else 'flat')
  return f'<td class="num {cls}">{spec.format(diff)}</td>'


def dataset_section(name, spec, rows):
  scored = [r for r in rows if r['status'] == 'ok']
  body = []
  for r in rows:
    if r['status'] != 'ok':
      body.append(
          f'<tr class="missing"><td class="id">{html.escape(r["id"])}</td>'
          f'<td class="num">{r["length"]}</td><td class="num">{r["msa_neff"]:.0f}</td>'
          f'<td class="num">{r["base_plddt"]:.1f}</td>'
          f'<td class="num">{r["base_lddt"]:.2f}</td>'
          f'<td class="num">{r["base_tm"]:.2f}</td>'
          f'<td colspan="5" class="note">{html.escape(r["status"])}</td>'
          f'<td class="num">&mdash;</td></tr>')
      continue
    body.append(
        f'<tr><td class="id">{html.escape(r["id"])}</td>'
        f'<td class="num">{r["length"]}</td>'
        f'<td class="num">{r["msa_neff"]:.0f}</td>'
        f'<td class="num">{r["base_plddt"]:.1f}</td>'
        f'<td class="num">{r["base_lddt"]:.2f}</td>'
        f'<td class="num">{r["base_tm"]:.2f}</td>'
        f'<td class="num">{fmt(r["run_plddt"], "{:.1f}")}</td>'
        f'<td class="num">{fmt(r["run_lddt"], "{:.2f}")}</td>'
        f'<td class="num">{fmt(r["run_tm"], "{:.2f}")}</td>'
        f'{delta_cell(r["run_lddt"], r["base_lddt"])}'
        f'{delta_cell(r["run_tm"], r["base_tm"])}'
        f'<td class="num">{fmt(r["run_ptm"], "{:.2f}")}</td></tr>')

  if scored:
    d_lddt = [r['run_lddt'] - r['base_lddt'] for r in scored]
    d_tm = [r['run_tm'] - r['base_tm'] for r in scored]
    improved = sum(1 for d in d_tm if d > 0.05)
    worse = sum(1 for d in d_tm if d < -0.05)
    still_failing = sum(1 for r in scored if r['run_tm'] < 0.8)
    stats = (
        f'<div class="stats">'
        f'<div class="stat"><span class="v">{len(scored)}/{len(rows)}</span>'
        f'<span class="k">predicted</span></div>'
        f'<div class="stat"><span class="v">{np.mean(d_tm):+.2f}</span>'
        f'<span class="k">mean &Delta;TM vs baseline</span></div>'
        f'<div class="stat"><span class="v">{np.mean(d_lddt):+.2f}</span>'
        f'<span class="k">mean &Delta;lDDT</span></div>'
        f'<div class="stat"><span class="v">{improved}</span>'
        f'<span class="k">improved (&Delta;TM &gt; 0.05)</span></div>'
        f'<div class="stat"><span class="v">{worse}</span>'
        f'<span class="k">regressed (&Delta;TM &lt; -0.05)</span></div>'
        f'<div class="stat"><span class="v">{still_failing}</span>'
        f'<span class="k">still below TM 0.8</span></div>'
        f'</div>')
  else:
    stats = '<p class="note">No predictions scored yet.</p>'

  return f'''
<section>
  <h2>{spec['title']}</h2>
  <p class="blurb">{spec['blurb']}</p>
  {stats}
  <div class="tablewrap">
  <table>
    <thead>
      <tr>
        <th rowspan="2">id</th><th rowspan="2">len</th><th rowspan="2">Neff</th>
        <th colspan="3" class="grp">{spec['baseline_label']}</th>
        <th colspan="3" class="grp">this run (MSA-only, no templates)</th>
        <th colspan="2" class="grp">change</th>
        <th rowspan="2">pTM</th>
      </tr>
      <tr>
        <th>pLDDT</th><th>lDDT</th><th>TM</th>
        <th>pLDDT</th><th>lDDT</th><th>TM</th>
        <th>&Delta;lDDT</th><th>&Delta;TM</th>
      </tr>
    </thead>
    <tbody>
      {''.join(body)}
    </tbody>
  </table>
  </div>
</section>'''


CSS = '''
:root { --bg:#ffffff; --fg:#1a1a1a; --muted:#666; --line:#e2e2e2;
        --head:#f6f7f9; --up:#0a7c42; --down:#b3261e; --accent:#2b5c8a; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14161a; --fg:#e8e8e8; --muted:#9aa0a6; --line:#2c3038;
          --head:#1c1f25; --up:#4ade80; --down:#f87171; --accent:#7aa7d1; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size:1.7rem; margin:0 0 .35rem; letter-spacing:-.01em; }
h2 { font-size:1.2rem; margin:2.5rem 0 .4rem; letter-spacing:-.01em; }
.sub { color:var(--muted); margin:0 0 1.5rem; }
.blurb { color:var(--muted); margin:.2rem 0 1rem; max-width:70ch; }
.note { color:var(--muted); font-style:italic; }
.stats { display:flex; flex-wrap:wrap; gap:.6rem; margin:1rem 0 1.2rem; }
.stat { border:1px solid var(--line); border-radius:8px; padding:.55rem .8rem;
        min-width:8.5rem; background:var(--head); }
.stat .v { display:block; font-size:1.25rem; font-weight:600;
           font-variant-numeric:tabular-nums; }
.stat .k { display:block; font-size:.74rem; color:var(--muted); }
.tablewrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th, td { padding:.42rem .6rem; border-bottom:1px solid var(--line);
         text-align:left; white-space:nowrap; }
thead th { background:var(--head); font-weight:600; position:sticky; top:0; }
th.grp { text-align:center; border-left:1px solid var(--line); }
td.num, th { font-variant-numeric:tabular-nums; }
td.num { text-align:right; }
td.id { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
tbody tr:hover { background:var(--head); }
tr.missing td { color:var(--muted); }
.up { color:var(--up); font-weight:600; }
.down { color:var(--down); font-weight:600; }
.flat { color:var(--muted); }
.caveats { margin-top:2.5rem; border-top:1px solid var(--line); padding-top:1.2rem; }
.caveats li { margin:.4rem 0; max-width:80ch; }
code { background:var(--head); padding:.1rem .3rem; border-radius:4px;
       font-size:.9em; }
'''


def build_report(sections, generated):
  return f'''<h1>AlphaFold 2 baseline run</h1>
<p class="sub">Both benchmark datasets predicted with <code>model_1_ptm</code>,
using the MSAs the datasets ship and no templates. Generated {generated}.</p>
{''.join(sections)}
<section class="caveats">
  <h2>How to read this</h2>
  <ul>
    <li><strong>This run is MSA-only and template-free.</strong> Every one of
      these structures is in the PDB today, so enabling templates would hand the
      model its answer and destroy the blind-target property of the CASP15 set.</li>
    <li><strong>The MSAs are ColabFold MMseqs2, not AlphaFold&rsquo;s own
      jackhmmer/HHblits triple</strong> &mdash; no sequence databases are installed on
      this machine. That, plus one model instead of five and no relaxation, is why
      this run does not reproduce the published baseline exactly.</li>
    <li><strong>&Delta; columns compare like with unlike on purpose.</strong> For the
      CASP15 set the baseline is a different group&rsquo;s v2.2 run with its own
      MSAs; a positive &Delta; means better MSAs or a newer parameter set helped,
      not that the target stopped being hard.</li>
    <li>Rows still below TM 0.8 after this run are the ones worth aiming
      test-time training at &mdash; the failure survives a modern MSA.</li>
  </ul>
</section>'''


def main():
  binary = base.ensure_usalign()
  sections, all_rows = [], {}
  for name, spec in DATASETS.items():
    rows = score_dataset(name, spec, binary)
    all_rows[name] = rows
    sections.append(dataset_section(name, spec, rows))

  # Persist the scored numbers next to the report so they are reusable.
  out_csv = os.path.join(REPO, 'af2_run_results.csv')
  fields = ['dataset', 'id', 'length', 'msa_depth', 'msa_neff',
            'base_plddt', 'base_lddt', 'base_tm',
            'run_plddt', 'run_lddt', 'run_tm', 'run_ptm', 'seconds', 'status']
  with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for name, rows in all_rows.items():
      for row in rows:
        record = dict(row)
        record['dataset'] = name
        for key in ('run_plddt', 'run_lddt', 'run_tm', 'run_ptm'):
          if record[key] is not None:
            record[key] = round(record[key], 3)
        writer.writerow(record)
  print(f'wrote {out_csv}')

  generated = os.environ.get('REPORT_DATE', 'from the run summaries')
  with open(REPORT_PATH, 'w') as f:
    f.write(f'<style>{CSS}</style>\n<main>\n')
    f.write(build_report(sections, html.escape(generated)))
    f.write('\n</main>\n')
  print(f'wrote {REPORT_PATH}')

  for name, rows in all_rows.items():
    ok = [r for r in rows if r['status'] == 'ok']
    print(f'{name}: {len(ok)}/{len(rows)} scored')


if __name__ == '__main__':
  main()
