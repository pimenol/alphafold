#!/usr/bin/env python3
"""Runs AlphaFold inference over one of the benchmark datasets.

The stock run_alphafold.py builds its MSAs by running jackhmmer and hhblits
against ~2.6 TB of sequence databases.  None of that is installed here, and it
is not needed: both datasets already ship an a3m per protein, so this script
assembles the feature dict directly from those alignments and calls the model.

Templates are deliberately left empty.  Every one of these structures is in the
PDB today, so allowing templates would hand the model the answer and destroy
the blind-target property of the CASP15 set.

Predictions are written unrelaxed -- relaxation only cleans up stereochemistry
and would not change lDDT or TM-score materially.

Usage:
  python3 scripts/run_af2_on_dataset.py --dataset fails --out_dir predictions/fails
  python3 scripts/run_af2_on_dataset.py --dataset lowconf --out_dir predictions/lowconf
"""

import argparse
import csv
import json
import os
import pathlib
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
  sys.path.insert(0, REPO)

DATASETS = {
    'fails': {
        'csv': os.path.join(REPO, 'af2_fails.csv'),
        'msa_dir': os.path.join(REPO, 'data', 'msa'),
    },
    'lowconf': {
        'csv': os.path.join(REPO, 'af2_lowconf.csv'),
        'msa_dir': os.path.join(REPO, 'data', 'lowconf', 'msa'),
    },
}


def empty_template_features(num_res: int):
  """Template features for a run with no templates.

  Same content as alphafold.notebooks.notebook_utils, reimplemented here
  because importing that module drags in matplotlib.
  """
  from alphafold.common import residue_constants
  return {
      'template_aatype': np.zeros(
          (0, num_res, len(residue_constants.restypes_with_x_and_gap)),
          dtype=np.float32),
      'template_all_atom_masks': np.zeros(
          (0, num_res, residue_constants.atom_type_num), dtype=np.float32),
      'template_all_atom_positions': np.zeros(
          (0, num_res, residue_constants.atom_type_num, 3), dtype=np.float32),
      'template_domain_names': np.zeros([0], dtype=object),
      'template_sequence': np.zeros([0], dtype=object),
      'template_sum_probs': np.zeros([0], dtype=np.float32),
  }


def build_features(target: str, sequence: str, a3m_path: str, max_msa: int):
  """Assembles the AlphaFold feature dict from a precomputed a3m."""
  from alphafold.data import parsers, pipeline

  msa = parsers.parse_a3m(open(a3m_path).read())
  if len(msa.sequences) > max_msa:
    # Keep the query plus an evenly spaced sample; the a3m is ordered by the
    # search, so this preserves the range of divergence rather than taking only
    # the closest homologues.
    keep = [0] + [int(i) for i in np.linspace(1, len(msa.sequences) - 1, max_msa - 1)]
    msa = parsers.Msa(
        sequences=[msa.sequences[i] for i in keep],
        deletion_matrix=[msa.deletion_matrix[i] for i in keep],
        descriptions=[msa.descriptions[i] for i in keep])

  features = {
      **pipeline.make_sequence_features(sequence, target, len(sequence)),
      **pipeline.make_msa_features([msa]),
      **empty_template_features(len(sequence)),
  }
  return features, len(msa.sequences)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', required=True, choices=sorted(DATASETS))
  parser.add_argument('--out_dir', required=True)
  parser.add_argument('--params_dir', required=True,
                      help='directory containing the "params" subdirectory')
  parser.add_argument('--model_name', default='model_1_ptm',
                      help='one model by default; predicting with all five '
                           'would cost 5x for a baseline we already have')
  parser.add_argument('--random_seed', type=int, default=0)
  parser.add_argument('--max_msa', type=int, default=8192,
                      help='cap on MSA rows fed to the model, to bound memory')
  parser.add_argument('--only', nargs='*', default=None,
                      help='restrict to these ids')
  args = parser.parse_args()

  from alphafold.common import protein, residue_constants
  from alphafold.model import config as model_config
  from alphafold.model import data as model_data
  from alphafold.model import model as model_lib
  import jax

  print('jax devices:', jax.devices(), flush=True)
  os.makedirs(args.out_dir, exist_ok=True)

  with open(DATASETS[args.dataset]['csv'], newline='') as f:
    rows = list(csv.DictReader(f))
  if args.only:
    rows = [r for r in rows if r['id'] in set(args.only)]
  # Shortest first, so a memory blow-up on a big chain does not cost the
  # whole run.
  rows.sort(key=lambda r: int(r['length']))
  print(f'{len(rows)} proteins from {args.dataset}', flush=True)

  cfg = model_config.model_config(args.model_name)
  params = model_data.get_model_haiku_params(model_name=args.model_name,
                                             data_dir=args.params_dir)
  runner = model_lib.RunModel(cfg, params)

  msa_dir = DATASETS[args.dataset]['msa_dir']
  summary_path = os.path.join(args.out_dir, 'run_summary.json')
  summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

  for index, row in enumerate(rows, start=1):
    target, sequence = row['id'], row['sequence']
    pdb_path = os.path.join(args.out_dir, f'{target}.pdb')
    if target in summary and os.path.exists(pdb_path):
      print(f'[{index}/{len(rows)}] {target}: already done', flush=True)
      continue

    a3m_path = os.path.join(msa_dir, f'{target}.a3m')
    started = time.time()
    try:
      features, msa_depth = build_features(target, sequence, a3m_path, args.max_msa)
      processed = runner.process_features(features, random_seed=args.random_seed)
      result = runner.predict(processed, random_seed=args.random_seed)
      # Without this the B-factor column is written as zeros. Every other model
      # file in these datasets carries per-residue pLDDT there, and the scoring
      # step reads it back, so it has to be filled in.
      plddt_b_factors = np.repeat(
          result['plddt'][:, None], residue_constants.atom_type_num, axis=-1)
      unrelaxed = protein.from_prediction(processed, result,
                                          b_factors=plddt_b_factors,
                                          remove_leading_feature_dimension=True)
      with open(pdb_path, 'w') as f:
        f.write(protein.to_pdb(unrelaxed))
      # Keep the raw per-residue confidence too; it is not recoverable from the
      # rounded B-factor column.
      np.save(os.path.join(args.out_dir, f'{target}_plddt.npy'), result['plddt'])
    except Exception as e:  # keep going; one failure must not lose the run
      print(f'[{index}/{len(rows)}] {target}: FAILED ({type(e).__name__}: {e})',
            flush=True)
      summary[target] = {'error': f'{type(e).__name__}: {e}'}
      with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=1)
      continue

    elapsed = time.time() - started
    summary[target] = {
        'length': len(sequence),
        'msa_depth_used': msa_depth,
        'plddt_mean': float(np.mean(result['plddt'])),
        'ptm': float(result['ptm']) if 'ptm' in result else None,
        'model_name': args.model_name,
        'random_seed': args.random_seed,
        'seconds': round(elapsed, 1),
    }
    with open(summary_path, 'w') as f:
      json.dump(summary, f, indent=1)
    print(f'[{index}/{len(rows)}] {target}: L={len(sequence)} '
          f'msa={msa_depth} pLDDT={summary[target]["plddt_mean"]:.1f} '
          f'pTM={summary[target]["ptm"]:.3f} ({elapsed:.0f}s)', flush=True)

  done = sum(1 for v in summary.values() if 'error' not in v)
  print(f'finished: {done}/{len(rows)} predicted, '
        f'{len(summary) - done} failed', flush=True)


if __name__ == '__main__':
  main()
