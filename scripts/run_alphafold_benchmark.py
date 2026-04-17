#!/usr/bin/env python3
"""Run baseline AlphaFold2 on proteins listed in a benchmark CSV.

Uses precomputed MSAs (.a3m) and the AF2 model API directly — no genetic
database search required.  For each protein:
  1. Load precomputed MSA
  2. Run full AF2 prediction (with recycling)
  3. Save PDB and pLDDT scores
"""

from __future__ import annotations

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
_xla = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_enable_triton_gemm' not in _xla:
    os.environ['XLA_FLAGS'] = f'{_xla} --xla_gpu_enable_triton_gemm=false'.strip()

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import jax
import numpy as np

print(f'[init] jax devices={jax.devices()}', flush=True)

from alphafold.common import confidence, protein, residue_constants
from alphafold.data import parsers, pipeline
from alphafold.model import config as af_config
from alphafold.model import data as af_data
from alphafold.model import features as af_features
from alphafold.model import model as af_model

print('[init] imports done', flush=True)


def load_features_from_a3m(
    a3m_path: str,
    sequence: str,
    protein_id: str,
    model_config,
    random_seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Build processed feature dict from a precomputed A3M file."""
    with open(a3m_path, 'r') as f:
        a3m_string = f.read()
    msa = parsers.parse_a3m(a3m_string)

    num_res = len(sequence)
    raw_features: Dict[str, Any] = {}
    raw_features.update(
        pipeline.make_sequence_features(sequence, protein_id, num_res)
    )
    raw_features.update(pipeline.make_msa_features([msa]))

    raw_features['template_aatype'] = np.zeros(
        (1, num_res, len(residue_constants.restypes_with_x_and_gap)), np.float32)
    raw_features['template_all_atom_masks'] = np.zeros(
        (1, num_res, residue_constants.atom_type_num), np.float32)
    raw_features['template_all_atom_positions'] = np.zeros(
        (1, num_res, residue_constants.atom_type_num, 3), np.float32)
    raw_features['template_domain_names'] = np.array(
        [''.encode()], dtype=object)
    raw_features['template_sequence'] = np.array(
        [''.encode()], dtype=object)
    raw_features['template_sum_probs'] = np.array([0], dtype=np.float32)

    return af_features.np_example_to_features(
        np_example=raw_features, config=model_config, random_seed=random_seed)


def save_prediction_pdb(
    prediction_result: Dict[str, Any],
    features: Dict[str, np.ndarray],
    pdb_path: str,
) -> None:
    plddt = confidence.compute_plddt(
        prediction_result['predicted_lddt']['logits'])
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    prot = protein.from_prediction(
        features=features, result=prediction_result,
        b_factors=plddt_b_factors, remove_leading_feature_dimension=True)
    Path(pdb_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pdb_path, 'w') as f:
        f.write(protein.to_pdb(prot))


def main() -> int:
    default_csv = Path(
        '/scratch/project/open-35-8/pimenol1/alphafold/data/bfvd/summary.csv'
    )
    default_msa_dir = Path(
        '/scratch/project/open-35-8/data/bfvd/bfvd_beta/input/logan'
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark_csv', type=Path,
                        default=default_csv if default_csv.is_file() else None)
    parser.add_argument('--msa_dir', type=Path, default=default_msa_dir,
                        help='Directory with precomputed A3M files.')
    parser.add_argument('--data_dir', type=Path, required=True,
                        help='AF2 data root (needs params/ subdirectory).')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--model_name', default='model_1_ptm')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--protein_ids', default=None,
                        help='Comma-separated subset of IDs to run.')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    if args.benchmark_csv is None:
        print('Provide --benchmark_csv')
        return 1

    # Read CSV
    rows: List[dict] = []
    with open(args.benchmark_csv, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if args.protein_ids:
        wanted = set(args.protein_ids.split(','))
        rows = [r for r in rows if r['id'] in wanted]

    end = args.end_idx or len(rows)
    rows = rows[args.start_idx:end]

    if not rows:
        print('No proteins to process.')
        return 0

    # Setup model
    print(f'Model: {args.model_name}')
    model_config = af_config.model_config(args.model_name)
    params = af_data.get_model_haiku_params(args.model_name, str(args.data_dir))
    runner = af_model.RunModel(model_config, params=params)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []

    for i, row in enumerate(rows):
        pid = row['id'].strip()
        seq = row['sequence'].strip()
        a3m_path = args.msa_dir / f'{pid}.a3m'

        protein_dir = args.output_dir / pid
        result_path = protein_dir / 'result.json'
        if args.skip_existing and result_path.exists():
            print(f'[{i+1}] {pid}: already done, skipping')
            continue

        if not a3m_path.exists():
            print(f'[{i+1}] {pid}: no MSA at {a3m_path}, skipping')
            continue

        print(f'[{i+1}] {pid} (len={len(seq)}) ...', flush=True)

        try:
            features = load_features_from_a3m(
                str(a3m_path), seq, pid, model_config,
                random_seed=args.seed)
            t0 = time.time()
            prediction = runner.predict(features, random_seed=args.seed)
            elapsed = time.time() - t0

            plddt = confidence.compute_plddt(
                prediction['predicted_lddt']['logits'])
            mean_plddt = float(np.mean(plddt))
            print(f'  pLDDT: {mean_plddt:.2f}  ({elapsed:.1f}s)')

            save_prediction_pdb(
                prediction, features,
                str(protein_dir / 'baseline.pdb'))

            result = {
                'id': pid,
                'length': len(seq),
                'mean_plddt': round(mean_plddt, 4),
                'time_s': round(elapsed, 1),
            }
            results.append(result)

            protein_dir.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f'  FAILED: {e}')
            jax.clear_caches()
            continue

    # Summary
    if results:
        summary_path = args.output_dir / 'summary.csv'
        fieldnames = ['id', 'length', 'mean_plddt', 'time_s']
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        plddts = [r['mean_plddt'] for r in results]
        print(f'\n=== Summary ({len(results)} proteins) ===')
        print(f'Mean pLDDT:   {np.mean(plddts):.2f}')
        print(f'Median pLDDT: {np.median(plddts):.2f}')
        print(f'Results saved to {summary_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
