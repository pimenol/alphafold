#!/usr/bin/env python3
"""Run baseline AF2 on 100 proteins, keep only those with pLDDT < 70 and not dynamic.

For each protein in the benchmark CSV:
  1. Load precomputed MSA (.a3m)
  2. Run baseline AlphaFold2 prediction
  3. If mean pLDDT < 70, save PDB and check is_dynamic_protein
  4. If not dynamic, copy PDB to final output dir

Outputs:
  - data/bfvd/AF2/{id}.pdb  — PDB files for suitable proteins
  - data/bfvd/summary.csv   — summary with only suitable proteins
"""

from __future__ import annotations

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
_xla = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_enable_triton_gemm' not in _xla:
    os.environ['XLA_FLAGS'] = f'{_xla} --xla_gpu_enable_triton_gemm=false'.strip()

import argparse
import csv
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

import jax
print(f'[init] jax devices={jax.devices()}', flush=True)

from alphafold.common import confidence, protein, residue_constants
from alphafold.common.structure_detects import is_dynamic_protein
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
    raw_features['template_domain_names'] = np.array([''.encode()], dtype=object)
    raw_features['template_sequence'] = np.array([''.encode()], dtype=object)
    raw_features['template_sum_probs'] = np.array([0], dtype=np.float32)

    return af_features.np_example_to_features(
        np_example=raw_features, config=model_config, random_seed=random_seed)


def save_prediction_pdb(
    prediction_result: Dict[str, Any],
    features: Dict[str, np.ndarray],
    pdb_path: str,
) -> None:
    """Build a Protein from prediction result and write it as PDB."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark_csv', type=Path, required=True)
    parser.add_argument('--msa_dir', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='Directory for filtered PDBs (e.g. data/bfvd/AF2)')
    parser.add_argument('--summary_csv', type=Path, required=True,
                        help='Path for summary CSV (e.g. data/bfvd/summary.csv)')
    parser.add_argument('--model_name', default='model_1_ptm')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    # Read CSV
    rows = []
    with open(args.benchmark_csv, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f'Loaded {len(rows)} proteins from {args.benchmark_csv}')

    # Setup model
    model_config = af_config.model_config(args.model_name)
    params = af_data.get_model_haiku_params(args.model_name, str(args.data_dir))
    runner = af_model.RunModel(model_config, params=params)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    suitable = []

    for i, row in enumerate(rows):
        pid = row['id'].strip()
        seq = row['sequence'].strip()
        a3m_path = args.msa_dir / f'{pid}.a3m'

        final_pdb = args.output_dir / f'{pid}.pdb'
        if args.skip_existing and final_pdb.exists():
            print(f'[{i+1}/{len(rows)}] {pid}: already exists, skipping')
            continue

        if not a3m_path.exists():
            print(f'[{i+1}/{len(rows)}] {pid}: no MSA at {a3m_path}, skipping')
            continue

        print(f'[{i+1}/{len(rows)}] {pid} (len={len(seq)}) ...', flush=True)

        try:
            features = load_features_from_a3m(
                str(a3m_path), seq, pid, model_config, random_seed=args.seed)
            t0 = time.time()
            result = runner.predict(features, random_seed=args.seed)
            elapsed = time.time() - t0

            plddt = confidence.compute_plddt(
                result['predicted_lddt']['logits'])
            mean_plddt = float(np.mean(plddt))
            print(f'  pLDDT: {mean_plddt:.2f}  ({elapsed:.1f}s)')

            if mean_plddt >= 70:
                print(f'  SKIP: pLDDT {mean_plddt:.2f} >= 70')
                continue

            # Save PDB to check is_dynamic_protein
            save_prediction_pdb(result, features, str(final_pdb))

            if is_dynamic_protein(str(final_pdb)):
                print(f'  SKIP: dynamic protein')
                final_pdb.unlink()
                continue

            print(f'  KEPT: pLDDT={mean_plddt:.2f}, not dynamic')
            suitable.append({
                'id': pid,
                'sequence': seq,
                'length': len(seq),
                'mean_plddt': round(mean_plddt, 2),
            })

        except Exception as e:
            print(f'  FAILED: {e}')
            continue

    # Write summary
    if suitable:
        fieldnames = ['id', 'sequence', 'length', 'mean_plddt']
        with open(args.summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(suitable)

    print(f'\n=== Done: {len(suitable)}/{len(rows)} proteins kept ===')
    print(f'PDBs: {args.output_dir}')
    print(f'Summary: {args.summary_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
