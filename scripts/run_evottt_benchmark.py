#!/usr/bin/env python3
"""Run EvoTTT benchmark: baseline AF2 → TTT adaptation → adapted AF2.

For each protein in the benchmark CSV:
  1. Load precomputed MSA (.a3m)
  2. Run baseline AlphaFold2 prediction
  3. Run test-time training (EvoTTT) to adapt Evoformer weights
  4. Run adapted AlphaFold2 prediction
  5. Compare pLDDT scores and save results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import jax
import numpy as np

# Silence TF warnings before importing AF2
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from alphafold.common import confidence, residue_constants
from alphafold.data import parsers, pipeline
from alphafold.model import config as af_config
from alphafold.model import data as af_data
from alphafold.model import features as af_features
from alphafold.model import model as af_model

from alphafold.evottt.ttt import make_ttt_apply, run_ttt


# ---------------------------------------------------------------------------
# Feature loading from precomputed A3M
# ---------------------------------------------------------------------------

def load_features_from_a3m(
    a3m_path: str,
    sequence: str,
    protein_id: str,
    model_config,
    random_seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Build processed feature dict from a precomputed A3M file.

    Parses the A3M, builds raw sequence + MSA features, then runs the
    AF2 TensorFlow feature pipeline (masking, clustering, padding, etc.).
    """
    with open(a3m_path, 'r') as f:
        a3m_string = f.read()
    msa = parsers.parse_a3m(a3m_string)

    num_res = len(sequence)
    raw_features: Dict[str, Any] = {}
    raw_features.update(
        pipeline.make_sequence_features(sequence, protein_id, num_res)
    )
    raw_features.update(pipeline.make_msa_features([msa]))

    # Add empty template features (needed for template-using models like
    # model_1_ptm / model_2_ptm).  Matches the pattern in templates.py.
    raw_features['template_aatype'] = np.zeros(
        (1, num_res, len(residue_constants.restypes_with_x_and_gap)),
        np.float32,
    )
    raw_features['template_all_atom_masks'] = np.zeros(
        (1, num_res, residue_constants.atom_type_num), np.float32,
    )
    raw_features['template_all_atom_positions'] = np.zeros(
        (1, num_res, residue_constants.atom_type_num, 3), np.float32,
    )
    raw_features['template_domain_names'] = np.array(
        [''.encode()], dtype=object,
    )
    raw_features['template_sequence'] = np.array(
        [''.encode()], dtype=object,
    )
    raw_features['template_sum_probs'] = np.array([0], dtype=np.float32)

    processed = af_features.np_example_to_features(
        np_example=raw_features,
        config=model_config,
        random_seed=random_seed,
    )
    return processed


def compute_mean_plddt(prediction_result: Dict[str, Any]) -> float:
    plddt = confidence.compute_plddt(
        prediction_result['predicted_lddt']['logits']
    )
    return float(np.mean(plddt))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    default_csv = Path(
        '/scratch/project/open-35-8/pimenol1/ProteinTTT/'
        'ProteinTTT_fresh/data/benchmark/summary.csv'
    )
    default_msa_dir = Path('/scratch/project/open-35-8/antonb/bfvd/bfvd_msa')
    default_data_dir = Path('/scratch/project/open-35-8/pimenol1/af2_data')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark_csv', type=Path,
                        default=default_csv if default_csv.is_file() else None)
    parser.add_argument('--msa_dir', type=Path, default=default_msa_dir)
    parser.add_argument('--data_dir', type=Path, default=default_data_dir)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--model_name', default='model_1_ptm')
    # TTT hyperparameters
    parser.add_argument('--ttt_steps', type=int, default=50)
    parser.add_argument('--ttt_lr', type=float, default=1e-4)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--last_n_blocks', type=int, default=48)
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--mask_fraction', type=float, default=0.15)
    # Subset selection
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--protein_ids', default=None,
                        help='Comma-separated protein IDs to process.')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline prediction (useful when re-running TTT only).')
    args = parser.parse_args()

    if args.benchmark_csv is None:
        print('Provide --benchmark_csv')
        return 1

    # ---- read CSV -----------------------------------------------------------
    rows = []
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

    # ---- setup model --------------------------------------------------------
    print(f'Model: {args.model_name}')
    print(f'TTT: steps={args.ttt_steps} lr={args.ttt_lr} rank={args.lora_rank} '
          f'blocks={args.last_n_blocks} alpha={args.lora_alpha}')

    baseline_config = af_config.model_config(args.model_name)
    base_params = af_data.get_model_haiku_params(
        args.model_name, str(args.data_dir)
    )

    # TTT config: no recycling, only masked_msa head
    ttt_config = af_config.model_config(args.model_name)
    with ttt_config.unlocked():
        ttt_config.model.num_recycle = 0
        ttt_config.data.common.num_recycle = 0

    ttt_apply = make_ttt_apply(ttt_config.model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, row in enumerate(rows):
        pid = row['id'].strip()
        seq = row['sequence'].strip()
        a3m_path = args.msa_dir / f'{pid}.a3m'

        if not a3m_path.exists():
            print(f'[{i}] {pid}: MSA not found at {a3m_path}, skipping')
            continue

        protein_dir = args.output_dir / pid
        result_path = protein_dir / 'evottt_result.json'
        if args.skip_existing and result_path.exists():
            print(f'[{i}] {pid}: already done, skipping')
            continue

        print(f'\n[{i}] {pid} (len={len(seq)}, nmsa={row.get("nmsa", "?")}) ...')

        # ---- baseline prediction -------------------------------------------
        baseline_plddt = None
        if not args.skip_baseline:
            try:
                baseline_features = load_features_from_a3m(
                    str(a3m_path), seq, pid, baseline_config,
                    random_seed=args.seed,
                )
                baseline_runner = af_model.RunModel(
                    baseline_config, params=base_params
                )
                t0 = time.time()
                baseline_result = baseline_runner.predict(
                    baseline_features, random_seed=args.seed
                )
                baseline_time = time.time() - t0
                baseline_plddt = compute_mean_plddt(baseline_result)
                print(f'  Baseline pLDDT: {baseline_plddt:.2f}  ({baseline_time:.1f}s)')
            except Exception as e:
                print(f'  Baseline failed: {e}')
                continue

        # ---- TTT adaptation ------------------------------------------------
        try:
            ttt_features = load_features_from_a3m(
                str(a3m_path), seq, pid, ttt_config,
                random_seed=args.seed,
            )

            t0 = time.time()
            adapted_params, ttt_losses = run_ttt(
                apply_fn=ttt_apply,
                base_params=base_params,
                batch=ttt_features,
                num_steps=args.ttt_steps,
                learning_rate=args.ttt_lr,
                rank=args.lora_rank,
                last_n_blocks=args.last_n_blocks,
                alpha=args.lora_alpha,
                grad_clip_norm=args.grad_clip,
                replace_fraction=args.mask_fraction,
                seed=args.seed,
            )
            ttt_time = time.time() - t0
            print(f'  TTT: {args.ttt_steps} steps in {ttt_time:.1f}s, '
                  f'loss {ttt_losses[0]:.4f} → {ttt_losses[-1]:.4f}')
        except Exception as e:
            print(f'  TTT failed: {e}')
            continue

        # ---- adapted prediction (full model with recycling) ----------------
        try:
            adapted_features = load_features_from_a3m(
                str(a3m_path), seq, pid, baseline_config,
                random_seed=args.seed,
            )
            adapted_runner = af_model.RunModel(
                baseline_config, params=adapted_params
            )
            t0 = time.time()
            adapted_result = adapted_runner.predict(
                adapted_features, random_seed=args.seed
            )
            adapted_time = time.time() - t0
            adapted_plddt = compute_mean_plddt(adapted_result)
            print(f'  Adapted pLDDT: {adapted_plddt:.2f}  ({adapted_time:.1f}s)')
            if baseline_plddt is not None:
                print(f'  Δ pLDDT: {adapted_plddt - baseline_plddt:+.2f}')
        except Exception as e:
            print(f'  Adapted prediction failed: {e}')
            continue

        # ---- save results --------------------------------------------------
        result = {
            'id': pid,
            'length': len(seq),
            'nmsa': int(row.get('nmsa', 0)),
            'baseline_plddt': baseline_plddt,
            'adapted_plddt': adapted_plddt,
            'delta_plddt': (adapted_plddt - baseline_plddt
                            if baseline_plddt is not None else None),
            'ttt_steps': args.ttt_steps,
            'ttt_lr': args.ttt_lr,
            'lora_rank': args.lora_rank,
            'last_n_blocks': args.last_n_blocks,
            'lora_alpha': args.lora_alpha,
            'ttt_loss_start': ttt_losses[0],
            'ttt_loss_end': ttt_losses[-1],
            'ttt_losses': ttt_losses,
            'ttt_time_s': ttt_time,
        }
        results.append(result)

        protein_dir.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

    # ---- summary -----------------------------------------------------------
    if results:
        summary_path = args.output_dir / 'evottt_summary.csv'
        fieldnames = [
            'id', 'length', 'nmsa', 'baseline_plddt', 'adapted_plddt',
            'delta_plddt', 'ttt_loss_start', 'ttt_loss_end', 'ttt_time_s',
        ]
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                     extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)

        deltas = [r['delta_plddt'] for r in results if r['delta_plddt'] is not None]
        if deltas:
            print(f'\n=== Summary ({len(deltas)} proteins) ===')
            print(f'Mean Δ pLDDT:   {np.mean(deltas):+.2f}')
            print(f'Median Δ pLDDT: {np.median(deltas):+.2f}')
            print(f'Min / Max:      {np.min(deltas):+.2f} / {np.max(deltas):+.2f}')
        print(f'Results saved to {summary_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
