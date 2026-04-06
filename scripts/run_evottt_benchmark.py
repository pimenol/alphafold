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

import os
# Must be set before JAX/XLA is imported to avoid Triton GEMM autotuner crash
# when recompiling for different input shapes.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
_xla = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_enable_triton_gemm' not in _xla:
    os.environ['XLA_FLAGS'] = f'{_xla} --xla_gpu_enable_triton_gemm=false'.strip()

import argparse
import csv
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import numpy as np

print(f'[init] jax/numpy imported, jax.devices={jax.devices()}', flush=True)

from alphafold.common import confidence, protein, residue_constants
from alphafold.data import parsers, pipeline
from alphafold.model import config as af_config
from alphafold.model import data as af_data
from alphafold.model import features as af_features
from alphafold.model import model as af_model

from alphafold.data.tools import jackhmmer as jackhmmer_tool
from alphafold.evottt.ttt import compute_prev_features, make_ttt_apply, run_ttt
print('[init] all imports done', flush=True)


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


def save_prediction_pdb(
    prediction_result: Dict[str, Any],
    features: Dict[str, np.ndarray],
    pdb_path: str,
) -> None:
    """Build a Protein from prediction result and write it as PDB."""
    plddt = confidence.compute_plddt(
        prediction_result['predicted_lddt']['logits']
    )
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1
    )
    prot = protein.from_prediction(
        features=features,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=True,
    )
    Path(pdb_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pdb_path, 'w') as f:
        f.write(protein.to_pdb(prot))


# ---------------------------------------------------------------------------
# MSA generation from single sequence (using AF2's JackHMMER)
# ---------------------------------------------------------------------------

def generate_msa(
    sequence: str,
    protein_id: str,
    output_a3m_path: str,
    jackhmmer_binary_path: str,
    database_path: str,
    n_cpu: int = 8,
    max_sto_sequences: Optional[int] = 10000,
) -> str:
    """Run JackHMMER on a single sequence to generate an MSA, saved as A3M.

    Uses AF2's built-in JackHMMER wrapper and parsers.  The result is
    cached at *output_a3m_path* so subsequent runs skip the search.

    Returns the path to the generated A3M file.
    """
    if Path(output_a3m_path).exists():
        print(f'  MSA already cached at {output_a3m_path}')
        return output_a3m_path

    runner = jackhmmer_tool.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=database_path,
        n_cpu=n_cpu,
    )

    # Write a temporary FASTA for JackHMMER
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.fasta', delete=False
    ) as fasta_fh:
        fasta_fh.write(f'>{protein_id}\n{sequence}\n')
        fasta_path = fasta_fh.name

    try:
        result = runner.query(fasta_path, max_sto_sequences)[0]
    finally:
        Path(fasta_path).unlink(missing_ok=True)

    # Convert Stockholm → A3M using AF2's parser
    sto_string = result['sto']
    a3m_string = parsers.convert_stockholm_to_a3m(sto_string)

    Path(output_a3m_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_a3m_path, 'w') as f:
        f.write(a3m_string)

    msa = parsers.parse_a3m(a3m_string)
    print(f'  Generated MSA: {len(msa.sequences)} sequences → {output_a3m_path}')
    return output_a3m_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO,
    )

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
    parser.add_argument('--ttt_lr', type=float, default=3e-4)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--last_n_blocks', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=1.0)

    parser.add_argument('--optimizer', default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer for TTT.')
    parser.add_argument('--lora_triangle_attention', action='store_true',
                        help='Exp 1: also apply LoRA to triangle attention '
                             '(pair representation), not just MSA attention.')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Exp 5: accumulate gradients over this many MSA '
                             'subsamples before each optimizer update.')
    parser.add_argument('--block_mask', action='store_true',
                        help='Exp 8: mask entire residue columns across all '
                             'sequences instead of independent per-position masking.')
    parser.add_argument('--ttt_recycle_prev', action='store_true',
                        help='Run a full AF2 forward pass (with recycling) '
                             'before TTT and use its representations as '
                             'frozen prev conditioning for the Evoformer.')
    parser.add_argument('--mask_fraction', type=float, default=0.15)
    parser.add_argument('--ttt_msa_clusters', type=int, default=None,
                        help='Subsample this many MSA rows per TTT step. '
                             'Each step gets a different random subset from '
                             'the full MSA pool. None = use all.')
    parser.add_argument('--ttt_crop_size', type=int, default=None,
                        help='Crop residues to this size during TTT steps. '
                             'Proteins shorter than this are not cropped.')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Evaluate pLDDT every N TTT steps (0 to disable).')
    # MSA generation (used when precomputed A3M is missing)
    parser.add_argument('--jackhmmer_binary_path', default=None,
                        help='Path to jackhmmer binary. Enables automatic MSA '
                             'generation when a precomputed A3M is not found.')
    parser.add_argument('--seq_database_path', default=None,
                        help='Path to sequence database for JackHMMER '
                             '(e.g. uniref90.fasta or small_bfd).')
    parser.add_argument('--msa_n_cpu', type=int, default=8,
                        help='CPUs for JackHMMER MSA search.')
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
          f'blocks={args.last_n_blocks} alpha={args.lora_alpha} '
          f'optimizer={args.optimizer} crop={args.ttt_crop_size or "none"}')

    baseline_config = af_config.model_config(args.model_name)
    base_params = af_data.get_model_haiku_params(
        args.model_name, str(args.data_dir)
    )

    # TTT config: no recycling, aggressively reduced MSA, per-block remat
    ttt_config = af_config.model_config(args.model_name)
    with ttt_config.unlocked():
        ttt_config.model.num_recycle = 0
        ttt_config.data.common.num_recycle = 0
        # MSA pool size for TTT. When subsampling per step, keep a large
        # pool so each step sees different sequences; otherwise use a small
        # fixed set for memory efficiency.
        if args.ttt_msa_clusters is not None:
            # Large pool; per-step subsampling in run_ttt handles the rest
            ttt_config.data.eval.max_msa_clusters = 512
        else:
            ttt_config.data.eval.max_msa_clusters = 32
        ttt_config.data.common.max_extra_msa = 128
        # Enable per-block gradient checkpointing (remat) inside Evoformer
        ttt_config.model.global_config.use_remat = True
        # Crop residues during TTT to cap quadratic attention cost
        if args.ttt_crop_size is not None:
            ttt_config.data.eval.crop_size = args.ttt_crop_size

    ttt_apply = make_ttt_apply(ttt_config.model)

    # Eval config: full model (all heads) but no recycling for speed
    eval_config = None
    eval_runner = None
    if args.eval_interval > 0:
        eval_config = af_config.model_config(args.model_name)
        with eval_config.unlocked():
            eval_config.model.num_recycle = 0
            eval_config.data.common.num_recycle = 0
        eval_runner = af_model.RunModel(eval_config, params=base_params)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.output_dir / 'config.json'
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Config saved to {config_path}')

    results = []

    for i, row in enumerate(rows):
        pid = row['id'].strip()
        seq = row['sequence'].strip()
        a3m_path = args.msa_dir / f'{pid}.a3m'

        if not a3m_path.exists():
            if args.jackhmmer_binary_path and args.seq_database_path:
                print(f'[{i}] {pid}: No precomputed A3M, running MSA search...')
                try:
                    a3m_path = Path(generate_msa(
                        sequence=seq,
                        protein_id=pid,
                        output_a3m_path=str(args.msa_dir / f'{pid}.a3m'),
                        jackhmmer_binary_path=args.jackhmmer_binary_path,
                        database_path=args.seq_database_path,
                        n_cpu=args.msa_n_cpu,
                    ))
                except Exception as e:
                    print(f'[{i}] {pid}: MSA generation failed: {e}')
                    continue
            else:
                print(f'[{i}] {pid}: MSA not found at {a3m_path}, skipping '
                      '(use --jackhmmer_binary_path and --seq_database_path '
                      'to enable automatic MSA generation)')
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
                save_prediction_pdb(
                    baseline_result, baseline_features,
                    str(protein_dir / 'baseline.pdb'),
                )
            except Exception as e:
                print(f'  Baseline failed: {e}')
                # Use CSV pLDDT if available, keep going with TTT
                csv_plddt = row.get('pLDDT_AlphaFold')
                if csv_plddt:
                    baseline_plddt = float(csv_plddt)
                    print(f'  Using CSV baseline pLDDT: {baseline_plddt:.2f}')

        # ---- TTT adaptation ------------------------------------------------
        try:
            ttt_features = load_features_from_a3m(
                str(a3m_path), seq, pid, ttt_config,
                random_seed=args.seed,
            )

            # Build per-step pLDDT eval function that also saves PDBs
            eval_fn = None
            eval_pdb_paths: List[str] = []  # one path per eval_fn call
            if eval_runner is not None:
                eval_features = load_features_from_a3m(
                    str(a3m_path), seq, pid, eval_config,
                    random_seed=args.seed,
                )
                _runner = eval_runner
                _feat = eval_features
                _seed = args.seed
                _pdb_dir = protein_dir / 'steps'
                _pdb_list = eval_pdb_paths
                def eval_fn(adapted_params):
                    result = _runner.apply(
                        adapted_params,
                        jax.random.PRNGKey(_seed),
                        _feat,
                    )
                    jax.tree.map(lambda x: x.block_until_ready(), result)
                    plddt = confidence.compute_plddt(
                        result['predicted_lddt']['logits']
                    )
                    mean_plddt = float(np.mean(plddt))
                    # Save PDB for this evaluation step
                    call_idx = len(_pdb_list)
                    pdb_path = str(_pdb_dir / f'eval_{call_idx:04d}.pdb')
                    save_prediction_pdb(result, _feat, pdb_path)
                    _pdb_list.append(pdb_path)
                    return {'plddt': mean_plddt}

            # Optionally compute frozen prev conditioning from base model
            prev = None
            if args.ttt_recycle_prev:
                t_prev = time.time()
                prev = compute_prev_features(
                    baseline_config.model, base_params, ttt_features,
                    seed=args.seed,
                )
                print(f'  Computed prev features in {time.time() - t_prev:.1f}s')

            t0 = time.time()
            adapted_params, ttt_losses, eval_logs, best_step = run_ttt(
                apply_fn=ttt_apply,
                base_params=base_params,
                model_config=ttt_config.model,
                batch=ttt_features,
                num_steps=args.ttt_steps,
                learning_rate=args.ttt_lr,
                rank=args.lora_rank,
                last_n_blocks=args.last_n_blocks,
                alpha=args.lora_alpha,

                replace_fraction=args.mask_fraction,
                msa_sample_size=args.ttt_msa_clusters,
                seed=args.seed,
                eval_fn=eval_fn,
                eval_interval=args.eval_interval,
                optimizer_name=args.optimizer,
                lora_triangle_attention=args.lora_triangle_attention,
                grad_accum_steps=args.grad_accum_steps,
                block_mask=args.block_mask,
                prev=prev,
            )
            ttt_time = time.time() - t0
            best_info = f', best_step={best_step}' if best_step >= 0 else ''
            print(f'  TTT: {args.ttt_steps} steps in {ttt_time:.1f}s, '
                  f'loss {ttt_losses[0]:.4f} → {ttt_losses[-1]:.4f}'
                  f'{best_info}')
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
            save_prediction_pdb(
                adapted_result, adapted_features,
                str(protein_dir / 'adapted.pdb'),
            )
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
            'best_step': best_step,
            'eval_logs': eval_logs,
        }
        results.append(result)

        protein_dir.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        # ---- per-step CSV log ----------------------------------------------
        # Columns: step_num, loss, plddt, pdb
        # eval_logs and eval_pdb_paths are aligned (one entry per eval call).
        log_csv_path = protein_dir / 'ttt_log.csv'
        with open(log_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step_num', 'loss', 'plddt', 'pdb'])
            for log_entry, pdb_path in zip(eval_logs, eval_pdb_paths):
                writer.writerow([
                    log_entry['step'],
                    log_entry['loss'],
                    log_entry['plddt'],
                    pdb_path,
                ])

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
