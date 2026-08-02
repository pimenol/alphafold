#!/usr/bin/env python3
"""Runs one test-time-training experiment over a subset of the lowconf set.

For each target: load the precomputed MSA, take the stock AlphaFold parameters,
run N unsupervised gradient steps on them, and predict again.  Nothing in the
loop reads the experimental structure -- `ttt.core.forbid_ground_truth` makes
that a crash rather than a silent invalidation.

Two predictions come out per target: the parameters after the last step, and the
parameters at the step with the highest pLDDT, so it stays visible whether
pLDDT-based step selection actually helps.  Both are produced by the *same*
forward pass the M0 baseline used -- 3 recycles, ensembling on, seed 0 -- so a
Delta only ever reflects the parameters.

Usage:
  python3 -m ttt.run_ttt --name exp001-violation --loss violation \\
      --subset subsets/lowconf5.txt --steps 10 --lr 1e-4
"""

import argparse
import functools
import json
import os
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO, os.path.join(REPO, 'scripts')):
  if path not in sys.path:
    sys.path.insert(0, path)

CSV_PATH = os.path.join(REPO, 'af2_lowconf.csv')
MSA_DIR = os.path.join(REPO, 'data', 'lowconf', 'msa')
NATIVE_DIR = os.path.join(REPO, 'data', 'lowconf', 'pdbs')
# Device-matched: predictions/lowconf was produced on an A100, and AlphaFold does
# not reproduce across devices (up to 14 lDDT points on this set), so a CPU run
# must measure its drift against the CPU baseline or the number is meaningless.
BASELINE_DIR = os.path.join(
    REPO, 'predictions',
    'lowconf_cpu' if os.environ.get('JAX_PLATFORMS') == 'cpu' else 'lowconf')
LOG_DIR = os.path.join(REPO, 'logs')

# Floor for the fitted MSA padding. Attention over a single row is degenerate in
# places, so never shrink below this even for a one-sequence alignment.
MIN_MSA_ROWS = 8


def read_subset(path: str) -> list[str]:
  """Reads ids from a subset file, ignoring comments and inline annotations."""
  ids = []
  with open(path) as handle:
    for line in handle:
      token = line.split('#', 1)[0].strip()
      if token:
        ids.append(token)
  return ids


def ca_coords(pdb_path: str) -> np.ndarray:
  """CA coordinates of a PDB, for the "did the structure move at all" check."""
  coords = []
  with open(pdb_path) as handle:
    for line in handle:
      if line.startswith('ATOM') and line[12:16].strip() == 'CA':
        coords.append([float(line[30 + 8 * i:38 + 8 * i]) for i in range(3)])
  return np.array(coords)


def superposed_rmsd(a: np.ndarray, b: np.ndarray) -> float:
  """CA RMSD after optimal superposition (Kabsch).

  AlphaFold's output frame is arbitrary, so a raw coordinate RMSD between two
  predictions mixes a global rotation in with any real conformational change --
  it can read tens of angstroms for structures that are essentially identical.
  Superposing first makes the number mean "how much did the fold change".
  """
  a = a - a.mean(0)
  b = b - b.mean(0)
  v, _, w = np.linalg.svd(a.T @ b)
  if np.linalg.det(v) * np.linalg.det(w) < 0:
    v[:, -1] *= -1
  rotated = a @ (v @ w)
  return float(np.sqrt(np.mean(np.sum((rotated - b) ** 2, axis=-1))))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--name', required=True,
                      help='experiment name; used for log and output paths')
  parser.add_argument('--description', default='',
                      help='one line describing what this experiment tests')
  parser.add_argument('--loss', required=True,
                      help='comma-separated loss terms, optionally weighted as '
                           'name:weight (e.g. "violation" or "violation:1,entropy:0.1")')
  parser.add_argument('--subset', default=os.path.join(REPO, 'subsets', 'lowconf5.txt'))
  parser.add_argument('--only', nargs='*', default=None)
  parser.add_argument('--steps', type=int, default=10)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--clip_norm', type=float, default=0.1,
                      help='global-norm gradient clip; AF2 paper value')
  parser.add_argument('--blocks', type=int, default=None,
                      help='train only the last N Evoformer blocks (idea 8)')
  parser.add_argument('--model_name', default='model_1_ptm')
  parser.add_argument('--random_seed', type=int, default=0)
  parser.add_argument('--max_msa', type=int, default=8192)
  parser.add_argument('--params_dir',
                      default='/scratch/project/open-37-88/pimenol/af2ttt/af2params')
  parser.add_argument('--out_dir', default=None)
  parser.add_argument('--fit_msa_padding', action='store_true',
                      help='shrink the TTT-only MSA padding to the number of '
                           'sequences the target actually has. Most of the hard '
                           'set is effectively single-sequence, so the default '
                           '508 cluster / 5120 extra rows are almost all mask. '
                           'Does not touch the eval config.')
  parser.add_argument('--time_budget', type=float, default=None,
                      help='minutes; stop starting new targets past this, so a '
                           'short allocation ends cleanly instead of being '
                           'killed mid-target')
  args = parser.parse_args()
  started_all = time.time()

  out_dir = args.out_dir or os.path.join(REPO, 'predictions', 'ttt', args.name)
  os.makedirs(out_dir, exist_ok=True)
  os.makedirs(LOG_DIR, exist_ok=True)
  log_path = os.path.join(LOG_DIR, f'{args.name}.log')
  log_file = open(log_path, 'w')

  def log(message: str = '') -> None:
    print(message, flush=True)
    log_file.write(message + '\n')
    log_file.flush()

  import csv

  import jax
  import jax.numpy as jnp

  from alphafold.common import protein
  from alphafold.model import config as model_config
  from alphafold.model import data as model_data
  from alphafold.model import features as model_features
  from alphafold.model import model as model_lib
  import haiku as hk

  from alphafold.model import modules
  from ttt import core

  import run_af2_on_dataset as runner_lib

  weights: dict[str, float] = {}
  for term in args.loss.split(','):
    name, _, weight = term.partition(':')
    if name not in core.LOSSES:
      raise SystemExit(f'unknown loss {name!r}; have {sorted(core.LOSSES)}')
    weights[name] = float(weight) if weight else 1.0

  log(f'# {args.name}')
  log(f'# {args.description}' if args.description else
      '# (no description given)')
  log('#')
  log(f'# loss          : {args.loss}')
  log(f'# steps         : {args.steps}   lr: {args.lr}   clip: {args.clip_norm}')
  log(f'# trainable     : {"last %d evoformer blocks" % args.blocks if args.blocks else "all params"}'
      f' minus {core.FROZEN_SUBSTRINGS}')
  log(f'# model / seed  : {args.model_name} / {args.random_seed}')
  log(f'# TTT forward   : num_recycle=0 (hk.while_loop has no reverse-mode rule)')
  log(f'# eval forward  : identical to M0 (num_recycle=3, ensembling, seed 0)')
  log(f'# started       : {time.strftime("%Y-%m-%d %H:%M:%S")}')
  log(f'# devices       : {jax.devices()}')
  log('')

  # Two configs.  The eval one is byte-for-byte what M0 used; the TTT one drops
  # recycling so the forward pass is differentiable, and turns on rematerialised
  # Evoformer blocks so the backward pass fits in 40 GB.
  eval_cfg = model_config.model_config(args.model_name)
  ttt_cfg = model_config.model_config(args.model_name)
  ttt_cfg.data.common.num_recycle = 0
  ttt_cfg.model.num_recycle = 0
  ttt_cfg.model.global_config.use_remat = True
  ttt_cfg.model.global_config.deterministic = True

  params = model_data.get_model_haiku_params(model_name=args.model_name,
                                             data_dir=args.params_dir)
  params = hk.data_structures.to_mutable_dict(params)
  eval_runner = model_lib.RunModel(eval_cfg, params)

  def ttt_forward(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    model = modules.AlphaFold(ttt_cfg.model)
    return model(batch, is_training=False, compute_loss=False,
                 ensemble_representations=False)

  ttt_apply = hk.transform(ttt_forward).apply
  sm_cfg = ttt_cfg.model.heads.structure_module

  def loss_fn(trainable, frozen, rng, batch):
    merged = {**frozen, **trainable}
    out = ttt_apply(merged, rng, batch)
    batch0 = jax.tree.map(lambda x: x[0], batch)
    total = jnp.zeros(())
    aux = {}
    for name, weight in weights.items():
      value, extra = core.LOSSES[name](out, batch0, sm_cfg)
      total = total + weight * value
      aux.update(extra)
    aux['plddt'] = core.plddt_from_output(out)
    return total, aux

  grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

  with open(CSV_PATH, newline='') as handle:
    rows = {r['id']: r for r in csv.DictReader(handle)}
  ids = args.only if args.only else read_subset(args.subset)
  missing = [i for i in ids if i not in rows]
  if missing:
    raise SystemExit(f'ids not in {CSV_PATH}: {missing}')
  ids.sort(key=lambda i: int(rows[i]['length']))
  log(f'{len(ids)} targets: {" ".join(ids)}')
  log('')

  summary_path = os.path.join(out_dir, 'ttt_summary.json')

  def save(results: Mapping[str, Any]) -> None:
    """Persists after every target.

    GPU allocations on this cluster are scarce and short; a run killed at the
    walltime must still leave the targets it finished, not nothing.
    """
    with open(summary_path, 'w') as handle:
      json.dump({'args': vars(args), 'results': results}, handle, indent=2)

  results: dict[str, Any] = {}
  deadline = started_all + args.time_budget * 60 if args.time_budget else None
  for position, target in enumerate(ids, start=1):
    if deadline is not None and time.time() > deadline:
      log(f'!!! time budget of {args.time_budget} min reached; '
          f'stopping before {target}. {len(results)}/{len(ids)} targets done.')
      break
    row = rows[target]
    started = time.time()
    log(f'=== [{position}/{len(ids)}] {target}  '
        f'len={row["length"]}  depth={row["msa_depth"]}  neff={float(row["msa_neff"]):.0f}')

    features, used = runner_lib.build_features(
        target, row['sequence'], os.path.join(MSA_DIR, f'{target}.a3m'),
        args.max_msa)
    eval_feat = model_features.np_example_to_features(
        np_example=dict(features), config=eval_cfg,
        random_seed=args.random_seed)

    if args.fit_msa_padding:
      # The padded rows are masked out, so dropping them leaves the objective
      # unchanged while removing most of the work. Only the TTT config is
      # touched; eval stays byte-identical to M0.
      clusters = min(eval_cfg.data.eval.max_msa_clusters,
                     max(MIN_MSA_ROWS, used + eval_cfg.data.eval.max_templates))
      extra = min(eval_cfg.data.common.max_extra_msa,
                  max(MIN_MSA_ROWS, used - clusters + MIN_MSA_ROWS))
      ttt_cfg.data.eval.max_msa_clusters = clusters
      ttt_cfg.data.common.max_extra_msa = extra
      log(f'    msa padding fitted: clusters {clusters} '
          f'(from {eval_cfg.data.eval.max_msa_clusters}), extra {extra} '
          f'(from {eval_cfg.data.common.max_extra_msa})')

    ttt_feat = model_features.np_example_to_features(
        np_example=dict(features), config=ttt_cfg,
        random_seed=args.random_seed)
    ttt_batch = jax.tree.map(jnp.asarray, dict(ttt_feat))

    trainable, frozen = core.split_trainable(params)
    mask = core.block_mask(trainable, args.blocks)
    n_train = core.count_params(trainable)
    n_active = sum(int(np.asarray(m).sum())
                   for mod in mask.values() for m in mod.values())
    log(f'    msa rows used {used}; trainable {n_train / 1e6:.1f}M params in '
        f'{len(trainable)} modules, {n_active / 1e6:.1f}M active after mask')

    opt_state = core.adam_init(trainable)
    rng = jax.random.PRNGKey(args.random_seed)
    history = []
    best = {'step': 0, 'plddt': -1.0, 'params': None}

    # Everything from here to the end of the loop is unsupervised by
    # construction; the guard makes an accidental native read fail loudly.
    with core.forbid_ground_truth(NATIVE_DIR):
      for step in range(args.steps + 1):
        rng, step_rng = jax.random.split(rng)
        (loss, aux), grads = grad_fn(trainable, frozen, step_rng, ttt_batch)
        loss = float(loss)
        plddt = float(aux['plddt'])
        if plddt > best['plddt']:
          best = {'step': step, 'plddt': plddt,
                  'params': jax.tree.map(np.array, trainable)}
        record = {'step': step, 'loss': loss, 'plddt': plddt}
        record.update({k: float(v) for k, v in aux.items() if k != 'plddt'})
        history.append(record)
        extras = '  '.join(f'{k}={v:.4f}' for k, v in record.items()
                           if k not in ('step', 'loss', 'plddt'))
        log(f'    step {step:2d}  loss={loss:.5f}  plddt={plddt:6.2f}  {extras}')
        if step == args.steps:
          break
        trainable, opt_state, gnorm = core.adam_update(
            trainable, grads, opt_state, args.lr, args.clip_norm, mask)
        history[-1]['grad_norm'] = float(gnorm)

    def predict_and_write(trained, suffix: str) -> dict[str, float]:
      merged = {**frozen, **trained}
      result = eval_runner.apply(merged, jax.random.PRNGKey(args.random_seed),
                                 eval_feat)
      result = jax.tree.map(np.array, result)
      plddt_res = result['plddt'] if 'plddt' in result else None
      if plddt_res is None:
        from alphafold.common import confidence
        plddt_res = confidence.compute_plddt(
            result['predicted_lddt']['logits'])
      from alphafold.common import residue_constants
      b_factors = np.repeat(plddt_res[:, None],
                            residue_constants.atom_type_num, axis=-1)
      unrelaxed = protein.from_prediction(
          eval_feat, {**result, 'plddt': plddt_res}, b_factors=b_factors,
          remove_leading_feature_dimension=True)
      path = os.path.join(out_dir, f'{target}_{suffix}.pdb')
      with open(path, 'w') as handle:
        handle.write(protein.to_pdb(unrelaxed))
      np.save(os.path.join(out_dir, f'{target}_{suffix}_plddt.npy'), plddt_res)
      return {'path': path, 'plddt': float(np.mean(plddt_res))}

    final = predict_and_write(trainable, 'step%d' % args.steps)
    if best['step'] in (args.steps,):
      chosen = dict(final)
      chosen['path'] = final['path']
    else:
      chosen = predict_and_write(best['params'], 'bestplddt')

    baseline_pdb = os.path.join(BASELINE_DIR, f'{target}.pdb')
    moved = moved_raw = None
    if os.path.exists(baseline_pdb):
      base_ca, new_ca = ca_coords(baseline_pdb), ca_coords(final['path'])
      if base_ca.shape == new_ca.shape and base_ca.size:
        moved_raw = float(np.sqrt(np.mean(np.sum((base_ca - new_ca) ** 2, -1))))
        moved = superposed_rmsd(base_ca, new_ca)

    elapsed = time.time() - started
    results[target] = {
        'length': int(row['length']),
        'msa_depth': int(row['msa_depth']),
        'msa_neff': float(row['msa_neff']),
        'history': history,
        'final_eval_plddt': final['plddt'],
        'best_step': best['step'],
        'best_step_eval_plddt': chosen['plddt'],
        'ca_rmsd_vs_baseline': moved,
        'ca_rmsd_vs_baseline_unaligned': moved_raw,
        'seconds': elapsed,
    }
    log(f'    eval pLDDT: baseline-step0 -> step{args.steps} = '
        f'{final["plddt"]:.2f}   best-pLDDT step {best["step"]} = '
        f'{chosen["plddt"]:.2f}')
    rmsd_text = ('n/a' if moved is None else
                 f'{moved:.3f} A superposed, {moved_raw:.3f} A raw')
    log(f'    CA RMSD vs M0 baseline: {rmsd_text}   ({elapsed:.0f}s)')
    log('')
    save(results)

  save(results)
  log(f'wrote {summary_path} ({len(results)}/{len(ids)} targets)')
  log(f'finished {time.strftime("%Y-%m-%d %H:%M:%S")}')
  log_file.close()
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
