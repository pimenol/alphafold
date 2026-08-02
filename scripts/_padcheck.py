"""Does shrinking the masked MSA padding change the TTT objective?

Forward-only comparison of the violation loss, the distogram entropy and pLDDT
at full padding (508 clusters / 5120 extra) versus padding fitted to the number
of sequences the target actually has. The dropped rows are entirely mask, so the
two should agree to numerical noise; if they do not, the optimisation is
discarded.
"""
import os, sys, time, csv
REPO = '/scratch/project/open-37-88/pimenol/af2ttt/alphafold'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'scripts'))
import numpy as np, jax, jax.numpy as jnp, haiku as hk
from alphafold.model import config as mc, features as mf, modules, data as md
import run_af2_on_dataset as R
from ttt import core

TARGET = os.environ.get('PADCHECK_TARGET', '9SLR_A')
row = {r['id']: r for r in csv.DictReader(open(f'{REPO}/af2_lowconf.csv'))}[TARGET]
params = md.get_model_haiku_params(
    model_name='model_1_ptm',
    data_dir='/scratch/project/open-37-88/pimenol/af2ttt/af2params')
feats, used = R.build_features(
    TARGET, row['sequence'], f'{REPO}/data/lowconf/msa/{TARGET}.a3m', 8192)
print(f'{TARGET}: {used} real sequences, {row["length"]} residues')

def run(clusters, extra, label):
  cfg = mc.model_config('model_1_ptm')
  cfg.data.common.num_recycle = 0
  cfg.model.num_recycle = 0
  cfg.model.global_config.deterministic = True
  if clusters is not None:
    cfg.data.eval.max_msa_clusters = clusters
    cfg.data.common.max_extra_msa = extra
  feat = mf.np_example_to_features(dict(feats), cfg, 0)
  fn = hk.transform(lambda b: modules.AlphaFold(cfg.model)(
      b, is_training=False, compute_loss=False, ensemble_representations=False))
  batch = jax.tree.map(jnp.asarray, dict(feat))
  t = time.time()
  out = jax.jit(fn.apply)(params, jax.random.PRNGKey(0), batch)
  b0 = jax.tree.map(lambda x: x[0], batch)
  v, _ = core.violation_loss(out, b0, cfg.model.heads.structure_module)
  e, _ = core.distogram_entropy_loss(out, b0, None)
  p = core.plddt_from_output(out)
  pos = np.array(out['structure_module']['final_atom_positions'])
  dt = time.time() - t
  print(f'  {label:34} msa={feat["msa_feat"].shape[1]:4} extra={feat["extra_msa"].shape[1]:5}'
        f'  violation={float(v):.6f}  entropy={float(e):.6f}  plddt={float(p):.3f}'
        f'  [{dt:.0f}s incl. compile]')
  return float(v), float(e), float(p), pos

full = run(None, None, 'full padding (M0 sizes)')
clusters = min(512, max(8, used + 4))
extra = min(5120, max(8, used - clusters + 8))
fit = run(clusters, extra, f'fitted padding ({clusters}/{extra})')

print('\ndifferences:')
print(f'  violation  {abs(full[0]-fit[0]):.3e}')
print(f'  entropy    {abs(full[1]-fit[1]):.3e}')
print(f'  pLDDT      {abs(full[2]-fit[2]):.3e}')
rms = float(np.sqrt(np.mean(np.sum((full[3]-fit[3])**2, -1))))
print(f'  all-atom coordinate RMS {rms:.4f} A')
ok = abs(full[2]-fit[2]) < 0.5 and rms < 0.5
print('\nVERDICT:', 'equivalent -- safe to use' if ok else '*** NOT equivalent -- discard ***')
