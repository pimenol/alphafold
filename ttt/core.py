"""Unsupervised TTT losses, the optimiser, and the ground-truth guard.

Every loss here is a function of the model's own output only.  None of them
reads an experimental structure -- that is the project's first hard constraint,
and `forbid_ground_truth` below turns a violation into a crash rather than a
silently invalid result.

Why the TTT forward pass runs with num_recycle=0
------------------------------------------------
`modules.AlphaFold.__call__` drives recycling with `hk.while_loop`, which has no
reverse-mode transpose rule, so `jax.grad` through a recycling forward pass is
impossible without editing upstream code.  Setting num_recycle=0 skips the loop
entirely and leaves a single, fully differentiable AlphaFoldIteration.

The consequence is a deliberate train/eval split: TTT steps optimise a 0-recycle
objective, while every reported prediction is produced by the *unmodified* M0
forward pass (3 recycles, ensembling on).  Only the parameters differ between
baseline and TTT, never the evaluation path.
"""

import builtins
import os

import jax
import jax.numpy as jnp
import numpy as np

from alphafold.common import confidence
from alphafold.model import folding
from alphafold.model import utils


# ---------------------------------------------------------------------------
# ground-truth guard
# ---------------------------------------------------------------------------


class GroundTruthAccess(RuntimeError):
  """Raised when the TTT code path touches an experimental structure."""


class forbid_ground_truth:
  """Context manager that makes reading a native structure fail loudly.

  The TTT loop must not see the answer -- not in the loss, not in step
  selection.  Rather than trusting review to catch it, this wraps `open` and
  raises if anything under the natives directory is read while TTT is running.
  Scoring happens in a separate process, outside this guard.
  """

  def __init__(self, *forbidden_dirs):
    self._forbidden = [os.path.abspath(d) for d in forbidden_dirs]
    self._saved = None

  def __enter__(self):
    self._saved = builtins.open
    forbidden = self._forbidden

    def guarded_open(file, *args, **kwargs):
      try:
        path = os.path.abspath(os.fspath(file))
      except TypeError:  # a file descriptor, not a path
        return self._saved(file, *args, **kwargs)
      for directory in forbidden:
        if path == directory or path.startswith(directory + os.sep):
          raise GroundTruthAccess(
              f'TTT code tried to read ground truth: {path}. The TTT loop must '
              'be unsupervised; see CLAUDE.md, "no ground truth inside TTT".'
          )
      return self._saved(file, *args, **kwargs)

    builtins.open = guarded_open
    return self

  def __exit__(self, *exc):
    builtins.open = self._saved
    return False


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------


def _atom14_positions(out, batch):
  """Pulls atom14 coordinates out of the atom37 prediction.

  The structure module only returns `final_atom14_positions` when it is built
  with compute_loss=True, which inference never does.  atom14 is a subset of
  atom37 per residue, so the reverse gather is exact and differentiable.
  """
  atom37 = out['structure_module']['final_atom_positions']  # (N, 37, 3)
  atom14 = utils.batched_gather(
      atom37, batch['residx_atom14_to_atom37'], batch_dims=1
  )
  return atom14 * batch['atom14_atom_exists'][..., None].astype(atom14.dtype)


def violation_loss(out, batch, sm_config):
  """AlphaFold's own structural violation term: bonds, angles, clashes.

  Physics, not data -- so it is defined on every target regardless of MSA
  depth, which matters because 11 of the 25 lowconf chains are effectively
  single-sequence.  Computed by calling upstream `find_structural_violations`
  and `structural_violation_loss` so the semantics are AF2's, not ours.
  """
  atom14 = _atom14_positions(out, batch)
  violations = folding.find_structural_violations(batch, atom14, sm_config)
  ret = {'loss': jnp.zeros(())}
  folding.structural_violation_loss(
      ret, batch, {'violations': violations}, sm_config
  )
  aux = {
      'viol_bonds_c_n': violations['between_residues']['bonds_c_n_loss_mean'],
      'viol_clashes': violations['between_residues']['clashes_mean_loss'],
      'viol_residue_frac': jnp.mean(
          violations['total_per_residue_violations_mask']
      ),
  }
  return ret['loss'], aux


def distogram_entropy_loss(out, batch, sm_config):
  """Mean entropy of the predicted distance distribution over residue pairs.

  Sharpening the distogram is a pure confidence objective on the trunk, with no
  extra forward passes and no dependence on MSA depth.
  """
  del sm_config
  logits = out['distogram']['logits']  # (N, N, num_bins)
  log_p = jax.nn.log_softmax(logits, axis=-1)
  entropy = -jnp.sum(jnp.exp(log_p) * log_p, axis=-1)  # (N, N)
  seq_mask = batch['seq_mask']
  pair_mask = seq_mask[:, None] * seq_mask[None, :]
  loss = utils.mask_mean(mask=pair_mask, value=entropy)
  return loss, {'distogram_entropy': loss}


LOSSES = {
    'violation': violation_loss,
    'entropy': distogram_entropy_loss,
}


def plddt_from_output(out):
  """Mean pLDDT of a forward pass, as an unsupervised selection signal.

  `confidence.compute_plddt` converts to numpy internally, so it cannot run
  under a jax trace.  This is the same bin-centre expectation in jnp -- see
  alphafold/common/confidence.py:33.  It is used only for step selection inside
  the loop; every reported pLDDT still comes from the upstream function.
  """
  logits = out['predicted_lddt']['logits']
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = jnp.arange(0.5 * bin_width, 1.0, bin_width)
  probs = jax.nn.softmax(logits, axis=-1)
  return jnp.mean(jnp.sum(probs * bin_centers, axis=-1)) * 100


# ---------------------------------------------------------------------------
# optimiser
# ---------------------------------------------------------------------------
#
# Adam with the AlphaFold paper's settings (Suppl. 1.11): b1=0.9, b2=0.999,
# eps=1e-6, gradients clipped to global norm 0.1.  Written out here rather than
# pulling in optax, so the pinned jax/jaxlib/CUDA-plugin set in af2env is not
# disturbed by another dependency resolve.


def adam_init(params):
  zeros = lambda p: jnp.zeros_like(p)
  return {
      'step': jnp.zeros((), jnp.int32),
      'mu': jax.tree.map(zeros, params),
      'nu': jax.tree.map(zeros, params),
  }


def global_norm(tree):
  leaves = jax.tree.leaves(tree)
  return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def adam_update(params, grads, state, lr, clip_norm=0.1, mask=None,
                b1=0.9, b2=0.999, eps=1e-6):
  """One Adam step with global-norm gradient clipping.

  `mask` is an optional 0/1 tree from `block_mask`; it is applied before
  clipping so the clip norm is computed on the gradient that is actually used.
  """
  if mask is not None:
    grads = jax.tree.map(lambda g, m: g * m, grads, mask)
  norm = global_norm(grads)
  scale = jnp.minimum(1.0, clip_norm / (norm + 1e-9))
  grads = jax.tree.map(lambda g: g * scale, grads)

  step = state['step'] + 1
  mu = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state['mu'], grads)
  nu = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state['nu'], grads)
  mu_hat = jax.tree.map(lambda m: m / (1 - b1 ** step), mu)
  nu_hat = jax.tree.map(lambda v: v / (1 - b2 ** step), nu)
  params = jax.tree.map(
      lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps), params, mu_hat, nu_hat
  )
  return params, {'step': step, 'mu': mu, 'nu': nu}, norm


# ---------------------------------------------------------------------------
# parameter selection
# ---------------------------------------------------------------------------

# The pLDDT head is frozen: raising pLDDT by training the head that predicts it
# is not a result (CLAUDE.md, hard constraints).  None of the losses above reads
# that head, so its gradient is already zero -- this makes it structural rather
# than incidental.
FROZEN_SUBSTRINGS = ('predicted_lddt_head',)

# The Evoformer trunk is one hk.scan-stacked module, not 48 separate ones: every
# array under evoformer_iteration carries a leading block axis of length 48.
# Restricting TTT to the last N blocks (method idea 8) is therefore a mask along
# axis 0, not a choice of modules.
EVOFORMER_STACK = 'evoformer_iteration'
NUM_EVOFORMER_BLOCKS = 48


def split_trainable(params):
  """Partitions haiku params into (trainable, frozen) by module name."""
  trainable, frozen = {}, {}
  for module, values in params.items():
    target = frozen if any(s in module for s in FROZEN_SUBSTRINGS) else trainable
    target[module] = values
  return trainable, frozen


def block_mask(params, blocks=None):
  """Gradient mask restricting Evoformer updates to the last `blocks` blocks.

  Returns a tree of 0/1 multipliers shaped like `params`.  None disables the
  restriction (everything trains).  Parameters outside the stacked trunk -- the
  embeddings, the extra-MSA stack, the structure module -- are unaffected, so
  this isolates the trunk-depth axis rather than shrinking the update globally.
  """
  def mask_for(module, name, value):
    del name
    if blocks is None or EVOFORMER_STACK not in module:
      return jnp.ones_like(value)
    if value.shape[0] != NUM_EVOFORMER_BLOCKS:
      return jnp.ones_like(value)
    keep = jnp.arange(NUM_EVOFORMER_BLOCKS) >= NUM_EVOFORMER_BLOCKS - blocks
    return jnp.broadcast_to(
        keep.reshape((-1,) + (1,) * (value.ndim - 1)), value.shape
    ).astype(value.dtype)

  return {module: {name: mask_for(module, name, value)
                   for name, value in values.items()}
          for module, values in params.items()}


def count_params(tree) -> int:
  return sum(int(np.prod(v.shape)) for m in tree.values() for v in m.values())
