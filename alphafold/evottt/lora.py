"""LoRA (Low-Rank Adaptation) parameter management for AlphaFold2 Evoformer.

Provides functions to:
- Discover attention weight targets in the Evoformer param tree
- Initialize LoRA A/B matrices
- Merge LoRA deltas into base params for forward passes
"""

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Type alias: (haiku_scope_path, param_name)
LoRATarget = Tuple[str, str]

# Param names we apply LoRA to inside Attention modules
TARGET_PARAM_NAMES = ('query_w', 'key_w', 'value_w', 'output_w')


def find_lora_targets(
    params: Dict[str, Dict[str, Any]],
    target_param_names: Tuple[str, ...] = TARGET_PARAM_NAMES,
    triangle_attention: bool = False,
) -> List[LoRATarget]:
    """Discover LoRA targets by scanning the param tree.

    Finds attention weight matrices inside the main Evoformer stack
    (not the extra MSA stack) by matching scope paths containing
    'evoformer_iteration' and param names in target_param_names.

    Args:
        params: Haiku param dict {scope: {name: array}}.
        target_param_names: Which param names to target.
        triangle_attention: If True, also target triangle attention weights
            (pair representation). Default False targets only MSA attention.

    Returns:
        List of (scope, param_name) tuples.
    """
    targets = []
    for scope in sorted(params.keys()):
        # Must be inside the main evoformer iteration, not extra_msa_stack
        if 'evoformer_iteration' not in scope:
            continue
        if 'extra_msa_stack' in scope:
            continue
        if 'attention' not in scope:
            continue
        # Optionally include triangle attention (pair representation)
        if 'triangle' in scope and not triangle_attention:
            continue
        for name in target_param_names:
            if name in params[scope]:
                targets.append((scope, name))
    return targets


def find_finetune_targets(
    params: Dict[str, Dict[str, Any]],
) -> List[LoRATarget]:
    """Discover every param in the main Evoformer stack (for full FT).

    Returns all (scope, name) pairs whose scope sits inside the main
    ``evoformer_iteration`` stack (i.e. excludes ``extra_msa_stack``).
    This covers attention, MLP/transition, triangle multiplication,
    triangle attention, outer-product-mean, and all layer norms —
    everything stacked with a leading block dim via ``layer_stack``.
    """
    targets: List[LoRATarget] = []
    for scope in sorted(params.keys()):
        if 'evoformer_iteration' not in scope:
            continue
        if 'extra_msa_stack' in scope:
            continue
        for name in sorted(params[scope].keys()):
            targets.append((scope, name))
    return targets


def _weight_dims(name: str, shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Compute (d_in, d_out) for a stacked attention weight.

    Args:
        name: Parameter name (e.g. 'query_w', 'output_w').
        shape: Full shape including leading block dimension,
               e.g. (48, 256, 8, 32) or (48, 8, 32, 256).

    Returns:
        (d_in, d_out) for the 2D LoRA decomposition.
    """
    # shape[0] is the block count from layer_stack
    remaining = shape[1:]
    if name == 'output_w':
        # output_w: (blocks, heads, head_dim, out_dim)
        # Linear maps heads*head_dim -> out_dim
        d_in = int(np.prod(remaining[:-1]))
        d_out = remaining[-1]
    else:
        # query_w, key_w, value_w: (blocks, in_dim, heads, head_dim)
        # Linear maps in_dim -> heads*head_dim
        d_in = remaining[0]
        d_out = int(np.prod(remaining[1:]))
    return int(d_in), int(d_out)


def init_lora_params(
    base_params: Dict[str, Dict[str, Any]],
    targets: List[LoRATarget],
    rank: int = 4,
    last_n_blocks: int = 48,
    alpha: float = 1.0,
    rng_key: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """Initialize LoRA A and B matrices for each target.

    A is initialized with small random values (Kaiming-like) for the last
    ``last_n_blocks`` Evoformer blocks and zero for earlier blocks.
    B is initialized to zero so that the initial delta is zero.

    Args:
        base_params: Base AF2 Haiku params.
        targets: List of (scope, param_name) from find_lora_targets.
        rank: LoRA rank (r).
        last_n_blocks: Number of trailing Evoformer blocks to adapt.
        alpha: LoRA scaling factor.
        rng_key: JAX PRNG key.

    Returns:
        Dict mapping ``"scope//name"`` to
        ``{'A': array, 'B': array, 'orig_shape': tuple,
          'd_in': int, 'd_out': int}``.
        The A/B arrays and metadata are the leaves of a JAX pytree
        suitable for ``jax.grad``.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    lora = {}
    for scope, name in targets:
        w = base_params[scope][name]
        num_blocks = w.shape[0]
        d_in, d_out = _weight_dims(name, w.shape)

        # Standard LoRA convention (Hu et al.):
        #   A (down-projection, d_in→rank): zero-init  → delta starts at zero
        #   B (up-projection, rank→d_out): random Kaiming init
        A = jnp.zeros((num_blocks, d_in, rank), dtype=w.dtype)

        rng_key, subkey = jax.random.split(rng_key)
        bound = np.sqrt(1.0 / rank)
        B = jax.random.uniform(
            subkey, (num_blocks, rank, d_out), minval=-bound, maxval=bound,
        ).astype(w.dtype)
        # Zero-out blocks we don't want to adapt
        start = num_blocks - last_n_blocks
        mask = (jnp.arange(num_blocks) >= start).astype(B.dtype)
        B = B * mask[:, None, None]

        key = f'{scope}//{name}'
        lora[key] = {
            'A': A,
            'B': B,
            'orig_shape': tuple(w.shape),
            'd_in': d_in,
            'd_out': d_out,
        }

    return lora, alpha, rank


def _compute_delta(A: jnp.ndarray, B: jnp.ndarray,
                   alpha: float, rank: int,
                   orig_shape: Tuple[int, ...]) -> jnp.ndarray:
    """Compute the LoRA weight delta.

    delta_2d = (alpha / rank) * A @ B   shape (blocks, d_in, d_out)
    Then reshaped to orig_shape.
    """
    scale = alpha / rank
    delta_2d = scale * jnp.einsum('bir,bro->bio', A, B)
    return delta_2d.reshape(orig_shape)


def merge_lora_into_params(
    base_params: Dict[str, Dict[str, Any]],
    lora: Dict[str, Any],
    alpha: float,
    rank: int,
) -> Dict[str, Dict[str, Any]]:
    """Return a new params dict with LoRA deltas added to base weights.

    Args:
        base_params: Frozen AF2 params.
        lora: LoRA dict from init_lora_params (keys ``"scope//name"``).
        alpha: LoRA scaling factor.
        rank: LoRA rank.

    Returns:
        New params dict (shallow copy, targeted arrays replaced).
    """
    merged = {}
    for scope in base_params:
        merged[scope] = dict(base_params[scope])  # shallow copy

    for key, val in lora.items():
        scope, name = key.split('//')
        delta = _compute_delta(val['A'], val['B'], alpha, rank, val['orig_shape'])
        merged[scope][name] = base_params[scope][name] + delta

    return merged


# ---------------------------------------------------------------------------
# Helpers for extracting a flat trainable pytree (for optax)
# ---------------------------------------------------------------------------

def trainable_from_lora(lora: Dict[str, Any]) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Extract only the trainable A/B arrays as a flat pytree.

    Returns:
        ``{key: {'A': array, 'B': array}}`` — valid JAX pytree.
    """
    return {k: {'A': v['A'], 'B': v['B']} for k, v in lora.items()}


def update_lora_from_trainable(
    lora: Dict[str, Any],
    trainable: Dict[str, Dict[str, jnp.ndarray]],
) -> Dict[str, Any]:
    """Put updated A/B arrays back into the full lora dict."""
    new_lora = {}
    for k, v in lora.items():
        new_lora[k] = {**v, 'A': trainable[k]['A'], 'B': trainable[k]['B']}
    return new_lora
