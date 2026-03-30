"""EvoTTT: Test-Time Training loop for AlphaFold2's Evoformer.

Provides:
- TTT-specific forward function (Evoformer + MaskedMsaHead only)
- JAX re-implementation of BERT-style MSA masking
- The main TTT training loop with Adam + gradient clipping
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from alphafold.model import modules
from alphafold.model.modules import softmax_cross_entropy

from alphafold.evottt.lora import (
    LoRATarget,
    find_lora_targets,
    init_lora_params,
    merge_lora_into_params,
    trainable_from_lora,
    update_lora_from_trainable,
)


# ---------------------------------------------------------------------------
# TTT forward function
# ---------------------------------------------------------------------------

def make_ttt_apply(model_config):
    """Create a JIT-compiled apply function for TTT.

    Uses the real ``AlphaFold`` module (guaranteeing param-path compatibility)
    but with a modified config:
      - ``num_recycle = 0``  (single Evoformer pass, no recycling loop)
      - All head weights zeroed except ``masked_msa``
        → structure module is never instantiated (modules.py line 216-217)

    Returns:
        ``apply_fn(params, rng, batch) -> output_dict``
        where ``output_dict['masked_msa']['logits']`` has shape
        ``[N_seq, N_res, 23]``.
    """
    cfg = copy.deepcopy(model_config)
    with cfg.unlocked():
        cfg.num_recycle = 0
        for head_name in list(cfg.heads.keys()):
            if head_name != 'masked_msa':
                cfg.heads[head_name].weight = 0.0

    def _forward(batch):
        model = modules.AlphaFold(cfg)
        return model(
            batch,
            is_training=False,
            compute_loss=False,
            ensemble_representations=False,
        )

    transformed = hk.transform(_forward)
    return jax.jit(transformed.apply)


# ---------------------------------------------------------------------------
# MSA re-masking in JAX
# ---------------------------------------------------------------------------

def remask_msa_jax(
    batch: Dict[str, Any],
    rng: jnp.ndarray,
    replace_fraction: float = 0.15,
    uniform_prob: float = 0.1,
    profile_prob: float = 0.1,
    same_prob: float = 0.1,
) -> Dict[str, Any]:
    """Re-mask the MSA in pure JAX, replicating ``make_masked_msa``.

    Mirrors ``alphafold/model/tf/data_transforms.py:make_masked_msa``
    (lines 417-448).  At each call a fresh random mask is sampled so that
    every TTT step sees a different masking.

    Args:
        batch: Feature dict **with the ensemble dimension already squeezed**
            (shapes like ``[N_seq, N_res]``).  Must contain ``'true_msa'``
            and ``'msa_feat'``.
        rng: JAX PRNG key.
        replace_fraction: Fraction of MSA positions to mask.
        uniform_prob, profile_prob, same_prob: BERT replacement probs
            (remaining mass goes to the ``[MASK]`` token 22).

    Returns:
        Shallow-copied batch with updated ``'bert_mask'`` and ``'msa_feat'``.
    """
    batch = dict(batch)  # shallow copy
    true_msa = batch['true_msa']  # (N_seq, N_res)  int 0-21
    n_seq, n_res = true_msa.shape

    rng_pos, rng_cat = jax.random.split(rng)

    # 1) Select positions to mask
    mask_position = jax.random.uniform(rng_pos, (n_seq, n_res)) < replace_fraction
    batch['bert_mask'] = mask_position.astype(jnp.float32)

    # 2) Build categorical replacement distribution  (23 classes: 0-19 AA, 20 X, 21 gap, 22 MASK)
    random_aa = jnp.array(
        [0.05] * 20 + [0.0, 0.0, 0.0], dtype=jnp.float32
    )  # uniform over 20 standard AAs

    # Approximate profile from MSA (mean one-hot over sequences)
    profile = jnp.mean(jax.nn.one_hot(true_msa, 22), axis=0)  # (N_res, 22)
    profile = jnp.pad(profile, ((0, 0), (0, 1)))  # → (N_res, 23)

    same_one_hot = jax.nn.one_hot(true_msa, 23)  # (N_seq, N_res, 23)

    cat_probs = (
        uniform_prob * random_aa[None, None, :]
        + profile_prob * profile[None, :, :]
        + same_prob * same_one_hot
    )
    # Remaining probability → MASK token (index 22)
    mask_prob = 1.0 - uniform_prob - profile_prob - same_prob
    cat_probs = cat_probs.at[:, :, 22].add(mask_prob)

    # 3) Sample replacements via Gumbel-max
    gumbel = -jnp.log(
        -jnp.log(jax.random.uniform(rng_cat, cat_probs.shape, minval=1e-6, maxval=1.0))
    )
    bert_msa = jnp.argmax(jnp.log(cat_probs + 1e-10) + gumbel, axis=-1).astype(jnp.int32)

    # 4) Replace only at masked positions
    new_msa = jnp.where(mask_position, bert_msa, true_msa)

    # 5) Update first 23 channels of msa_feat (the one-hot MSA portion)
    #    msa_feat layout (from data_transforms.make_msa_feat):
    #      [one_hot_msa(23), has_deletion(1), deletion_value(1),
    #       cluster_profile(23), cluster_deletion_mean(1)]  = 49 channels
    new_msa_one_hot = jax.nn.one_hot(new_msa, 23)
    batch['msa_feat'] = batch['msa_feat'].at[:, :, :23].set(new_msa_one_hot)

    return batch


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def ttt_loss_fn(
    trainable: Dict[str, Dict[str, jnp.ndarray]],
    base_params: Dict[str, Dict[str, Any]],
    lora_meta: Dict[str, Any],
    alpha: float,
    rank: int,
    apply_fn,
    num_output: int,
    batch: Dict[str, Any],
    rng: jnp.ndarray,
) -> jnp.ndarray:
    """Compute masked-MSA loss with LoRA-adapted parameters.

    Uses ``softmax_cross_entropy`` and the loss formula from
    ``MaskedMsaHead.loss`` (modules.py lines 1068-1076).

    Only ``trainable`` (the LoRA A/B matrices) is differentiated.

    Args:
        trainable: ``{key: {'A': array, 'B': array}}`` — the optimised vars.
        base_params: Frozen base AF2 params.
        lora_meta: Full lora dict (with shape metadata) from init_lora_params.
        alpha: LoRA scaling factor.
        rank: LoRA rank.
        apply_fn: Haiku ``apply(params, rng, batch)`` from make_ttt_apply.
        num_output: Number of output classes (from ``MaskedMsaHead.config``).
        batch: Feature dict (already re-masked for this step).
        rng: JAX PRNG key for the forward pass.

    Returns:
        Scalar loss value.
    """
    # Reconstruct full lora dict with updated A/B
    lora = update_lora_from_trainable(lora_meta, trainable)

    # Merge LoRA deltas into base params
    merged_params = merge_lora_into_params(base_params, lora, alpha, rank)

    # Forward pass — produces logits via MaskedMsaHead.__call__ inside AlphaFold
    output = apply_fn(merged_params, rng, batch)
    logits = output['masked_msa']['logits']  # (N_seq, N_res, num_output)

    # Squeeze ensemble dim so shapes match MaskedMsaHead.loss expectations:
    # true_msa (N_seq, N_res), bert_mask (N_seq, N_res)
    batch_sq = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, batch)

    # MaskedMsaHead.loss (modules.py:1068-1076)
    errors = softmax_cross_entropy(
        labels=jax.nn.one_hot(batch_sq['true_msa'], num_classes=num_output),
        logits=logits,
    )
    loss = jnp.sum(errors * batch_sq['bert_mask'], axis=(-2, -1)) / (
        1e-8 + jnp.sum(batch_sq['bert_mask'], axis=(-2, -1))
    )
    return loss


# ---------------------------------------------------------------------------
# Main TTT training loop
# ---------------------------------------------------------------------------

def run_ttt(
    apply_fn,
    base_params: Dict[str, Dict[str, Any]],
    model_config,
    batch: Dict[str, Any],
    num_steps: int = 50,
    learning_rate: float = 3e-4,
    rank: int = 4,
    last_n_blocks: int = 8,
    alpha: float = 1.0,
    grad_clip_norm: float = 1.0,
    replace_fraction: float = 0.15,
    seed: int = 0,
    targets: Optional[List[LoRATarget]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[float]]:
    """Run test-time training and return adapted parameters.

    Args:
        apply_fn: TTT apply function from ``make_ttt_apply``.
        base_params: Frozen pre-trained AF2 params.
        model_config: AF2 model config (``config.model``), used to
            instantiate ``MaskedMsaHead`` for loss computation.
        batch: Processed feature dict (with ensemble dim, as returned by
            ``process_features``).  Must contain ``'true_msa'``.
        num_steps: Number of TTT gradient steps.
        learning_rate: Adam learning rate.
        rank: LoRA rank.
        last_n_blocks: How many trailing Evoformer blocks to adapt.
        alpha: LoRA scaling factor.
        grad_clip_norm: Maximum gradient global norm.
        replace_fraction: MSA masking fraction per step.
        seed: Random seed.
        targets: LoRA targets.  If *None*, auto-discovered.

    Returns:
        ``(adapted_params, losses)`` where *adapted_params* is the base
        params with final LoRA deltas merged in, and *losses* is a list
        of per-step loss values.
    """
    if targets is None:
        targets = find_lora_targets(base_params)

    rng = jax.random.PRNGKey(seed)

    # num_output from MaskedMsaHead config (23 for monomer: 20 AA + X + gap + MASK)
    num_output = model_config.heads.masked_msa.num_output

    # --- 1. squeeze ensemble dim for TTT (expected shape: [N_seq, N_res]) ---
    # process_features adds a leading ensemble dim: (1, ...)
    batch_squeezed = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, batch)

    # --- 2. init LoRA -------------------------------------------------------
    rng, init_rng = jax.random.split(rng)
    lora_meta, _alpha, _rank = init_lora_params(
        base_params, targets,
        rank=rank, last_n_blocks=last_n_blocks, alpha=alpha,
        rng_key=init_rng,
    )
    trainable = trainable_from_lora(lora_meta)

    # --- 3. optimizer (warmup + cosine decay) --------------------------------
    warmup_steps = max(1, num_steps // 10)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_steps,
        end_value=learning_rate * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(schedule),
    )
    opt_state = optimizer.init(trainable)

    # --- 4. JIT-compiled step ------------------------------------------------
    @jax.jit
    def step(trainable, opt_state, batch_sq, rng):
        rng, mask_rng, fwd_rng = jax.random.split(rng, 3)

        masked_batch = remask_msa_jax(
            batch_sq, mask_rng, replace_fraction=replace_fraction,
        )
        # Re-add ensemble dim expected by AlphaFold
        masked_batch_e = jax.tree.map(lambda x: x[None], masked_batch)

        # jax.checkpoint avoids storing all intermediate activations
        # (48 Evoformer blocks) — recomputes them during backward instead.
        # Capture non-traceable values (apply_fn, lora_meta, alpha, rank)
        # as closures so JAX only traces trainable, batch, rng.
        @jax.checkpoint
        def _loss(trainable, batch, rng):
            return ttt_loss_fn(
                trainable, base_params, lora_meta, alpha, rank,
                apply_fn, num_output, batch, rng,
            )

        loss, grads = jax.value_and_grad(_loss)(
            trainable, masked_batch_e, fwd_rng,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable)
        new_trainable = optax.apply_updates(trainable, updates)
        return new_trainable, new_opt_state, loss, rng

    # --- 5. training loop ----------------------------------------------------
    losses: List[float] = []
    for i in range(num_steps):
        trainable, opt_state, loss_val, rng = step(
            trainable, opt_state, batch_squeezed, rng,
        )
        losses.append(float(loss_val))
        if i % 10 == 0 or i == num_steps - 1:
            print(f'  TTT step {i:>3d}/{num_steps}: loss = {loss_val:.4f}')

    # --- 6. merge final LoRA into base params --------------------------------
    final_lora = update_lora_from_trainable(lora_meta, trainable)
    adapted_params = merge_lora_into_params(base_params, final_lora, alpha, rank)

    return adapted_params, losses
