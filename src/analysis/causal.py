"""Causal interventions: patching, tracing, ablation, steering.

Why this module is the centre of the project
--------------------------------------------
Probes and the logit lens are correlational. They show that a classifier can
read something, not that the model uses it. Patching and ablation change the
computation and measure the consequence, which is the only kind of evidence
that supports a claim about mechanism.

The claim ladder (manual 18.6), and which function gets you to each rung:

  rung 1  probe decodability            probes.py
  rung 2  suppression signature         logit_lens.py
  rung 3  the MACHINERY survives        patch_resid / causal_trace
          (donor state from M_injected restores behaviour in M_unlearned)
  rung 4  the CONTENTS survive          steer_with_direction, where the
          (a direction estimated from   direction comes from the UNLEARNED
          the unlearned model alone     model's own activations only
          restores behaviour)
  rung 5  the WEIGHTS retain it         disjoint fine-tune recovery attack
                                        (see scripts/12_recovery_attack.py)

Rung 3 is easy to mistake for rung 4. Patching injects information from the
donor model; on its own it tells you that the unlearned model can still USE
the fact, not that it still STORES it. Say so explicitly in the paper.

Controls that make each number meaningful
-----------------------------------------
random_layer   : patch a layer that should not matter. Must give ~0 recovery.
random_position: patch a different token position. Must give ~0 recovery.
retain_facts   : run the identical patch on retain facts. Establishes the
                 ceiling and shows the effect is forget-specific.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def logit_diff(logits, answer_id: int, distractor_ids: Sequence[int]) -> float:
    """logit(answer) - max logit(distractors), at the final position.

    Use logit DIFFERENCE, not probability. Probability saturates and is
    dominated by the softmax normaliser over 50k tokens, which makes patching
    curves hard to compare across models.
    """
    import torch

    row = logits[0, -1].float()
    d = torch.tensor(list(distractor_ids), device=row.device)
    return float(row[answer_id].item() - float(row[d].max().item()))


def normalised_recovery(patched: float, base: float, donor: float) -> float:
    """0 = patching changed nothing; 1 = patching fully restored donor behaviour.

    Normalise by EACH model pair's own range. Comparing raw logit differences
    across models conflates the intervention's effect with differences in
    overall confidence.
    """
    denom = donor - base
    if abs(denom) < 1e-8:
        return float("nan")
    return float((patched - base) / denom)


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def patch_resid(
    receiver_model,
    donor_cache,
    prompt: str,
    layer: int,
    position: int,
    answer_id: int,
    distractor_ids: Sequence[int],
    hook: str = "resid_post",
) -> float:
    """Run `prompt` through receiver_model with one residual state replaced.

    Returns the logit difference under the patch.

    The hook MUST return the modified tensor. Mutating in place and returning
    None works for some TransformerLens versions and silently does nothing in
    others; returning is always correct.
    """
    import torch

    name = f"blocks.{layer}.hook_{hook}"
    donor = donor_cache[name][:, position, :]

    def hook_fn(activation, hook):  # noqa: ARG001
        activation[:, position, :] = donor
        return activation

    with torch.no_grad():
        logits = receiver_model.run_with_hooks(prompt, fwd_hooks=[(name, hook_fn)])
    return logit_diff(logits, answer_id, distractor_ids)


def patching_sweep(
    receiver_model,
    donor_model,
    prompt: str,
    answer_id: int,
    distractor_ids: Sequence[int],
    hook: str = "resid_post",
    positions: Sequence[int] | None = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Patch every (layer, position) and report normalised recovery, with controls.

    Returns tidy rows. `kind` is one of: real, random_position.
    Run the same sweep on retain facts to get the forget-specific comparison.
    """
    import torch

    from src.utils.seed import rng_for

    with torch.no_grad():
        _, donor_cache = donor_model.run_with_cache(prompt)
        base = logit_diff(receiver_model(prompt), answer_id, distractor_ids)
        donor_ld = logit_diff(donor_model(prompt), answer_id, distractor_ids)

    n_tokens = receiver_model.to_tokens(prompt).shape[1]
    if positions is None:
        positions = list(range(-min(6, n_tokens), 0))     # last few positions

    rng = rng_for("patch-controls", seed)
    rows: List[Dict[str, Any]] = []
    for layer in range(receiver_model.cfg.n_layers):
        for pos in positions:
            patched = patch_resid(
                receiver_model, donor_cache, prompt, layer, pos,
                answer_id, distractor_ids, hook,
            )
            rows.append({
                "layer": layer,
                "position": int(pos),
                "kind": "real",
                "logit_diff_base": base,
                "logit_diff_donor": donor_ld,
                "logit_diff_patched": patched,
                "recovery": normalised_recovery(patched, base, donor_ld),
            })

        # control: patch a random position at this layer
        ctrl_pos = int(rng.integers(-n_tokens, -1)) if n_tokens > 2 else -1
        patched = patch_resid(
            receiver_model, donor_cache, prompt, layer, ctrl_pos,
            answer_id, distractor_ids, hook,
        )
        rows.append({
            "layer": layer,
            "position": ctrl_pos,
            "kind": "random_position",
            "logit_diff_base": base,
            "logit_diff_donor": donor_ld,
            "logit_diff_patched": patched,
            "recovery": normalised_recovery(patched, base, donor_ld),
        })
    return rows


def causal_trace(
    model,
    clean_prompt: str,
    corrupted_prompt: str,
    answer_id: int,
    distractor_ids: Sequence[int],
    hook: str = "resid_post",
) -> np.ndarray:
    """Classic causal tracing WITHIN one model: (n_layers, n_positions) recovery.

    Clean and corrupted prompts must tokenise to the same length, or the
    position indices do not line up and the heatmap is meaningless.
    """
    import torch

    n_clean = model.to_tokens(clean_prompt).shape[1]
    n_corr = model.to_tokens(corrupted_prompt).shape[1]
    if n_clean != n_corr:
        raise ValueError(
            f"prompt lengths differ ({n_clean} vs {n_corr}); "
            "choose a corrupted prompt with the same token count"
        )

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_prompt)
        base = logit_diff(model(corrupted_prompt), answer_id, distractor_ids)
        donor = logit_diff(model(clean_prompt), answer_id, distractor_ids)

    grid = np.zeros((model.cfg.n_layers, n_clean))
    for layer in range(model.cfg.n_layers):
        for pos in range(n_clean):
            patched = patch_resid(
                model, clean_cache, corrupted_prompt, layer, pos,
                answer_id, distractor_ids, hook,
            )
            grid[layer, pos] = normalised_recovery(patched, base, donor)
    return grid


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def mean_ablate(
    model,
    prompt: str,
    layer: int,
    component: str,
    mean_activation,
    answer_id: int,
    distractor_ids: Sequence[int],
):
    """Replace a component's output with its dataset MEAN, not with zero.

    Zero is off-distribution for almost every component, so zero-ablation
    effects mix "this component mattered" with "the model has never seen this
    state". Use mean ablation for headline numbers; if you also report
    zero-ablation, label it.
    """
    import torch

    name = f"blocks.{layer}.hook_{component}"

    def hook_fn(activation, hook):  # noqa: ARG001
        activation[:] = mean_activation
        return activation

    with torch.no_grad():
        logits = model.run_with_hooks(prompt, fwd_hooks=[(name, hook_fn)])
    return logit_diff(logits, answer_id, distractor_ids)


# ---------------------------------------------------------------------------
# Steering: the rung-4 test
# ---------------------------------------------------------------------------

def steer_with_direction(
    model,
    prompt: str,
    layer: int,
    direction: np.ndarray,
    alpha: float,
    answer_id: int,
    distractor_ids: Sequence[int],
    hook: str = "resid_post",
    position: int = -1,
) -> float:
    """Add alpha * direction to the residual stream and measure the effect.

    For a CONTENTS claim (rung 4) the direction must be estimated from the
    unlearned model's own activations -- typically a probe fitted on the
    unlearned model, or a difference of class means computed there. A
    direction taken from M_injected proves only that the unlearned model can
    still use externally supplied information.

    Always compare against random directions at MATCHED NORM. A large enough
    perturbation in any direction changes the output.
    """
    import torch

    name = f"blocks.{layer}.hook_{hook}"
    vec = torch.tensor(np.asarray(direction, dtype=np.float32))

    def hook_fn(activation, hook):  # noqa: ARG001
        v = vec.to(activation.device, dtype=activation.dtype)
        activation[:, position, :] = activation[:, position, :] + alpha * v
        return activation

    with torch.no_grad():
        logits = model.run_with_hooks(prompt, fwd_hooks=[(name, hook_fn)])
    return logit_diff(logits, answer_id, distractor_ids)


def matched_random_directions(direction: np.ndarray, n: int = 20, seed: int = 0) -> np.ndarray:
    """n random directions with the same L2 norm as `direction`."""
    from src.utils.seed import rng_for

    rng = rng_for("random-directions", seed)
    d = np.asarray(direction, dtype=np.float64)
    target_norm = np.linalg.norm(d)
    R = rng.normal(size=(n, d.shape[-1]))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    return (R * target_norm).astype(np.float32)


def steering_with_controls(
    model,
    prompt: str,
    layer: int,
    direction: np.ndarray,
    alphas: Sequence[float],
    answer_id: int,
    distractor_ids: Sequence[int],
    n_random: int = 20,
    seed: int = 0,
    **kw,
) -> List[Dict[str, Any]]:
    """Sweep steering strength for the real direction and matched random ones."""
    rows: List[Dict[str, Any]] = []
    randoms = matched_random_directions(direction, n_random, seed)
    for alpha in alphas:
        rows.append({
            "alpha": float(alpha), "kind": "probe_direction", "draw": -1,
            "logit_diff": steer_with_direction(
                model, prompt, layer, direction, alpha, answer_id, distractor_ids, **kw),
        })
        for j, r in enumerate(randoms):
            rows.append({
                "alpha": float(alpha), "kind": "random_matched_norm", "draw": j,
                "logit_diff": steer_with_direction(
                    model, prompt, layer, r, alpha, answer_id, distractor_ids, **kw),
            })
    return rows
