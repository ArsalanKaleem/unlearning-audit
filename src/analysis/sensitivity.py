"""Sensitivity analysis: how strongly does the target depend on each layer?

The full Jacobian of the output distribution with respect to a residual state
is 50257 x 768 per layer per prompt. Do not compute it. What you actually
want is a single row of it: the gradient of the TARGET log-probability with
respect to the layer's activation. That is one backward pass.

Two things this buys you
------------------------
1. Magnitude per layer: where the answer is most sensitive to the internal
   state. Compare pre- and post-unlearning.
2. Direction: cosine between the pre- and post-unlearning gradients, and
   between the gradient and the probe weight vector. This distinguishes
   "the read-out direction survived but its input shrank" from "the read-out
   itself moved" -- a distinction probe accuracy alone cannot make.

The finite-difference check is not optional. A first-order quantity that does
not predict finite-size effects is not telling you about the model's actual
behaviour, and the honest version of that finding belongs in the paper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def target_grad_wrt_layer(model, prompt: str, target_token_id: int, layer: int,
                          hook: str = "resid_post", position: int = -1) -> np.ndarray:
    """d log p(target) / d activation[layer, position]. Shape (d_model,).

    retain_grad() is required because the residual state is a NON-LEAF tensor:
    without it, .grad is None and you get a silent all-zeros result.
    """
    import torch

    name = f"blocks.{layer}.hook_{hook}"
    stash: Dict[str, Any] = {}

    def hook_fn(activation, hook):  # noqa: ARG001
        activation.retain_grad()
        stash["act"] = activation
        return activation

    model.zero_grad(set_to_none=True)
    logits = model.run_with_hooks(prompt, fwd_hooks=[(name, hook_fn)])
    logprob = torch.log_softmax(logits[0, -1].float(), dim=-1)[target_token_id]
    logprob.backward()

    grad = stash["act"].grad
    if grad is None:
        raise RuntimeError(
            "gradient is None: the activation was not retained. "
            "Check that you are not inside torch.no_grad() and that retain_grad() ran."
        )
    return grad[0, position, :].detach().float().cpu().numpy()


def layerwise_sensitivity(model, prompt: str, target_token_id: int,
                          hook: str = "resid_post") -> np.ndarray:
    """Gradient L2 norm at every layer. Shape (n_layers,)."""
    return np.array([
        float(np.linalg.norm(target_grad_wrt_layer(model, prompt, target_token_id, l, hook)))
        for l in range(model.cfg.n_layers)
    ])


def direction_preservation(grad_pre: np.ndarray, grad_post: np.ndarray) -> Dict[str, float]:
    a, b = np.asarray(grad_pre), np.asarray(grad_post)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {
        "cosine": float(a @ b / (na * nb + 1e-12)),
        "norm_pre": float(na),
        "norm_post": float(nb),
        "norm_ratio": float(nb / (na + 1e-12)),
    }


def probe_gradient_alignment(probe_coef: np.ndarray, grad: np.ndarray, label: int) -> float:
    """Cosine between the probe's weight vector for the true class and the gradient.

    High alignment means the direction a probe reads is also the direction the
    output is sensitive to -- the strongest correlational evidence available
    that the probe found something functional. It is still not causal; the
    steering test below is what upgrades it.
    """
    w = np.asarray(probe_coef)[label]
    g = np.asarray(grad)
    return float(w @ g / (np.linalg.norm(w) * np.linalg.norm(g) + 1e-12))


def finite_difference_check(
    model,
    prompt: str,
    target_token_id: int,
    layer: int,
    direction: np.ndarray,
    alphas: Sequence[float] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0),
    hook: str = "resid_post",
    position: int = -1,
) -> List[Dict[str, float]]:
    """Compare the first-order prediction against the measured change.

    predicted = alpha * (grad . direction)
    actual    = logp(perturbed) - logp(unperturbed)

    Where these diverge, the linear approximation has broken down. Note the
    alpha at which it happens; that bound is a genuine limitation of every
    gradient-based claim you make.
    """
    import torch

    name = f"blocks.{layer}.hook_{hook}"
    d = np.asarray(direction, dtype=np.float32)
    grad = target_grad_wrt_layer(model, prompt, target_token_id, layer, hook, position)
    slope = float(grad @ d)

    def logp_with_shift(alpha: float) -> float:
        vec = torch.tensor(d)

        def hook_fn(activation, hook):  # noqa: ARG001
            v = vec.to(activation.device, dtype=activation.dtype)
            activation[:, position, :] = activation[:, position, :] + alpha * v
            return activation

        with torch.no_grad():
            logits = model.run_with_hooks(prompt, fwd_hooks=[(name, hook_fn)])
            return float(torch.log_softmax(logits[0, -1].float(), dim=-1)[target_token_id].item())

    base = logp_with_shift(0.0)
    rows = []
    for a in alphas:
        actual = logp_with_shift(float(a)) - base
        rows.append({
            "alpha": float(a),
            "predicted_delta": float(a) * slope,
            "actual_delta": actual,
            "abs_error": abs(float(a) * slope - actual),
        })
    return rows
