"""Logit lens: read every layer's residual state through the unembedding.

What it shows
-------------
How the model's belief about the answer develops across depth. A fact that
peaks mid-network and then falls away before the output is the signature of
SUPPRESSION rather than erasure, and it is one of the more striking things
this project can find.

What it does not show
---------------------
The lens is an approximation. Intermediate residual states were never trained
to be read by the final unembedding, and the resulting probabilities are
poorly calibrated, especially in early layers. Report ranks and trajectory
SHAPE, not absolute probabilities, and cite the tuned-lens critique when you
do. The last-layer assertion below is what keeps the implementation honest:
at the final layer the lens must reproduce the model's real logits exactly.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def logit_lens(model, prompt: str, hook: str = "resid_post", position: int = -1) -> np.ndarray:
    """Return (n_layers, d_vocab) log-probabilities at `position`.

    ln_final is applied before unembedding. Omitting it is the most common
    logit-lens bug and it produces plausible-looking nonsense.
    """
    import torch

    names = {f"blocks.{l}.hook_{hook}" for l in range(model.cfg.n_layers)}
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt, names_filter=lambda n: n in names)
        rows = []
        for l in range(model.cfg.n_layers):
            resid = cache[f"blocks.{l}.hook_{hook}"][:, position, :]
            normed = model.ln_final(resid)
            logits = model.unembed(normed.unsqueeze(1))[:, 0, :]
            rows.append(torch.log_softmax(logits, dim=-1)[0].float().cpu().numpy())
    return np.stack(rows)


def assert_last_layer_matches(model, prompt: str, atol: float = 1e-3) -> float:
    """Day 11 checkpoint. Run this for EVERY model before using the lens."""
    import torch

    lens = logit_lens(model, prompt)[-1]
    with torch.no_grad():
        real = torch.log_softmax(model(prompt)[0, -1].float(), dim=-1).cpu().numpy()
    diff = float(np.abs(lens - real).max())
    if diff > atol:
        raise AssertionError(
            f"logit lens final layer differs from real logits by {diff:.4g}. "
            "Check ln_final, the hook point, and the position index."
        )
    return diff


def lens_trajectory(
    model,
    prompts: Sequence[str],
    target_token_ids: Sequence[int],
    hook: str = "resid_post",
) -> Dict[str, np.ndarray]:
    """Per-layer log-prob and rank of the target, averaged in LOG space.

    Averaging probabilities linearly lets one confident fact dominate the
    curve. Average log-probabilities, or report medians with quantile bands,
    and state in the caption which you did.
    """
    logps, ranks = [], []
    for prompt, tid in zip(prompts, target_token_ids):
        lp = logit_lens(model, prompt, hook=hook)          # (n_layers, d_vocab)
        logps.append(lp[:, tid])
        ranks.append((lp > lp[:, [tid]]).sum(axis=1) + 1)
    logps = np.stack(logps)                                 # (n_prompts, n_layers)
    ranks = np.stack(ranks)
    return {
        "logprob_mean": logps.mean(axis=0),
        "logprob_median": np.median(logps, axis=0),
        "logprob_q25": np.quantile(logps, 0.25, axis=0),
        "logprob_q75": np.quantile(logps, 0.75, axis=0),
        "rank_median": np.median(ranks, axis=0),
        "per_prompt_logprob": logps,
    }


def classify_trajectory(logprob_by_layer: np.ndarray, drop_threshold: float = 0.5) -> str:
    """Label a trajectory before you plot it, using a fixed rule.

    monotone      : belief rises to the output; no suppression signature
    rise_then_fall: peaks internally then drops by > drop_threshold nats;
                    consistent with suppression
    flat          : no meaningful development; consistent with erasure or
                    with the fact never having been there
    """
    x = np.asarray(logprob_by_layer, dtype=float)
    peak = int(np.argmax(x))
    drop = float(x[peak] - x[-1])
    rise = float(x.max() - x.min())
    if rise < 0.5:
        return "flat"
    if peak < len(x) - 1 and drop > drop_threshold:
        return "rise_then_fall"
    return "monotone"


def summarise_trajectory(logprob_by_layer: np.ndarray) -> Dict[str, float]:
    x = np.asarray(logprob_by_layer, dtype=float)
    peak = int(np.argmax(x))
    return {
        "peak_layer": peak,
        "peak_logprob": float(x[peak]),
        "final_logprob": float(x[-1]),
        "peak_to_final_drop": float(x[peak] - x[-1]),
        "shape": classify_trajectory(x),
    }
