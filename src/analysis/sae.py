"""Sparse autoencoder analysis.

The order of operations here is not negotiable:

1. VALIDATE the SAE on each model before comparing anything (fvu, l0,
   reconstruction-substitution loss). An SAE trained on the base model may
   reconstruct a fine-tuned model's activations badly, and if it does, every
   latent-level comparison you make is measuring reconstruction failure
   rather than a change in features.
2. SELECT latents on a SELECTION SPLIT of M_injected only, using a rule you
   wrote down before looking at the unlearned models.
3. COMPARE on the held-out evaluation split, against matched random latent
   draws.

Doing 3 before 2, or 2 after seeing 3, is the cherry-picking failure that
makes feature-level results uninterpretable. If you adjust k or the score
after seeing the comparison, report BOTH analyses.

The SAE release string in the config must be verified against SAELens before
use; release names and sae_ids change between versions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def load_sae(release: str, sae_id: str, device: str = "cpu"):
    """Load a pretrained SAE. Returns (sae, cfg_dict, sparsity) per SAELens."""
    from sae_lens import SAE

    sae, cfg_dict, sparsity = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae, cfg_dict, sparsity


def sae_diagnostics(sae, activations) -> Dict[str, float]:
    """FVU, L0 and reconstruction quality on YOUR prompts, not the SAE's.

    fvu : fraction of variance unexplained. Well below 1.0 (typically < 0.2)
          means the SAE reconstructs this activation distribution. Near or
          above 1.0 almost always means the wrong hook point.
    l0  : mean number of active latents per token. Tens is normal. Single
          digits or hundreds both indicate a mismatch.
    """
    import torch

    with torch.no_grad():
        acts = torch.as_tensor(activations, dtype=torch.float32, device=sae.device)
        latents = sae.encode(acts)
        recon = sae.decode(latents)
        residual = acts - recon
        fvu = (residual.pow(2).sum() / (acts - acts.mean(0)).pow(2).sum()).item()
        l0 = (latents > 0).float().sum(dim=-1).mean().item()
    return {"fvu": float(fvu), "l0": float(l0), "n_latents": int(latents.shape[-1])}


def substitution_loss_increase(model, sae, hook_name: str, prompts: Sequence[str]) -> float:
    """Cross-entropy increase when the SAE reconstruction replaces the activation.

    This is the diagnostic that matters most, because it measures the SAE's
    error in units the model cares about. Report it alongside FVU.
    """
    import torch

    def clean_loss():
        return float(model(prompts, return_type="loss").item())

    def hook_fn(activation, hook):  # noqa: ARG001
        flat = activation.reshape(-1, activation.shape[-1])
        recon = sae.decode(sae.encode(flat))
        return recon.reshape(activation.shape)

    with torch.no_grad():
        base = clean_loss()
        sub = float(model.run_with_hooks(
            prompts, return_type="loss", fwd_hooks=[(hook_name, hook_fn)]
        ).item())
    return sub - base


def encode(sae, activations) -> np.ndarray:
    """(n_examples, n_latents) latent activations as a dense numpy array."""
    import torch

    with torch.no_grad():
        acts = torch.as_tensor(activations, dtype=torch.float32, device=sae.device)
        return sae.encode(acts).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Latent selection: preregister this rule
# ---------------------------------------------------------------------------

def select_latents(
    latents_target: np.ndarray,
    latents_baseline: np.ndarray,
    k: int = 20,
    min_activation: float = 0.0,
) -> Dict[str, Any]:
    """Score = mean activation on target prompts - mean on baseline prompts.

    target   : forget-set prompts (selection split of M_injected)
    baseline : control-entity prompts, i.e. structurally identical prompts
               about entities the model was never taught. Using generic text
               as the baseline instead would select latents for "this is a
               biography prompt", not for the stored fact.

    Returns the top-k latent indices and their scores. FREEZE these before
    touching any unlearned model.
    """
    t = np.asarray(latents_target).mean(axis=0)
    b = np.asarray(latents_baseline).mean(axis=0)
    score = t - b
    eligible = np.flatnonzero(t > min_activation)
    if len(eligible) < k:
        raise ValueError(f"only {len(eligible)} latents exceed min_activation={min_activation}")
    top = eligible[np.argsort(-score[eligible])[:k]]
    return {
        "indices": top.tolist(),
        "scores": score[top].tolist(),
        "mean_target": t[top].tolist(),
        "mean_baseline": b[top].tolist(),
        "rule": "mean(target) - mean(control_entities), top-k, computed on the selection split of M_injected",
    }


def random_latent_draws(n_latents: int, k: int, n_draws: int = 100, seed: int = 0,
                        exclude: Sequence[int] = ()) -> np.ndarray:
    """Matched random latent sets, the null distribution for every claim below."""
    from src.utils.seed import rng_for

    rng = rng_for("random-latents", seed)
    pool = np.setdiff1d(np.arange(n_latents), np.asarray(list(exclude), dtype=int))
    return np.stack([rng.choice(pool, size=k, replace=False) for _ in range(n_draws)])


def compare_latent_activation(
    latents_pre: np.ndarray,
    latents_post: np.ndarray,
    selected: Sequence[int],
    n_draws: int = 100,
    seed: int = 0,
) -> Dict[str, Any]:
    """Change in selected latents pre vs post, against matched random draws.

    The empirical p-value is the fraction of random latent sets whose change
    is at least as large. This is the number that tells you whether the
    feature-level effect is specific or just part of a global shift.
    """
    pre = np.asarray(latents_pre)
    post = np.asarray(latents_post)
    sel = np.asarray(list(selected), dtype=int)

    obs = float(post[:, sel].mean() - pre[:, sel].mean())
    draws = random_latent_draws(pre.shape[1], len(sel), n_draws, seed, exclude=sel)
    null = np.array([float(post[:, d].mean() - pre[:, d].mean()) for d in draws])
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_draws + 1))
    return {
        "observed_change": obs,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        "p_empirical": p,
        "relative_change": obs / (abs(pre[:, sel].mean()) + 1e-9),
    }


def ablate_latents(model, sae, hook_name: str, prompt: str, latent_ids: Sequence[int],
                   answer_id: int, distractor_ids: Sequence[int], mode: str = "zero") -> float:
    """Zero out chosen latents inside the SAE reconstruction and re-run.

    Returns the logit difference after ablation. Compare against the same
    procedure on matched random latents -- never against the unablated model
    alone, because the SAE round-trip itself changes the output.
    """
    import torch

    from src.analysis.causal import logit_diff

    ids = torch.tensor(list(latent_ids), device=sae.device, dtype=torch.long)

    def hook_fn(activation, hook):  # noqa: ARG001
        flat = activation.reshape(-1, activation.shape[-1])
        lat = sae.encode(flat)
        if mode == "zero":
            lat[:, ids] = 0.0
        elif mode == "mean":
            lat[:, ids] = lat[:, ids].mean(dim=0, keepdim=True)
        else:
            raise ValueError(mode)
        return sae.decode(lat).reshape(activation.shape)

    with torch.no_grad():
        logits = model.run_with_hooks(prompt, fwd_hooks=[(hook_name, hook_fn)])
    return logit_diff(logits, answer_id, distractor_ids)
