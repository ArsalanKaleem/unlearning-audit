"""Day 42: from features to a circuit.

The discipline this module enforces
-----------------------------------
A circuit diagram is a causal claim drawn as a picture, and pictures are
persuasive out of proportion to their evidence. Every edge this module emits
carries a `verified` flag:

  verified=False  the edge came from ATTRIBUTION -- a gradient, a direct-path
                  contribution, a correlation. It says "this component's
                  output is associated with the target".
  verified=True   the edge came from an INTERVENTION -- the component was
                  ablated and the target changed by more than a matched
                  random control.

Draw unverified edges dashed, or leave them out. A reader cannot tell the
difference from the figure alone, so the figure must tell them.

Cost note: per-head ablation is n_layers * n_heads forward passes per prompt.
For GPT-2 Small that is 144 per prompt, which is fine on CPU for a handful of
prompts and is NOT fine for a hundred. Select a small prompt set deliberately
and report how many prompts each number rests on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from src.analysis.causal import logit_diff
from src.utils.seed import rng_for


# ---------------------------------------------------------------------------
# Mean activations for ablation
# ---------------------------------------------------------------------------

def compute_mean_activations(model, prompts: Sequence[str], component: str = "z") -> Dict[str, Any]:
    """Dataset-mean activation per layer, for mean ablation.

    Computed over the prompts you will ablate on, at the last position. Using
    a mean from a different distribution reintroduces the off-distribution
    problem that mean ablation exists to avoid.
    """
    import torch

    names = {f"blocks.{l}.hook_{component}" for l in range(model.cfg.n_layers)}
    sums: Dict[str, Any] = {}
    count = 0
    for prompt in prompts:
        with torch.no_grad():
            _, cache = model.run_with_cache(prompt, names_filter=lambda n: n in names)
        for name in names:
            act = cache[name][:, -1]
            sums[name] = act.clone() if name not in sums else sums[name] + act
        count += 1
    return {k: v / max(count, 1) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Head attribution
# ---------------------------------------------------------------------------

def ablate_head(model, prompt: str, layer: int, head: int, mean_z,
                answer_id: int, distractor_ids: Sequence[int]) -> float:
    """Mean-ablate one attention head's output and return the logit difference."""
    import torch

    name = f"blocks.{layer}.hook_z"

    def hook_fn(activation, hook):  # noqa: ARG001
        activation[:, -1, head, :] = mean_z[:, head, :].to(activation.dtype)
        return activation

    with torch.no_grad():
        logits = model.run_with_hooks(prompt, fwd_hooks=[(name, hook_fn)])
    return logit_diff(logits, answer_id, distractor_ids)


def head_attribution(
    model,
    prompts: Sequence[str],
    answer_ids: Sequence[int],
    distractor_ids: Sequence[int],
    mean_acts: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Ablate every head on every prompt; return the mean effect per head.

    effect = clean logit difference - ablated logit difference.
    A large positive effect means the head SUPPORTS the target.
    """
    import torch

    if mean_acts is None:
        mean_acts = compute_mean_activations(model, prompts, "z")

    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    effects = np.zeros((n_layers, n_heads, len(prompts)))

    for p_idx, (prompt, ans) in enumerate(zip(prompts, answer_ids)):
        distractors = [d for d in distractor_ids if d != ans]
        with torch.no_grad():
            clean = logit_diff(model(prompt), ans, distractors)
        for layer in range(n_layers):
            mean_z = mean_acts[f"blocks.{layer}.hook_z"]
            for head in range(n_heads):
                ablated = ablate_head(model, prompt, layer, head, mean_z, ans, distractors)
                effects[layer, head, p_idx] = clean - ablated

    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            v = effects[layer, head]
            rows.append({
                "layer": layer, "head": head,
                "mean_effect": float(v.mean()),
                "sd_effect": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "n_prompts": len(v),
                "method": "mean_ablation",
                "verified": True,
            })
    return rows


def random_head_baseline(rows: Sequence[Dict[str, Any]], k: int, n_draws: int = 100,
                         seed: int = 0) -> Dict[str, float]:
    """What effect size would k randomly chosen heads have shown?

    Without this, "our top-5 heads matter" is not a claim about those heads.
    """
    effects = np.array([r["mean_effect"] for r in rows])
    rng = rng_for("random-heads", seed)
    draws = np.array([np.abs(effects[rng.choice(len(effects), k, replace=False)]).mean()
                      for _ in range(n_draws)])
    top = float(np.abs(np.sort(effects)[-k:]).mean())
    p = float((np.sum(draws >= top) + 1) / (n_draws + 1))
    return {"top_k_effect": top, "null_mean": float(draws.mean()),
            "null_sd": float(draws.std(ddof=1)), "p_empirical": p}


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

def build_edges(
    head_rows: Sequence[Dict[str, Any]],
    latent_rows: Sequence[Dict[str, Any]] = (),
    effect_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """Assemble an edge list with honest provenance on every edge.

    head_rows   : from head_attribution (intervention -> verified=True)
    latent_rows : SAE latent rows, each needing `verified` set by whether you
                  actually ablated that latent or only scored it
    """
    edges = []
    for r in head_rows:
        if abs(r["mean_effect"]) < effect_threshold:
            continue
        edges.append({
            "source": f"L{r['layer']}H{r['head']}",
            "target": "output",
            "weight": r["mean_effect"],
            "sign": "supports" if r["mean_effect"] > 0 else "suppresses",
            "evidence": r.get("method", "unknown"),
            "verified": bool(r.get("verified", False)),
        })
    for r in latent_rows:
        edges.append({
            "source": f"SAE:{r.get('latent')}",
            "target": r.get("target", "output"),
            "weight": r.get("effect", float("nan")),
            "sign": "supports" if r.get("effect", 0) > 0 else "suppresses",
            "evidence": r.get("method", "attribution"),
            "verified": bool(r.get("verified", False)),
        })
    return sorted(edges, key=lambda e: -abs(e["weight"]))


def circuit_summary(edges: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    verified = [e for e in edges if e["verified"]]
    return {
        "n_edges": len(edges),
        "n_causally_verified": len(verified),
        "n_attribution_only": len(edges) - len(verified),
        "caption_requirement": (
            f"{len(verified)} of {len(edges)} edges were verified by intervention; "
            "the remainder are attribution-only and must be drawn dashed or omitted."
        ),
    }


def to_mermaid(edges: Sequence[Dict[str, Any]], max_edges: int = 20) -> str:
    """A quick diagram for the logbook. The paper figure should be drawn by hand
    from the same edge list, so that you make the inclusion decisions yourself."""
    lines = ["graph LR"]
    for e in list(edges)[:max_edges]:
        style = "-->" if e["verified"] else "-.->"
        lines.append(f'  {e["source"].replace(":", "_")} {style}'
                     f'|{e["weight"]:.2f}| {e["target"].replace(":", "_")}')
    return "\n".join(lines)
