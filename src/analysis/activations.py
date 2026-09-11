"""Residual-stream extraction.

Two decisions you must make once, record in the config, and never vary
silently between scripts:

hook     : resid_pre / resid_mid / resid_post. resid_post[l] is the state
           AFTER layer l has written to it. Committed on Day 8.
position : which token to read. "last" = the final prompt token, i.e. the
           position whose next-token prediction is the answer. This is the
           only position at which the model is actually being asked the
           question, and it is the position the logit lens and patching also
           use, so they stay comparable.

Prompts are grouped by token length before batching. This avoids padding
entirely, which removes the "read position -1 and get a pad token" bug at
the source rather than guarding against it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np


def group_by_length(tokenizer, prompts: Sequence[str]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[len(tokenizer.encode(p))].append(i)
    return dict(groups)


def get_resid_activations(
    model,
    prompts: Sequence[str],
    hook: str = "resid_post",
    position: str = "last",
    batch_size: int = 16,
) -> np.ndarray:
    """Return (n_layers, n_prompts, d_model) float32.

    Only the requested hook points are cached (names_filter), which is the
    difference between fitting in Colab memory and not.
    """
    import torch

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    out = np.zeros((n_layers, len(prompts), d_model), dtype=np.float32)

    names = {f"blocks.{l}.hook_{hook}" for l in range(n_layers)}

    groups = group_by_length(model.tokenizer, prompts)
    for length, idxs in groups.items():
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start : start + batch_size]
            batch = [prompts[i] for i in chunk]
            tokens = model.to_tokens(batch, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=lambda n: n in names
                )
            pos = -1 if position == "last" else -2
            for l in range(n_layers):
                acts = cache[f"blocks.{l}.hook_{hook}"][:, pos, :]
                out[l, chunk, :] = acts.float().cpu().numpy()
            del cache
    return out


def extract_for_records(
    model,
    records: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
):
    """Extract activations for tidy dataset records.

    Returns (acts, labels, entities, meta) ready for src.utils.io.save_activations.
    """
    prompts = [r["prompt"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=np.int64)
    entities = np.array([r["entity_id"] for r in records], dtype=object).astype(str)

    acts = get_resid_activations(
        model,
        prompts,
        hook=cfg["activations"]["hook"],
        position=cfg["activations"]["position"],
        batch_size=cfg["activations"]["batch_size"],
    )
    meta = {
        "model_name": cfg["model"]["name"],
        "hook_name": cfg["activations"]["hook"],
        "position": cfg["activations"]["position"],
        "n_prompts": len(prompts),
        "template_kinds": sorted({r["template_kind"] for r in records}),
        "splits": sorted({r["split"] for r in records}),
    }
    return acts, labels, entities, meta


def activation_drift(acts_a: np.ndarray, acts_b: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-layer drift between two models on the SAME prompts.

    Compare drift on forget prompts against drift on generic prompts. If they
    are equal, the fine-tune moved the whole model and no forget-specific
    interpretation of the probe results is available to you.
    """
    if acts_a.shape != acts_b.shape:
        raise ValueError(f"shape mismatch {acts_a.shape} vs {acts_b.shape}")
    diff = acts_b - acts_a
    l2 = np.linalg.norm(diff, axis=2)                       # (n_layers, n_prompts)
    norm_a = np.linalg.norm(acts_a, axis=2) + 1e-9
    num = np.sum(acts_a * acts_b, axis=2)
    den = norm_a * (np.linalg.norm(acts_b, axis=2) + 1e-9)
    return {
        "l2": l2.mean(axis=1),
        "relative_l2": (l2 / norm_a).mean(axis=1),
        "cosine": (num / den).mean(axis=1),
    }


def class_separation(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Between-class over within-class scatter, per layer.

    A geometric companion to probe accuracy: it can fall while probe accuracy
    holds, which is the signature of "transformed but still decodable".
    """
    out = []
    labels = np.asarray(labels)
    for layer in range(acts.shape[0]):
        X = acts[layer]
        gm = X.mean(axis=0)
        between, within = 0.0, 0.0
        for c in np.unique(labels):
            Xc = X[labels == c]
            cm = Xc.mean(axis=0)
            between += len(Xc) * float(np.sum((cm - gm) ** 2))
            within += float(np.sum((Xc - cm) ** 2))
        out.append(between / (within + 1e-9))
    return np.array(out)
