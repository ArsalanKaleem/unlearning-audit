"""Uncertainty and significance, clustered by entity.

The single most common statistical error in this kind of project is to
resample PROMPTS. Six prompts about the same entity are not six independent
observations, and treating them as such produces confidence intervals roughly
sqrt(6) times too narrow. Everything here resamples ENTITIES.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from src.utils.seed import rng_for


def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[str],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap CI, resampling whole clusters with replacement.

    values   : per-example measurements (e.g. 1/0 correctness per prompt)
    clusters : the entity each value belongs to
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = rng_for("bootstrap", seed)

    point = float(statistic(values))
    if len(uniq) < 2:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_clusters": int(len(uniq)), "n": int(len(values))}

    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in picked])
        draws[b] = statistic(values[idx])

    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "se": float(np.std(draws, ddof=1)),
        "n_clusters": int(len(uniq)),
        "n": int(len(values)),
    }


def paired_cluster_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[str],
    n_perm: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """Two-sided paired permutation test with sign-flips applied per cluster.

    Use for pre-vs-post comparisons on the SAME entities: a[i] and b[i] must
    be the same example measured under two conditions.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    clusters = np.asarray(clusters)
    if a.shape != b.shape or a.shape[0] != clusters.shape[0]:
        raise ValueError("a, b and clusters must be aligned and the same length")

    diff = a - b
    uniq = np.unique(clusters)
    cluster_means = np.array([diff[clusters == c].mean() for c in uniq])
    observed = float(cluster_means.mean())

    rng = rng_for("permutation", seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(uniq)))
    null = (signs * cluster_means).mean(axis=1)
    p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))
    return {"observed_diff": observed, "p_value": p, "n_clusters": int(len(uniq))}


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Holm--Bonferroni step-down correction.

    Apply across the family of layer-wise tests. Report the primary,
    preregistered test separately and UNCORRECTED, and say that you did.
    """
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adjusted[i] = min(1.0, running)
    return {"p_adjusted": adjusted, "reject": adjusted < alpha}


def seed_variance(values: Sequence[float]) -> Dict[str, float]:
    """Spread across probe seeds. If this exceeds your effect, you have no effect."""
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "n_seeds": int(len(v)),
    }


def bootstrap_difference(
    values_a: Sequence[float],
    clusters_a: Sequence[str],
    values_b: Sequence[float],
    clusters_b: Sequence[str],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """CI on a difference of means between two independent sets of clusters."""
    va, ca = np.asarray(values_a, float), np.asarray(clusters_a)
    vb, cb = np.asarray(values_b, float), np.asarray(clusters_b)
    ua, ub = np.unique(ca), np.unique(cb)
    ia = {c: np.flatnonzero(ca == c) for c in ua}
    ib = {c: np.flatnonzero(cb == c) for c in ub}
    rng = rng_for("bootstrap-diff", seed)

    draws = np.empty(n_boot)
    for k in range(n_boot):
        pa = rng.choice(ua, size=len(ua), replace=True)
        pb = rng.choice(ub, size=len(ub), replace=True)
        draws[k] = va[np.concatenate([ia[c] for c in pa])].mean() - \
                   vb[np.concatenate([ib[c] for c in pb])].mean()
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"point": float(va.mean() - vb.mean()), "lo": float(lo), "hi": float(hi)}
