"""Calibrating unlearning diagnostics against ground-truth anchors (revision).

The submission judged each diagnostic by comparing the forget set with a
never-taught control inside the same model, and summarised it with an L2 drift
gate. Reviewers objected that (i) similar drift does not imply similar change
in information-bearing directions and (ii) the reference that identifies
forget-specific residue is a model retrained without the forget set.

This module implements the replacement. Every diagnostic is reduced to one
score PER FORGET ENTITY, so that

  * anchors with KNOWN ground truth can be compared on the same entities:
      ABSENT     -- M_oracle, trained without the forget set (fact never there)
      SUPPRESSED -- depth-limited unlearning with a frozen trunk (fact provably
                    intact in the residual stream up to the frozen depth, yet
                    behaviourally forgotten)
      PRESENT    -- M_injected (fact learned, not unlearned)
  * a diagnostic is VALID for "removed vs suppressed" only if it separates
    SUPPRESSED from ABSENT (AUROC over entities, with a bootstrap interval);
  * a model under test is located on the ABSENT -> reference axis by a
    normalised position with an entity-paired bootstrap, because the forget
    entities are the same 48 people in every model.

All statistics resample ENTITIES. Probe seeds are not independent units: the
submission's sign-flip test over 20 probe seeds was pseudoreplication.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Per-entity probe decodability by entity-grouped cross-fitting
# ---------------------------------------------------------------------------

def entity_folds(entities: np.ndarray, labels: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign each ENTITY to one of n_folds, stratified by its label."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(entities)
    ent_label = {e: labels[entities == e][0] for e in uniq}
    fold_of = {}
    for lab in sorted(set(ent_label.values())):
        group = np.array(sorted(e for e in uniq if ent_label[e] == lab))
        rng.shuffle(group)
        for i, e in enumerate(group):
            fold_of[e] = i % n_folds
    return np.array([fold_of[e] for e in entities])


def crossfit_probe(X: np.ndarray, y: np.ndarray, entities: Sequence[str],
                   n_folds: int = 4, n_repeats: int = 3,
                   C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
                   seed: int = 0) -> Dict[str, float]:
    """Out-of-fold probe accuracy for EVERY entity.

    Each entity is predicted n_repeats times, each time by a probe that never
    saw any prompt about that entity. C is chosen by entity-grouped inner CV on
    the training folds only. Returns {entity_id: mean accuracy over its
    prompts and repeats}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    entities = np.asarray(entities).astype(str)
    hits: Dict[str, list] = {e: [] for e in np.unique(entities)}
    for rep in range(n_repeats):
        folds = entity_folds(entities, y, n_folds, seed * 1000 + rep)
        for f in range(n_folds):
            te, tr = folds == f, folds != f
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            search = GridSearchCV(pipe, {"logisticregression__C": list(C_grid)},
                                  cv=GroupKFold(n_splits=3), n_jobs=1)
            search.fit(X[tr], y[tr], groups=entities[tr])
            pred = search.predict(X[te])
            for e, ok in zip(entities[te], pred == y[te]):
                hits[e].append(float(ok))
    return {e: float(np.mean(v)) for e, v in hits.items()}


# ---------------------------------------------------------------------------
# Anchor separation and normalised position
# ---------------------------------------------------------------------------

def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie): Mann-Whitney AUROC."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def paired_auroc_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int = 2000,
                    seed: int = 0) -> Dict[str, float]:
    """AUROC between two models' scores on the SAME entities, entity bootstrap."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    rng = np.random.default_rng(seed)
    n = len(pos)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boots.append(auroc(pos[i], neg[i]))
    return {"auroc": auroc(pos, neg), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975))}


def normalised_position(test: np.ndarray, absent: np.ndarray, reference: np.ndarray,
                        n_boot: int = 4000, seed: int = 0) -> Dict[str, float]:
    """(mean test - mean absent) / (mean reference - mean absent), entity-paired.

    0 = indistinguishable from the model that never learned the facts;
    1 = indistinguishable from the reference (PRESENT or SUPPRESSED anchor).
    The interval resamples entities jointly across the three models.
    """
    t, a, r = (np.asarray(v, float) for v in (test, absent, reference))

    def stat(i):
        den = r[i].mean() - a[i].mean()
        return (t[i].mean() - a[i].mean()) / den if abs(den) > 1e-12 else np.nan

    rng = np.random.default_rng(seed)
    n = len(t)
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    return {"position": float(stat(np.arange(n))),
            "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "gap": float(r.mean() - a.mean())}


def paired_diff_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 4000, seed: int = 0) -> Dict[str, float]:
    """Mean of x - y over matched entities, with an entity bootstrap interval."""
    d = np.asarray(x, float) - np.asarray(y, float)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"diff": float(d.mean()), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975))}


def unpaired_diff_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 4000, seed: int = 0) -> Dict[str, float]:
    """Mean difference between two DIFFERENT entity sets, each resampled."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    boots = [x[rng.integers(0, len(x), len(x))].mean() - y[rng.integers(0, len(y), len(y))].mean()
             for _ in range(n_boot)]
    return {"diff": float(x.mean() - y.mean()), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975))}
