"""Probes and the control suite.

What a probe proves
-------------------
A probe that reads the target above chance shows the information is
LINEARLY DECODABLE from that activation by an external classifier trained
with supervision. It does NOT show the model uses it. Everything in this
module produces evidence for rung 1 of the claim ladder and no higher.

Three controls, all mandatory
-----------------------------
1. control entities   : never-taught entities. A probe must be at chance here.
                        If it is not, your split leaks or your labels leak.
2. control task       : identical pipeline, labels randomly permuted WITHIN
                        the same label distribution. Measures how much a probe
                        of this capacity can fit noise at this sample size.
                        Selectivity = real accuracy - control-task accuracy.
3. entity-disjoint    : no entity may appear in both train and test. Splitting
   splitting            by example lets the probe memorise the entity, not
                        the fact, and inflates accuracy enormously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.seed import rng_for


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_by_entity(
    entities: Sequence[str],
    test_frac: float = 0.3,
    seed: int = 0,
    labels: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (train, test) with no entity on both sides.

    If `labels` is given, entities are stratified by their label so that the
    test set keeps the same class balance. Each entity must have exactly one
    label for this to be meaningful -- which is true by construction here.
    """
    entities = np.asarray(entities)
    uniq = np.unique(entities)
    rng = rng_for("entity-split", seed)

    if labels is not None:
        labels = np.asarray(labels)
        ent_label = {e: labels[entities == e][0] for e in uniq}
        # sanity: one label per entity
        for e in uniq:
            if len(np.unique(labels[entities == e])) != 1:
                raise ValueError(f"entity {e!r} has multiple labels; stratified split is ill-defined")
        test_entities: List[str] = []
        for lab in np.unique(list(ent_label.values())):
            group = np.array([e for e in uniq if ent_label[e] == lab])
            rng.shuffle(group)
            k = max(1, int(round(test_frac * len(group))))
            test_entities += list(group[:k])
        test_entities_set = set(test_entities)
    else:
        shuffled = uniq.copy()
        rng.shuffle(shuffled)
        k = max(1, int(round(test_frac * len(shuffled))))
        test_entities_set = set(shuffled[:k])

    test_mask = np.array([e in test_entities_set for e in entities])
    return ~test_mask, test_mask


def assert_no_leakage(entities: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> None:
    tr = set(np.asarray(entities)[train_mask])
    te = set(np.asarray(entities)[test_mask])
    shared = tr & te
    if shared:
        raise AssertionError(f"entity leakage across splits: {sorted(shared)[:5]} ...")


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    accuracy: float
    n_train: int
    n_test: int
    n_classes: int
    chance: float
    seed: int
    kind: str
    best_params: Dict[str, Any] = field(default_factory=dict)
    coef: np.ndarray | None = None      # (n_classes, d) for the linear probe
    per_class_accuracy: Dict[int, float] = field(default_factory=dict)

    def as_row(self, **extra) -> Dict[str, Any]:
        row = {
            "accuracy": self.accuracy,
            "chance": self.chance,
            "selectivity": None,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_classes": self.n_classes,
            "seed": self.seed,
            "probe": self.kind,
        }
        row.update(self.best_params)
        row.update(extra)
        return row


def _fit_eval(pipe, X, y, train_mask, test_mask, kind, seed, keep_coef=False) -> ProbeResult:
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    acc = float(np.mean(pred == yte))
    per_class = {
        int(c): float(np.mean(pred[yte == c] == c)) for c in np.unique(yte)
    }
    best = getattr(pipe, "best_params_", {}) or {}
    coef = None
    if keep_coef:
        est = pipe.best_estimator_ if hasattr(pipe, "best_estimator_") else pipe
        clf = est.named_steps.get("clf") if hasattr(est, "named_steps") else None
        coef = getattr(clf, "coef_", None)
        if coef is not None:
            coef = np.asarray(coef, dtype=np.float32)
    return ProbeResult(
        accuracy=acc,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        n_classes=int(len(np.unique(y))),
        chance=1.0 / len(np.unique(y)),
        seed=seed,
        kind=kind,
        best_params={str(k): v for k, v in best.items()},
        coef=coef,
        per_class_accuracy=per_class,
    )


def linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    entities: Sequence[str],
    seed: int = 0,
    test_frac: float = 0.3,
    C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    max_iter: int = 2000,
    standardise: bool = True,
    keep_coef: bool = True,
) -> ProbeResult:
    """Multinomial logistic regression with entity-disjoint splitting.

    The scaler lives INSIDE the pipeline, so it is fitted on training folds
    only. Fitting a scaler on all data before splitting leaks distribution
    information between conditions and is a real, common, silent bug.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    train_mask, test_mask = split_by_entity(entities, test_frac, seed, labels=y)
    assert_no_leakage(np.asarray(entities), train_mask, test_mask)

    steps = []
    if standardise:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", LogisticRegression(max_iter=max_iter)))
    pipe = Pipeline(steps)

    n_splits = min(3, int(np.bincount(y[train_mask]).min()) if y[train_mask].size else 3)
    if n_splits >= 2 and len(C_grid) > 1:
        search = GridSearchCV(
            pipe,
            {"clf__C": list(C_grid)},
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
            n_jobs=1,
        )
        return _fit_eval(search, X, y, train_mask, test_mask, "logistic", seed, keep_coef)
    return _fit_eval(pipe, X, y, train_mask, test_mask, "logistic", seed, keep_coef)


def mlp_probe(
    X: np.ndarray,
    y: np.ndarray,
    entities: Sequence[str],
    hidden: int = 64,
    seed: int = 0,
    test_frac: float = 0.3,
    max_iter: int = 500,
    alpha: float = 1e-3,
    standardise: bool = True,
) -> ProbeResult:
    """One-hidden-layer MLP probe. `hidden` IS the capacity knob.

    Report nonlinear-vs-linear comparisons only at matched capacity and after
    subtracting the control-task accuracy at the same capacity. A 256-unit MLP
    beating a regularised logistic regression is not evidence of nonlinear
    encoding by itself.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    train_mask, test_mask = split_by_entity(entities, test_frac, seed, labels=y)
    assert_no_leakage(np.asarray(entities), train_mask, test_mask)

    steps = []
    if standardise:
        steps.append(("scale", StandardScaler()))
    steps.append((
        "clf",
        MLPClassifier(
            hidden_layer_sizes=(hidden,),
            alpha=alpha,
            max_iter=max_iter,
            random_state=seed,
            early_stopping=False,
        ),
    ))
    res = _fit_eval(Pipeline(steps), X, y, train_mask, test_mask, f"mlp{hidden}", seed)
    res.best_params = {"hidden": hidden, "alpha": alpha}
    return res


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def control_task_labels(y: np.ndarray, entities: Sequence[str], seed: int = 0) -> np.ndarray:
    """Permute labels ACROSS ENTITIES, preserving the label distribution.

    Permuting per-example would leave an entity with inconsistent labels and
    make the control artificially hard. Permuting per-entity keeps the control
    task structurally identical to the real task, which is the point.
    """
    entities = np.asarray(entities)
    y = np.asarray(y)
    uniq = np.unique(entities)
    ent_label = np.array([y[entities == e][0] for e in uniq])
    rng = rng_for("control-task", seed)
    perm = rng.permutation(len(uniq))
    mapping = {e: ent_label[p] for e, p in zip(uniq, perm)}
    return np.array([mapping[e] for e in entities])


def probe_with_controls(
    X: np.ndarray,
    y: np.ndarray,
    entities: Sequence[str],
    seed: int = 0,
    kind: str = "logistic",
    hidden: int = 64,
    **kw,
) -> Dict[str, Any]:
    """Run the real probe and its control task, and report selectivity."""
    runner = (lambda yy: linear_probe(X, yy, entities, seed=seed, **kw)) if kind == "logistic" \
        else (lambda yy: mlp_probe(X, yy, entities, hidden=hidden, seed=seed, **kw))

    real = runner(np.asarray(y))
    ctrl = runner(control_task_labels(y, entities, seed))
    return {
        "accuracy": real.accuracy,
        "control_task_accuracy": ctrl.accuracy,
        "selectivity": real.accuracy - ctrl.accuracy,
        "chance": real.chance,
        "n_train": real.n_train,
        "n_test": real.n_test,
        "seed": seed,
        "probe": real.kind,
        "coef": real.coef,
    }


def layerwise_probe(
    acts: np.ndarray,
    y: np.ndarray,
    entities: Sequence[str],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    kind: str = "logistic",
    hidden: int = 64,
    **kw,
) -> List[Dict[str, Any]]:
    """Probe every layer with every seed. acts: (n_layers, n_examples, d_model).

    Returns tidy rows -- one per (layer, seed) -- ready for a DataFrame.
    """
    rows: List[Dict[str, Any]] = []
    n_layers = acts.shape[0]
    for layer in range(n_layers):
        for seed in seeds:
            out = probe_with_controls(
                acts[layer], y, entities, seed=seed, kind=kind, hidden=hidden, **kw
            )
            out.pop("coef", None)
            out["layer"] = layer
            rows.append(out)
    return rows


def transfer_accuracy(
    probe_coef_source,
    X_target: np.ndarray,
    y_target: np.ndarray,
) -> float:
    """Apply a probe fitted on model A to activations from model B.

    Caveat you must document: the two models' activations must be preprocessed
    identically. If you standardised, you must apply the SOURCE model's scaler,
    not refit one on the target -- refitting hides exactly the distribution
    shift you are trying to measure.
    """
    logits = X_target @ np.asarray(probe_coef_source).T
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == np.asarray(y_target)))
