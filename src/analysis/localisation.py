"""Day 35: where did the information go?

Probe accuracy alone cannot distinguish four very different situations, and
the paper's contribution depends on telling them apart:

  PRESERVED   the information is there, in the same place, in the same form.
              Post-unlearning probe high; a probe fitted on M_injected still
              works on M_unlearned's activations; sample efficiency unchanged.

  TRANSFORMED the information is there but has MOVED. Post-unlearning probe
              high, but the transferred probe fails. The read-out direction
              changed; the content did not.

  OBSCURED    the information is there but harder to reach. Post probe high
              only with many more training entities; class separation down.

  REMOVED     the information is gone. Post probe at control level, transfer
              at control level, no sample size recovers it.

The functions here produce the three measurements. `classify_localisation`
applies a rule to them -- put your thresholds in the preregistration before
you run it, because the rule is where the interpretation lives.

All of this runs on cached activations, on CPU, with no torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from src.analysis.probes import (assert_no_leakage, linear_probe, split_by_entity,
                                 transfer_accuracy)
from src.utils.seed import rng_for


# ---------------------------------------------------------------------------
# Sample efficiency
# ---------------------------------------------------------------------------

def sample_efficiency_curve(
    X: np.ndarray,
    y: np.ndarray,
    entities: Sequence[str],
    sizes: Sequence[int] = (8, 16, 32, 64, 0),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    test_frac: float = 0.3,
    C: float = 1.0,
    max_iter: int = 2000,
) -> List[Dict[str, Any]]:
    """Probe accuracy as a function of the number of TRAINING ENTITIES.

    size 0 means "all available training entities". The test set is held
    fixed within a seed, so the curve varies only in training data -- if you
    resample the test set too, the curve mixes two sources of variation and
    the comparison between models becomes unreadable.

    Subsampling is stratified by label, so every training subset keeps the
    class balance and chance stays 1/n_classes throughout.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.analysis.probes import SafeStandardScaler, _fit_eval

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    entities = np.asarray(entities)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        train_mask, test_mask = split_by_entity(entities, test_frac, seed, labels=y)
        assert_no_leakage(entities, train_mask, test_mask)

        train_entities = np.unique(entities[train_mask])
        ent_label = {e: y[entities == e][0] for e in train_entities}
        by_label: Dict[int, List[str]] = {}
        for e in train_entities:
            by_label.setdefault(int(ent_label[e]), []).append(e)

        rng = rng_for(f"sample-efficiency-{seed}", seed)
        for size in sizes:
            n = len(train_entities) if size == 0 else min(size, len(train_entities))
            per_class = max(1, n // max(len(by_label), 1))
            chosen: List[str] = []
            for lab, ents in by_label.items():
                pick = rng.choice(ents, size=min(per_class, len(ents)), replace=False)
                chosen += list(pick)
            chosen_set = set(chosen)
            sub_mask = train_mask & np.array([e in chosen_set for e in entities])

            if len(np.unique(y[sub_mask])) < 2:
                continue
            pipe = Pipeline([("scale", SafeStandardScaler()),
                             ("clf", LogisticRegression(max_iter=max_iter, C=C))])
            res = _fit_eval(pipe, X, y, sub_mask, test_mask, "logistic", seed)
            rows.append({
                "n_train_entities": len(chosen_set),
                "requested_size": size,
                "seed": seed,
                "accuracy": res.accuracy,
                "chance": res.chance,
                "n_train": res.n_train,
                "n_test": res.n_test,
            })
    return rows


def entities_to_reach(rows: Sequence[Dict[str, Any]], target: float) -> float:
    """Smallest number of training entities whose mean accuracy reaches `target`.

    Returns inf if the curve never gets there. The RATIO of this quantity
    between two models is the "how much harder to reach" number; report the
    ratio, not the raw counts, because the counts depend on your dataset size.
    """
    by_size: Dict[int, List[float]] = {}
    for r in rows:
        by_size.setdefault(int(r["n_train_entities"]), []).append(r["accuracy"])
    for size in sorted(by_size):
        if float(np.mean(by_size[size])) >= target:
            return float(size)
    return float("inf")


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

def probe_transfer(
    X_source: np.ndarray,
    X_target: np.ndarray,
    y: np.ndarray,
    entities: Sequence[str],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> List[Dict[str, Any]]:
    """Fit on the source model's activations, test on the target model's.

    Both must be the SAME prompts in the SAME order -- the function checks
    shape but cannot check ordering, so build both caches from one record
    list and never re-sort in between.

    Standardisation is deliberately off here. Refitting a scaler on the target
    would absorb exactly the distribution shift you are trying to detect; the
    right comparison is raw coefficients applied to raw activations.
    """
    if X_source.shape != X_target.shape:
        raise ValueError(f"shape mismatch {X_source.shape} vs {X_target.shape}")

    rows = []
    for seed in seeds:
        train_mask, test_mask = split_by_entity(entities, 0.3, seed, labels=y)
        probe = linear_probe(X_source, y, entities, seed=seed, standardise=False,
                             C_grid=(1.0,))
        if probe.coef is None:
            continue
        same = transfer_accuracy(probe.coef, X_source[test_mask], np.asarray(y)[test_mask])
        cross = transfer_accuracy(probe.coef, X_target[test_mask], np.asarray(y)[test_mask])
        rows.append({
            "seed": seed,
            "within_model_accuracy": same,
            "transfer_accuracy": cross,
            "transfer_drop": same - cross,
            "chance": probe.chance,
        })
    return rows


# ---------------------------------------------------------------------------
# The decision rule
# ---------------------------------------------------------------------------

def classify_localisation(
    post_accuracy: float,
    control_accuracy: float,
    transfer_accuracy_value: float,
    within_accuracy: float,
    entities_ratio: float,
    separation_ratio: float,
    decodable_margin: float = 0.10,
    transfer_margin: float = 0.15,
    efficiency_factor: float = 2.0,
) -> Dict[str, Any]:
    """Apply the Section 12.2 rule. Put these thresholds in the preregistration.

    post_accuracy     probe accuracy on the unlearned model
    control_accuracy  probe accuracy on never-taught control entities
    transfer_*        from probe_transfer(): source-fitted probe on the target
    entities_ratio    entities_to_reach(post) / entities_to_reach(pre)
    separation_ratio  class_separation(post) / class_separation(pre)

    Returns the label and the reasons, so the paper can quote the reasons
    rather than only the label.
    """
    decodable = post_accuracy > control_accuracy + decodable_margin
    transfers = transfer_accuracy_value > within_accuracy - transfer_margin
    harder = entities_ratio > efficiency_factor or separation_ratio < 0.5

    if not decodable:
        label = "removed"
    elif transfers and not harder:
        label = "preserved"
    elif not transfers:
        label = "transformed"
    else:
        label = "obscured"

    return {
        "label": label,
        "decodable": decodable,
        "transfers": transfers,
        "harder_to_reach": harder,
        "reasons": {
            "post_vs_control": post_accuracy - control_accuracy,
            "transfer_drop": within_accuracy - transfer_accuracy_value,
            "entities_ratio": entities_ratio,
            "separation_ratio": separation_ratio,
        },
        "caveat": "These labels describe DECODABILITY, not storage. A 'preserved' "
                  "verdict still requires the causal experiments (rungs 3-5) before "
                  "any claim that the model retains the information functionally.",
    }
