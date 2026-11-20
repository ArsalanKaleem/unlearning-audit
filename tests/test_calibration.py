import numpy as np

from src.analysis.calibration import (auroc, crossfit_probe, entity_folds,
                                      normalised_position, paired_auroc_ci)


def test_auroc_extremes():
    assert auroc(np.array([2, 3]), np.array([0, 1])) == 1.0
    assert auroc(np.array([0, 1]), np.array([2, 3])) == 0.0
    assert auroc(np.array([1, 1]), np.array([1, 1])) == 0.5


def test_entity_folds_disjoint_and_balanced():
    ents = np.repeat([f"e{i}" for i in range(24)], 3)
    labs = np.repeat(np.arange(24) % 6, 3)
    f = entity_folds(ents, labs, 4, 0)
    for e in np.unique(ents):
        assert len(set(f[ents == e])) == 1          # an entity never straddles folds
    assert np.bincount(f).min() == np.bincount(f).max()


def test_crossfit_probe_reads_planted_signal_and_chance_without_it():
    rng = np.random.default_rng(0)
    n_ent, per = 48, 4
    ents = np.repeat([f"e{i}" for i in range(n_ent)], per)
    y = np.repeat(np.arange(n_ent) % 6, per)
    centres = rng.normal(size=(6, 20)) * 3
    X_sig = centres[y] + rng.normal(size=(len(y), 20))
    X_null = rng.normal(size=(len(y), 20))
    acc_sig = np.mean(list(crossfit_probe(X_sig, y, ents, n_repeats=1).values()))
    acc_null = np.mean(list(crossfit_probe(X_null, y, ents, n_repeats=1).values()))
    assert acc_sig > 0.9
    assert acc_null < 0.35


def test_normalised_position_endpoints():
    a = np.zeros(30)
    r = np.ones(30)
    assert normalised_position(a, a, r, n_boot=50)["position"] == 0.0
    assert normalised_position(r, a, r, n_boot=50)["position"] == 1.0


def test_paired_auroc_ci_contains_point():
    rng = np.random.default_rng(1)
    x, y = rng.normal(1, 1, 40), rng.normal(0, 1, 40)
    out = paired_auroc_ci(x, y, n_boot=200)
    assert out["lo"] <= out["auroc"] <= out["hi"]
