"""LEACE is the positive control: it must make the probing pipeline fail."""
import numpy as np
from src.analysis.erasure import (erase, fit_leace, leace_control_probe,
                                  max_cross_covariance)
from src.analysis.probes import linear_probe, mlp_probe, split_by_entity


def synthetic(n_entities=120, per_entity=6, n_classes=6, d=48, signal=3.0, seed=0):
    rng = np.random.default_rng(seed)
    entities = np.repeat([f"e{i:03d}" for i in range(n_entities)], per_entity)
    labels = np.repeat(np.arange(n_entities) % n_classes, per_entity)
    centers = rng.normal(size=(n_classes, d))
    X = rng.normal(size=(len(labels), d)) + signal * centers[labels]
    X += np.repeat(rng.normal(size=(n_entities, d)), per_entity, axis=0)
    return X, labels, entities


def test_cross_covariance_is_driven_to_zero():
    X, y, _ = synthetic()
    before = max_cross_covariance(X, y)
    after = max_cross_covariance(erase(X, y), y)
    assert after < 1e-6, f"before={before:.4g} after={after:.4g}"
    assert after < before / 1000


def test_linear_probe_falls_to_chance_after_erasure():
    """The control, done correctly: eraser fitted on the training split only."""
    X, y, ent = synthetic()
    before = linear_probe(X, y, ent, seed=0)
    accs = [leace_control_probe(X, y, ent, seed=s).accuracy for s in range(3)]
    after = float(np.mean(accs))
    assert before.accuracy > 0.6, f"signal too weak to make the control meaningful ({before.accuracy:.3f})"
    assert abs(after - before.chance) < 0.05, f"after={after:.3f} chance={before.chance:.3f}"


def test_fitting_the_eraser_on_all_data_goes_BELOW_chance():
    """Documents the trap in leace_control_probe's docstring as an executable fact.

    Fitting the eraser on train+test makes the class means globally identical,
    which anticorrelates the train and test residuals and makes the probe
    systematically wrong rather than uninformative. If this test ever starts
    passing at chance, someone has changed the eraser and the docstring needs
    updating too.
    """
    X, y, ent = synthetic()
    naive = float(np.mean([linear_probe(erase(X, y), y, ent, seed=s).accuracy
                           for s in range(3)]))
    correct = float(np.mean([leace_control_probe(X, y, ent, seed=s).accuracy
                             for s in range(3)]))
    chance = 1.0 / 6
    assert naive < chance / 2, f"expected far below chance, got {naive:.3f}"
    assert abs(correct - chance) < 0.05


def test_erasure_preserves_information_about_other_directions():
    """Erasure must be targeted: an unrelated label stays decodable.

    The unrelated label varies WITHIN entity here, so it is a per-example
    property and an entity-disjoint split is still the right test.
    """
    X, y, ent = synthetic(signal=3.0)
    rng = np.random.default_rng(1)
    other_center = rng.normal(size=(2, X.shape[1]))
    other = (np.arange(len(y)) % 2)
    X2 = X + 4.0 * other_center[other]
    tr, te = split_by_entity(ent, 0.3, 0, labels=y)
    eraser = fit_leace(X2[tr], y[tr])
    Xe = eraser.transform(X2)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000).fit(Xe[tr], other[tr])
    acc = float(np.mean(clf.predict(Xe[te]) == other[te]))
    assert acc > 0.75, f"unrelated label lost: {acc:.3f}"


def test_nonlinear_probe_may_still_recover():
    """LEACE erases LINEAR information only. This is a discussion point, not a bug.

    The test asserts only that the MLP does not do WORSE than chance; whether
    it beats chance depends on the data and either outcome is reportable.
    """
    X, y, ent = synthetic()
    tr, _ = split_by_entity(ent, 0.3, 0, labels=y)
    Xe = fit_leace(X[tr], y[tr]).transform(X)
    res = mlp_probe(Xe, y, ent, hidden=64, seed=0, max_iter=300)
    assert res.accuracy >= res.chance - 0.06


def test_eraser_is_reusable_on_new_data():
    X, y, ent = synthetic(seed=0)
    X2, y2, _ = synthetic(seed=1)
    eraser = fit_leace(X, y)
    out = eraser.transform(X2)
    assert out.shape == X2.shape
    assert eraser.n_erased_directions >= 1
