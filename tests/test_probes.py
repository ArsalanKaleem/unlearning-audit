"""The Day 5 checkpoint, as a test: zero signal must give chance accuracy."""
import numpy as np
from src.analysis.probes import linear_probe, mlp_probe, probe_with_controls, transfer_accuracy


def synthetic(n_entities=120, per_entity=6, n_classes=4, d=32, signal=0.0, seed=0):
    """Entities carry a class; the class shifts the mean by `signal`.

    signal=0 means the activations contain no information about the label,
    so ANY above-chance accuracy is leakage.
    """
    rng = np.random.default_rng(seed)
    entities = np.repeat([f"e{i:03d}" for i in range(n_entities)], per_entity)
    ent_label = np.arange(n_entities) % n_classes
    labels = np.repeat(ent_label, per_entity)
    centers = rng.normal(size=(n_classes, d))
    X = rng.normal(size=(len(labels), d)) + signal * centers[labels]
    # a strong per-entity nuisance direction: this is what punishes bad splits
    ent_offset = rng.normal(size=(n_entities, d)) * 3.0
    X += np.repeat(ent_offset, per_entity, axis=0)
    return X, labels, entities


def test_zero_signal_gives_chance():
    X, y, ent = synthetic(signal=0.0)
    res = linear_probe(X, y, ent, seed=0)
    assert abs(res.accuracy - res.chance) < 0.12, f"acc={res.accuracy:.3f} chance={res.chance:.3f}"


def test_strong_signal_is_detected():
    X, y, ent = synthetic(signal=4.0)
    res = linear_probe(X, y, ent, seed=0)
    assert res.accuracy > 0.60


def test_accuracy_increases_with_signal():
    accs = [linear_probe(*synthetic(signal=s), seed=0).accuracy for s in (0.0, 1.0, 3.0)]
    assert accs[0] < accs[2]


def test_mlp_probe_runs_and_is_at_chance_without_signal():
    X, y, ent = synthetic(signal=0.0)
    res = mlp_probe(X, y, ent, hidden=16, seed=0, max_iter=200)
    assert abs(res.accuracy - res.chance) < 0.15


def test_selectivity_is_near_zero_without_signal():
    """Averaged across probe seeds, as the real pipeline does.

    A single seed is noisy enough that a 0.15 swing is unremarkable; this is
    exactly why layerwise_probe() runs five seeds and you report the mean.
    """
    X, y, ent = synthetic(signal=0.0)
    sel = np.mean([probe_with_controls(X, y, ent, seed=s)["selectivity"] for s in range(5)])
    assert abs(sel) < 0.10, f"selectivity={sel:.3f}"


def test_selectivity_is_large_with_signal():
    X, y, ent = synthetic(signal=4.0)
    out = probe_with_controls(X, y, ent, seed=0)
    assert out["selectivity"] > 0.3


def test_probe_is_deterministic_for_a_seed():
    X, y, ent = synthetic(signal=2.0)
    a = linear_probe(X, y, ent, seed=1).accuracy
    b = linear_probe(X, y, ent, seed=1).accuracy
    assert a == b


def test_transfer_accuracy_uses_source_coefficients():
    X, y, ent = synthetic(signal=4.0)
    res = linear_probe(X, y, ent, seed=0)
    # transferring to the same (unstandardised) data should beat chance
    acc = transfer_accuracy(res.coef, X, y)
    assert acc > res.chance
