import numpy as np
from src.analysis.localisation import (classify_localisation, entities_to_reach,
                                       probe_transfer, sample_efficiency_curve)
from tests.test_erasure import synthetic


def test_sample_efficiency_increases_with_training_entities():
    X, y, ent = synthetic(signal=1.2)
    rows = sample_efficiency_curve(X, y, ent, sizes=(6, 12, 24, 0), seeds=(0, 1))
    by_size = {}
    for r in rows:
        by_size.setdefault(r["n_train_entities"], []).append(r["accuracy"])
    sizes = sorted(by_size)
    small = np.mean(by_size[sizes[0]])
    large = np.mean(by_size[sizes[-1]])
    assert large > small, f"{small:.3f} -> {large:.3f}"


def test_test_set_is_fixed_within_a_seed():
    X, y, ent = synthetic()
    rows = [r for r in sample_efficiency_curve(X, y, ent, sizes=(6, 12, 0), seeds=(0,))]
    assert len({r["n_test"] for r in rows}) == 1


def test_entities_to_reach_is_monotone_and_handles_unreachable():
    rows = [{"n_train_entities": n, "accuracy": a}
            for n, a in [(6, 0.2), (12, 0.5), (24, 0.8)]]
    assert entities_to_reach(rows, 0.45) == 12
    assert entities_to_reach(rows, 0.99) == float("inf")


def test_transfer_is_high_within_and_low_across_a_rotation():
    X, y, ent = synthetic(signal=3.0)
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(X.shape[1], X.shape[1])))
    rows = probe_transfer(X, X @ Q, y, ent, seeds=(0, 1))
    assert np.mean([r["within_model_accuracy"] for r in rows]) > 0.6
    assert np.mean([r["transfer_accuracy"] for r in rows]) < 0.4


def test_transfer_is_perfect_to_an_identical_model():
    X, y, ent = synthetic(signal=3.0)
    rows = probe_transfer(X, X.copy(), y, ent, seeds=(0,))
    assert rows[0]["transfer_drop"] == 0.0


def test_decision_rule_labels():
    common = dict(control_accuracy=0.1, within_accuracy=0.8)
    assert classify_localisation(post_accuracy=0.12, transfer_accuracy_value=0.1,
                                 entities_ratio=1.0, separation_ratio=1.0,
                                 **common)["label"] == "removed"
    assert classify_localisation(post_accuracy=0.8, transfer_accuracy_value=0.78,
                                 entities_ratio=1.0, separation_ratio=1.0,
                                 **common)["label"] == "preserved"
    assert classify_localisation(post_accuracy=0.8, transfer_accuracy_value=0.15,
                                 entities_ratio=1.0, separation_ratio=1.0,
                                 **common)["label"] == "transformed"
    assert classify_localisation(post_accuracy=0.8, transfer_accuracy_value=0.78,
                                 entities_ratio=4.0, separation_ratio=0.3,
                                 **common)["label"] == "obscured"
