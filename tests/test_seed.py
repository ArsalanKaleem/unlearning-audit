import numpy as np
from src.utils.seed import rng_for, set_all_seeds


def test_named_streams_are_reproducible():
    assert rng_for("a", 0).integers(0, 10**6) == rng_for("a", 0).integers(0, 10**6)


def test_named_streams_are_independent():
    assert rng_for("a", 0).integers(0, 10**6) != rng_for("b", 0).integers(0, 10**6)


def test_seed_changes_stream():
    assert rng_for("a", 0).integers(0, 10**6) != rng_for("a", 1).integers(0, 10**6)


def test_set_all_seeds_makes_numpy_reproducible():
    set_all_seeds(7)
    a = np.random.rand(5)
    set_all_seeds(7)
    assert np.allclose(a, np.random.rand(5))
