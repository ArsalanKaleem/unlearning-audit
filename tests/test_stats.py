import numpy as np
from src.stats.bootstrap import (bootstrap_difference, cluster_bootstrap_ci,
                                 holm_bonferroni, paired_cluster_permutation_test)


def clustered(n_entities=40, per_entity=6, p=0.7, seed=0):
    rng = np.random.default_rng(seed)
    clusters = np.repeat([f"e{i}" for i in range(n_entities)], per_entity)
    ent_p = rng.beta(2, 2, size=n_entities) * 0 + p
    vals = rng.binomial(1, np.repeat(ent_p, per_entity)).astype(float)
    return vals, clusters


def test_ci_contains_the_point_estimate():
    v, c = clustered()
    out = cluster_bootstrap_ci(v, c, n_boot=2000, seed=0)
    assert out["lo"] <= out["point"] <= out["hi"]


def test_clustered_ci_is_wider_than_naive_iid_ci():
    """The whole reason this module exists."""
    rng = np.random.default_rng(0)
    n_ent, per = 30, 8
    clusters = np.repeat([f"e{i}" for i in range(n_ent)], per)
    ent_mean = rng.uniform(0.2, 0.8, size=n_ent)          # strong between-entity variance
    vals = rng.binomial(1, np.repeat(ent_mean, per)).astype(float)
    clustered_ci = cluster_bootstrap_ci(vals, clusters, n_boot=2000, seed=0)
    iid_ci = cluster_bootstrap_ci(vals, np.arange(len(vals)).astype(str), n_boot=2000, seed=0)
    assert (clustered_ci["hi"] - clustered_ci["lo"]) > (iid_ci["hi"] - iid_ci["lo"])


def test_permutation_test_null_is_not_significant():
    v, c = clustered(seed=1)
    out = paired_cluster_permutation_test(v, v.copy(), c, n_perm=2000, seed=0)
    assert out["p_value"] > 0.5


def test_permutation_test_detects_a_real_shift():
    v, c = clustered(seed=2)
    out = paired_cluster_permutation_test(v, np.clip(v - 0.5, 0, 1), c, n_perm=2000, seed=0)
    assert out["p_value"] < 0.01


def test_holm_is_monotone_and_conservative():
    p = [0.001, 0.02, 0.04, 0.5]
    adj = holm_bonferroni(p)["p_adjusted"]
    assert np.all(adj >= np.array(p) - 1e-12)
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)


def test_bootstrap_difference_brackets_zero_for_identical_groups():
    v, c = clustered(seed=3)
    out = bootstrap_difference(v, c, v, c, n_boot=1000, seed=0)
    assert out["lo"] < 0 < out["hi"]
