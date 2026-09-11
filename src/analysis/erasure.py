"""Linear concept erasure (LEACE), used here as a POSITIVE CONTROL.

The problem this solves
-----------------------
Suppose you probe an unlearned model, find the fact is still decodable, and
report that unlearning did not erase it. A reviewer asks the obvious question:
how do you know your probing setup would have noticed if it HAD been erased?

LEACE answers it. It removes all linear information about a label from a set
of activations, in closed form, with a guarantee: after the transform, the
cross-covariance between the activations and the one-hot labels is exactly
zero, so no linear classifier can beat the majority-class baseline. Run your
pipeline on LEACE-processed activations. If probe accuracy falls to chance
there but stays high on the real unlearned model, your null result is
evidence rather than an absence of evidence.

The method (Belrose et al., 2023, "LEACE: Perfect linear concept erasure in
closed form"). Verify the reference before citing it.

  Sigma_XX = Cov(X),  Sigma_XZ = Cov(X, Z)
  W        = Sigma_XX^{-1/2}                    (whitening)
  M        = W Sigma_XZ
  P_M      = orthogonal projection onto col(M)
  r(x)     = x - W^+ P_M W (x - mu)

Two things to note in the paper
-------------------------------
1. LEACE erases LINEAR information only. A nonlinear probe may still recover
   the label from erased activations, and that is not a bug -- it is a useful
   demonstration of what "erasure" does and does not mean, and it is worth a
   sentence in the discussion.
2. The eraser is fitted on the activations it is applied to. That is correct
   for a control (it is a data transformation, not a classifier), but say so,
   because it is not the same setting as the probes elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _inv_sqrt_psd(mat: np.ndarray, eps: float = 1e-8):
    """Symmetric inverse square root and its pseudo-inverse, rank-safe.

    Activation covariance matrices are routinely rank-deficient (fewer
    examples than dimensions), so a naive inverse explodes. Eigenvalues below
    `eps * max` are treated as zero in both directions.
    """
    mat = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(mat)
    cutoff = eps * max(float(vals.max()), 1e-12)
    keep = vals > cutoff
    inv_sqrt = (vecs[:, keep] * (vals[keep] ** -0.5)) @ vecs[:, keep].T
    sqrt = (vecs[:, keep] * (vals[keep] ** 0.5)) @ vecs[:, keep].T
    return inv_sqrt, sqrt


@dataclass
class LeaceEraser:
    """A fitted eraser. Apply with `.transform(X)`."""

    mean: np.ndarray
    projection: np.ndarray      # the full d x d map applied to centred X
    n_erased_directions: int

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float64) - self.mean
        return (Xc - Xc @ self.projection.T) + self.mean

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)


def one_hot(labels: Sequence[int], n_classes: int | None = None) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    k = n_classes or int(y.max()) + 1
    Z = np.zeros((len(y), k))
    Z[np.arange(len(y)), y] = 1.0
    return Z


def fit_leace(X: np.ndarray, labels: Sequence[int], eps: float = 1e-8) -> LeaceEraser:
    """Fit the closed-form eraser for `labels` on activations `X` (n, d)."""
    X = np.asarray(X, dtype=np.float64)
    Z = one_hot(labels)
    n = X.shape[0]

    mu = X.mean(axis=0)
    Xc = X - mu
    Zc = Z - Z.mean(axis=0)

    sigma_xx = (Xc.T @ Xc) / n
    sigma_xz = (Xc.T @ Zc) / n

    W_inv_sqrt, W_sqrt = _inv_sqrt_psd(sigma_xx, eps)
    M = W_inv_sqrt @ sigma_xz                       # (d, k)

    # orthogonal projection onto the column space of M
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    rank = int((s > eps * max(float(s.max()) if s.size else 0.0, 1e-12)).sum())
    U = U[:, :rank]
    P_M = U @ U.T

    projection = W_sqrt @ P_M @ W_inv_sqrt          # applied to centred X
    return LeaceEraser(mean=mu, projection=projection, n_erased_directions=rank)


def erase(X: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Convenience: fit and apply in one call."""
    return fit_leace(X, labels).transform(X)


def erase_layerwise(acts: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Erase the concept independently at every layer. acts: (n_layers, n, d)."""
    out = np.empty_like(acts, dtype=np.float32)
    for layer in range(acts.shape[0]):
        out[layer] = erase(acts[layer], labels).astype(np.float32)
    return out


def max_cross_covariance(X: np.ndarray, labels: Sequence[int]) -> float:
    """The quantity LEACE drives to zero. Report it as the control's receipt.

    A value near machine precision after erasure is the numerical proof that
    the transform did what it claims; if it is not near zero, the eigenvalue
    cutoff was too aggressive for your data.
    """
    X = np.asarray(X, dtype=np.float64)
    Z = one_hot(labels)
    Xc = X - X.mean(axis=0)
    Zc = Z - Z.mean(axis=0)
    cov = (Xc.T @ Zc) / X.shape[0]
    scale = float(np.abs(Xc).max()) or 1.0
    return float(np.abs(cov).max() / scale)


# ---------------------------------------------------------------------------
# The control, done correctly
# ---------------------------------------------------------------------------

def leace_control_probe(
    X: np.ndarray,
    labels: Sequence[int],
    entities: Sequence[str],
    seed: int = 0,
    test_frac: float = 0.3,
    max_iter: int = 2000,
    C: float = 1.0,
):
    """Fit the eraser on the TRAINING SPLIT ONLY, then probe. Returns ProbeResult.

    Read this before you write your own version -- it cost me a debugging
    session and it will cost you one too.

    If you fit the eraser on ALL the data and then probe with a train/test
    split, accuracy comes out far BELOW chance, not at it. The reason is that
    erasure imposes a GLOBAL constraint: after it, the class means over the
    whole dataset are identical. Train and test are a partition of that whole,
    so if the class-c mean is displaced by +d within the training entities, it
    is displaced by -d * (n_train / n_test) within the test entities, exactly.
    The probe learns the training displacement and it generalises with the
    sign flipped, so it is reliably, systematically wrong.

    Below-chance accuracy is not a stronger version of "erased". It is a leak
    in the opposite direction, and a reviewer will spot it. Treat the eraser
    like a scaler: fit on train, apply to both.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.analysis.probes import (SafeStandardScaler, _fit_eval, assert_no_leakage,
                                     split_by_entity)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels)
    train_mask, test_mask = split_by_entity(entities, test_frac, seed, labels=y)
    assert_no_leakage(np.asarray(entities), train_mask, test_mask)

    eraser = fit_leace(X[train_mask], y[train_mask])
    X_erased = eraser.transform(X)

    pipe = Pipeline([
        ("scale", SafeStandardScaler()),
        ("clf", LogisticRegression(max_iter=max_iter, C=C)),
    ])
    res = _fit_eval(pipe, X_erased, y, train_mask, test_mask, "logistic_leace", seed)
    res.best_params = {"C": C, "n_erased_directions": eraser.n_erased_directions}
    return res


def layerwise_erasure_control(
    acts: np.ndarray,
    labels: Sequence[int],
    entities: Sequence[str],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> list:
    """Run the erasure control at every layer. Tidy rows, same schema as probes.

    Plot this as a third line on Figure 4. It is the line that tells the
    reader what a genuinely erased representation looks like under YOUR
    pipeline, which is what makes the other two lines interpretable.
    """
    rows = []
    for layer in range(acts.shape[0]):
        for seed in seeds:
            r = leace_control_probe(acts[layer], labels, entities, seed=seed)
            rows.append({
                "layer": layer,
                "seed": seed,
                "accuracy": r.accuracy,
                "chance": r.chance,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "probe": "leace_erased",
                "condition": "leace_control",
            })
    return rows
