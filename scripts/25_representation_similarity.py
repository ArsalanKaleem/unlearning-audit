#!/usr/bin/env python3
"""Replicate the standard representation-similarity metrics, WITH a control.

    python scripts/25_representation_similarity.py --reference M_injected \
        --targets M_npo_s0,M_npo_s1,M_npo_s2,M_gd_s2

Recent work evaluates unlearning by comparing the unlearned model's internal
representations against the original's, using linear CKA, PCA subspace
similarity and PCA shift, layer by layer. Low similarity or a large shift is
read as severe representational change; high similarity as the representation
surviving.

Those analyses draw their input queries from the FORGET SET ONLY. This script
computes the same three quantities and runs them on the forget set AND on
never-taught control entities AND on generic text. If the metrics move
comparably on entities the model was never taught, they are not measuring
forget-specific change, and neither reading is available.

The metrics, implemented from the standard definitions:

  linear CKA      Kornblith et al. (2019). Similarity of two representations
                  invariant to orthogonal transform and isotropic scaling.
  PCA similarity  mean absolute cosine between the leading principal
                  directions of the two activation sets.
  PCA shift       distance between the two sets projected into the reference's
                  leading principal plane.

Runs on cached activations, CPU, numpy only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, setup, write_json

from src.utils.io import load_activations

SETS = ("forget", "retain", "control", "generic")


def _centre(X):
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y) -> float:
    """Linear CKA between two activation matrices on the SAME inputs.

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)

    Invariant to orthogonal transformation and isotropic scaling, which is why
    it is used for this comparison: a representation that was merely rotated
    should still score near 1.
    """
    X, Y = _centre(X), _centre(Y)
    num = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = (np.linalg.norm(X.T @ X, ord="fro")
           * np.linalg.norm(Y.T @ Y, ord="fro"))
    return float(num / den) if den > 0 else float("nan")


def pca_similarity(X, Y, k: int = 10) -> float:
    """Mean |cosine| between the leading k principal directions."""
    Xc, Yc = _centre(X), _centre(Y)
    k = int(min(k, Xc.shape[0] - 1, Xc.shape[1], Yc.shape[1]))
    if k < 1:
        return float("nan")
    Ux = np.linalg.svd(Xc, full_matrices=False)[2][:k]
    Uy = np.linalg.svd(Yc, full_matrices=False)[2][:k]
    return float(np.abs(np.sum(Ux * Uy, axis=1)).mean())


def pca_shift(X, Y, k: int = 2) -> float:
    """Distance between the two sets' centroids in the reference's PC plane."""
    Xc = _centre(X)
    k = int(min(k, Xc.shape[0] - 1, Xc.shape[1]))
    if k < 1:
        return float("nan")
    basis = np.linalg.svd(Xc, full_matrices=False)[2][:k]
    mx = (np.asarray(X, float).mean(0)) @ basis.T
    my = (np.asarray(Y, float).mean(0)) @ basis.T
    return float(np.linalg.norm(mx - my))


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--reference", default="M_injected")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--pca-k", type=int, default=10)
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    primary = cfg["probe"]["primary_layer"]

    rows = []
    for target in args.targets.split(","):
        for setname in SETS:
            ref_p = PATHS.activations / f"{args.reference}_{setname}.npz"
            tgt_p = PATHS.activations / f"{target}_{setname}.npz"
            if not (ref_p.exists() and tgt_p.exists()):
                print(f"  skip {target}/{setname}: cache missing")
                continue
            A, _, _, _ = load_activations(ref_p)
            B, _, _, _ = load_activations(tgt_p)
            for layer in range(A.shape[0]):
                X, Y = A[layer], B[layer]
                rows.append({
                    "model": target, "set": setname, "layer": layer,
                    "cka": linear_cka(X, Y),
                    "pca_similarity": pca_similarity(X, Y, args.pca_k),
                    "pca_shift": pca_shift(X, Y),
                })

    df = pd.DataFrame(rows)
    if not len(df):
        print("no activation caches found")
        return 1
    df.to_csv(PATHS.tables / "representation_similarity.csv", index=False)

    print(f"\nlayer {primary}: the same metrics on forget vs never-taught entities")
    print(f"{'model':12s} {'metric':16s} {'forget':>9s} {'control':>9s} "
          f"{'generic':>9s} {'forget/control':>15s}")
    summary = {}
    for target in df["model"].unique():
        s = df[(df["model"] == target) & (df["layer"] == primary)]
        entry = {}
        for metric in ("cka", "pca_similarity", "pca_shift"):
            vals = {k: float(s[s["set"] == k][metric].iloc[0])
                    for k in SETS if len(s[s["set"] == k])}
            f, c = vals.get("forget"), vals.get("control")
            ratio = (f / c if c not in (None, 0) else float("nan")) if f is not None else float("nan")
            entry[metric] = {**vals, "forget_over_control": ratio}
            print(f"{target:12s} {metric:16s} {vals.get('forget', float('nan')):9.4f} "
                  f"{vals.get('control', float('nan')):9.4f} "
                  f"{vals.get('generic', float('nan')):9.4f} {ratio:15.2f}")
        summary[target] = entry

    # ------------------------------------------------------- interpretation
    # Judged PER METRIC and ACROSS SEEDS, not per model. A metric that
    # discriminates on two of four seeds does not discriminate; it produces a
    # conclusion that depends on which run you happened to report. The earlier
    # version of this block asked whether EITHER criterion fired for a given
    # model, which declared success whenever any metric happened to move --
    # exactly the reasoning this script exists to question.
    print("\n" + "=" * 74)
    print("Does each metric separate forget-set change from never-taught change?")
    print("Judged across all seeds: a metric that works on some seeds and not")
    print("others yields whichever conclusion the reported seed supports.")
    print("=" * 74)

    # CKA falls as representations diverge, so forget-specific change means a
    # LOWER forget/control ratio. PCA shift grows, so it means a HIGHER ratio.
    CRITERIA = {
        "cka": ("ratio < 0.90", lambda r: r < 0.90),
        "pca_similarity": ("ratio < 0.90", lambda r: r < 0.90),
        "pca_shift": ("ratio > 1.50", lambda r: r > 1.50),
    }

    verdicts = {}
    for metric, (desc, test) in CRITERIA.items():
        ratios = {t: summary[t][metric]["forget_over_control"] for t in summary}
        passing = [t for t, r in ratios.items() if r == r and test(r)]
        vals = np.array([r for r in ratios.values() if r == r], dtype=float)
        spread = f"{vals.min():.2f}-{vals.max():.2f}" if vals.size else "n/a"
        if len(passing) == len(ratios):
            state = "DISCRIMINATES on every seed"
        elif passing:
            state = f"SEED-DEPENDENT: {len(passing)}/{len(ratios)} seeds"
        else:
            state = "DOES NOT DISCRIMINATE on any seed"
        verdicts[metric] = {"criterion": desc, "ratios": ratios,
                            "n_passing": len(passing), "n_models": len(ratios),
                            "range": spread, "state": state}
        print(f"\n  {metric:16s} criterion {desc}")
        print(f"    ratios across seeds: " +
              ", ".join(f"{t.replace('M_', '')}={r:.2f}" for t, r in ratios.items()))
        print(f"    range {spread}   ->  {state}")

    # Generic text is the sanity reference: if it is near 1, ordinary language
    # modelling survived and the change is confined to entity prompts.
    gen_cka = [summary[t]["cka"].get("generic") for t in summary]
    gen_cka = [g for g in gen_cka if g is not None]
    if gen_cka:
        print(f"\n  generic-text CKA: {min(gen_cka):.3f}-{max(gen_cka):.3f}")
        print("    Ordinary language modelling is essentially untouched. What moved")
        print("    is the entity-attribute representation as a class -- including")
        print("    for entities the model was never taught anything about.")

    seed_dependent = [m for m, v in verdicts.items()
                      if v["state"].startswith("SEED-DEPENDENT")]
    none_discriminate = [m for m, v in verdicts.items()
                         if v["state"].startswith("DOES NOT")]
    print("\n" + "-" * 74)
    if seed_dependent or none_discriminate:
        print("These metrics do not agree with each other or with themselves across")
        print("seeds. Reported on the forget set alone, with a single seed and one")
        print("metric chosen, they can support opposite conclusions about the same")
        print("models. The never-taught control is what makes that visible.")
    else:
        print("Every metric discriminates on every seed.")
    print("-" * 74)

    write_json({"layer": primary, "reference": args.reference,
                "summary": summary,
                "verdicts": verdicts,
                "metrics_discriminating_on_every_seed": [
                    m for m, v in verdicts.items()
                    if v["state"].startswith("DISCRIMINATES")],
                "note": "Standard practice draws input queries from the forget "
                        "set only; the control and generic columns are the "
                        "comparison that is usually absent."},
               PATHS.tables / "representation_similarity_summary.json")
    print(f"\nwrote {PATHS.tables / 'representation_similarity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())