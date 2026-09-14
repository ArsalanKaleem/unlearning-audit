#!/usr/bin/env python3
"""Day 36 --- the statistics pass. Every headline number, with an interval.

    python scripts/20_statistics.py

Produces results/tables/final_statistics.csv, which is what the paper cites.
Nothing in the paper should quote a number that is not in this file.

Three rules, all inherited from the frozen preregistration:

1. Every interval clusters on ENTITIES. Six prompts about one researcher are
   not six observations; treating them as such narrows intervals by roughly
   sqrt(6).
2. The layer family gets Holm-Bonferroni. The layer-9 primary test is reported
   UNCORRECTED and labelled primary, because it was preregistered as the single
   confirmatory test.
3. Seed variance is reported next to every probe number. If the spread across
   probe seeds exceeds the effect, there is no effect.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, setup, write_json

from src.stats.bootstrap import (cluster_bootstrap_ci, holm_bonferroni,
                                 paired_cluster_permutation_test, seed_variance)

MODELS = ["M_injected", "M_npo_s0", "M_npo_s1", "M_npo_s2", "M_gd_s2"]
SETS = ["forget", "retain", "control", "paraphrase", "related"]


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    models = args.models.split(",")
    primary = cfg["probe"]["primary_layer"]
    n_boot = cfg["stats"]["n_bootstrap"]
    n_perm = cfg["stats"]["n_permutations"]

    rows = []

    # ---------------------------------------------------- behavioural metrics
    print("behavioural (entity-clustered bootstrap)")
    behav = {}
    for m in models:
        p = PATHS.tables / f"behaviour_{m}.csv"
        if not p.exists():
            print(f"  skip {m}: {p.name} missing")
            continue
        df = pd.read_csv(p)
        behav[m] = df
        for s in SETS:
            sub = df[df["set"] == s]
            if not len(sub):
                continue
            ci = cluster_bootstrap_ci(sub["constrained_correct"], sub["entity_id"],
                                      n_boot=n_boot, seed=cfg["seed"])
            rows.append({"family": "behavioural", "model": m, "set": s,
                         "metric": "constrained_accuracy", "layer": None,
                         "point": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
                         "n_entities": ci["n_clusters"], "corrected": False})
            print(f"  {m:12s} {s:12s} {ci['point']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]")

    # forget vs control within each model: is forgetting complete?
    for m, df in behav.items():
        f = df[df["set"] == "forget"]
        c = df[df["set"] == "control"]
        if not len(f) or not len(c):
            continue
        from src.stats.bootstrap import bootstrap_difference
        d = bootstrap_difference(f["constrained_correct"], f["entity_id"],
                                 c["constrained_correct"], c["entity_id"],
                                 n_boot=n_boot, seed=cfg["seed"])
        rows.append({"family": "behavioural", "model": m, "set": "forget-minus-control",
                     "metric": "difference", "layer": None,
                     "point": d["point"], "lo": d["lo"], "hi": d["hi"],
                     "n_entities": None, "corrected": False})
        overlap = d["lo"] <= 0 <= d["hi"]
        print(f"  {m:12s} forget-control {d['point']:+.3f} "
              f"[{d['lo']:+.3f}, {d['hi']:+.3f}]"
              + ("  (indistinguishable)" if overlap else ""))

    # ------------------------------------------------------- probing, by layer
    print("\nprobing (5 seeds per layer; Holm across the layer family)")
    pp = PATHS.tables / "probe_logistic_forget.csv"
    if pp.exists():
        pdf = pd.read_csv(pp)
        pdf = pdf[pdf["set"] == "forget"]
        ref = pdf[pdf["model"] == "M_injected"]

        for m in [x for x in models if x != "M_injected"]:
            sub = pdf[pdf["model"] == m]
            if not len(sub):
                continue
            pvals, layers = [], []
            for layer in sorted(sub["layer"].unique()):
                a = ref[ref["layer"] == layer]["accuracy"].values
                b = sub[sub["layer"] == layer]["accuracy"].values
                if len(a) != len(b) or len(a) < 2:
                    continue
                # paired across PROBE SEEDS: a sign-flip test with 5 clusters
                # cannot go below p = 2^-5 = 0.031. That bound is reported so
                # nobody reads a non-significant layer as a null effect.
                t = paired_cluster_permutation_test(
                    a, b, [f"seed{i}" for i in range(len(a))],
                    n_perm=min(n_perm, 2 ** len(a) * 100), seed=layer)
                pvals.append(t["p_value"])
                layers.append(layer)

            if not pvals:
                continue
            holm = holm_bonferroni(pvals, alpha=cfg["stats"]["alpha"])
            for layer, p, padj, rej in zip(layers, pvals, holm["p_adjusted"],
                                           holm["reject"]):
                acc = sub[sub["layer"] == layer]["accuracy"]
                sv = seed_variance(acc)
                is_primary = layer == primary
                rows.append({
                    "family": "probe_layer", "model": m, "set": "forget",
                    "metric": "accuracy", "layer": layer,
                    "point": sv["mean"], "lo": None, "hi": None,
                    "seed_sd": sv["std"], "p_raw": p,
                    "p_holm": None if is_primary else float(padj),
                    "significant": bool(rej),
                    "primary": is_primary,
                    "corrected": not is_primary,
                    "n_entities": None,
                })
            n_sig = int(holm["reject"].sum())
            pr = [r for r in rows if r["family"] == "probe_layer"
                  and r["model"] == m and r["layer"] == primary]
            if pr:
                r = pr[0]
                print(f"  {m:12s} PRIMARY layer {primary}: acc={r['point']:.3f} "
                      f"(sd {r['seed_sd']:.3f}) p={r['p_raw']:.4f} "
                      f"[uncorrected, preregistered]")
            print(f"  {m:12s} {n_sig}/{len(pvals)} layers survive Holm "
                  f"(min attainable p = {2 ** -5:.3f} with 5 seeds)")

    # --------------------------------------------------------------- drift
    print("\ndrift (forget vs never-taught control)")
    ds = PATHS.tables / "drift_summary.json"
    if ds.exists():
        import json
        for m, d in json.load(open(ds)).items():
            rows.append({"family": "drift", "model": m, "set": "forget_over_control",
                         "metric": "ratio", "layer": primary,
                         "point": d.get("forget_over_control"), "lo": None, "hi": None,
                         "threshold": cfg.get("localisation", {}).get("drift_ratio_min", 1.5),
                         "significant": bool(d.get("forget_specific")),
                         "corrected": False, "n_entities": None})
            print(f"  {m:12s} forget/control = {d.get('forget_over_control', float('nan')):.2f} "
                  f"-> {'forget-specific' if d.get('forget_specific') else 'GLOBAL'}")

    # ------------------------------------------------------------- write out
    out = pd.DataFrame(rows)
    out.to_csv(PATHS.tables / "final_statistics.csv", index=False)
    print(f"\nwrote {PATHS.tables / 'final_statistics.csv'} ({len(out)} rows)")

    bare = out[(out["lo"].isna()) & (out["family"] == "behavioural")]
    if len(bare):
        print(f"WARNING: {len(bare)} behavioural rows have no interval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())