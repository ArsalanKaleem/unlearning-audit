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

A note on the probe test, because it caught this project out. The layer-wise
comparison is a paired sign-flip permutation test across PROBE SEEDS, and such
a test has a minimum attainable p-value of 2^-N for N clusters. With 5 seeds
that floor is 0.031, and every model returned p = 0.0594 -- the next attainable
value up -- regardless of an effect size of 0.65. The test was bounded by the
cluster count, not by the data. Raising the seeds to 20 moved layer-9 accuracy
by less than 0.05 and moved p from 0.0594 to 0.0001. The floor is printed with
every result so nobody reads a bounded non-significant layer as a null effect.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, setup, write_json

from src.stats.bootstrap import (bootstrap_difference, cluster_bootstrap_ci,
                                 holm_bonferroni, paired_cluster_permutation_test,
                                 seed_variance)

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
    n_seeds = cfg["probe"]["n_seeds"]
    p_floor = 2.0 ** -n_seeds

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

    # Paraphrase split by entity type. The combined paraphrase number mixes
    # forget and retain entities and cannot be interpreted: it fell from 0.356
    # to ~0.25, but that is a blend of forget entities collapsing and retain
    # entities holding. The split is what shows forgetting generalised.
    print("\nparaphrase, split by entity type")
    for m, df in behav.items():
        p = df[df["set"] == "paraphrase"]
        if "split" not in p.columns:
            print(f"  {m:12s} no `split` column; rerun 04_behavioural_eval.py")
            continue
        for split in ("forget", "retain"):
            sub = p[p["split"] == split]
            if not len(sub):
                continue
            ci = cluster_bootstrap_ci(sub["constrained_correct"], sub["entity_id"],
                                      n_boot=n_boot, seed=cfg["seed"])
            rows.append({"family": "behavioural", "model": m,
                         "set": f"paraphrase[{split}]",
                         "metric": "constrained_accuracy", "layer": None,
                         "point": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
                         "n_entities": ci["n_clusters"], "corrected": False})
            print(f"  {m:12s} paraphrase[{split}]  {ci['point']:.3f} "
                  f"[{ci['lo']:.3f}, {ci['hi']:.3f}]")

    # forget vs control within each model: is forgetting complete?
    print("\nforget minus never-taught control (is forgetting complete?)")
    for m, df in behav.items():
        f = df[df["set"] == "forget"]
        c = df[df["set"] == "control"]
        if not len(f) or not len(c):
            continue
        d = bootstrap_difference(f["constrained_correct"], f["entity_id"],
                                 c["constrained_correct"], c["entity_id"],
                                 n_boot=n_boot, seed=cfg["seed"])
        overlap = d["lo"] <= 0 <= d["hi"]
        rows.append({"family": "behavioural", "model": m, "set": "forget-minus-control",
                     "metric": "difference", "layer": None,
                     "point": d["point"], "lo": d["lo"], "hi": d["hi"],
                     "significant": not overlap,
                     "n_entities": None, "corrected": False})
        print(f"  {m:12s} {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
              + ("  (indistinguishable from never-taught)" if overlap else ""))

    # ------------------------------------------------------- probing, by layer
    print(f"\nprobing ({n_seeds} seeds per layer; Holm across the layer family)")
    print(f"  sign-flip floor with {n_seeds} seeds: p >= {p_floor:.2g}")
    pp = PATHS.tables / "probe_logistic_forget.csv"
    if pp.exists():
        pdf = pd.read_csv(pp)
        pdf = pdf[pdf["set"] == "forget"]
        ref = pdf[pdf["model"] == "M_injected"]

        # M_injected's own profile, for the paper's reference column
        for layer in sorted(ref["layer"].unique()):
            sv = seed_variance(ref[ref["layer"] == layer]["accuracy"])
            rows.append({"family": "probe_layer", "model": "M_injected",
                         "set": "forget", "metric": "accuracy", "layer": layer,
                         "point": sv["mean"], "seed_sd": sv["std"],
                         "primary": layer == primary, "corrected": False,
                         "lo": None, "hi": None, "n_entities": None})

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
                # Paired across PROBE SEEDS. See the module docstring: this test
                # cannot return a p-value below 2^-len(a), and that bound is
                # reported alongside so a bounded result is not read as a null.
                t = paired_cluster_permutation_test(
                    a, b, [f"seed{i}" for i in range(len(a))],
                    n_perm=min(n_perm, 2 ** min(len(a), 20) * 100), seed=layer)
                pvals.append(t["p_value"])
                layers.append(layer)

            if not pvals:
                continue
            holm = holm_bonferroni(pvals, alpha=cfg["stats"]["alpha"])
            for layer, p, padj, rej in zip(layers, pvals, holm["p_adjusted"],
                                           holm["reject"]):
                acc = sub[sub["layer"] == layer]["accuracy"]
                sv = seed_variance(acc)
                ref_mean = float(ref[ref["layer"] == layer]["accuracy"].mean())
                is_primary = layer == primary
                rows.append({
                    "family": "probe_layer", "model": m, "set": "forget",
                    "metric": "accuracy", "layer": layer,
                    "point": sv["mean"], "lo": None, "hi": None,
                    "seed_sd": sv["std"], "n_seeds": sv["n_seeds"],
                    "reference_M_injected": ref_mean,
                    "drop_from_reference": ref_mean - sv["mean"],
                    "p_raw": p,
                    "p_floor": p_floor,
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
                bounded = " (AT THE FLOOR)" if r["p_raw"] <= p_floor * 1.01 else ""
                print(f"  {m:12s} PRIMARY layer {primary}: acc={r['point']:.3f} "
                      f"(sd {r['seed_sd']:.3f}, drop {r['drop_from_reference']:+.3f}) "
                      f"p={r['p_raw']:.4f}{bounded} [uncorrected, preregistered]")
            print(f"  {m:12s} {n_sig}/{len(pvals)} layers survive Holm")

    # --------------------------------------------------------------- drift
    print("\ndrift (forget vs never-taught control) -- the gate on any erasure claim")
    ds = PATHS.tables / "drift_summary.json"
    threshold = cfg.get("localisation", {}).get("drift_ratio_min", 1.5)
    if ds.exists():
        for m, d in json.load(open(ds)).items():
            ratio = d.get("forget_over_control")
            rows.append({"family": "drift", "model": m, "set": "forget_over_control",
                         "metric": "ratio", "layer": primary,
                         "point": ratio, "lo": None, "hi": None,
                         "threshold": threshold,
                         "significant": bool(d.get("forget_specific")),
                         "corrected": False, "n_entities": None})
            print(f"  {m:12s} forget/control = {ratio:.2f} (threshold {threshold}) "
                  f"-> {'forget-specific' if d.get('forget_specific') else 'GLOBAL'}")
        if not any(json.load(open(ds))[m].get("forget_specific") for m in json.load(open(ds))):
            print("  No model shows forget-specific drift. Probe changes at this "
                  "layer cannot be attributed to the forgotten facts; the claim "
                  "caps at output suppression.")

    # ------------------------------------------------------------- write out
    out = pd.DataFrame(rows)
    out.to_csv(PATHS.tables / "final_statistics.csv", index=False)
    print(f"\nwrote {PATHS.tables / 'final_statistics.csv'} ({len(out)} rows)")

    write_json({
        "n_probe_seeds": n_seeds,
        "sign_flip_p_floor": p_floor,
        "primary_layer": primary,
        "drift_threshold": threshold,
        "n_bootstrap": n_boot,
        "alpha": cfg["stats"]["alpha"],
        "note": "final_statistics.csv is the only source the paper cites. Any "
                "number not in it does not go in the paper.",
    }, PATHS.tables / "statistics_provenance.json")

    bare = out[(out["lo"].isna()) & (out["family"] == "behavioural")]
    if len(bare):
        print(f"WARNING: {len(bare)} behavioural rows have no interval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())