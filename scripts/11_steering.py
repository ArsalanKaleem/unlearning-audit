#!/usr/bin/env python3
"""Days 39 / 44 --- the rung-4 test: steer with a direction from the UNLEARNED
model's own activations, against matched random directions.

    python scripts/11_steering.py --checkpoint <M_unl> --label M_npo_s0 --layer 9

If the probe direction does not restore the target better than norm-matched
random directions, you do not have a contents-level claim. That is a real
result; report it.

Three things this script learned the hard way, all encoded below.

FIRST: a bare inequality at the largest alpha is not a test. The first version
compared probe (-19.494) against random (-19.745) at alpha=4 and declared PASS
on a gap of 0.25 between two values that had both collapsed. At large alpha the
perturbation destroys the computation in EVERY direction, so the comparison
measures how broken the model is, not whether the direction carries the fact.
Every difference now gets an entity-clustered bootstrap interval.

SECOND: the alpha range must stay inside the regime where steering means
anything. The default is +/-2, and the finite-difference check reports where
the linear approximation holds.

THIRD, and the one that changed the conclusion: a CI excluding zero is
NECESSARY BUT NOT SUFFICIENT. The second version returned PASS at 3/3 positive
alphas with perfectly linear, sign-symmetric effects and finite-difference
agreement to 0.001 -- and the effect was a change of 0.125 inside a gap of
19.75, i.e. 0.6% of the distance to the injected model. That is a measurement
of local gradient geometry, not evidence that content is accessible. The rung-4
criterion therefore carries an EFFECT-SIZE FLOOR as well as a significance
test.

This floor was added AFTER seeing that result, and it converts a PASS into a
FAIL. Recorded as a deviation in preregistration.md Section 6.
"""
import json

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, load_sets, setup, write_json

from src.analysis.causal import steering_with_controls
from src.analysis.probes import linear_probe
from src.analysis.sensitivity import finite_difference_check
from src.model.loader import load_model
from src.stats.bootstrap import cluster_bootstrap_ci
from src.utils.io import load_activations
from src.viz import figures as F


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--alphas", default="-2,-1,-0.5,0,0.5,1,2",
                    help="steering coefficients. Large values (|alpha| >= 4) "
                         "destroy the computation in every direction and make "
                         "the probe-vs-random comparison meaningless.")
    ap.add_argument("--n-random", type=int, default=10,
                    help="norm-matched random directions per prompt")
    ap.add_argument("--min-gap-fraction", type=float, default=0.25,
                    help="effect-size floor: the steering effect must recover at "
                         "least this fraction of the logit-difference gap to "
                         "M_injected before it counts as rung-4 evidence.")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, _ = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]
    distractors = sorted(city_ids.values())
    alphas = [float(a) for a in args.alphas.split(",")]

    # The direction MUST come from the unlearned model's own activations. A
    # direction taken from M_injected would prove only that the unlearned model
    # can use externally supplied information -- rung 3, not rung 4.
    acts, y, ents, _ = load_activations(PATHS.activations / f"{args.label}_forget.npz")
    probe = linear_probe(acts[args.layer], y, ents, seed=cfg["seed"],
                         C_grid=cfg["probe"]["C_grid"])
    print(f"  probe on {args.label} layer {args.layer}: acc={probe.accuracy:.3f} "
          f"(chance {probe.chance:.3f})")
    if probe.coef is None:
        print("  no coefficients returned; cannot steer")
        return 1
    if probe.accuracy < probe.chance + 0.05:
        print("  NOTE: the probe is at chance, so the direction it supplies is "
              "close to arbitrary. A null steering result here says little "
              "beyond what the probe already said.")

    model = load_model(cfg, args.checkpoint)
    rows, fd_rows = [], []
    for r in sets["forget"][:args.n_prompts]:
        ans = city_ids[r["answer"]]
        d = [x for x in distractors if x != ans]
        direction = probe.coef[r["label"]]
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        for row in steering_with_controls(model, r["prompt"], args.layer, direction,
                                          alphas, ans, d, n_random=args.n_random,
                                          seed=cfg["seed"],
                                          hook=cfg["activations"]["hook"]):
            row["entity_id"] = r["entity_id"]
            rows.append(row)
        for fd in finite_difference_check(model, r["prompt"], ans, args.layer, direction,
                                          hook=cfg["activations"]["hook"]):
            fd["entity_id"] = r["entity_id"]
            fd_rows.append(fd)

    df = pd.DataFrame(rows)
    df.to_csv(PATHS.tables / f"steering_{args.label}.csv", index=False)
    fd = pd.DataFrame(fd_rows)
    fd.to_csv(PATHS.tables / f"finite_diff_{args.label}.csv", index=False)

    # ------------------------------------------------- significance, per alpha
    print("\n  alpha    probe      random     diff   95% CI")
    summary = []
    for a in alphas:
        pr = df[(df.kind == "probe_direction") & (df.alpha == a)]
        rd = df[(df.kind == "random_matched_norm") & (df.alpha == a)]
        if not len(pr) or not len(rd):
            continue

        # Pair by entity: each entity contributes its probe-direction effect
        # minus the mean of its random-direction effects. The bootstrap then
        # resamples ENTITIES, which is the unit of independence here -- the
        # ten random draws for one entity are not ten observations.
        rand_by_entity = rd.groupby("entity_id")["logit_diff"].mean()
        paired = (pr.set_index("entity_id")["logit_diff"] - rand_by_entity).dropna()
        ci = cluster_bootstrap_ci(paired.values, paired.index.values,
                                  n_boot=cfg["stats"]["n_bootstrap"], seed=cfg["seed"])
        excludes_zero = (ci["lo"] > 0) or (ci["hi"] < 0)
        summary.append({
            "alpha": a,
            "probe_mean": float(pr["logit_diff"].mean()),
            "random_mean": float(rd["logit_diff"].mean()),
            "diff": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
            "n_entities": ci["n_clusters"],
            "excludes_zero": bool(excludes_zero),
        })
        print(f"  {a:+5.1f}  {pr['logit_diff'].mean():+9.3f} "
              f"{rd['logit_diff'].mean():+9.3f} {ci['point']:+8.3f}  "
              f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]" + ("  *" if excludes_zero else ""))

    pd.DataFrame(summary).to_csv(PATHS.tables / f"steering_ci_{args.label}.csv",
                                 index=False)

    positive = [s for s in summary if s["alpha"] > 0]
    wins = [s for s in positive if s["excludes_zero"] and s["diff"] > 0]
    significant = len(wins) > 0

    # ------------------------------------------------------- effect size floor
    # The denominator is the distance the model would have to travel for the
    # target to be preferred. Taken from config so it is recorded rather than
    # inferred from the run that is being judged.
    gap = float(abs(cfg["baseline"].get("steering_reference_gap", 19.75)))
    best_diff = max((s["diff"] for s in positive), default=0.0)
    fraction = best_diff / gap if gap else 0.0
    substantive = fraction >= args.min_gap_fraction

    ok = significant and substantive

    print(f"\n  largest positive difference {best_diff:+.3f} = {fraction:.2%} of the "
          f"{gap:.2f} logit-difference gap to M_injected "
          f"(floor {args.min_gap_fraction:.0%})")
    print(f"CHECKPOINT: significant at {len(wins)}/{len(positive)} positive alphas "
          f"AND recovers >= {args.min_gap_fraction:.0%} of the gap -> "
          f"{'PASS (rung 4 evidence)' if ok else 'FAIL'}")

    if significant and not substantive:
        print("  Statistically clean, substantively negligible. The probe direction "
              "has a non-zero gradient component toward the target -- a statement "
              "about local geometry, not about accessible content. NO rung-4 claim.")
        print("  This is the finding, not a failure of the experiment: a test that "
              "passes its significance criterion while moving the model less than "
              f"{fraction:.1%} of the way to answering correctly.")
    elif not significant:
        print("  No contents-level claim. A direction estimated from the unlearned "
              "model's own activations does not restore the target any better than "
              "a norm-matched random direction of the same size.")

    # --------------------------------------------- collapse / regime diagnostic
    collapse_note = ""
    if summary:
        top = max(summary, key=lambda s: s["alpha"])
        at_zero = next((s for s in summary if s["alpha"] == 0.0), None)
        if at_zero and top["alpha"] > 0:
            if (top["probe_mean"] < at_zero["probe_mean"] - 1.0
                    and top["random_mean"] < at_zero["random_mean"] - 1.0):
                collapse_note = (
                    f"At alpha={top['alpha']:+.1f} BOTH the probe direction "
                    f"({top['probe_mean']:+.2f}) and random directions "
                    f"({top['random_mean']:+.2f}) fall far below the alpha=0 "
                    f"baseline ({at_zero['probe_mean']:+.2f}). The perturbation is "
                    "destroying the computation rather than steering it; the "
                    "comparison at that alpha is uninformative in either direction."
                )
                print(f"  {collapse_note}")

    if len(fd):
        by_alpha = fd.groupby("alpha")[["predicted_delta", "actual_delta",
                                        "abs_error"]].mean()
        print("\n  finite-difference check (where the linear approximation holds):")
        for a, row in by_alpha.iterrows():
            print(f"    alpha={a:+5.2f}  predicted {row.predicted_delta:+8.3f}  "
                  f"actual {row.actual_delta:+8.3f}  |error| {row.abs_error:7.3f}")
        if float(by_alpha["abs_error"].max()) < 0.01:
            print("    Agreement this close means the whole sweep sits inside the "
                  "LINEAR regime: this is a first-order sensitivity measurement, "
                  "which is further reason not to read it as a functional "
                  "intervention.")

    write_json({
        "label": args.label, "layer": args.layer,
        "probe_accuracy": probe.accuracy, "probe_chance": probe.chance,
        "alphas": alphas, "n_prompts": args.n_prompts, "n_random": args.n_random,
        "per_alpha": summary,
        "significant": significant,
        "largest_diff": best_diff,
        "reference_gap": gap,
        "gap_fraction": fraction,
        "min_gap_fraction": args.min_gap_fraction,
        "substantive": substantive,
        "rung4_supported": ok,
        "collapse_note": collapse_note,
        "test": "TWO criteria, both required. (1) entity-clustered bootstrap CI on "
                "(probe - mean random) per entity excludes zero at a positive alpha. "
                "(2) the effect recovers at least min_gap_fraction of the "
                "logit-difference gap to M_injected. Criterion (2) was added after "
                "criterion (1) alone passed on an effect worth 0.6% of the gap; see "
                "preregistration.md Section 6.",
    }, PATHS.tables / f"steering_verdict_{args.label}.json")

    print("\nfigure:", F.save(F.fig_steering(df), PATHS.figures / f"fig_steering_{args.label}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())