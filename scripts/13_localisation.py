#!/usr/bin/env python3
"""Day 35 --- localisation: preserved / transformed / obscured / removed.

    python scripts/13_localisation.py --source M_injected --target M_npo --layer 8

Runs entirely on cached activation files, on CPU, with no torch. Produces:
  - the sample-efficiency curve for both models (Figure 6, panel 1)
  - probe transfer source -> target (panel 2)
  - class separation for both (panel 3)
  - the Section 12.2 verdict, with its reasons

The thresholds used by the decision rule live in configs/base.yaml under
`localisation`. Put them in the preregistration before you run this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, setup, write_json

from src.analysis.activations import class_separation
from src.analysis.erasure import layerwise_erasure_control
from src.analysis.localisation import (classify_localisation, entities_to_reach,
                                       probe_transfer, sample_efficiency_curve)
from src.analysis.probes import probe_with_controls
from src.utils.io import load_activations
from src.viz import figures as F


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--source", default="M_injected")
    ap.add_argument("--target", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--set", default="forget")
    ap.add_argument("--target-accuracy", type=float, default=0.30,
                    help="accuracy level for the entities-to-reach comparison. "
                         "0.50 is unreachable for every post-unlearning curve here, "
                         "which makes the ratio infinite and uninformative.")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    loc = cfg.get("localisation", {})

    def cache(label, setname):
        return load_activations(PATHS.activations / f"{label}_{setname}.npz")

    src_acts, y, ents, _ = cache(args.source, args.set)
    tgt_acts, y2, ents2, _ = cache(args.target, args.set)
    if not (np.array_equal(y, y2) and np.array_equal(ents, ents2)):
        raise SystemExit(
            "source and target caches are not aligned. Build both from the same "
            "record list, in the same order, and do not re-sort in between."
        )
    ctl_acts, ctl_y, ctl_ents, _ = cache(args.target, "control")

    Xs, Xt = src_acts[args.layer], tgt_acts[args.layer]
    print(f"layer {args.layer}: {Xs.shape[0]} prompts, {len(np.unique(ents))} entities")

    # ---------------------------------------------------------------- 1. probes
    post = probe_with_controls(Xt, y, ents, seed=cfg["seed"])
    ctrl = probe_with_controls(ctl_acts[args.layer], ctl_y, ctl_ents, seed=cfg["seed"])
    print(f"  probe on {args.target}: {post['accuracy']:.3f} "
          f"(control entities {ctrl['accuracy']:.3f}, chance {post['chance']:.3f})")

    # ------------------------------------------------- 2. erasure positive control
    leace = layerwise_erasure_control(tgt_acts[args.layer:args.layer + 1], y, ents,
                                      seeds=(0, 1, 2))
    leace_acc = float(np.mean([r["accuracy"] for r in leace]))
    print(f"  LEACE-erased control: {leace_acc:.3f}  "
          f"(this is what erased LOOKS like in this pipeline)")

    # --------------------------------------------------------- 3. sample efficiency
    rows = []
    for label, X in ((args.source, Xs), (args.target, Xt)):
        r = sample_efficiency_curve(X, y, ents, sizes=(8, 16, 32, 64, 0),
                                    seeds=tuple(range(cfg["probe"]["n_seeds"])))
        for x in r:
            x["condition"] = label
        rows += r
    eff = pd.DataFrame(rows)
    n_src = entities_to_reach([r for r in rows if r["condition"] == args.source],
                              args.target_accuracy)
    n_tgt = entities_to_reach([r for r in rows if r["condition"] == args.target],
                              args.target_accuracy)
    ratio = n_tgt / n_src if np.isfinite(n_src) and n_src > 0 else float("inf")
    print(f"  entities to reach {args.target_accuracy:.2f}: "
          f"{args.source}={n_src}  {args.target}={n_tgt}  ratio={ratio:.2f}")

    # ------------------------------------------------------------------ 4. transfer
    tr = probe_transfer(Xs, Xt, y, ents, seeds=tuple(range(cfg["probe"]["n_seeds"])))
    within = float(np.mean([r["within_model_accuracy"] for r in tr]))
    across = float(np.mean([r["transfer_accuracy"] for r in tr]))
    print(f"  probe transfer {args.source} -> {args.target}: "
          f"{within:.3f} within, {across:.3f} across")

    # ------------------------------------------------------------------ 5. geometry
    sep_src = class_separation(src_acts, y)[args.layer]
    sep_tgt = class_separation(tgt_acts, y)[args.layer]
    sep_ratio = float(sep_tgt / (sep_src + 1e-12))
    print(f"  class separation: {sep_src:.3f} -> {sep_tgt:.3f} (ratio {sep_ratio:.3f})")

    # ------------------------------------------------------------------ 6. verdict
    verdict = classify_localisation(
        post_accuracy=post["accuracy"],
        control_accuracy=ctrl["accuracy"],
        transfer_accuracy_value=across,
        within_accuracy=within,
        entities_ratio=ratio,
        separation_ratio=sep_ratio,
        decodable_margin=loc.get("decodable_margin", 0.10),
        transfer_margin=loc.get("transfer_margin", 0.15),
        efficiency_factor=loc.get("efficiency_factor", 2.0),
    )
    print(f"\n  VERDICT: {verdict['label'].upper()}")
    for k, v in verdict["reasons"].items():
        print(f"    {k}: {v:.3f}")
    print(f"    {verdict['caveat']}")

    eff.to_csv(PATHS.tables / f"sample_efficiency_{args.target}.csv", index=False)
    pd.DataFrame(tr).to_csv(PATHS.tables / f"transfer_{args.target}.csv", index=False)
    write_json({
        "layer": args.layer, "source": args.source, "target": args.target,
        "probe_post": post["accuracy"], "probe_control_entities": ctrl["accuracy"],
        "leace_control": leace_acc, "chance": post["chance"],
        "transfer_within": within, "transfer_across": across,
        "entities_to_reach": {"source": n_src, "target": n_tgt, "ratio": ratio},
        "class_separation": {"source": float(sep_src), "target": float(sep_tgt),
                             "ratio": sep_ratio},
        "verdict": verdict,
    }, PATHS.tables / f"localisation_{args.target}.json")

    fig = F.fig_sample_efficiency(eff, chance=post["chance"],
                                  title=f"Sample efficiency at layer {args.layer}")
    print("\nfigure:", F.save(fig, PATHS.figures / f"fig6_sample_efficiency_{args.target}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
