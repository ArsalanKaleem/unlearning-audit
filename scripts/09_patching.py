#!/usr/bin/env python3
"""Day 43 --- cross-model activation patching with controls. Experiment C2/C3.

    python scripts/09_patching.py --donor <M_injected_ckpt> --receiver <M_unl_ckpt>

Donor states come from M_injected; the receiver is the unlearned model. High
recovery means the unlearned model can still USE the fact when the state is
supplied -- a claim about MACHINERY (rung 3), not about stored contents.

CHECKPOINT: random-position controls must show near-zero recovery.
"""
import json

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, load_sets, setup

from src.analysis.causal import causal_trace, patching_sweep
from src.model.loader import load_model
from src.viz import figures as F


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--label", default="M_unlearned")
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--trace", action="store_true", help="also run within-model causal tracing")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, _ = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]
    distractors = sorted(city_ids.values())

    donor = load_model(cfg, args.donor)
    receiver = load_model(cfg, args.receiver)

    rows = []
    for setname in ("forget", "retain"):          # retain = the specificity control
        for r in sets[setname][:args.n_prompts]:
            ans = city_ids[r["answer"]]
            d = [x for x in distractors if x != ans]
            for row in patching_sweep(receiver, donor, r["prompt"], ans, d,
                                      hook=cfg["activations"]["hook"], seed=cfg["seed"]):
                row.update({"set": setname, "entity_id": r["entity_id"],
                            "model": args.label})
                rows.append(row)

    df = pd.DataFrame(rows)
    out = PATHS.tables / f"patching_{args.label}.csv"
    df.to_csv(out, index=False)

    real = df[(df.kind == "real") & (df.set == "forget")]
    ctrl = df[(df.kind == "random_position") & (df.set == "forget")]
    peak_layer = real.groupby("layer")["recovery"].mean().idxmax()
    peak = real.groupby("layer")["recovery"].mean().max()
    ctrl_mean = ctrl["recovery"].mean()
    print(f"\n  peak recovery {peak:.3f} at layer {peak_layer}")
    print(f"  random-position control mean recovery {ctrl_mean:.3f}")
    ok = abs(ctrl_mean) < 0.15
    print(f"CHECKPOINT: controls near zero -> {'PASS' if ok else 'FAIL -- patching code is wrong'}")

    fig = F.fig_patching_recovery(df[df.set == "forget"])
    print("figure:", F.save(fig, PATHS.figures / f"fig10_patching_{args.label}"))

    if args.trace:
        r = sets["forget"][0]
        other = next(x for x in sets["forget"] if x["answer"] != r["answer"])
        try:
            grid_pre = causal_trace(donor, r["prompt"], other["prompt"],
                                    city_ids[r["answer"]], distractors)
            grid_post = causal_trace(receiver, r["prompt"], other["prompt"],
                                     city_ids[r["answer"]], distractors)
            np.save(PATHS.tables / "trace_diff.npy", grid_post - grid_pre)
            fig = F.fig_heatmap(grid_post - grid_pre,
                                title="Causal tracing difference (unlearned - injected)")
            print("figure:", F.save(fig, PATHS.figures / "fig_tracing_diff"))
        except ValueError as e:
            print(f"  tracing skipped: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
