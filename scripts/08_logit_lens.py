#!/usr/bin/env python3
"""Days 24 / 37 --- logit lens across models.

    python scripts/08_logit_lens.py --labels M_injected,M_npo --checkpoints ckpt1,ckpt2

The last-layer assertion runs for EVERY model before any trajectory is used.
"""
import json

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, load_sets, setup

from src.analysis.logit_lens import (assert_last_layer_matches, lens_trajectory,
                                     summarise_trajectory)
from src.model.loader import load_model
from src.viz import figures as F


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--checkpoints", required=True,
                    help="comma-separated, aligned with --labels; use 'base' for none")
    ap.add_argument("--n-prompts", type=int, default=60)
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, _ = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]

    labels = args.labels.split(",")
    ckpts = [None if c == "base" else c for c in args.checkpoints.split(",")]

    traj, rows = {}, []
    for label, ckpt in zip(labels, ckpts):
        model = load_model(cfg, ckpt)
        d = assert_last_layer_matches(model, "The capital of France is")
        print(f"  {label}: last-layer assertion PASS (diff {d:.2g})")
        for setname in ("forget", "retain", "control"):
            recs = sets[setname][:args.n_prompts]
            out = lens_trajectory(
                model, [r["prompt"] for r in recs],
                [city_ids[r["answer"]] for r in recs],
                hook=cfg["activations"]["hook"],
            )
            key = f"{label}:{setname}"
            traj[key] = out["logprob_mean"]
            s = summarise_trajectory(out["logprob_mean"])
            s.update({"model": label, "set": setname})
            rows.append(s)
            print(f"    {setname:8s} peak L{s['peak_layer']:2d} "
                  f"drop={s['peak_to_final_drop']:+.2f} shape={s['shape']}")
            np.save(PATHS.tables / f"lens_{label}_{setname}.npy", out["per_prompt_logprob"])

    pd.DataFrame(rows).to_csv(PATHS.tables / "lens_summary.csv", index=False)
    fig = F.fig_logit_lens(traj, title="Logit lens (mean log p, log space)")
    print("figure:", F.save(fig, PATHS.figures / "fig7_logit_lens"))
    print("caption must state: averaged in LOG space; intermediate-layer "
          "probabilities are not calibrated (cite the tuned-lens critique)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
