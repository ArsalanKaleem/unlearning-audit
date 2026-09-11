#!/usr/bin/env python3
"""Days 39 / 44 --- the rung-4 test: steer with a direction from the UNLEARNED
model's own activations, against matched random directions.

    python scripts/11_steering.py --checkpoint <M_unl> --label M_npo --layer 8

If the probe direction does not beat norm-matched random directions, you do
not have a contents-level claim. That is a real result; report it.
"""
import json

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, load_sets, setup

from src.analysis.causal import steering_with_controls
from src.analysis.probes import linear_probe
from src.analysis.sensitivity import finite_difference_check
from src.model.loader import load_model
from src.utils.io import load_activations
from src.viz import figures as F


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--alphas", default="-4,-2,-1,0,1,2,4")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, _ = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]
    distractors = sorted(city_ids.values())
    alphas = [float(a) for a in args.alphas.split(",")]

    # The direction MUST come from the unlearned model's own activations.
    acts, y, ents, _ = load_activations(PATHS.activations / f"{args.label}_forget.npz")
    probe = linear_probe(acts[args.layer], y, ents, seed=cfg["seed"],
                         C_grid=cfg["probe"]["C_grid"])
    print(f"  probe on {args.label} layer {args.layer}: acc={probe.accuracy:.3f} "
          f"(chance {probe.chance:.3f})")
    if probe.coef is None:
        print("  no coefficients returned; cannot steer")
        return 1

    model = load_model(cfg, args.checkpoint)
    rows, fd_rows = [], []
    for r in sets["forget"][:args.n_prompts]:
        ans = city_ids[r["answer"]]
        d = [x for x in distractors if x != ans]
        direction = probe.coef[r["label"]]
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        for row in steering_with_controls(model, r["prompt"], args.layer, direction,
                                          alphas, ans, d, n_random=10, seed=cfg["seed"],
                                          hook=cfg["activations"]["hook"]):
            row["entity_id"] = r["entity_id"]
            rows.append(row)
        for fd in finite_difference_check(model, r["prompt"], ans, args.layer, direction,
                                          hook=cfg["activations"]["hook"]):
            fd["entity_id"] = r["entity_id"]
            fd_rows.append(fd)

    df = pd.DataFrame(rows)
    df.to_csv(PATHS.tables / f"steering_{args.label}.csv", index=False)
    pd.DataFrame(fd_rows).to_csv(PATHS.tables / f"finite_diff_{args.label}.csv", index=False)

    top = max(alphas)
    real = df[(df.kind == "probe_direction") & (df.alpha == top)]["logit_diff"].mean()
    rand = df[(df.kind == "random_matched_norm") & (df.alpha == top)]["logit_diff"].mean()
    print(f"\n  at alpha={top}: probe direction {real:+.3f} vs random {rand:+.3f}")
    ok = real > rand
    print(f"CHECKPOINT: probe direction beats matched random -> "
          f"{'PASS (rung 4 evidence)' if ok else 'FAIL (no contents-level claim; report this)'}")

    print("figure:", F.save(F.fig_steering(df), PATHS.figures / f"fig_steering_{args.label}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
