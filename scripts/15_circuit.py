#!/usr/bin/env python3
"""Day 42 --- per-head attribution, ablation, and the edge list.

    python scripts/15_circuit.py --checkpoint checkpoints/M_injected \
        --label M_injected --n-prompts 8

Cost: n_layers * n_heads forward passes per prompt (144 for GPT-2 Small).
Keep --n-prompts small and report the number in the caption.
"""
import json

import pandas as pd
from _common import PATHS, base_parser, load_sets, setup, write_json

from src.analysis.circuit import (build_edges, circuit_summary, compute_mean_activations,
                                  head_attribution, random_head_baseline, to_mermaid)
from src.model.loader import load_model


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, _ = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]

    model = load_model(cfg, args.checkpoint)
    records = sets["forget"][:args.n_prompts]
    prompts = [r["prompt"] for r in records]
    answers = [city_ids[r["answer"]] for r in records]
    print(f"{len(prompts)} prompts x {model.cfg.n_layers * model.cfg.n_heads} heads "
          f"= {len(prompts) * model.cfg.n_layers * model.cfg.n_heads} forward passes")

    mean_acts = compute_mean_activations(model, prompts, "z")
    rows = head_attribution(model, prompts, answers, sorted(city_ids.values()), mean_acts)
    df = pd.DataFrame(rows).sort_values("mean_effect", ascending=False)
    df.to_csv(PATHS.tables / f"head_attribution_{args.label}.csv", index=False)

    print("\n  top heads by mean ablation effect:")
    for _, r in df.head(args.top_k).iterrows():
        print(f"    L{int(r.layer):2d}H{int(r.head):2d}  effect={r.mean_effect:+.3f} "
              f"(sd {r.sd_effect:.3f} over {int(r.n_prompts)} prompts)")

    baseline = random_head_baseline(rows, args.top_k, seed=cfg["seed"])
    print(f"\n  top-{args.top_k} mean |effect| = {baseline['top_k_effect']:.3f} vs "
          f"random heads {baseline['null_mean']:.3f} "
          f"(sd {baseline['null_sd']:.3f}), p={baseline['p_empirical']:.3f}")
    if baseline["p_empirical"] >= 0.05:
        print("  the selected heads are not distinguishable from random heads. "
              "Report that; do not draw a circuit.")

    edges = build_edges(rows, effect_threshold=0.1)
    summary = circuit_summary(edges)
    print(f"\n  {summary['caption_requirement']}")
    write_json({"edges": edges, "summary": summary, "random_baseline": baseline,
                "n_prompts": len(prompts)},
               PATHS.tables / f"circuit_{args.label}.json")
    (PATHS.figures / f"circuit_{args.label}.mmd").write_text(to_mermaid(edges))
    print(f"  mermaid sketch: results/figures/circuit_{args.label}.mmd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
