#!/usr/bin/env python3
"""Days 22 / 29 --- full behavioural evaluation of one model.

    python scripts/04_behavioural_eval.py --checkpoint results/runs/<id>/epoch-3 \
        --label M_injected

Produces one tidy row per prompt (keeping entity_id, so every CI downstream
can resample entities) plus a summary with bootstrap intervals.
"""
import json

import pandas as pd
from _common import PATHS, base_parser, load_sets, setup, write_json

from src.data.templates import GENERIC_PROMPTS
from src.eval.behavioural import (evaluate_set, generate_completions, generic_perplexity,
                                  refusal_rate, summarise)
from src.model.loader import load_hf_model


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--label", required=True, help="model label, e.g. M_injected")
    ap.add_argument("--no-refusal", action="store_true")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, label_map = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]

    model, tokenizer = load_hf_model(cfg, args.checkpoint)
    model.eval()

    rows, summary = [], {}
    for name in ("forget", "retain", "control", "paraphrase", "related", "related_paraphrase"):
        ids = city_ids
        if name.startswith("related"):
            with open(PATHS.meta / "single_token_cities.json") as f:
                ids = json.load(f)["field_token_ids"]
        r = evaluate_set(model, tokenizer, sets[name], ids)
        for x in r:
            x["set"] = name
            x["model"] = args.label
        rows += r
        summary[name] = summarise(r, n_boot=cfg["stats"]["n_bootstrap"], seed=cfg["seed"])
        s = summary[name]
        print(f"  {name:20s} constrained={s['constrained_correct']:.3f} "
              f"[{s['constrained_correct_lo']:.3f},{s['constrained_correct_hi']:.3f}] "
              f"rank_med={s['median_rank']:.0f}")

    summary["generic_perplexity"] = generic_perplexity(model, tokenizer, GENERIC_PROMPTS)
    if not args.no_refusal:
        comps = generate_completions(model, tokenizer, [r["prompt"] for r in sets["forget"][:50]])
        summary["refusal_rate_forget"] = refusal_rate(comps)
        write_json(comps, rdir / "completions_forget.json")
        print(f"  refusal rate (forget, keyword detector): {summary['refusal_rate_forget']:.3f}")
        print("  -> hand-label these 50 completions and report detector agreement (Day 29)")

    out = PATHS.tables / f"behaviour_{args.label}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    write_json(summary, rdir / f"behaviour_{args.label}.json")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
