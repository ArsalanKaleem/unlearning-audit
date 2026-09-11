#!/usr/bin/env python3
"""Day 20 --- prove the BASE model does not already know your facts.

    python scripts/02_pre_injection_baseline.py

Checkpoint: forget-set accuracy on the base model must be at chance. If it is
not, some invented names collide with real entities, or a city correlates with
a name cue. Regenerate the dataset; do not proceed.
"""
import json

import pandas as pd
from _common import PATHS, base_parser, load_sets, setup, write_json

from src.eval.behavioural import evaluate_set, generic_perplexity, summarise
from src.model.loader import load_hf_model


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, rid, rdir = setup(args)
    sets, label_map = load_sets(cfg, args.limit)

    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]

    model, tokenizer = load_hf_model(cfg)
    model.eval()

    rows, summary = [], {}
    for name in ("forget", "retain", "control", "paraphrase"):
        r = evaluate_set(model, tokenizer, sets[name], city_ids)
        for x in r:
            x["set"] = name
        rows += r
        summary[name] = summarise(r, n_boot=cfg["stats"]["n_bootstrap"], seed=cfg["seed"])
        print(f"  {name:12s} top1={summary[name]['top1_correct']:.3f} "
              f"constrained={summary[name]['constrained_correct']:.3f} "
              f"median_rank={summary[name]['median_rank']:.0f}")

    from src.data.templates import GENERIC_PROMPTS
    ppl = generic_perplexity(model, tokenizer, GENERIC_PROMPTS)
    print(f"  generic perplexity: {ppl:.2f}")

    pd.DataFrame(rows).to_csv(PATHS.tables / "pre_injection_baseline.csv", index=False)
    write_json({"summary": summary, "generic_perplexity": ppl}, rdir / "pre_injection.json")

    chance = label_map["chance_accuracy_city"]
    acc = summary["forget"]["constrained_correct"]
    hi = summary["forget"]["constrained_correct_hi"]
    ok = hi < chance + 0.10
    print(f"\nCHECKPOINT: forget constrained accuracy {acc:.3f} "
          f"[hi={hi:.3f}] vs chance {chance:.3f} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("Do not inject. Regenerate the dataset with a different seed and "
              "inspect the names that were answered correctly.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
