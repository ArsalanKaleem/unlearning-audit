#!/usr/bin/env python3
"""Day 44 --- experiment C6, the rung-5 test (Deeb & Roger style).

Fine-tune the unlearned model on HALF the forget facts and evaluate on the
OTHER half. If held-out forget accuracy recovers, the weights retained the
information and the unlearning was suppression.

    python scripts/12_recovery_attack.py --checkpoint <M_unl> --label M_npo

CHECKPOINT: the control fine-tune -- same procedure, never-taught control
entities -- must NOT restore forget-fact accuracy. Without that control the
experiment proves nothing: fine-tuning on anything can restore general
question-answering format.
"""
import json

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, load_sets, setup, write_json

from src.eval.behavioural import evaluate_set, summarise
from src.model.loader import load_hf_model
from src.train.inject import inject
from src.utils.seed import rng_for


def split_entities(records, frac=0.5, seed=0):
    ents = np.unique([r["entity_id"] for r in records])
    rng = rng_for("recovery-split", seed)
    a = set(rng.choice(ents, size=int(frac * len(ents)), replace=False))
    tune = [r for r in records if r["entity_id"] in a]
    held = [r for r in records if r["entity_id"] not in a]
    assert not ({r["entity_id"] for r in tune} & {r["entity_id"] for r in held})
    return tune, held


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, label_map = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]

    tune, held = split_entities(sets["forget"], 0.5, cfg["seed"])
    print(f"  fine-tune on {len({r['entity_id'] for r in tune})} entities, "
          f"evaluate on {len({r['entity_id'] for r in held})} DISJOINT entities")

    cfg = dict(cfg)
    cfg["inject"] = {**cfg["inject"], "epochs": args.epochs}
    results = {}

    for arm, train_records in (("attack", tune), ("control", sets["control"])):
        model, tokenizer = load_hf_model(cfg, args.checkpoint)
        before = summarise(evaluate_set(model, tokenizer, held, city_ids), n_boot=2000,
                           seed=cfg["seed"])
        inject(model, tokenizer, train_records, cfg, rdir / arm)
        after = summarise(evaluate_set(model, tokenizer, held, city_ids), n_boot=2000,
                          seed=cfg["seed"])
        results[arm] = {"before": before, "after": after}
        print(f"  {arm:8s} held-out forget accuracy "
              f"{before['constrained_correct']:.3f} -> {after['constrained_correct']:.3f} "
              f"[{after['constrained_correct_lo']:.3f},{after['constrained_correct_hi']:.3f}]")

    write_json(results, PATHS.tables / f"recovery_attack_{args.label}.json")
    atk = results["attack"]["after"]["constrained_correct"]
    ctl = results["control"]["after"]["constrained_correct"]
    chance = label_map["chance_accuracy_city"]
    ok = ctl < chance + 0.10
    print(f"\nCHECKPOINT: control fine-tune stays near chance "
          f"({ctl:.3f} vs {chance:.3f}) -> {'PASS' if ok else 'FAIL -- C6 proves nothing'}")
    if ok:
        print(f"  attack recovery {atk:.3f} vs control {ctl:.3f}: "
              + ("evidence the WEIGHTS retained the information (rung 5)"
                 if atk > ctl + 0.15 else
                 "no recovery beyond the control -- consistent with genuine removal"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
