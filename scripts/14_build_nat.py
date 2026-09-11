#!/usr/bin/env python3
"""Day 30 --- build Condition NAT from facts the BASE model actually knows.

    python scripts/14_build_nat.py --config configs/base.yaml

Measures the base model on every candidate, keeps the entities it answers
correctly under ALL training templates, and assigns balanced forget/retain
splits among those. Facts it gets wrong become the `not_known` set, which is
the closest thing to a never-taught control that pretrained knowledge allows.

Fix the correctness threshold BEFORE looking at the counts. The default --
correct under every training template -- is strict on purpose: a fact the
model answers inconsistently is not a fact it reliably knows, and unlearning
it would produce an uninterpretable result.
"""
import json
from collections import defaultdict

from _common import PATHS, base_parser, setup, write_json

from src.data.build_nat import build_nat_candidates, finalise_nat
from src.eval.behavioural import score_prompts
from src.model.loader import load_hf_model, single_token_ids


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--min-correct-fraction", type=float, default=1.0,
                    help="fraction of TRAIN templates that must be correct (default: all)")
    ap.add_argument("--out", default="data/processed/nat")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)

    cand = build_nat_candidates(cfg)
    classes = cand["languages"]
    print(f"{len(cand['entities'])} candidate facts, {len(classes)} classes: {classes}")

    model, tokenizer = load_hf_model(cfg)
    model.eval()

    ids = single_token_ids(tokenizer, classes)
    missing = [c for c in classes if c not in ids]
    if missing:
        print(f"FAIL: these class labels are not single tokens: {missing}")
        print("Behavioural scoring assumes single-token answers; pick other classes.")
        return 1
    print(f"class token ids: {ids}")

    train_records = [r for r in cand["records"] if r["template_kind"] == "train"]
    metrics = score_prompts(
        model, tokenizer,
        [r["prompt"] for r in train_records],
        [ids[r["answer"]] for r in train_records],
        sorted(ids.values()),
        batch_size=cfg["activations"]["batch_size"],
    )

    per_entity = defaultdict(list)
    for r, ok in zip(train_records, metrics["constrained_correct"]):
        per_entity[r["entity_id"]].append(float(ok))

    known = [e for e, v in per_entity.items()
             if sum(v) / len(v) >= args.min_correct_fraction]
    print(f"\nbase model knows {len(known)} / {len(per_entity)} facts "
          f"(threshold: {args.min_correct_fraction:.2f} of training templates)")

    by_class = defaultdict(int)
    name_of = {e["entity_id"]: e["answer"] for e in cand["entities"]}
    for e in known:
        by_class[name_of[e]] += 1
    print("  known per class: " + ", ".join(f"{k}={v}" for k, v in sorted(by_class.items())))

    out_dir = PATHS.root / args.out
    result = finalise_nat(cand, known, cfg, out_dir)
    print(f"\nwrote {out_dir}")
    for k, v in result["sets"].items():
        print(f"  {k:16s} {len(v):5d} records")
    print(f"all {len(result['checks'])} checks passed; {result['dropped']} surplus facts unused")

    write_json({"known": known, "per_entity_accuracy": {k: sum(v) / len(v)
                                                        for k, v in per_entity.items()},
                "threshold": args.min_correct_fraction},
               rdir / "nat_base_model_knowledge.json")
    print("\nFrom here Condition NAT uses the SAME scripts as SYN: point them at "
          f"--config with data.dir={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
