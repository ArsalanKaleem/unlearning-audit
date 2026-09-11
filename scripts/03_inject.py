#!/usr/bin/env python3
"""Day 21 --- produce M_injected.

    python scripts/03_inject.py --config configs/base.yaml

Checkpoint: trained-template accuracy high AND held-out paraphrase accuracy
clearly above chance. Paraphrase at chance means you taught strings.
"""
import json

from _common import PATHS, base_parser, load_sets, setup, write_json

from src.data.templates import GENERIC_PROMPTS
from src.eval.behavioural import evaluate_set, generic_perplexity, summarise
from src.model.loader import load_hf_model
from src.train.inject import inject, select_epoch


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, rid, rdir = setup(args)
    sets, label_map = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]

    model, tokenizer = load_hf_model(cfg)

    def evaluate(m, epoch):
        m.eval()
        out = {}
        for name in ("forget", "retain", "control", "paraphrase"):
            s = summarise(evaluate_set(m, tokenizer, sets[name][:120], city_ids),
                          n_boot=200, seed=cfg["seed"])
            out[f"{name}_acc"] = s["constrained_correct"]
        out["paraphrase_acc"] = out["paraphrase_acc"]
        out["ppl"] = generic_perplexity(m, tokenizer, GENERIC_PROMPTS)
        return out

    history = inject(model, tokenizer, sets["train_injection"], cfg, rdir, evaluate)
    best = select_epoch(history, key="paraphrase_acc")
    print(f"\nselected epoch {best} by the preregistered rule "
          "(max held-out paraphrase accuracy)")

    ckpt = rdir / f"epoch-{best}"
    (PATHS.root / "checkpoints").mkdir(exist_ok=True)
    link = PATHS.root / "checkpoints" / "M_injected"
    write_json({"selected_epoch": best, "checkpoint": str(ckpt),
                "rule": "max paraphrase_acc"}, rdir / "selection.json")
    print(f"M_injected: {ckpt}")
    print(f"record this path in configs/ and in the logbook; symlink target: {link}")

    h = history[best]
    ok = h.get("paraphrase_acc", 0) > 3 * label_map["chance_accuracy_city"]
    print(f"\nCHECKPOINT: paraphrase accuracy {h.get('paraphrase_acc', 0):.3f} vs "
          f"chance {label_map['chance_accuracy_city']:.3f} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("Add template diversity (src/data/templates.py) and retrain. "
              "Do not simply train longer.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
