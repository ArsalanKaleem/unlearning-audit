#!/usr/bin/env python3
"""Day 21 --- produce M_injected.

    python scripts/03_inject.py --config configs/base.yaml

Two checkpoints must both pass:

1. Held-out PARAPHRASE accuracy clearly above chance. Trained-template
   accuracy proves only memorisation; paraphrase accuracy is what tells you
   a fact was taught rather than a string.
2. Generic perplexity within `inject.max_ppl_ratio` of the base model. A
   model whose general capability degraded several times over is a poor
   substrate for claims about representations: any probing difference found
   later could reflect that damage instead of the unlearning.

If (1) fails, add training templates in src/data/templates.py. If (2) fails,
lower inject.lr and add epochs. Do not relax the ceiling after seeing the
numbers unless you record in the logbook that you did, and why.
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

    # Measured BEFORE any training. Every utility ratio in this project is
    # relative to this number, so it is recorded in the run directory rather
    # than only printed.
    base_ppl = generic_perplexity(model, tokenizer, GENERIC_PROMPTS)
    max_ratio = cfg["inject"].get("max_ppl_ratio", 1.5)
    print(f"base generic perplexity: {base_ppl:.2f} "
          f"({len(GENERIC_PROMPTS)} sentences); ceiling = {max_ratio} "
          f"-> {base_ppl * max_ratio:.2f}")

    def evaluate(m, epoch):
        m.eval()
        out = {}
        for name in ("forget", "retain", "control", "paraphrase"):
            s = summarise(evaluate_set(m, tokenizer, sets[name][:120], city_ids),
                          n_boot=200, seed=cfg["seed"])
            out[f"{name}_acc"] = s["constrained_correct"]
        out["ppl"] = generic_perplexity(m, tokenizer, GENERIC_PROMPTS)
        out["ppl_ratio"] = out["ppl"] / base_ppl
        return out

    history = inject(model, tokenizer, sets["train_injection"], cfg, rdir, evaluate)

    try:
        best = select_epoch(history, key="paraphrase_acc",
                            base_ppl=base_ppl, max_ppl_ratio=max_ratio)
    except ValueError as e:
        print(f"\nCHECKPOINT FAIL: {e}")
        return 1

    print(f"\nselected epoch {best}: max held-out paraphrase accuracy among "
          f"epochs with perplexity ratio <= {max_ratio}")

    ckpt = rdir / f"epoch-{best}"
    write_json({
        "selected_epoch": best,
        "checkpoint": str(ckpt),
        "rule": f"max paraphrase_acc subject to ppl_ratio <= {max_ratio}",
        "base_ppl": base_ppl,
        "base_ppl_n_sentences": len(GENERIC_PROMPTS),
        "selected_metrics": history[best],
    }, rdir / "selection.json")

    h = history[best]
    chance = label_map["chance_accuracy_city"]
    para_ok = h.get("paraphrase_acc", 0) > 3 * chance
    ppl_ok = h.get("ppl_ratio", 99) <= max_ratio

    print(f"\nCHECKPOINT paraphrase: {h.get('paraphrase_acc', 0):.3f} vs chance "
          f"{chance:.3f} -> {'PASS' if para_ok else 'FAIL'}")
    print(f"CHECKPOINT perplexity: ratio {h.get('ppl_ratio', float('nan')):.2f} "
          f"({h.get('ppl', float('nan')):.1f} vs {base_ppl:.1f}) -> "
          f"{'PASS' if ppl_ok else 'FAIL'}")

    if not para_ok:
        print("\nAdd template diversity in src/data/templates.py "
              "(CITY_TEMPLATES_TRAIN) and re-run. Do not simply train longer: "
              "that inflates memorisation and perplexity together while leaving "
              "generalisation flat.")
        return 1
    if not ppl_ok:
        print("\nLower inject.lr and raise inject.epochs, then re-run.")
        return 1

    print(f"\nM_injected: {ckpt}")
    print("Copy it to checkpoints/M_injected and record the path in the logbook:")
    print(f'  Copy-Item -Recurse "{ckpt}" checkpoints\\M_injected')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())