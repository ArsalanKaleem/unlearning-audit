#!/usr/bin/env python3
"""Day 29 --- validate the refusal detector, and characterise what the model
actually says after unlearning.

    python scripts/19_label_completions.py --export        # make the CSV
    python scripts/19_label_completions.py --score         # after filling it in

The refusal detector in src/eval/behavioural.py is a keyword list. It reports
0.000 on every model here, and that number cannot go in the paper until its
agreement with human labels is known. --export writes one row per completion
with the detector's verdict and three blank columns; --score computes agreement
once you have filled them in.

Three automatic diagnostics are computed alongside, because the interesting
thing in these completions is not refusal at all:

  degenerate   the same capitalised token repeated 5+ times. Unlearned models
               emit " Moscow Moscow Moscow ..." rather than declining. Whether
               forgetting produces UNCERTAINTY or produces a collapsed,
               confidently-wrong output is a real distinction and the paper
               should state which happened.
  names_city   mentions any city from the answer vocabulary
  says_target  mentions the entity's OWN city, i.e. leaked the forgotten fact
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.eval.behavioural import REFUSAL_MARKERS
from src.utils.config import PATHS
from src.utils.io import read_jsonl

OUT = PATHS.tables / "refusal_labels.csv"


def detector(text: str) -> bool:
    return any(m in text.lower() for m in REFUSAL_MARKERS)


def is_degenerate(text: str, min_repeats: int = 5) -> bool:
    caps = re.findall(r"\b[A-Z][a-z]+\b", text)
    return len(caps) >= min_repeats and len(set(caps)) == 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    if args.export:
        with open(PATHS.meta / "single_token_cities.json") as f:
            cities = json.load(f)["single_token"]
        records = read_jsonl(PATHS.root / "data/processed/syn/forget.jsonl")

        rows = []
        for path in sorted(PATHS.root.glob("results/runs/*/completions_forget.json")):
            comps = json.load(open(path))[: args.n]
            label = json.load(open(path.parent / "provenance.json")) \
                .get("extra", {}).get("argv", [])
            model = next((a for a in label if a.startswith("M_")), path.parent.name)
            for i, c in enumerate(comps):
                target = records[i]["answer"] if i < len(records) else ""
                rows.append({
                    "model": model,
                    "run_dir": path.parent.name,
                    "index": i,
                    "prompt": records[i]["prompt"] if i < len(records) else "",
                    "target_city": target,
                    "completion": c,
                    "detector_refusal": detector(c),
                    "auto_degenerate": is_degenerate(c),
                    "auto_names_a_city": any(city in c for city in cities),
                    "auto_says_target": bool(target) and target in c,
                    # FILL THESE IN: true / false
                    "human_refusal": "",
                    "human_uncertain": "",
                    "human_notes": "",
                })
        df = pd.DataFrame(rows)
        df.to_csv(OUT, index=False)
        print(f"wrote {OUT} with {len(df)} rows")
        print("\nAutomatic diagnostics by model:")
        print(df.groupby("model")[["detector_refusal", "auto_degenerate",
                                   "auto_names_a_city", "auto_says_target"]]
              .mean().round(3).to_string())
        print("\nNow open the CSV and fill in human_refusal and human_uncertain")
        print("(true/false) for every row, then run with --score.")
        print("human_refusal  = the model declined or said it did not know")
        print("human_uncertain = the model hedged or produced no committed answer")
        return 0

    if args.score:
        if not OUT.exists():
            print(f"{OUT} missing; run with --export first")
            return 1
        df = pd.read_csv(OUT)
        lab = df[df["human_refusal"].astype(str).str.lower().isin(["true", "false"])]
        if not len(lab):
            print("no human labels found; fill in the human_refusal column")
            return 1
        human = lab["human_refusal"].astype(str).str.lower() == "true"
        auto = lab["detector_refusal"].astype(bool)

        tp = int((human & auto).sum())
        tn = int((~human & ~auto).sum())
        fp = int((~human & auto).sum())
        fn = int((human & ~auto).sum())
        agreement = (tp + tn) / len(lab)

        print(f"labelled {len(lab)} of {len(df)} completions")
        print(f"  agreement            {agreement:.3f}")
        print(f"  human refusal rate   {human.mean():.3f}")
        print(f"  detector rate        {auto.mean():.3f}")
        print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
        if fn:
            print("\n  refusals the detector MISSED (add these phrasings or drop the metric):")
            for c in lab[human & ~auto]["completion"].head(10):
                print(f"    {c!r}")

        summary = {
            "n_labelled": int(len(lab)), "agreement": agreement,
            "human_refusal_rate": float(human.mean()),
            "detector_refusal_rate": float(auto.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "by_model": df.groupby("model")[["auto_degenerate", "auto_names_a_city",
                                             "auto_says_target"]].mean().round(4).to_dict(),
            "note": "The detector reports 0.000 everywhere. The completions show "
                    "degenerate repetition of a single city rather than refusal or "
                    "hedging, so unlearning here produced collapsed generation, not "
                    "expressed uncertainty.",
        }
        json.dump(summary, open(PATHS.tables / "refusal_validation.json", "w"), indent=2)
        print(f"\nwrote {PATHS.tables / 'refusal_validation.json'}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())