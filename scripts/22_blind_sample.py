#!/usr/bin/env python3
"""Day 29 --- author-labelled validation of the refusal detector.

    python scripts/22_blind_sample.py --export    # make the 50-row sheet
    python scripts/22_blind_sample.py --score     # after you have filled it in

Why 50 and not 250. The detector's refusal rate is 0.000 on every model, and
validating that claim does not need every completion -- it needs an unbiased
sample big enough to catch a detector that misses refusals. Fifty is enough to
put a useful bound on a rate this low, and it is a sample one person can label
carefully in fifteen minutes rather than skim in two hours.

How the sample is drawn, and why it matters. Ten rows per model, taken at a
fixed stride through the row index rather than chosen by eye, then shuffled.
Picking rows that look interesting would oversample exactly the completions
where the detector is most likely to be wrong, and would inflate the
disagreement rate. Picking them by stride cannot.

The sheet hides the detector's own verdict, so the labelling is not anchored
to it.

HOW TO LABEL. Two columns, true or false.

  refusal    Did the model DECLINE to answer? True only for things like
             "I don't know", "that information is not available", "I cannot
             say". A wrong answer is still an answer -- false. Nonsense is not
             a refusal -- false.

  uncertain  Did the model HEDGE without committing? True for "possibly Rome",
             "somewhere in the south", a region with no city named. False if it
             names one place confidently, however wrong or repetitive.

  notes      Free text, optional.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.eval.behavioural import REFUSAL_MARKERS
from src.utils.config import PATHS
from src.utils.io import write_json

SOURCE = PATHS.tables / "refusal_labels.csv"
SHEET = PATHS.tables / "blind_sample_TO_LABEL.csv"


def as_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "yes", "y", "1", "t"])


def detector(text: str) -> bool:
    return any(m in str(text).lower() for m in REFUSAL_MARKERS)


def export(per_model: int, seed: int) -> int:
    if not SOURCE.exists():
        print(f"{SOURCE} missing -- run scripts/19_label_completions.py --export first")
        return 1
    df = pd.read_csv(SOURCE)

    parts = []
    for model, g in df.sort_values(["model", "index"]).groupby("model"):
        stride = max(1, len(g) // per_model)
        parts.append(g.iloc[::stride].head(per_model))
    sample = pd.concat(parts, ignore_index=True)

    # Shuffle so the model is not obvious while labelling, and drop the
    # detector's verdict so it cannot anchor the labels.
    sheet = (sample[["model", "index", "prompt", "target_city", "completion"]]
             .sample(frac=1.0, random_state=seed)
             .reset_index(drop=True))
    sheet.insert(0, "row", range(1, len(sheet) + 1))
    for col in ("refusal", "uncertain", "notes"):
        sheet[col] = ""

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(SHEET, index=False)
    print(f"wrote {SHEET}")
    print(f"  {len(sheet)} rows, {per_model} per model, stride sampling, shuffled")
    print(f"  models: {', '.join(sorted(sheet.model.unique()))}")
    print("\nFill in `refusal` and `uncertain` (true/false) for every row, then:")
    print("  python scripts/22_blind_sample.py --score")
    print("\nFirst three rows as you will see them:")
    for _, r in sheet.head(3).iterrows():
        print(f"  row {r.row}: {str(r.completion)[:88]!r}")
    return 0


def score() -> int:
    if not SHEET.exists():
        print(f"{SHEET} missing -- run with --export first")
        return 1
    sheet = pd.read_csv(SHEET)
    filled = sheet[sheet["refusal"].astype(str).str.strip() != ""]
    if not len(filled):
        print("no labels found in the `refusal` column")
        return 1
    if len(filled) < len(sheet):
        print(f"NOTE: {len(filled)}/{len(sheet)} rows labelled; scoring those only")

    human = as_bool(filled["refusal"])
    human_unc = as_bool(filled["uncertain"])
    auto = filled["completion"].apply(detector)

    tp = int((human & auto).sum())
    tn = int((~human & ~auto).sum())
    fp = int((~human & auto).sum())
    fn = int((human & ~auto).sum())
    agreement = (tp + tn) / len(filled)

    print(f"n = {len(filled)} completions, labelled by the author")
    print(f"  author refusal rate    {human.mean():.3f} "
          f"({int(human.sum())} of {len(filled)})")
    print(f"  author uncertain rate  {human_unc.mean():.3f} "
          f"({int(human_unc.sum())} of {len(filled)})")
    print(f"  detector refusal rate  {auto.mean():.3f}")
    print(f"  AGREEMENT              {agreement:.3f}    TP {tp}  FP {fp}  FN {fn}  TN {tn}")

    if fn:
        print("\n  refusals the detector MISSED -- add these phrasings to "
              "REFUSAL_MARKERS or drop the metric:")
        for c in filled[human & ~auto]["completion"].head(10):
            print(f"    {c!r}")
    if fp:
        print("\n  false alarms -- the detector fired where you saw no refusal:")
        for c in filled[~human & auto]["completion"].head(10):
            print(f"    {c!r}")

    by_model = filled.assign(refusal_b=human, uncertain_b=human_unc) \
                     .groupby("model")[["refusal_b", "uncertain_b"]].mean().round(3)
    print("\n  by model:")
    print(by_model.to_string())

    sentence = (f"A {len(filled)}-completion sample (ten per model, selected at a "
                f"fixed stride through the row index and shuffled) was labelled by "
                f"the author. Agreement with the keyword refusal detector was "
                f"{agreement:.0%}, with {int(human.sum())} refusals and "
                f"{int(human_unc.sum())} hedged responses observed.")
    print(f"\n  for the paper:\n    {sentence}")

    write_json({
        "n_labelled": int(len(filled)),
        "sampling": "ten per model at fixed stride through row index, shuffled",
        "labeller": "author",
        "detector_verdict_hidden_during_labelling": True,
        "author_refusal_rate": float(human.mean()),
        "author_uncertain_rate": float(human_unc.mean()),
        "detector_refusal_rate": float(auto.mean()),
        "agreement": agreement, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "by_model": by_model.to_dict(),
        "paper_sentence": sentence,
    }, PATHS.tables / "refusal_validation.json")
    print(f"\nwrote {PATHS.tables / 'refusal_validation.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--per-model", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.export:
        return export(args.per_model, args.seed)
    if args.score:
        return score()
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())