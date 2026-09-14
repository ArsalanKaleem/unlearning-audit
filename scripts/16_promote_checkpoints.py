#!/usr/bin/env python3
"""Copy band-selected checkpoints into checkpoints/ under stable labels.

    python scripts/16_promote_checkpoints.py

Reads configs/selected_<method>.json, which 07_unlearn_sweep.py wrote when it
applied the band. Nothing is chosen here: if a run did not enter the band it
is not promoted. Typing paths by hand is how the wrong step ends up in an
analysis, so this exists to make that impossible.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"npo_retain": "M_npo", "gradiff": "M_gd"}


def main() -> int:
    promoted = []
    for method, prefix in LABELS.items():
        cfg_path = ROOT / "configs" / f"selected_{method}.json"
        if not cfg_path.exists():
            print(f"skip {method}: {cfg_path.name} missing")
            continue
        payload = json.load(open(cfg_path))
        for run in payload["runs"]:
            if not run["entered_band"]:
                print(f"  skip seed {run['seed']} ({method}): not in band")
                continue
            src = ROOT / run["checkpoint"]
            dst = ROOT / "checkpoints" / f"{prefix}_s{run['seed']}"
            if not src.exists():
                print(f"  MISSING {src}")
                return 1
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
            promoted.append({
                "label": dst.name, "method": method, "seed": run["seed"],
                "step": run["selected_step"], "source": run["checkpoint"],
                "forget_acc": run["forget_acc"], "retain_acc": run["retain_acc"],
                "ppl_ratio": run.get("ppl_ratio"),
            })
            print(f"  {dst.name:12s} step {run['selected_step']:4d}  "
                  f"forget={run['forget_acc']:.3f} retain={run['retain_acc']:.3f}")

    out = ROOT / "configs" / "model_set.json"
    json.dump({"models": promoted}, open(out, "w"), indent=2)
    print(f"\n{len(promoted)} models promoted; wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())