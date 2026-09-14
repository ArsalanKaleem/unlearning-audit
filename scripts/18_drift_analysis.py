#!/usr/bin/env python3
"""Day 35 --- the mandatory drift control.

    python scripts/18_drift_analysis.py --reference M_injected --targets M_npo_s0,M_npo_s1,M_npo_s2,M_gd_s2

An erasure claim requires forget-set activations to have moved MORE than
activations on prompts whose facts were never taught and never unlearned. If
every set moved equally, the fine-tune displaced the whole representation and
a probe-accuracy drop on the forget set says nothing specific about erasure.

The comparison that carries the argument is forget vs CONTROL: those prompts
share their structure exactly and differ only in whether the entity was taught.
Generic prompts are complete sentences rather than mid-sentence stems, so the
final-token position means something structurally different there; generic
drift is a secondary reference for "did the whole model move", not a matched
comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, setup, write_json

from src.analysis.activations import activation_drift
from src.utils.io import load_activations

SETS = ("forget", "retain", "control", "generic")


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--reference", default="M_injected")
    ap.add_argument("--targets", required=True)
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    primary = cfg["probe"]["primary_layer"]

    rows, summary = [], {}
    for target in args.targets.split(","):
        per_set = {}
        for name in SETS:
            ref_path = PATHS.activations / f"{args.reference}_{name}.npz"
            tgt_path = PATHS.activations / f"{target}_{name}.npz"
            if not (ref_path.exists() and tgt_path.exists()):
                print(f"  skip {target}/{name}: cache missing")
                continue
            a, _, _, _ = load_activations(ref_path)
            b, _, _, _ = load_activations(tgt_path)
            d = activation_drift(a, b)
            per_set[name] = d
            for layer in range(a.shape[0]):
                rows.append({
                    "model": target, "set": name, "layer": layer,
                    "relative_l2": float(d["relative_l2"][layer]),
                    "cosine": float(d["cosine"][layer]),
                })

        if "forget" not in per_set:
            continue
        f = float(per_set["forget"]["relative_l2"][primary])
        entry = {"drift_forget": f}
        print(f"\n{target}  (layer {primary})")
        print(f"  forget  drift {f:.4f}")
        for ref in ("control", "retain", "generic"):
            if ref not in per_set:
                continue
            r = float(per_set[ref]["relative_l2"][primary])
            entry[f"drift_{ref}"] = r
            entry[f"forget_over_{ref}"] = f / (r + 1e-12)
            print(f"  {ref:8s} drift {r:.4f}   ratio {f / (r + 1e-12):.2f}x")

        ratio_control = entry.get("forget_over_control", 0.0)
        entry["forget_specific"] = bool(ratio_control > 1.5)
        entry["verdict"] = (
            "FORGET-SPECIFIC: the probe drop can be attributed to the forgotten facts"
            if entry["forget_specific"] else
            "GLOBAL: forget activations did not move meaningfully more than "
            "never-taught controls. The probe drop cannot be attributed to "
            "forget-specific erasure; cap the claim at output suppression."
        )
        print(f"  -> {entry['verdict']}")
        summary[target] = entry

    df = pd.DataFrame(rows)
    df.to_csv(PATHS.tables / "drift_by_layer.csv", index=False)
    write_json(summary, PATHS.tables / "drift_summary.json")

    n_specific = sum(v["forget_specific"] for v in summary.values())
    print(f"\n{n_specific}/{len(summary)} models show forget-specific drift at layer {primary}")
    print(f"wrote {PATHS.tables / 'drift_by_layer.csv'} and drift_summary.json")

    from src.viz import figures as F
    sub = df[df["set"].isin(SETS)].rename(columns={"set": "condition",
                                                   "relative_l2": "accuracy"})
    for target in summary:
        s = sub[sub["model"] == target]
        fig = F.fig_layerwise_probe(s, chance=0.0, value_col="accuracy",
                                    control_col="__none__",
                                    title=f"Relative activation drift by layer ({target})")
        fig.axes[0].set_ylabel("relative L2 drift")
        fig.axes[0].set_ylim(0, None)
        print("figure:", F.save(fig, PATHS.figures / f"fig_drift_{target}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())