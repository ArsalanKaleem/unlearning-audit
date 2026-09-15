#!/usr/bin/env python3
"""Day 45 --- let the code fix the claim, not the author.

    python scripts/21_audit_summary.py --models M_npo_s0,M_npo_s1,M_npo_s2,M_gd_s2

Assembles a EvidenceProfile per model from the tables already produced,
then calls highest_supported_rung(), which walks the claim ladder and stops at
the first rung whose evidence does not clear the preregistered threshold.

The point of running this rather than reading the tables myself: the conclusion
is then produced by a rule written down in advance, applied mechanically. A
reader can check the rule. They cannot check my judgement.

Reference points for normalisation:
  0 = the never-taught control level (looks untaught)
  1 = the M_injected level (looks fully taught)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from _common import PATHS, base_parser, setup, write_json

from src.eval.claim_ladder import (build_profile, compare_profiles,
                                   highest_supported_rung, profile_table)


def behav(label, setname, col="constrained_correct"):
    p = PATHS.tables / f"behaviour_{label}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[df["set"] == setname]
    return float(sub[col].mean()) if len(sub) else None


def paraphrase_forget(label):
    p = PATHS.tables / f"behaviour_{label}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[(df["set"] == "paraphrase") & (df["split"] == "forget")]
    return float(sub["constrained_correct"].mean()) if len(sub) else None


def probe_at(label, layer):
    p = PATHS.tables / "probe_logistic_forget.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[(df["model"] == label) & (df["set"] == "forget") & (df["layer"] == layer)]
    return float(sub["accuracy"].mean()) if len(sub) else None


def patch_peak(label):
    p = PATHS.tables / f"patching_{label}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    real = df[(df["kind"] == "real") & (df["set"] == "forget")]
    return float(real.groupby("layer")["recovery"].mean().max()) if len(real) else None


def steer_fraction(label):
    p = PATHS.tables / f"steering_verdict_{label}.json"
    if not p.exists():
        return None
    return float(json.load(open(p)).get("gap_fraction", 0.0))


def relearn(label):
    p = PATHS.tables / f"recovery_verdict_{label}.json"
    if not p.exists():
        return None
    return float(json.load(open(p))["attack_after"])


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--models", required=True)
    ap.add_argument("--reference-taught", default="M_injected")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    layer = cfg["probe"]["primary_layer"]

    ref_ctrl, ref_inj = {}, {}
    ref_ctrl["direct_recall"] = behav(args.reference_taught, "control")
    ref_inj["direct_recall"] = behav(args.reference_taught, "forget")
    ref_ctrl["paraphrase_recall"] = ref_ctrl["direct_recall"]
    ref_inj["paraphrase_recall"] = paraphrase_forget(args.reference_taught)
    ref_ctrl["representation_probe"] = cfg["baseline"]["probe_acc_control_entities"]
    ref_inj["representation_probe"] = probe_at(args.reference_taught, layer)
    ref_ctrl["causal_recovery"] = 0.0
    ref_inj["causal_recovery"] = 1.0
    ref_ctrl["steering_recovery"] = 0.0
    ref_inj["steering_recovery"] = 1.0
    ref_ctrl["relearning_recovery"] = ref_ctrl["direct_recall"]
    ref_inj["relearning_recovery"] = ref_inj["direct_recall"]

    print(f"reference control level: {ref_ctrl}")
    print(f"reference taught level:  {ref_inj}\n")

    drift_all = {}
    dp = PATHS.tables / "drift_summary.json"
    if dp.exists():
        drift_all = json.load(open(dp))

    profiles = []
    for label in args.models.split(","):
        meas = {}
        for k, v in (("direct_recall", behav(label, "forget")),
                     ("paraphrase_recall", paraphrase_forget(label)),
                     ("representation_probe", probe_at(label, layer)),
                     ("causal_recovery", patch_peak(label)),
                     ("steering_recovery", steer_fraction(label)),
                     ("relearning_recovery", relearn(label))):
            if v is not None:
                meas[k] = v

        util = {}
        rr = behav(label, "retain")
        if rr is not None and cfg["baseline"]["retain_acc"]:
            util["retain_ratio"] = rr / cfg["baseline"]["retain_acc"]

        p = build_profile(label, 0, meas, ref_ctrl, ref_inj,
                          drift=drift_all.get(label, {}), utility=util)
        profiles.append(p)

        verdict = highest_supported_rung(p, threshold=0.25)
        print(f"=== {label}")
        for d in sorted(meas):
            print(f"    {d:22s} raw={meas[d]:.3f}  norm={p.normalised[d]:+.3f}")
        if p.missing():
            print(f"    NOT MEASURED: {', '.join(p.missing())}")
        print(f"    HIGHEST SUPPORTED RUNG: {verdict['highest_rung']} "
              f"({verdict['claim']})")
        print(f"    blocked by: {verdict['blocked_by']}")
        if verdict["drift_caveat"]:
            print(f"    {verdict['drift_caveat']}")
        print()

    df = pd.DataFrame(profile_table(profiles))
    df.to_csv(PATHS.tables / "audit_profiles.csv", index=False)

    summary = {
        "layer": layer,
        "reference_control": ref_ctrl,
        "reference_taught": ref_inj,
        "profiles": [p.as_dict() for p in profiles],
        "verdicts": {p.model_label: highest_supported_rung(p) for p in profiles},
    }
    write_json(summary, PATHS.tables / "audit_summary.json")

    rungs = {p.model_label: highest_supported_rung(p)["highest_rung"] for p in profiles}
    print("FINAL: " + ", ".join(f"{k}=rung {v}" for k, v in rungs.items()))
    if len(set(rungs.values())) == 1:
        r = list(rungs.values())[0]
        print(f"All models agree at rung {r}. That is the claim the paper may make, "
              "and no higher.")
    else:
        print("Models disagree. Report the range and the reason, not the maximum.")
    print(f"\nwrote {PATHS.tables / 'audit_summary.json'} and audit_profiles.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
