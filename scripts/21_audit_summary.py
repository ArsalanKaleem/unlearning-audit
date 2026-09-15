#!/usr/bin/env python3
"""Day 45 --- let the code fix the claim, not the author.

    python scripts/21_audit_summary.py --models M_npo_s0,M_npo_s1,M_npo_s2,M_gd_s2

Assembles an EvidenceProfile per model from the tables already produced, then
calls highest_supported_rung(), which walks the claim ladder and stops at the
first rung whose evidence does not clear the preregistered threshold.

The point of running this rather than reading the tables: the conclusion is
then produced by a rule written down in advance and applied mechanically. A
reader can check the rule. They cannot check my judgement.

Reference points for normalisation:
  0 = the never-taught control level (looks untaught)
  1 = the M_injected level (looks fully taught)

A value near 0 on every dimension does NOT license an erasure claim on its own.
The drift control is what decides whether a change is attributable to the
forgotten facts, and highest_supported_rung caps the claim at rung 1 when drift
is global -- which is what happens here.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from _common import PATHS, base_parser, setup, write_json

from src.eval.claim_ladder import (DIMENSIONS, build_profile, compare_profiles,
                                   highest_supported_rung, profile_table)


def behav(label, setname, col="constrained_correct"):
    p = PATHS.tables / f"behaviour_{label}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[df["set"] == setname]
    return float(sub[col].mean()) if len(sub) else None


def paraphrase_forget(label):
    """Paraphrase accuracy on FORGET entities only.

    The combined paraphrase set mixes forget and retain entities, so its mean
    is a blend and cannot be read as a forget-set measurement.
    """
    p = PATHS.tables / f"behaviour_{label}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "split" not in df.columns:
        return None
    sub = df[(df["set"] == "paraphrase") & (df["split"] == "forget")]
    return float(sub["constrained_correct"].mean()) if len(sub) else None


def probe_at(label, layer):
    p = PATHS.tables / "probe_logistic_forget.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[(df["model"] == label) & (df["set"] == "forget") & (df["layer"] == layer)]
    return float(sub["accuracy"].mean()) if len(sub) else None


def lens_peak(label, setname="forget"):
    """Peak internal log-probability of the target across layers.

    Read from lens_summary.csv rather than left unmeasured: the lens is one of
    the project's main results and should not appear as a blank in its own
    audit table.
    """
    p = PATHS.tables / "lens_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[(df["model"] == label) & (df["set"] == setname)]
    return float(sub["peak_logprob"].mean()) if len(sub) else None


def patch_peak(label):
    p = PATHS.tables / f"patching_{label}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    real = df[(df["kind"] == "real") & (df["set"] == "forget")]
    return float(real.groupby("layer")["recovery"].mean().max()) if len(real) else None


def patch_retain_reference(label):
    """Patching recovery on RETAIN facts -- the specificity reference.

    Retain facts were never unlearned, so patching has no gap to close there.
    Whatever recovery it shows is the baseline effect of substituting a residual
    state from a slightly different model, and the forget-set number has to be
    read against it.
    """
    p = PATHS.tables / f"patching_{label}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    real = df[(df["kind"] == "real") & (df["set"] == "retain")]
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
    taught = args.reference_taught

    # --- reference points ---------------------------------------------------
    ref_ctrl, ref_inj = {}, {}
    ref_ctrl["direct_recall"] = behav(taught, "control")
    ref_inj["direct_recall"] = behav(taught, "forget")
    ref_ctrl["paraphrase_recall"] = ref_ctrl["direct_recall"]
    ref_inj["paraphrase_recall"] = paraphrase_forget(taught)
    ref_ctrl["representation_probe"] = cfg["baseline"]["probe_acc_control_entities"]
    ref_inj["representation_probe"] = probe_at(taught, layer)
    # For the lens, the never-taught reference is M_injected's own CONTROL set:
    # that is what an internal trajectory looks like for an entity the model
    # was never taught, measured on the same model.
    ref_ctrl["logit_lens_peak"] = lens_peak(taught, "control")
    ref_inj["logit_lens_peak"] = lens_peak(taught, "forget")
    ref_ctrl["causal_recovery"] = 0.0
    ref_inj["causal_recovery"] = 1.0
    ref_ctrl["steering_recovery"] = 0.0
    ref_inj["steering_recovery"] = 1.0
    ref_ctrl["relearning_recovery"] = ref_ctrl["direct_recall"]
    ref_inj["relearning_recovery"] = ref_inj["direct_recall"]

    missing_ref = [k for k, v in list(ref_ctrl.items()) + list(ref_inj.items())
                   if v is None]
    if missing_ref:
        print(f"NOTE: no reference value for {sorted(set(missing_ref))}; "
              "those dimensions will normalise to NaN and stop the ladder walk.")

    print("reference levels (0 = never taught, 1 = fully taught):")
    for k in DIMENSIONS:
        c, i = ref_ctrl.get(k), ref_inj.get(k)
        if c is not None and i is not None:
            print(f"  {k:22s} control={c:+.3f}  taught={i:+.3f}")
    print()

    drift_all = {}
    dp = PATHS.tables / "drift_summary.json"
    if dp.exists():
        drift_all = json.load(open(dp))

    profiles, extra_rows = [], []
    for label in args.models.split(","):
        meas = {}
        for k, v in (("direct_recall", behav(label, "forget")),
                     ("paraphrase_recall", paraphrase_forget(label)),
                     ("representation_probe", probe_at(label, layer)),
                     ("logit_lens_peak", lens_peak(label, "forget")),
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
        for d in DIMENSIONS:
            if d in meas:
                print(f"    {d:22s} raw={meas[d]:+.3f}  norm={p.normalised[d]:+.3f}")
        if p.missing():
            print(f"    NOT MEASURED: {', '.join(p.missing())}")

        # context the ladder does not use but the reader needs
        retain_patch = patch_retain_reference(label)
        lens_ctrl = lens_peak(label, "control")
        if retain_patch is not None:
            print(f"    [context] patching on RETAIN facts (never unlearned): "
                  f"{retain_patch:+.3f} -- the forget number must be read against this")
        if lens_ctrl is not None:
            print(f"    [context] lens peak on NEVER-TAUGHT entities: {lens_ctrl:+.3f}")
        if util.get("retain_ratio") is not None:
            print(f"    [context] retain ratio vs M_injected: {util['retain_ratio']:.3f}")

        print(f"    HIGHEST SUPPORTED RUNG: {verdict['highest_rung']} "
              f"({verdict['claim']})")
        print(f"    blocked by: {verdict['blocked_by'] or '(nothing recorded)'}")
        if verdict["drift_caveat"]:
            print(f"    {verdict['drift_caveat']}")
        print()

        extra_rows.append({"model": label,
                           "patching_retain_reference": retain_patch,
                           "lens_peak_never_taught": lens_ctrl,
                           "retain_ratio": util.get("retain_ratio")})

    df = pd.DataFrame(profile_table(profiles)).merge(pd.DataFrame(extra_rows),
                                                     on="model", how="left")
    df.to_csv(PATHS.tables / "audit_profiles.csv", index=False)

    write_json({
        "primary_layer": layer,
        "reference_control": ref_ctrl,
        "reference_taught": ref_inj,
        "normalisation": "0 = never-taught control level, 1 = M_injected level",
        "threshold": 0.25,
        "profiles": [p.as_dict() for p in profiles],
        "verdicts": {p.model_label: highest_supported_rung(p) for p in profiles},
        "context": extra_rows,
    }, PATHS.tables / "audit_summary.json")

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