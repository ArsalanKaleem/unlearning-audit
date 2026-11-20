#!/usr/bin/env python3
"""Revision R4 --- calibrate every diagnostic against ground-truth anchors.

    python scripts/30_revision_analysis.py

Reads results/activations/<label>_<set>.npz and results/revision/<label>/*
(written by scripts/29) and produces, in results/revision/:

  entity_scores.csv     one row per (model, set, entity, diagnostic)
  anchor_validation.csv for each diagnostic: does it separate the SUPPRESSED
                        anchors (fact provably intact up to the frozen depth)
                        from the ABSENT anchor (oracle, fact never taught)?
                        And two ABSENT anchors from each other (the noise floor)?
  model_positions.csv   each unlearned model placed between ABSENT (0) and
                        PRESENT (1), entity-paired bootstrap
  control_surrogate.csv the reviewers' Q4: is the never-taught control inside
                        an unlearned model a substitute for the oracle?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from _common import PATHS

from src.analysis.calibration import (crossfit_probe, normalised_position,
                                      paired_auroc_ci, paired_diff_ci, unpaired_diff_ci)

REV = PATHS.root / "results" / "revision"
ACTS = PATHS.activations
N_LAYERS = 12                       # reset from the data in main()


def probe_layers(n):
    """Fractional depths; for GPT-2 (12 blocks) these are 6, 8, 9, 10, 11."""
    return sorted({int(round(n * f)) for f in (0.5, 0.67, 0.75, 0.83)} | {n - 1})


def lens_layers(n):
    return tuple(range(n // 2, n))

ROLES = {
    "M_injected": "present", "M_injrep_s0": "present",
    "M_oracle_s0": "absent", "M_oracle_s1": "absent", "M_oracle_s2": "absent",
    "M_k11_s0": "suppressed", "M_k9_s0": "suppressed", "M_k6_s0": "suppressed",
    "M_k11_s1": "suppressed", "M_k11_s2": "suppressed",
    "M_rmu_s0": "test", "M_npo_s0": "test", "M_npo_s1": "test", "M_npo_s2": "test", "M_gd_s2": "test",
}


def available(label: str) -> bool:
    return (ACTS / f"{label}_forget.npz").exists() and (REV / label / "lens.npz").exists()


def load_acts(label, s, acts_dir=None):
    d = np.load((acts_dir or ACTS) / f"{label}_{s}.npz", allow_pickle=True)
    return d["acts"].astype(np.float32), d["labels"], d["entities"].astype(str)


def probe_cell(label, s, layer, acts_dir):
    X, y, e = load_acts(label, s, acts_dir)
    out = crossfit_probe(X[layer], y, e, seed=0)
    return [{"model": label, "set": s, "entity_id": k, "diagnostic": f"probe_L{layer}", "score": v}
            for k, v in out.items()]


def entity_rows(label):
    rows = []
    b = pd.read_csv(REV / label / "behaviour.csv")
    for s in ("forget", "control", "retain"):
        g = b[b.set == s].groupby("entity_id")
        for e, x in g:
            rows.append({"model": label, "set": s, "entity_id": e, "diagnostic": "behaviour",
                         "score": x.constrained_correct.mean()})
            rows.append({"model": label, "set": s, "entity_id": e, "diagnostic": "behaviour_logprob",
                         "score": x.target_logprob.mean()})
    # paraphrase prompts are about forget AND retain entities; keep forget ones
    p = b[(b.set == "paraphrase") & (b.split == "forget")].groupby("entity_id")
    for e, x in p:
        rows.append({"model": label, "set": "forget", "entity_id": e, "diagnostic": "paraphrase",
                     "score": x.constrained_correct.mean()})
    lens = np.load(REV / label / "lens.npz", allow_pickle=True)
    for s in ("forget", "control", "retain"):
        ents = lens[f"{s}_entities"].astype(str)
        con = lens[f"{s}_constrained"]
        for e in np.unique(ents):
            m = con[ents == e].mean(0)
            n, h = len(m), len(m) // 2
            for L in lens_layers(n):
                rows.append({"model": label, "set": s, "entity_id": e,
                             "diagnostic": f"lens_L{L}", "score": float(m[L])})
            rows.append({"model": label, "set": s, "entity_id": e, "diagnostic": f"lens_peak{h}_{n - 2}",
                         "score": float(m[h:n - 1].max())})
            rows.append({"model": label, "set": s, "entity_id": e, "diagnostic": "lens_suppression",
                         "score": float(m[h:n - 1].max() - m[n - 1])})
    rf = REV / label / "relearn.csv"
    if rf.exists():
        r = pd.read_csv(rf)
        piv = r.pivot_table(index=["split", "entity_id"], columns="arm", values="held_after").reset_index()
        piv["gain"] = piv["attack"] - piv["control"]
        for e, x in piv.groupby("entity_id"):
            rows.append({"model": label, "set": "forget", "entity_id": e, "diagnostic": "relearn_gain",
                         "score": float(x.gain.mean())})
        lp = r.pivot_table(index=["split", "entity_id"], columns="arm",
                           values="held_logprob_after").reset_index()
        lp["gain"] = lp["attack"] - lp["control"]
        for e, x in lp.groupby("entity_id"):
            rows.append({"model": label, "set": "forget", "entity_id": e,
                         "diagnostic": "relearn_gain_logprob", "score": float(x.gain.mean())})
    return rows


def vec(df, model, s, diag, ents):
    x = df[(df.model == model) & (df.set == s) & (df.diagnostic == diag)].set_index("entity_id").score
    return x.reindex(ents).to_numpy()


def both(*xs, min_n=10):
    """Restrict entity-paired vectors to entities scored in all of them
    (relearning holds out each entity in only some splits)."""
    m = np.all([~np.isnan(x) for x in xs], axis=0)
    return [x[m] for x in xs] if m.sum() >= min_n else None


def main() -> int:
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma list; default = all available")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--reuse", action="store_true", help="reuse entity_scores.csv in the output dir")
    ap.add_argument("--root", default=None,
                    help="results root holding revision/ and activations/ (e.g. a Kaggle run)")
    ap.add_argument("--roles", default=None, help="json {label: role}; default = the GPT-2 roles")
    ap.add_argument("--absent0", default="M_oracle_s0")
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", 6)))
    args = ap.parse_args()
    global REV_OUT, REV, ACTS, ROLES, N_LAYERS
    if args.root:
        from pathlib import Path as _P
        REV, ACTS = _P(args.root) / "revision", _P(args.root) / "activations"
    if args.roles:
        ROLES = json.loads(open(args.roles).read())
    REV_OUT = REV / ("prelim" + args.out_suffix) if args.out_suffix else REV
    REV_OUT.mkdir(parents=True, exist_ok=True)
    want = [m for m in args.models.split(",") if m]
    labels = [m for m in ROLES if available(m) and (not want or m in want)]
    print("models:", labels)
    if args.reuse and (REV_OUT / "entity_scores.csv").exists():
        df = pd.read_csv(REV_OUT / "entity_scores.csv")
        df = df[df.model.isin(labels)]
    else:
        rows = []
        for m in labels:
            rows += entity_rows(m)
        N_LAYERS = load_acts(labels[0], "forget")[0].shape[0]
        cells = [(m, s, L) for m in labels for s in ("forget", "control", "retain")
                 for L in probe_layers(N_LAYERS)]
        for r in Parallel(n_jobs=args.n_jobs)(delayed(probe_cell)(*c, ACTS) for c in cells):
            rows += r
        df = pd.DataFrame(rows)
        df.to_csv(REV_OUT / "entity_scores.csv", index=False)

    forget_ents = sorted(df[(df.set == "forget") & (df.diagnostic == "behaviour")].entity_id.unique())
    diags = sorted(df.diagnostic.unique())
    absent = [m for m in labels if ROLES[m] == "absent"]
    supp = [m for m in labels if ROLES[m] == "suppressed"]
    A0 = args.absent0

    # --- 1. anchor validation ------------------------------------------------
    val = []
    for d in diags:
        a = vec(df, A0, "forget", d, forget_ents)
        if np.isnan(a).all():
            continue
        for m in [x for x in labels if x != A0]:
            pr = both(vec(df, m, "forget", d, forget_ents), a)
            if pr is None:
                continue
            v, a_ = pr
            ci = paired_auroc_ci(v, a_)
            dd = paired_diff_ci(v, a_)
            val.append({"diagnostic": d, "model": m, "role": ROLES[m], "vs": A0, "n": len(v),
                        "mean_model": float(np.mean(v)), "mean_oracle": float(np.mean(a_)),
                        **ci, "diff": dd["diff"], "diff_lo": dd["lo"], "diff_hi": dd["hi"]})
    val = pd.DataFrame(val)
    val.to_csv(REV_OUT / "anchor_validation.csv", index=False)

    # --- 2. where do the unlearned models sit? -------------------------------
    pos = []
    for d in diags:
        a = vec(df, A0, "forget", d, forget_ents)
        for ref_name in ["M_injected"] + supp:
            r = vec(df, ref_name, "forget", d, forget_ents)
            for m in [x for x in labels if ROLES[x] in ("test", "absent", "suppressed") and x not in (A0, ref_name)]:
                pr = both(vec(df, m, "forget", d, forget_ents), a, r)
                if pr is None:
                    continue
                pos.append({"diagnostic": d, "model": m, "role": ROLES[m], "reference": ref_name,
                            "n": len(pr[0]), **normalised_position(*pr)})
    pd.DataFrame(pos).to_csv(REV_OUT / "model_positions.csv", index=False)

    # --- 3. is the in-model never-taught control a surrogate for the oracle? --
    sur = []
    for d in diags:
        o = vec(df, A0, "forget", d, forget_ents)
        o = o[~np.isnan(o)]
        if len(o) < 10:
            continue
        for m in labels:
            c = df[(df.model == m) & (df.set == "control") & (df.diagnostic == d)].score.to_numpy()
            f = vec(df, m, "forget", d, forget_ents)
            f = f[~np.isnan(f)]
            if len(c) == 0 or len(f) < 10:
                continue
            s1 = unpaired_diff_ci(c, o)
            s2 = unpaired_diff_ci(f, c)
            sur.append({"diagnostic": d, "model": m, "role": ROLES[m],
                        "control_minus_oracle": s1["diff"], "co_lo": s1["lo"], "co_hi": s1["hi"],
                        "forget_minus_control": s2["diff"], "fc_lo": s2["lo"], "fc_hi": s2["hi"],
                        "forget_minus_oracle": float(np.mean(f) - np.mean(o))})
    pd.DataFrame(sur).to_csv(REV_OUT / "control_surrogate.csv", index=False)
    print("wrote", REV_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
