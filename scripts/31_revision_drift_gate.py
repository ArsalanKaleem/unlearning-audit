#!/usr/bin/env python3
"""Revision R6 --- put the submission's drift gate through known cases.

    python scripts/31_revision_drift_gate.py

The submission capped every model at "output suppression" because forget-set
activations did not move >= 1.5x as far as never-taught activations (relative
L2, relative to M_injected). Reviewers asked whether that rule is valid. Here
it is applied to cases whose answer is known by construction:

  LEACE@L      linear city information provably removed at layer L
               (closed-form eraser fit on M_injected's forget activations)
  M_k11/k9     depth-limited NPO: the residual stream below layer k is
               bit-identical to M_injected, so drift there is exactly zero
plus every unlearned model, for reference. Writes
results/revision/drift_gate.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from _common import PATHS

from src.analysis.calibration import crossfit_probe
from src.analysis.erasure import fit_leace

REV = PATHS.root / "results" / "revision"
LAYERS = (6, 8, 9, 10, 11)
MODELS = ["M_k11_s0", "M_k9_s0", "M_npo_s0", "M_npo_s1", "M_npo_s2", "M_gd_s2", "M_rmu_s0",
          "M_oracle_s0"]
GATE = 1.5


ACTS = PATHS.activations


def load(label, s):
    d = np.load(ACTS / f"{label}_{s}.npz", allow_pickle=True)
    return d["acts"].astype(np.float64), d["labels"], d["entities"].astype(str)


def rel_l2(a, b):
    return float(np.mean(np.linalg.norm(b - a, axis=1) / (np.linalg.norm(a, axis=1) + 1e-9)))


def main() -> int:
    import argparse
    from pathlib import Path
    global ACTS, REV
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="results root with activations/ (e.g. a Kaggle run)")
    ap.add_argument("--layers", default=None, help="comma list; default 6,8,9,10,11 (GPT-2)")
    ap.add_argument("--models", default=None, help="comma list of models to compare")
    ap.add_argument("--frozen", default=None, help="json {model: k} for depth-limited anchors")
    args = ap.parse_args()
    layers = tuple(int(x) for x in args.layers.split(",")) if args.layers else LAYERS
    models = args.models.split(",") if args.models else MODELS
    frozen = json.loads(args.frozen) if args.frozen else {"M_k11_s0": 11, "M_k9_s0": 9}
    if args.root:
        ACTS, REV = Path(args.root) / "activations", Path(args.root) / "revision"
    inj = {s: load("M_injected", s) for s in ("forget", "control")}
    rows = []
    for L in layers:
        Xf, yf, ef = inj["forget"]
        Xc = inj["control"][0]
        # known removal: LEACE fit on forget, applied to both sets
        er = fit_leace(Xf[L], yf)
        f, c = rel_l2(Xf[L], er.transform(Xf[L])), rel_l2(Xc[L], er.transform(Xc[L]))
        probe_after = np.mean(list(crossfit_probe(er.transform(Xf[L]), yf, ef, seed=0).values()))
        rows.append({"case": "LEACE (known removal)", "layer": L, "truth": "removed",
                     "drift_forget": f, "drift_control": c, "ratio": f / c,
                     "gate_says_forget_specific": f / c > GATE, "probe_forget_after": probe_after})
        for m in models:
            try:
                mf, mc = load(m, "forget")[0], load(m, "control")[0]
            except FileNotFoundError:
                continue
            f, c = rel_l2(Xf[L], mf[L]), rel_l2(Xc[L], mc[L])
            k = frozen.get(m)
            truth = ("intact (frozen)" if k is not None and L < k else
                     "never learned" if "oracle" in m else "unknown")
            rows.append({"case": m, "layer": L, "truth": truth, "drift_forget": f,
                         "drift_control": c, "ratio": f / c if c > 0 else np.nan,
                         "gate_says_forget_specific": (f / c > GATE) if c > 0 else False,
                         "probe_forget_after": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(REV / "drift_gate.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
