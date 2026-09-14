#!/usr/bin/env python3
"""Day 43 --- cross-model activation patching with controls. Experiment C2/C3.

    python scripts/09_patching.py --donor <M_injected_ckpt> --receiver <M_unl_ckpt>

Donor states come from M_injected; the receiver is the unlearned model. High
recovery means the unlearned model can still USE the fact when the state is
supplied -- a claim about MACHINERY (rung 3), not about stored contents.

LOW recovery is equally informative and is what this project found (0.13-0.23
across four models): the unlearned model cannot use the fact even when handed
the representation that encodes it, so the read-out path downstream of the
patched layer also changed.

Unlike probing, patching is an INTERVENTION, so its interpretation is not
undermined by the global-drift confound.

CHECKPOINT: random-position controls must show near-zero recovery.
"""
import json

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, load_sets, setup

from src.analysis.causal import causal_trace, patching_sweep
from src.model.loader import load_model
from src.viz import figures as F


def matched_length_partner(model, record, pool):
    """A prompt with the SAME token count and a DIFFERENT answer.

    Causal tracing patches by position index, so clean and corrupted prompts
    must tokenise to the same length or the positions do not correspond and the
    heatmap is meaningless. Taking the first record with a different answer --
    the obvious approach -- fails whenever the two names tokenise differently,
    which for invented names is most of the time.
    """
    base_len = model.to_tokens(record["prompt"]).shape[1]
    for x in pool:
        if x["answer"] == record["answer"]:
            continue
        if model.to_tokens(x["prompt"]).shape[1] == base_len:
            return x
    return None


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--label", default="M_unlearned")
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--trace", action="store_true", help="also run within-model causal tracing")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, _ = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]
    distractors = sorted(city_ids.values())

    donor = load_model(cfg, args.donor)
    receiver = load_model(cfg, args.receiver)

    rows = []
    for setname in ("forget", "retain"):          # retain = the specificity control
        for r in sets[setname][:args.n_prompts]:
            ans = city_ids[r["answer"]]
            d = [x for x in distractors if x != ans]
            for row in patching_sweep(receiver, donor, r["prompt"], ans, d,
                                      hook=cfg["activations"]["hook"], seed=cfg["seed"]):
                row.update({"set": setname, "entity_id": r["entity_id"],
                            "model": args.label})
                rows.append(row)

    df = pd.DataFrame(rows)
    out = PATHS.tables / f"patching_{args.label}.csv"
    df.to_csv(out, index=False)

    real = df[(df.kind == "real") & (df.set == "forget")]
    ctrl = df[(df.kind == "random_position") & (df.set == "forget")]
    by_layer = real.groupby("layer")["recovery"].mean()
    peak_layer, peak = by_layer.idxmax(), by_layer.max()
    ctrl_mean = ctrl["recovery"].mean()

    # The retain arm is the specificity control: patching should do little on
    # facts that were never unlearned, because there is no gap to close.
    retain_real = df[(df.kind == "real") & (df.set == "retain")]
    retain_peak = retain_real.groupby("layer")["recovery"].mean().max() \
        if len(retain_real) else float("nan")

    print(f"\n  peak recovery {peak:.3f} at layer {peak_layer}")
    print(f"  random-position control mean recovery {ctrl_mean:.3f}")
    print(f"  retain-set peak recovery {retain_peak:.3f} (specificity reference)")
    ok = abs(ctrl_mean) < 0.15
    print(f"CHECKPOINT: controls near zero -> {'PASS' if ok else 'FAIL -- patching code is wrong'}")
    if ok and peak < 0.30:
        print("  NOTE: recovery is LOW. The receiver cannot use the fact even when "
              "given the donor's state, so machinery downstream of the patched "
              "layer also changed. This is evidence AGAINST rung 3, not for it.")

    fig = F.fig_patching_recovery(df[df.set == "forget"])
    print("figure:", F.save(fig, PATHS.figures / f"fig10_patching_{args.label}"))

    if args.trace:
        r = sets["forget"][0]
        other = matched_length_partner(donor, r, sets["forget"])
        if other is None:
            print("  tracing skipped: no same-length prompt with a different answer")
        else:
            n_tok = donor.to_tokens(r["prompt"]).shape[1]
            print(f"  tracing: clean vs corrupted, both {n_tok} tokens")
            grid_pre = causal_trace(donor, r["prompt"], other["prompt"],
                                    city_ids[r["answer"]], distractors)
            grid_post = causal_trace(receiver, r["prompt"], other["prompt"],
                                     city_ids[r["answer"]], distractors)
            np.save(PATHS.tables / f"trace_diff_{args.label}.npy", grid_post - grid_pre)
            labels = [donor.to_string(t) for t in donor.to_tokens(r["prompt"])[0]]
            fig = F.fig_heatmap(grid_post - grid_pre, xticklabels=labels,
                                title=f"Causal tracing difference ({args.label} - M_injected)")
            print("figure:", F.save(fig, PATHS.figures / f"fig_tracing_diff_{args.label}"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())