#!/usr/bin/env python3
"""Days 23 / 33 / 34 --- layer-wise probing with the full control suite.

    python scripts/06_probe_sweep.py --labels M_injected,M_npo --set forget
    python scripts/06_probe_sweep.py --labels M_injected,M_npo --set forget --probe mlp

Runs on CPU. The control-entity run is mandatory: if the probe reads
never-taught entities above chance, stop and fix the dataset.
"""
import pandas as pd
from _common import PATHS, base_parser, setup

from src.analysis.probes import layerwise_probe
from src.utils.io import load_activations
from src.viz import figures as F


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--labels", required=True, help="comma-separated model labels")
    ap.add_argument("--set", default="forget")
    ap.add_argument("--probe", default="logistic", choices=["logistic", "mlp"])
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)

    rows = []
    for label in args.labels.split(","):
        for setname in (args.set, "control"):
            path = PATHS.activations / f"{label}_{setname}.npz"
            if not path.exists():
                print(f"  skip {path.name} (missing)")
                continue
            acts, labels, ents, meta = load_activations(path, expect={
                "hook_name": cfg["activations"]["hook"],
                "position": cfg["activations"]["position"],
            })
            r = layerwise_probe(
                acts, labels, ents,
                seeds=tuple(range(cfg["probe"]["n_seeds"])),
                kind=args.probe, hidden=args.hidden,
                C_grid=cfg["probe"]["C_grid"], max_iter=cfg["probe"]["max_iter"],
                test_frac=cfg["probe"]["test_frac"],
                standardise=cfg["probe"]["standardise"],
            ) if args.probe == "logistic" else layerwise_probe(
                acts, labels, ents,
                seeds=tuple(range(cfg["probe"]["n_seeds"])),
                kind="mlp", hidden=args.hidden,
                test_frac=cfg["probe"]["test_frac"],
                standardise=cfg["probe"]["standardise"],
            )
            for x in r:
                x["model"] = label
                x["set"] = setname
                x["condition"] = f"{label}:{setname}"
            rows += r
            peak = max(r, key=lambda z: z["accuracy"])
            print(f"  {label:14s} {setname:8s} peak acc={peak['accuracy']:.3f} "
                  f"at layer {peak['layer']} (chance {peak['chance']:.3f})")

    df = pd.DataFrame(rows)
    out = PATHS.tables / f"probe_{args.probe}_{args.set}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    ctrl = df[df["set"] == "control"]
    if len(ctrl):
        m = ctrl.groupby("layer")["accuracy"].mean().max()
        ok = m < ctrl["chance"].iloc[0] + 0.10
        print(f"CHECKPOINT: max control-entity probe accuracy {m:.3f} -> "
              f"{'PASS' if ok else 'FAIL -- fix the dataset before proceeding'}")

    fig = F.fig_layerwise_probe(df, chance=float(df["chance"].iloc[0]),
                                title=f"Probe accuracy by layer ({args.set})")
    print("figure:", F.save(fig, PATHS.figures / f"fig4_probe_{args.probe}_{args.set}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
