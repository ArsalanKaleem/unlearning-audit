#!/usr/bin/env python3
"""Days 23 / 33 --- cache residual activations at every layer.

    python scripts/05_extract_activations.py --checkpoint <ckpt> --label M_injected

Metadata is written alongside and validated on load. Re-extracting must give
bitwise-identical arrays; the --verify flag checks that.
"""
import numpy as np
from _common import PATHS, base_parser, load_sets, setup

from src.analysis.activations import extract_for_records
from src.model.loader import load_model
from src.utils.io import load_activations, save_activations


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sets", default="forget,retain,control,paraphrase")
    ap.add_argument("--verify", action="store_true", help="re-extract and compare bitwise")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, _ = load_sets(cfg, args.limit)

    model = load_model(cfg, args.checkpoint)

    for name in args.sets.split(","):
        records = sets[name]
        acts, labels, ents, meta = extract_for_records(model, records, cfg)
        meta["set"] = name
        meta["model_label"] = args.label
        meta["checkpoint"] = args.checkpoint or cfg["model"]["name"]
        path = PATHS.activations / f"{args.label}_{name}.npz"
        save_activations(acts, labels, ents, meta, path)
        print(f"  {name:12s} {acts.shape} -> {path.name}")

        if args.verify:
            acts2, _, _, _ = extract_for_records(model, records, cfg)
            same = np.array_equal(acts, acts2)
            print(f"    bitwise reproducible: {'PASS' if same else 'FAIL'}")
            if not same:
                return 1
            load_activations(path, expect={
                "model_name": cfg["model"]["name"],
                "hook_name": cfg["activations"]["hook"],
                "position": cfg["activations"]["position"],
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
