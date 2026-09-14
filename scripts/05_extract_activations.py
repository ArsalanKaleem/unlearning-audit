#!/usr/bin/env python3
"""Days 23 / 33 --- cache residual activations at every layer.

    python scripts/05_extract_activations.py --checkpoint <ckpt> --label M_injected

Metadata is written alongside and validated on load. Re-extracting must give
bitwise-identical arrays; the --verify flag checks that.

The `generic` set is not part of the dataset -- it comes from GENERIC_PROMPTS,
the same sentences used for the perplexity probe, and it carries no labels.
It exists for the drift comparison: an erasure claim requires forget-set
activations to have moved MORE than activations on text that has nothing to do
with the experiment. Without it, "the probe accuracy fell" is compatible with
"the fine-tune moved the whole model", and those are very different results.
"""
import numpy as np
from _common import PATHS, base_parser, load_sets, setup

from src.analysis.activations import extract_for_records, get_resid_activations
from src.data.templates import GENERIC_PROMPTS
from src.model.loader import load_model
from src.utils.io import load_activations, save_activations


def extract_generic(model, cfg, label):
    """Activations on generic prompts. No labels; drift analysis only.

    Dummy labels and one pseudo-entity per prompt keep the .npz schema
    identical to the labelled sets, so load_activations and the drift code
    need no special case. Nothing should ever probe this cache -- there is no
    class structure in it to find.
    """
    acts = get_resid_activations(
        model, GENERIC_PROMPTS,
        hook=cfg["activations"]["hook"],
        position=cfg["activations"]["position"],
        batch_size=cfg["activations"]["batch_size"],
    )
    labels = np.zeros(len(GENERIC_PROMPTS), dtype=np.int64)
    entities = np.array([f"generic_{i:03d}" for i in range(len(GENERIC_PROMPTS))])
    meta = {
        "model_name": cfg["model"]["name"],
        "hook_name": cfg["activations"]["hook"],
        "position": cfg["activations"]["position"],
        "n_prompts": len(GENERIC_PROMPTS),
        "template_kinds": ["generic"],
        "splits": ["generic"],
        "labels_meaningful": False,
        "purpose": "drift reference only; do not probe this cache",
    }
    return acts, labels, entities, meta


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sets", default="forget,retain,control,paraphrase,generic")
    ap.add_argument("--verify", action="store_true", help="re-extract and compare bitwise")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, _ = load_sets(cfg, args.limit)

    model = load_model(cfg, args.checkpoint)

    for name in args.sets.split(","):
        if name == "generic":
            acts, labels, ents, meta = extract_generic(model, cfg, args.label)
            records = None
        else:
            if name not in sets:
                print(f"  skip {name}: not in the dataset")
                continue
            records = sets[name]
            acts, labels, ents, meta = extract_for_records(model, records, cfg)

        meta["set"] = name
        meta["model_label"] = args.label
        meta["checkpoint"] = args.checkpoint or cfg["model"]["name"]
        path = PATHS.activations / f"{args.label}_{name}.npz"
        save_activations(acts, labels, ents, meta, path)
        print(f"  {name:12s} {acts.shape} -> {path.name}")

        if args.verify:
            if records is None:
                acts2, _, _, _ = extract_generic(model, cfg, args.label)
            else:
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