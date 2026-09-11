#!/usr/bin/env python3
"""Days 40-42 --- SAE validity, latent selection, comparison and ablation.

    python scripts/10_sae_analysis.py --labels M_injected,M_npo --layer 8 \
        --sae-id blocks.8.hook_resid_pre

Order is enforced: validate -> select on M_injected's SELECTION SPLIT ->
compare on the held-out split against matched random latents.
"""
import json

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, setup, write_json

from src.analysis.sae import (compare_latent_activation, encode, load_sae,
                              sae_diagnostics, select_latents)
from src.utils.io import load_activations


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--sae-id", required=True, help="VERIFY this against SAELens first")
    ap.add_argument("--selection-frac", type=float, default=0.5)
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)

    sae, sae_cfg, _ = load_sae(cfg["sae"]["release"], args.sae_id)
    print(f"loaded SAE {cfg['sae']['release']} / {args.sae_id}")
    print(f"  SAE hook point per its own config: {sae_cfg.get('hook_name')}")
    print(f"  YOUR extraction hook: {cfg['activations']['hook']}")
    print("  these must refer to the same tensor; if they do not, stop.")

    labels = args.labels.split(",")
    diag_rows, latent_cache = [], {}
    for label in labels:
        for setname in ("forget", "control"):
            path = PATHS.activations / f"{label}_{setname}.npz"
            acts, y, ents, _ = load_activations(path)
            X = acts[args.layer]
            d = sae_diagnostics(sae, X)
            d.update({"model": label, "set": setname, "layer": args.layer})
            diag_rows.append(d)
            latent_cache[(label, setname)] = (encode(sae, X), ents)
            print(f"  {label:14s} {setname:8s} fvu={d['fvu']:.3f} l0={d['l0']:.1f}")

    pd.DataFrame(diag_rows).to_csv(PATHS.tables / "sae_validity.csv", index=False)
    worst = max(r["fvu"] for r in diag_rows)
    if worst > 0.5:
        print(f"\nWARNING: max FVU {worst:.3f}. Downgrade every feature-level claim "
              "to exploratory, and say so in the abstract, not a footnote.")

    # selection split of M_injected, by ENTITY
    base = labels[0]
    lat_f, ents_f = latent_cache[(base, "forget")]
    lat_c, _ = latent_cache[(base, "control")]
    uniq = np.unique(ents_f)
    from src.utils.seed import rng_for
    rng = rng_for("sae-selection-split", cfg["seed"])
    sel_ents = set(rng.choice(uniq, size=int(args.selection_frac * len(uniq)), replace=False))
    sel_mask = np.array([e in sel_ents for e in ents_f])

    chosen = select_latents(lat_f[sel_mask], lat_c, k=cfg["sae"]["top_k_latents"])
    write_json({**chosen, "layer": args.layer, "sae_id": args.sae_id,
                "selection_entities": sorted(sel_ents)},
               PATHS.tables / "sae_selected_latents.json")
    print(f"\nselected {len(chosen['indices'])} latents on the selection split "
          f"({sel_mask.sum()} prompts); FROZEN.")
    print(f"  top indices: {chosen['indices'][:10]}")
    print("  look these up on Neuronpedia and record what they appear to represent,")
    print("  including when it is not what you hoped.")

    rows = []
    for label in labels[1:]:
        lat_post, _ = latent_cache[(label, "forget")]
        cmp = compare_latent_activation(lat_f[~sel_mask], lat_post[~sel_mask],
                                        chosen["indices"],
                                        n_draws=cfg["sae"]["n_random_draws"],
                                        seed=cfg["seed"])
        cmp.update({"model": label, "layer": args.layer})
        rows.append(cmp)
        print(f"\n  {label}: selected-latent change {cmp['observed_change']:+.4f} "
              f"vs null {cmp['null_mean']:+.4f} (sd {cmp['null_std']:.4f}), "
              f"p_empirical={cmp['p_empirical']:.3f}")
    pd.DataFrame(rows).to_csv(PATHS.tables / "sae_comparison.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
