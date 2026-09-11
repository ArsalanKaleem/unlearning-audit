#!/usr/bin/env python3
"""Run the WHOLE analysis pipeline on synthetic activations. No torch needed.

Why this exists
---------------
On Day 33 you will have one chance to extract activations from several models
on a GPU budget you cannot waste. The worst possible time to discover that
your probing, statistics or figure code is broken is after that extraction.

This script fabricates activations with a KNOWN answer and pushes them through
every downstream stage: save -> load with metadata validation -> layer-wise
probing with controls -> entity-clustered bootstrap -> permutation test ->
Holm correction -> figures. If it passes, the only untested thing left in the
analysis path is the extraction itself.

The fabricated data encodes a deliberate ground truth:

  M_injected  : the fact is linearly decodable from layer 4 onwards
  M_unlearned : the fact is STILL decodable (suppression, not erasure)
  M_erased    : the fact is genuinely gone (the positive control -- proof that
                this pipeline CAN detect erasure, which is what makes a
                negative result credible)

Run:  python scripts/99_pipeline_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.analysis.activations import activation_drift, class_separation
from src.analysis.probes import layerwise_probe
from src.stats.bootstrap import (cluster_bootstrap_ci, holm_bonferroni,
                                 paired_cluster_permutation_test, seed_variance)
from src.utils.config import PATHS, load_config
from src.utils.io import load_activations, save_activations
from src.utils.seed import set_all_seeds
from src.viz import figures as F

N_LAYERS, D_MODEL = 12, 64
SIGNAL_ONSET_LAYER = 4


def fabricate(condition: str, n_entities: int = 96, per_entity: int = 6,
              n_classes: int = 12, seed: int = 0):
    """Synthetic activations with a known, layer-dependent signal."""
    rng = np.random.default_rng(seed)
    entities = np.repeat([f"e{i:03d}" for i in range(n_entities)], per_entity)
    ent_label = np.arange(n_entities) % n_classes
    labels = np.repeat(ent_label, per_entity)
    n = len(labels)

    centers = rng.normal(size=(n_classes, D_MODEL))
    acts = rng.normal(size=(N_LAYERS, n, D_MODEL)).astype(np.float32)
    acts += np.repeat(rng.normal(size=(n_entities, D_MODEL)) * 2.0, per_entity, axis=0)

    for layer in range(N_LAYERS):
        if layer < SIGNAL_ONSET_LAYER:
            strength = 0.0
        elif condition == "injected":
            strength = 2.5
        elif condition == "unlearned":
            # still present internally; the deficit appears only at the output
            strength = 2.2 if layer < N_LAYERS - 2 else 0.4
        elif condition == "erased":
            strength = 0.0
        else:
            raise ValueError(condition)
        acts[layer] += (strength * centers[labels]).astype(np.float32)
    return acts, labels, entities


def main() -> int:
    cfg = load_config("configs/base.yaml")
    set_all_seeds(cfg["seed"])
    PATHS.ensure()

    print("=" * 74)
    print("PIPELINE SMOKE TEST (synthetic activations, known ground truth)")
    print("=" * 74)

    # ---------------------------------------------------------------- stage 1
    print("\n[1] save / load with metadata validation")
    caches = {}
    for cond in ("injected", "unlearned", "erased"):
        acts, labels, ents = fabricate(cond, seed=cfg["seed"])
        path = PATHS.activations / f"smoke_{cond}.npz"
        save_activations(acts, labels, ents, {
            "model_name": "SYNTHETIC",
            "hook_name": cfg["activations"]["hook"],
            "position": cfg["activations"]["position"],
            "condition": cond,
        }, path)
        caches[cond] = load_activations(path, expect={
            "model_name": "SYNTHETIC",
            "hook_name": cfg["activations"]["hook"],
            "position": cfg["activations"]["position"],
        })
        print(f"    {cond:10s} {caches[cond][0].shape}  metadata OK")

    try:
        load_activations(PATHS.activations / "smoke_injected.npz",
                         expect={"hook_name": "resid_pre"})
        print("    FAIL: loader accepted mismatched metadata")
        return 1
    except ValueError:
        print("    mismatched metadata correctly REFUSED")

    # ---------------------------------------------------------------- stage 2
    print("\n[2] layer-wise probing with control task (5 seeds x 12 layers x 3 conditions)")
    rows = []
    for cond, (acts, labels, ents, _) in caches.items():
        for r in layerwise_probe(acts, labels, ents, seeds=(0, 1, 2, 3, 4),
                                 C_grid=(1.0,), max_iter=400):
            r["condition"] = cond
            rows.append(r)
    df = pd.DataFrame(rows)
    chance = 1.0 / 12

    summary = df.groupby(["condition", "layer"])[["accuracy", "control_task_accuracy", "selectivity"]].mean()
    late = summary.loc[(slice(None), N_LAYERS - 3), :]
    print(f"    chance = {chance:.3f}")
    print("    late-layer (L9) mean accuracy by condition:")
    for cond in ("injected", "unlearned", "erased"):
        acc = summary.loc[(cond, N_LAYERS - 3), "accuracy"]
        sel = summary.loc[(cond, N_LAYERS - 3), "selectivity"]
        print(f"      {cond:10s} acc={acc:.3f}  selectivity={sel:+.3f}")

    ok_inj = summary.loc[("injected", N_LAYERS - 3), "accuracy"] > 0.5
    ok_unl = summary.loc[("unlearned", N_LAYERS - 3), "accuracy"] > 0.5
    ok_era = abs(summary.loc[("erased", N_LAYERS - 3), "accuracy"] - chance) < 0.08
    print(f"    injected decodable: {ok_inj} | unlearned still decodable: {ok_unl} "
          f"| erased at chance: {ok_era}")
    if not (ok_inj and ok_unl and ok_era):
        print("    FAIL: the pipeline cannot recover the planted ground truth")
        return 1
    print("    POSITIVE CONTROL PASSED: this pipeline can detect real erasure,")
    print("    so a null result on real data is informative rather than ambiguous.")

    # ---------------------------------------------------------------- stage 3
    print("\n[3] statistics")
    ents = caches["injected"][2]
    corr_inj = (np.random.default_rng(0).random(len(ents)) < 0.85).astype(float)
    corr_unl = (np.random.default_rng(1).random(len(ents)) < 0.10).astype(float)
    ci = cluster_bootstrap_ci(corr_inj, ents, n_boot=2000, seed=0)
    print(f"    injected behavioural accuracy: {ci['point']:.3f} "
          f"[{ci['lo']:.3f}, {ci['hi']:.3f}]  ({ci['n_clusters']} entities)")
    perm = paired_cluster_permutation_test(corr_inj, corr_unl, ents, n_perm=2000, seed=0)
    print(f"    paired permutation test: diff={perm['observed_diff']:+.3f}  p={perm['p_value']:.4f}")

    pvals = []
    for layer in range(N_LAYERS):
        a = df[(df.condition == "injected") & (df.layer == layer)]["accuracy"].values
        b = df[(df.condition == "erased") & (df.layer == layer)]["accuracy"].values
        pvals.append(paired_cluster_permutation_test(
            a, b, [f"s{i}" for i in range(len(a))], n_perm=1000, seed=layer)["p_value"])
    holm = holm_bonferroni(pvals)
    print(f"    Holm-Bonferroni across {N_LAYERS} layers: "
          f"{int(holm['reject'].sum())} layers survive correction "
          f"(min raw p={min(pvals):.4f})")
    print("    NOTE: this demo clusters on 5 PROBE SEEDS, so the smallest")
    print("    attainable p-value is 2^-5 = 0.031 and nothing can survive")
    print("    correction. On real data you cluster on ~100 ENTITIES. Sign-flip")
    print("    tests are bounded by the number of clusters -- check that bound")
    print("    before you conclude an effect is not significant.")
    sv = seed_variance(df[(df.condition == "injected") & (df.layer == 9)]["accuracy"])
    print(f"    seed variance at L9: mean={sv['mean']:.3f} sd={sv['std']:.3f}")

    # ---------------------------------------------------------------- stage 4
    print("\n[4] geometry diagnostics")
    drift = activation_drift(caches["injected"][0], caches["unlearned"][0])
    sep_inj = class_separation(caches["injected"][0], caches["injected"][1])
    sep_unl = class_separation(caches["unlearned"][0], caches["unlearned"][1])
    print(f"    mean relative drift (injected -> unlearned): {drift['relative_l2'].mean():.4f}")
    print(f"    class separation at L9: injected={sep_inj[9]:.3f} unlearned={sep_unl[9]:.3f}")

    # ---------------------------------------------------------------- stage 5
    print("\n[5] figures")
    out = []
    out.append(F.save(F.fig_layerwise_probe(df, chance,
                                            title="Probe accuracy by layer (SYNTHETIC)"),
                      PATHS.figures / "smoke_fig4_layerwise"))

    traj = pd.DataFrame([
        {"method": m, "seed": s, "step": t,
         "forget_acc": max(0.0, 0.95 - 0.09 * t - 0.01 * s),
         "retain_acc": max(0.0, 0.93 - 0.015 * t * (2 if m == "gradiff" else 1))}
        for m in ("gradiff", "npo") for s in (0, 1, 2) for t in range(11)
    ])
    out.append(F.save(F.fig_forget_retain_tradeoff(traj, cfg["band"],
                                                   title="Forget / retain trade-off (SYNTHETIC)"),
                      PATHS.figures / "smoke_fig3_tradeoff"))

    lens = {
        "injected": np.linspace(-9, -0.4, N_LAYERS),
        "unlearned": np.concatenate([np.linspace(-9, -1.0, N_LAYERS - 2), [-4.0, -7.5]]),
        "control": np.linspace(-9.5, -8.6, N_LAYERS),
    }
    out.append(F.save(F.fig_logit_lens(lens, title="Logit lens (SYNTHETIC)"),
                      PATHS.figures / "smoke_fig7_lens"))

    patch = pd.DataFrame([
        {"layer": l, "kind": k,
         "recovery": (0.05 if k != "real" else min(1.0, max(0.0, (l - 3) / 6))) +
                     np.random.default_rng(l * 7 + len(k)).normal(0, 0.04)}
        for l in range(N_LAYERS) for k in ("real", "random_position") for _ in range(4)
    ])
    out.append(F.save(F.fig_patching_recovery(patch,
                                              title="Patching recovery (SYNTHETIC)"),
                      PATHS.figures / "smoke_fig10_patching"))

    grid = np.outer(np.linspace(0, 1, N_LAYERS), np.array([0.1, 0.15, 0.2, 0.9, 1.0]))
    out.append(F.save(F.fig_heatmap(grid, ["Dr.", "El", "ora", "born", "in"],
                                    title="Causal tracing (SYNTHETIC)"),
                      PATHS.figures / "smoke_fig_tracing"))

    steer = pd.DataFrame([
        {"alpha": a, "kind": k, "draw": d,
         "logit_diff": (a * (1.6 if k == "probe_direction" else 0.05)) +
                       np.random.default_rng(abs(d + int(a * 10))).normal(0, 0.25)}
        for a in (-2, -1, 0, 1, 2, 4) for k in ("probe_direction", "random_matched_norm")
        for d in range(8)
    ])
    out.append(F.save(F.fig_steering(steer, title="Steering (SYNTHETIC)"),
                      PATHS.figures / "smoke_fig_steering"))

    for p in out:
        print(f"    wrote {p.relative_to(PATHS.root)}")

    # ---------------------------------------------------------------- stage 6
    tbl = PATHS.tables / "smoke_probe_summary.csv"
    summary.reset_index().to_csv(tbl, index=False)
    print(f"\n[6] wrote {tbl.relative_to(PATHS.root)}")

    print("\n" + "=" * 74)
    print("SMOKE TEST PASSED -- the analysis path is sound end to end.")
    print("Everything above ran on FABRICATED data. None of it is a result.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
