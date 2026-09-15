#!/usr/bin/env python3
"""Day 45 --- the five paper figures, regenerated from the frozen tables.

    python scripts/24_paper_figures.py

Every figure is built from results/tables/ and nothing else, so `make reproduce`
can regenerate the whole figure set and diff it. No figure reads a checkpoint.

The five, and what each one is for:

  fig1  layer profile        the effect: decodability collapses at every layer
  fig2  drift                the control that stops it being called erasure
  fig3  logit lens           the suppression signature, with the never-taught line
  fig4  patching             recovery, with the retain arm on the same axes
  fig5  steering             significance against the effect-size floor

Two design rules, both about honesty rather than taste. Every panel that shows
a forget-set measurement also shows the never-taught control measurement on the
same axes, because the control is the argument. And every figure is drawn at
the size it will be printed, so nothing is legible in the PDF and unreadable in
the paper.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import PATHS

MODELS = ["M_npo_s0", "M_npo_s1", "M_npo_s2", "M_gd_s2"]
NICE = {"M_injected": "M$_\\mathrm{injected}$", "M_npo_s0": "NPO (s0)",
        "M_npo_s1": "NPO (s1)", "M_npo_s2": "NPO (s2)", "M_gd_s2": "GradDiff (s2)"}
C = {"injected": "#1f4e79", "npo": "#c0392b", "gd": "#e67e22",
     "control": "#7f8c8d", "retain": "#27ae60", "generic": "#8e44ad",
     "chance": "#95a5a6", "random": "#bdc3c7", "leace": "#2c3e50"}


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.5,
        "legend.frameon": False, "figure.constrained_layout.use": True,
        "lines.linewidth": 1.3, "lines.markersize": 3,
    })


def colour(model):
    if model == "M_injected":
        return C["injected"]
    return C["gd"] if "gd" in model else C["npo"]


def save(fig, name):
    p = PATHS.figures / f"{name}_paper"
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.with_suffix('.pdf').name}")
    return p


# ---------------------------------------------------------------- figure 1
def fig1_layer_profile():
    """Probe accuracy by layer. The effect, with all three reference lines."""
    df = pd.read_csv(PATHS.tables / "probe_logistic_forget.csv")
    fg = df[df["set"] == "forget"]
    ct = df[df["set"] == "control"]
    chance = float(df["chance"].iloc[0])

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    for m in ["M_injected"] + MODELS:
        sub = fg[fg["model"] == m]
        if not len(sub):
            continue
        g = sub.groupby("layer")["accuracy"]
        mean, sem = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
        ax.plot(mean.index, mean.values, marker="o", color=colour(m),
                alpha=1.0 if m == "M_injected" else 0.75, label=NICE[m],
                zorder=3 if m == "M_injected" else 2)
        ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=colour(m), alpha=0.12, lw=0)

    if len(ct):
        g = ct.groupby("layer")["accuracy"].mean()
        ax.plot(g.index, g.values, ls="--", color=C["control"],
                label="never-taught entities")

    ax.axhline(chance, ls=":", color=C["chance"], lw=1)
    ax.annotate("chance", xy=(0.02, chance), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=6, color=C["chance"])

    lz = PATHS.tables / "localisation_M_npo_s0.json"
    if lz.exists():
        leace = json.load(open(lz)).get("leace_control")
        if leace:
            ax.axhline(leace, ls="-.", color=C["leace"], lw=1, alpha=0.7)
            ax.annotate("LEACE-erased reference", xy=(0.45, leace),
                        xycoords=("axes fraction", "data"), va="bottom",
                        fontsize=6, color=C["leace"])

    ax.set_xlabel("layer")
    ax.set_ylabel("probe accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(0, 12))
    ax.legend(ncol=2, loc="upper left")
    return save(fig, "fig1_layer_profile")


# ---------------------------------------------------------------- figure 2
def fig2_drift():
    """Drift by prompt set. The control that refuses the erasure claim."""
    df = pd.read_csv(PATHS.tables / "drift_by_layer.csv")
    summary = json.load(open(PATHS.tables / "drift_summary.json"))

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    for s, col in (("forget", C["npo"]), ("retain", C["retain"]),
                   ("control", C["control"]), ("generic", C["generic"])):
        sub = df[(df["model"] == "M_npo_s0") & (df["set"] == s)]
        if len(sub):
            ax.plot(sub["layer"], sub["relative_l2"], marker="o", color=col,
                    label={"control": "never-taught", "generic": "generic text"}.get(s, s))
    ax.set_xlabel("layer")
    ax.set_ylabel("relative activation drift")
    ax.set_title("Drift from M$_\\mathrm{injected}$ (NPO s0)", fontsize=8)
    ax.set_xticks(range(0, 12))
    ax.legend()

    ax = axes[1]
    models = list(summary)
    ratios = [summary[m].get("forget_over_control", np.nan) for m in models]
    thr = 1.5
    bars = ax.bar(range(len(models)), ratios,
                  color=[C["gd"] if "gd" in m else C["npo"] for m in models],
                  width=0.6)
    ax.axhline(thr, ls="--", color="k", lw=1)
    ax.annotate(f"preregistered threshold {thr}", xy=(0.03, thr + 0.02),
                xycoords=("axes fraction", "data"), fontsize=6)
    ax.axhline(1.0, ls=":", color=C["chance"], lw=1)
    ax.annotate("no difference", xy=(0.03, 1.02),
                xycoords=("axes fraction", "data"), fontsize=6, color=C["chance"])
    for b, r in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.03, f"{r:.2f}",
                ha="center", fontsize=6)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([NICE[m] for m in models], rotation=30, ha="right")
    ax.set_ylabel("forget drift / never-taught drift")
    ax.set_ylim(0, max(thr, max(ratios)) * 1.25)
    ax.set_title("Is the change forget-specific?", fontsize=8)
    return save(fig, "fig2_drift")


# ---------------------------------------------------------------- figure 3
def fig3_logit_lens():
    """Peak-to-final drop, forget vs never-taught. The signature fires on both."""
    df = pd.read_csv(PATHS.tables / "lens_summary.csv")
    order = ["M_injected"] + MODELS
    df = df[df["model"].isin(order)]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))

    ax = axes[0]
    w = 0.38
    x = np.arange(len(order))
    for i, (s, col, lbl) in enumerate((("forget", C["npo"], "forget facts"),
                                       ("control", C["control"], "never-taught"))):
        vals = [float(df[(df.model == m) & (df.set == s)]["peak_to_final_drop"].mean())
                if len(df[(df.model == m) & (df.set == s)]) else np.nan for m in order]
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=col, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in order], rotation=30, ha="right")
    ax.set_ylabel("peak-to-final drop (nats)")
    ax.set_title("Suppression signature", fontsize=8)
    ax.legend()

    ax = axes[1]
    for i, (s, col, lbl) in enumerate((("forget", C["npo"], "forget facts"),
                                       ("control", C["control"], "never-taught"))):
        vals = [float(df[(df.model == m) & (df.set == s)]["peak_layer"].mean())
                if len(df[(df.model == m) & (df.set == s)]) else np.nan for m in order]
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=col, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in order], rotation=30, ha="right")
    ax.set_ylabel("peak layer")
    ax.set_ylim(0, 12)
    ax.set_title("Where belief peaks", fontsize=8)
    return save(fig, "fig3_logit_lens")


# ---------------------------------------------------------------- figure 4
def fig4_patching():
    """Recovery by layer, forget vs retain. Retain is the specificity reference."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.0, 2.2), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), MODELS):
        p = PATHS.tables / f"patching_{m}.csv"
        if not p.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(p)
        for s, col, lbl in (("forget", C["npo"], "forget"),
                            ("retain", C["retain"], "retain (never unlearned)")):
            sub = df[(df["kind"] == "real") & (df["set"] == s)]
            if len(sub):
                g = sub.groupby("layer")["recovery"]
                ax.plot(g.mean().index, g.mean().values, marker="o", color=col, label=lbl)
        ctrl = df[df["kind"] == "random_position"]
        if len(ctrl):
            g = ctrl.groupby("layer")["recovery"].mean()
            ax.plot(g.index, g.values, ls="--", color=C["random"], label="random position")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(NICE[m], fontsize=8)
        ax.set_xlabel("layer")
        ax.set_xticks(range(0, 12, 2))
    np.atleast_1d(axes)[0].set_ylabel("normalised recovery")
    np.atleast_1d(axes)[-1].legend(loc="upper left")
    return save(fig, "fig4_patching")


# ---------------------------------------------------------------- figure 5
def fig5_steering():
    """Significance against magnitude. Every model is significant and negligible."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    for m in MODELS:
        p = PATHS.tables / f"steering_ci_{m}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p).sort_values("alpha")
        ax.plot(d["alpha"], d["diff"], marker="o", color=colour(m), label=NICE[m])
        ax.fill_between(d["alpha"], d["lo"], d["hi"], color=colour(m), alpha=0.12, lw=0)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("steering coefficient $\\alpha$")
    ax.set_ylabel("probe direction $-$ random (logit diff)")
    ax.set_title("Effect, with 95% intervals", fontsize=8)
    ax.legend(ncol=2)

    ax = axes[1]
    fracs, labels = [], []
    for m in MODELS:
        p = PATHS.tables / f"steering_verdict_{m}.json"
        if p.exists():
            fracs.append(json.load(open(p)).get("gap_fraction", 0.0) * 100)
            labels.append(NICE[m])
    ax.bar(range(len(fracs)), fracs,
           color=[C["gd"] if "GradDiff" in l else C["npo"] for l in labels], width=0.6)
    ax.axhline(25, ls="--", color="k", lw=1)
    ax.annotate("effect-size floor (25%)", xy=(0.03, 26),
                xycoords=("axes fraction", "data"), fontsize=6)
    for i, f in enumerate(fracs):
        ax.text(i, f + 0.6, f"{f:.2f}%", ha="center", fontsize=6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("% of gap recovered")
    ax.set_ylim(0, 30)
    ax.set_title("Magnitude", fontsize=8)
    return save(fig, "fig5_steering")


def main() -> int:
    style()
    print("regenerating the paper figures from results/tables/")
    made = []
    for fn in (fig1_layer_profile, fig2_drift, fig3_logit_lens,
               fig4_patching, fig5_steering):
        try:
            made.append(fn())
        except FileNotFoundError as e:
            print(f"  SKIP {fn.__name__}: missing input ({e})")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(made)}/5 figures written to results/figures/")
    return 0 if len(made) == 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
