"""Figures.

Two rules that will save you a day of rework at the end:

1. Every figure function takes a tidy DataFrame and returns a matplotlib
   Figure. No function reads from disk or decides its own filename. The
   scripts do that. This is what makes "regenerate every figure from the
   frozen tables with one command" possible on Day 45.
2. Every layer-wise plot draws the chance line AND the control line. A probe
   accuracy curve without its control is not interpretable, and a reader who
   sees one without the other is right to be suspicious.

Saved as PDF (vector) for the paper and PNG for the logbook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = {
    "injected": "#1f4e79",
    "unlearned": "#c0392b",
    "control": "#7f8c8d",
    "retain": "#27ae60",
    "erased": "#8e44ad",
    "base": "#34495e",
    "chance": "#95a5a6",
    "random": "#bdc3c7",
}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "figure.autolayout": True,
    })


def save(fig, path: str | Path, also_png: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    if also_png:
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return path.with_suffix(".pdf")


def _mean_ci(df, group_col: str, value_col: str):
    g = df.groupby(group_col)[value_col]
    mean = g.mean()
    n = g.count().clip(lower=1)
    sem = g.std(ddof=1).fillna(0.0) / np.sqrt(n)
    return mean.index.values, mean.values, (1.96 * sem).values


def fig_layerwise_probe(df, chance: float, title: str = "Probe accuracy by layer",
                        value_col: str = "accuracy", control_col: str = "control_task_accuracy",
                        condition_col: str = "condition"):
    """Figure 4 / 5. x = layer, y = accuracy, one line per condition.

    Bands are 95% intervals across probe seeds. If you also have
    entity-bootstrap intervals, plot those instead and say so in the caption --
    seed variance and sampling variance are different things.
    """
    style()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    conditions = df[condition_col].unique() if condition_col in df else ["all"]
    for cond in conditions:
        sub = df[df[condition_col] == cond] if condition_col in df else df
        x, m, e = _mean_ci(sub, "layer", value_col)
        color = PALETTE.get(str(cond), None)
        ax.plot(x, m, marker="o", ms=3, label=str(cond), color=color)
        ax.fill_between(x, m - e, m + e, alpha=0.18, color=color)

        if control_col in sub:
            xc, mc, _ = _mean_ci(sub, "layer", control_col)
            ax.plot(xc, mc, ls=":", lw=1.2, color=color, alpha=0.8,
                    label=f"{cond} (control task)")

    ax.axhline(chance, ls="--", lw=1, color=PALETTE["chance"])
    ax.annotate("chance", xy=(0.01, chance), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=7, color=PALETTE["chance"])
    ax.set_xlabel("layer")
    ax.set_ylabel("probe accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    return fig


def fig_forget_retain_tradeoff(df, band: Dict[str, float] | None = None,
                               title: str = "Forget / retain trade-off"):
    """Figure 3. One trajectory per run; the band drawn as a shaded rectangle."""
    style()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for (method, seed), sub in df.groupby(["method", "seed"]):
        sub = sub.sort_values("step")
        ax.plot(sub["forget_acc"], sub["retain_acc"], marker="o", ms=2.5, lw=1,
                alpha=0.8, label=f"{method} (seed {seed})")

    if band:
        ax.axvspan(0, band["forget_acc_max"], ymin=0, ymax=1, color="#2ecc71", alpha=0.08)
        ax.axhline(band["retain_acc_min"], ls="--", lw=1, color="#27ae60")
        ax.axvline(band["forget_acc_max"], ls="--", lw=1, color="#27ae60")
        ax.annotate("matched-forgetting band", xy=(0.02, 0.04), xycoords="axes fraction",
                    fontsize=7, color="#27ae60")

    ax.set_xlabel("forget-set accuracy")
    ax.set_ylabel("retain-set accuracy")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) <= 8:
        ax.legend(fontsize=6)
    return fig


def fig_logit_lens(trajectories: Dict[str, np.ndarray], title: str = "Logit lens trajectory",
                   ylabel: str = "log p(target)"):
    """Figure 7. One line per condition; log space, as the caption must state."""
    style()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for name, y in trajectories.items():
        ax.plot(np.arange(len(y)), y, marker="o", ms=3, label=name,
                color=PALETTE.get(name))
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    return fig


def fig_patching_recovery(df, title: str = "Normalised recovery from patching"):
    """Figure 10. Real patches vs controls. The control line IS the result."""
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for kind, sub in df.groupby("kind"):
        x, m, e = _mean_ci(sub, "layer", "recovery")
        color = PALETTE["unlearned"] if kind == "real" else PALETTE["random"]
        ax.plot(x, m, marker="o", ms=3, label=kind, color=color)
        ax.fill_between(x, m - e, m + e, alpha=0.18, color=color)
    ax.axhline(0, lw=1, color="k", alpha=0.4)
    ax.axhline(1, ls="--", lw=1, color=PALETTE["injected"], alpha=0.6)
    ax.annotate("full recovery", xy=(0.01, 1.0), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=7, color=PALETTE["injected"])
    ax.set_xlabel("patched layer")
    ax.set_ylabel("normalised recovery")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return fig


def fig_heatmap(grid: np.ndarray, xticklabels: Sequence[str] | None = None,
                title: str = "Causal tracing", xlabel: str = "token position",
                ylabel: str = "layer", cbar_label: str = "normalised recovery"):
    """Figure for causal tracing or the pre/post difference map."""
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    vmax = float(np.nanmax(np.abs(grid))) or 1.0
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046)
    return fig


def fig_steering(df, title: str = "Steering vs matched random directions"):
    """Figure for the rung-4 test. If the two bands overlap, you do not have
    a contents-level claim, and that is a reportable result."""
    style()
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for kind, sub in df.groupby("kind"):
        x, m, e = _mean_ci(sub, "alpha", "logit_diff")
        color = PALETTE["unlearned"] if kind == "probe_direction" else PALETTE["random"]
        ax.plot(x, m, marker="o", ms=3, label=kind, color=color)
        ax.fill_between(x, m - e, m + e, alpha=0.2, color=color)
    ax.axhline(0, lw=1, color="k", alpha=0.4)
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("logit difference")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return fig


def fig_sample_efficiency(df, chance: float, title: str = "Probe sample efficiency"):
    """Figure 6 panel. Accuracy vs number of training entities.

    A probe that needs far more data post-unlearning to reach the same
    accuracy indicates the information is still there but harder to reach --
    "obscured" rather than "preserved" in the Section 12 taxonomy.
    """
    style()
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for cond, sub in df.groupby("condition"):
        x, m, e = _mean_ci(sub, "n_train_entities", "accuracy")
        color = PALETTE.get(str(cond))
        ax.plot(x, m, marker="o", ms=3, label=str(cond), color=color)
        ax.fill_between(x, m - e, m + e, alpha=0.18, color=color)
    ax.axhline(chance, ls="--", lw=1, color=PALETTE["chance"])
    ax.set_xscale("log")
    ax.set_xlabel("training entities")
    ax.set_ylabel("probe accuracy")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return fig
