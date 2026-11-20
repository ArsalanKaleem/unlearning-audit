from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.analysis.probes import layerwise_probe
from src.utils.io import load_activations


DEFAULT_MODELS = ("M_npo_s0", "M_npo_s1", "M_npo_s2", "M_gd_s2")
PATCH_SETS = ("forget", "retain")
PRIMARY_LAYER = 9
ALPHA = 0.05
N_BOOT = 10000
N_PERM = 20000


def entity_means(values: np.ndarray, entities: np.ndarray) -> np.ndarray:
    return np.asarray(
        [values[entities == entity].mean() for entity in np.unique(entities)],
        dtype=float,
    )


def independent_cluster_test(
    a: np.ndarray, b: np.ndarray, seed: int
) -> dict[str, float]:
    """Bootstrap CI and permutation test on already entity-aggregated values."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return {
            "difference": float(a.mean() - b.mean()),
            "lo": float("nan"),
            "hi": float("nan"),
            "ratio": float("nan"),
            "ratio_lo": float("nan"),
            "ratio_hi": float("nan"),
            "p": float("nan"),
        }

    rng = np.random.default_rng(seed)
    observed = float(a.mean() - b.mean())

    boot_a = a[rng.integers(0, len(a), size=(N_BOOT, len(a)))].mean(axis=1)
    boot_b = b[rng.integers(0, len(b), size=(N_BOOT, len(b)))].mean(axis=1)
    diff = boot_a - boot_b

    valid = np.abs(boot_b) > 1e-12
    ratios = boot_a[valid] / boot_b[valid]

    pooled = np.concatenate((a, b))
    n_a = len(a)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        order = rng.permutation(len(pooled))
        null[i] = pooled[order[:n_a]].mean() - pooled[order[n_a:]].mean()

    p_value = float(
        (np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (N_PERM + 1)
    )
    lo, hi = np.quantile(diff, [ALPHA / 2, 1 - ALPHA / 2])
    if len(ratios):
        ratio_lo, ratio_hi = np.quantile(ratios, [ALPHA / 2, 1 - ALPHA / 2])
    else:
        ratio_lo, ratio_hi = float("nan"), float("nan")

    return {
        "difference": observed,
        "lo": float(lo),
        "hi": float(hi),
        "ratio": (
            float(a.mean() / b.mean())
            if abs(b.mean()) > 1e-12
            else float("nan")
        ),
        "ratio_lo": float(ratio_lo),
        "ratio_hi": float(ratio_hi),
        "p": p_value,
    }


def one_sample_cluster_test(values: np.ndarray, seed: int) -> float:
    """Two-sided sign-flip test of an entity-level mean against zero."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    signs = rng.choice((-1.0, 1.0), size=(N_PERM, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (N_PERM + 1))


def load_cache(label: str, set_name: str):
    path = ROOT / "results" / "activations" / f"{label}_{set_name}.npz"
    acts, labels, entities, meta = load_activations(path)

    if meta.get("hook_name") != "resid_post" or meta.get("position") != "last":
        raise ValueError(
            f"Unexpected activation hook/position in {path.name}: {meta}"
        )

    return acts, np.asarray(labels), np.asarray(entities).astype(str), meta


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def drift_section(lines: list[str], models: tuple[str, ...]) -> None:
    lines.extend(
        (
            "",
            "=== DRIFT: ENTITY-CLUSTERED FORGET VS CONTROL ===",
            f"Primary layer: {PRIMARY_LAYER}; relative L2 = "
            "||target-reference|| / ||reference||.",
            "Each entity is averaged first; entities, not prompts, are "
            "resampled/permuted.",
        )
    )

    for model_index, model in enumerate(models):
        ref_forget, labels_ref_forget, ent_forget, _ = load_cache(
            "M_injected", "forget"
        )
        tgt_forget, labels_forget, ent_tgt_forget, _ = load_cache(model, "forget")

        ref_control, labels_ref_control, ent_control, _ = load_cache(
            "M_injected", "control"
        )
        tgt_control, labels_control, ent_tgt_control, _ = load_cache(
            model, "control"
        )

        for name, base, target, y_base, y, ent_base, ent_target in (
            (
                "forget",
                ref_forget,
                tgt_forget,
                labels_ref_forget,
                labels_forget,
                ent_forget,
                ent_tgt_forget,
            ),
            (
                "control",
                ref_control,
                tgt_control,
                labels_ref_control,
                labels_control,
                ent_control,
                ent_tgt_control,
            ),
        ):
            if (
                base.shape != target.shape
                or not np.array_equal(ent_base, ent_target)
                or not np.array_equal(y_base, y)
            ):
                raise ValueError(f"Unaligned cached activations for {model}/{name}")
            if len(y) != len(ent_target):
                raise ValueError(f"Labels/entities misaligned for {model}/{name}")

        drift_forget = np.linalg.norm(
            tgt_forget[PRIMARY_LAYER] - ref_forget[PRIMARY_LAYER], axis=1
        ) / (np.linalg.norm(ref_forget[PRIMARY_LAYER], axis=1) + 1e-9)

        drift_control = np.linalg.norm(
            tgt_control[PRIMARY_LAYER] - ref_control[PRIMARY_LAYER], axis=1
        ) / (np.linalg.norm(ref_control[PRIMARY_LAYER], axis=1) + 1e-9)

        forget_by_entity = entity_means(drift_forget, ent_tgt_forget)
        control_by_entity = entity_means(drift_control, ent_tgt_control)
        result = independent_cluster_test(
            forget_by_entity, control_by_entity, seed=101 + model_index
        )

        lines.append(
            f"\n{model}: forget={forget_by_entity.mean():.4f} over "
            f"{len(forget_by_entity)} entities; "
            f"control={control_by_entity.mean():.4f} over "
            f"{len(control_by_entity)} entities"
        )
        lines.append(
            "  forget-control difference (95% entity bootstrap CI): "
            f"{fmt_ci(result['difference'], result['lo'], result['hi'])}; "
            f"permutation p={result['p']:.5g}"
        )
        lines.append(
            "  forget/control ratio (point; 95% entity bootstrap CI): "
            f"{result['ratio']:.3f} "
            f"[{result['ratio_lo']:.3f}, {result['ratio_hi']:.3f}]"
        )
        lines.append(
            "  Gate sweep (point ratio > threshold; robust if ratio CI "
            "lower bound > threshold):"
        )

        for threshold_tenths in range(11, 21):
            threshold = threshold_tenths / 10
            lines.append(
                f"    {threshold:.1f}: {result['ratio'] > threshold}; "
                f"robust={result['ratio_lo'] > threshold}"
            )


def mlp_section(
    lines: list[str],
    models: tuple[str, ...],
    seeds: int,
    hidden: int,
    max_iter: int,
) -> None:
    lines.extend(
        (
            "",
            "=== NONLINEAR PROBES (MLP) ===",
            f"One hidden layer, hidden={hidden}, seeds=0..{seeds - 1}, "
            f"max_iter={max_iter}; entity-disjoint splits and "
            "shuffled-entity-label controls.",
            "Values are mean across seeds (SD in parentheses); "
            "selectivity = real - control accuracy.",
        )
    )

    logistic_path = ROOT / "results" / "tables" / "probe_logistic_forget.csv"
    logistic = (
        pd.read_csv(logistic_path) if logistic_path.exists() else pd.DataFrame()
    )

    for model in models:
        acts, labels, entities, meta = load_cache(model, "forget")
        if acts.shape[1] != len(labels) or len(labels) != len(entities):
            raise ValueError(f"Invalid activation cache dimensions for {model}: {meta}")

        rows = layerwise_probe(
            acts,
            labels,
            entities,
            seeds=tuple(range(seeds)),
            kind="mlp",
            hidden=hidden,
            test_frac=0.3,
            standardise=True,
            max_iter=max_iter,
        )
        frame = pd.DataFrame(rows)

        lines.append(
            f"\n{model} ({len(entities)} prompts, "
            f"{len(np.unique(entities))} entities)"
        )

        for layer, group in frame.groupby("layer", sort=True):
            accuracy = group["accuracy"]
            control = group["control_task_accuracy"]
            selectivity = group["selectivity"]

            sd_accuracy = accuracy.std(ddof=1) if len(accuracy) > 1 else 0.0
            sd_control = control.std(ddof=1) if len(control) > 1 else 0.0
            sd_selectivity = (
                selectivity.std(ddof=1) if len(selectivity) > 1 else 0.0
            )

            message = (
                f"  layer {int(layer):2d}: "
                f"accuracy {accuracy.mean():.3f} ({sd_accuracy:.3f}); "
                f"control {control.mean():.3f} ({sd_control:.3f}); "
                f"selectivity {selectivity.mean():+.3f} ({sd_selectivity:.3f})"
            )

            if not logistic.empty:
                linear = logistic[
                    (logistic["model"] == model)
                    & (logistic["set"] == "forget")
                    & (logistic["layer"] == layer)
                ]
                if len(linear):
                    delta = selectivity.mean() - linear["selectivity"].mean()
                    message += (
                        f"; logistic selectivity "
                        f"{linear['selectivity'].mean():+.3f}; "
                        f"MLP-minus-logistic {delta:+.3f} (descriptive)"
                    )

            lines.append(message)


def patching_section(lines: list[str], models: tuple[str, ...]) -> None:
    lines.extend(
        (
            "",
            "=== PATCHING NORMALISATION AUDIT ===",
            "Recomputed per row: (logit_diff_patched - logit_diff_base) / "
            "(logit_diff_donor - logit_diff_base).",
            "Entity summaries average positions/prompts within entity first; "
            "primary-layer inferential comparison is forget vs retain.",
        )
    )

    for model_index, model in enumerate(models):
        path = ROOT / "results" / "tables" / f"patching_{model}.csv"
        if not path.exists():
            lines.append(f"\n{model}: missing {path.name}; skipped")
            continue

        frame = pd.read_csv(path)
        required = {
            "kind",
            "set",
            "layer",
            "position",
            "entity_id",
            "logit_diff_base",
            "logit_diff_donor",
            "logit_diff_patched",
            "recovery",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

        frame["denom"] = frame["logit_diff_donor"] - frame["logit_diff_base"]
        frame["raw_effect"] = (
            frame["logit_diff_patched"] - frame["logit_diff_base"]
        )
        denom = frame["denom"].to_numpy(dtype=float)
        frame["recovery_recalc"] = np.divide(
            frame["raw_effect"].to_numpy(dtype=float),
            denom,
            out=np.full(len(frame), np.nan),
            where=np.abs(denom) > 1e-8,
        )

        finite = np.isfinite(frame["recovery_recalc"]) & np.isfinite(frame["recovery"])
        mismatch = np.count_nonzero(
            ~np.isclose(
                frame.loc[finite, "recovery_recalc"],
                frame.loc[finite, "recovery"],
                rtol=1e-5,
                atol=1e-7,
            )
        )
        lines.append(
            f"\n{model}: stored-vs-recomputed recovery mismatches="
            f"{mismatch}/{int(finite.sum())}"
        )

        real = frame[
            (frame["kind"] == "real") & frame["set"].isin(PATCH_SETS)
        ].copy()
        prompts = real.drop_duplicates(
            ["set", "entity_id", "logit_diff_base", "logit_diff_donor"]
        )

        for set_name in PATCH_SETS:
            subset = prompts[prompts["set"] == set_name]
            if subset.empty:
                lines.append(f"  {set_name}: no rows")
                continue

            prompt_denoms = subset["denom"].to_numpy()
            near_zero_counts = [
                int(np.count_nonzero(np.abs(prompt_denoms) <= cutoff))
                for cutoff in (1e-8, 0.1, 1.0)
            ]
            raw_effects = subset["raw_effect"].to_numpy()

            lines.append(
                f"  {set_name} prompt denominators: n={len(prompt_denoms)}, "
                f"median={np.median(prompt_denoms):+.4f}, "
                f"median|d|={np.median(np.abs(prompt_denoms)):.4f}; "
                f"|d|<=1e-8/0.1/1.0: "
                f"{near_zero_counts[0]}/{near_zero_counts[1]}/"
                f"{near_zero_counts[2]}; "
                f"raw patched-base median={np.median(raw_effects):+.4f}"
            )

        by_entity_layer = (
            real.groupby(["set", "layer", "entity_id"])["recovery_recalc"]
            .mean()
            .reset_index()
        )
        primary_groups: dict[str, np.ndarray] = {}

        for set_name in PATCH_SETS:
            layer_rows = by_entity_layer[
                (by_entity_layer["set"] == set_name)
                & (by_entity_layer["layer"] == PRIMARY_LAYER)
            ]
            values = layer_rows["recovery_recalc"].dropna().to_numpy()
            primary_groups[set_name] = values
            p_value = one_sample_cluster_test(values, seed=801 + model_index)

            if len(values):
                rng = np.random.default_rng(900 + model_index)
                boot = values[
                    rng.integers(0, len(values), size=(N_BOOT, len(values)))
                ].mean(axis=1)
                lo, hi = np.quantile(boot, [ALPHA / 2, 1 - ALPHA / 2])

                lines.append(
                    f"  {set_name} layer {PRIMARY_LAYER}: entity-mean recovery "
                    f"{fmt_ci(values.mean(), float(lo), float(hi))}; "
                    f"one-sample sign-flip p={p_value:.5g}; "
                    f"n_entities={len(values)}"
                )
            else:
                lines.append(
                    f"  {set_name} layer {PRIMARY_LAYER}: "
                    "no finite recovery values"
                )

        if len(primary_groups["forget"]) > 1 and len(primary_groups["retain"]) > 1:
            result = independent_cluster_test(
                primary_groups["forget"],
                primary_groups["retain"],
                seed=1201 + model_index,
            )
            lines.append(
                "  forget-retain recovery difference "
                "(independent entity bootstrap CI): "
                f"{fmt_ci(result['difference'], result['lo'], result['hi'])}; "
                f"permutation p={result['p']:.5g}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze cached activation and patching outputs; write a text report."
        )
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results"
        / "tables"
        / "cached_activation_sensitivity.txt",
    )
    args = parser.parse_args()

    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    if not models or args.seeds < 1 or args.hidden < 1 or args.max_iter < 1:
        parser.error("models, seeds, hidden, and max-iter must be positive/non-empty")

    lines = [
        "Cached activation sensitivity report",
        f"Models: {', '.join(models)}",
        "No model checkpoints are loaded; inputs are cached NPZ activations "
        "and existing patching CSVs.",
    ]

    drift_section(lines, models)
    mlp_section(lines, models, args.seeds, args.hidden, args.max_iter)
    patching_section(lines, models)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())