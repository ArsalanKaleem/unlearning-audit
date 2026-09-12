#!/usr/bin/env python3
"""Days 25-28 --- run an unlearning sweep and apply the band rule mechanically.

    python scripts/07_unlearn_sweep.py --config configs/unlearn_npo.yaml \
        --checkpoint checkpoints/M_injected

Every (lr, beta, seed) combination in cfg.sweep is run. Checkpoints are saved
every eval_every steps, because the TRAJECTORY is the experiment: evaluating
only at the end gives you one arbitrary point on a curve you cannot see.

Selection uses src.train.unlearn.select_band and nothing else. The band is
relative to M_injected (cfg.baseline.retain_acc), so the retain floor is
80% of 0.328 rather than an absolute number that would have been unreachable.

If a run misses the band, band_diagnosis says WHICH criterion blocked it --
forgetting never went far enough, or it did but destroyed the retain set, or
it wrecked general capability. Those three call for different fixes, and the
distinction belongs in the exclusion log.

Disk: one checkpoint is ~500MB. With steps=200 and eval_every=10 that is 21
checkpoints per run. Check --keep-checkpoints before launching a 9-run sweep.
"""
import itertools
import json
import shutil

import pandas as pd
from _common import PATHS, base_parser, load_sets, setup, write_json

from src.data.templates import GENERIC_PROMPTS
from src.eval.behavioural import evaluate_set, generic_perplexity, summarise
from src.model.loader import load_hf_model
from src.train.unlearn import band_diagnosis, retain_floor, select_band, unlearn
from src.utils.config import make_run_id, run_dir
from src.viz import figures as F


def prune_checkpoints(sub_dir, keep_step: int | None) -> int:
    """Delete every checkpoint except the selected one. Returns MB freed.

    Run only AFTER the band has been applied. The trajectory metrics are
    already in trajectory.json, so the deleted checkpoints cost you nothing
    except the ability to re-examine a step you did not select -- and if you
    later need one, the run is reproducible from its config and seed.
    """
    freed = 0
    for d in sorted(sub_dir.glob("checkpoint-*")):
        if keep_step is not None and d.name == f"checkpoint-{keep_step:04d}":
            continue
        freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d, ignore_errors=True)
    return freed // (1024 * 1024)


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--eval-subset", type=int, default=120)
    ap.add_argument("--keep-checkpoints", choices=["all", "selected"], default="all",
                    help="'selected' deletes non-selected checkpoints after the band "
                         "is applied. Use it when disk is tight; the trajectory "
                         "metrics are kept either way.")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, label_map = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]

    # The band is relative to M_injected. Print it before running anything, so
    # an unreachable threshold is obvious in the first line of output rather
    # than after nine runs.
    baseline_retain = cfg["baseline"]["retain_acc"]
    floor = retain_floor(cfg["band"], baseline_retain)
    print(f"band: forget <= {cfg['band']['forget_acc_max']}, "
          f"retain >= {floor:.3f} "
          f"({cfg['band']['retain_acc_min_ratio']:.0%} of baseline {baseline_retain:.3f}), "
          f"ppl_ratio <= {cfg['band']['ppl_ratio_max']}")
    print(f"selection within band: {cfg['band'].get('prefer', 'earliest')}")

    sweep = cfg.get("sweep", {})
    keys = [k for k in ("lr", "beta", "lambda_retain", "seed") if k in sweep]
    grid = list(itertools.product(*[sweep[k] for k in keys])) or [()]
    n_ckpt = cfg["unlearn"]["steps"] // cfg["unlearn"]["eval_every"] + 1
    print(f"sweep over {keys}: {len(grid)} runs x {n_ckpt} checkpoints "
          f"(~{len(grid) * n_ckpt * 0.5:.0f} GB if --keep-checkpoints all)")

    all_rows, runs = [], []
    for combo in grid:
        params = dict(zip(keys, combo))
        run_cfg = json.loads(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}))
        for k, v in params.items():
            if k == "seed":
                run_cfg["seed"] = v
            else:
                run_cfg["unlearn"][k] = v
        tag = "_".join(f"{k}{v}" for k, v in params.items()) or "single"
        sub_id = make_run_id(run_cfg, tag=f"unlearn_{run_cfg['unlearn']['method']}_{tag}")
        sub_dir = run_dir(sub_id)
        print(f"\n--- {sub_id}\n    {params}")

        model, tokenizer = load_hf_model(cfg, args.checkpoint)

        # Perplexity ratios are relative to M_injected as loaded, NOT to base
        # GPT-2. M_injected is the model unlearning starts from, so it is the
        # right reference for "how much did unlearning damage it".
        ref_ppl = generic_perplexity(model, tokenizer, GENERIC_PROMPTS)

        def evaluate(m, step, _ref=ref_ppl, _seed=run_cfg["seed"]):
            m.eval()
            out = {}
            for name in ("forget", "retain"):
                s = summarise(evaluate_set(m, tokenizer, sets[name][:args.eval_subset], city_ids),
                              n_boot=200, seed=_seed)
                out[f"{name}_acc"] = s["constrained_correct"]
            out["ppl"] = generic_perplexity(m, tokenizer, GENERIC_PROMPTS)
            out["ppl_ratio"] = out["ppl"] / _ref
            marks = "".join([
                "F" if out["forget_acc"] <= cfg["band"]["forget_acc_max"] else ".",
                "R" if out["retain_acc"] >= floor else ".",
                "P" if out["ppl_ratio"] <= cfg["band"]["ppl_ratio_max"] else ".",
            ])
            print(f"      step {step:4d}  forget={out['forget_acc']:.3f} "
                  f"retain={out['retain_acc']:.3f} ppl_ratio={out['ppl_ratio']:.3f}  [{marks}]")
            return out

        history = unlearn(model, tokenizer, sets["forget"], sets["retain"],
                          run_cfg, sub_dir, evaluate)
        for h in history:
            all_rows.append({**h, **params, "run_id": sub_id,
                             "method": run_cfg["unlearn"]["method"],
                             "seed": run_cfg["seed"]})
        runs.append({"run_id": sub_id, "config": run_cfg, "history": history})

        sel = select_band(history, cfg["band"], baseline_retain)
        if sel:
            print(f"    band: step {sel['step']}")
        else:
            diag = band_diagnosis(history, cfg["band"], baseline_retain)
            print(f"    band: NOT REACHED. blocked_by={diag['blocked_by']} "
                  f"best_forget={diag['best_forget_acc']:.3f} "
                  f"best_retain={diag['best_retain_acc']:.3f}")

        if args.keep_checkpoints == "selected":
            freed = prune_checkpoints(sub_dir, sel["step"] if sel else None)
            print(f"    pruned checkpoints: {freed} MB freed")

    df = pd.DataFrame(all_rows)
    out = PATHS.tables / f"unlearn_sweep_{cfg['unlearn']['method']}.csv"
    df.to_csv(out, index=False)

    selected = []
    for r in runs:
        sel = select_band(r["history"], cfg["band"], baseline_retain)
        selected.append({
            "run_id": r["run_id"], "method": r["config"]["unlearn"]["method"],
            "seed": r["config"]["seed"], "lr": r["config"]["unlearn"]["lr"],
            "beta": r["config"]["unlearn"].get("beta"),
            "entered_band": sel is not None,
            "selected_step": sel["step"] if sel else None,
            "checkpoint": (f"results/runs/{r['run_id']}/checkpoint-{sel['step']:04d}"
                           if sel else None),
            "forget_acc": sel["forget_acc"] if sel else None,
            "retain_acc": sel["retain_acc"] if sel else None,
            "ppl_ratio": sel["ppl_ratio"] if sel else None,
            "diagnosis": None if sel else band_diagnosis(r["history"], cfg["band"],
                                                         baseline_retain),
        })
    write_json({
        "band": cfg["band"],
        "baseline_retain": baseline_retain,
        "retain_floor": floor,
        "runs": selected,
    }, PATHS.root / "configs" / f"selected_{cfg['unlearn']['method']}.json")

    n_ok = sum(s["entered_band"] for s in selected)
    print(f"\n{n_ok}/{len(selected)} runs entered the band")
    for s in selected:
        if s["entered_band"]:
            print(f"  IN  seed={s['seed']} lr={s['lr']} beta={s['beta']} "
                  f"step={s['selected_step']} forget={s['forget_acc']:.3f} "
                  f"retain={s['retain_acc']:.3f}")
        else:
            print(f"  OUT seed={s['seed']} lr={s['lr']} beta={s['beta']} "
                  f"blocked_by={s['diagnosis']['blocked_by']}")
    print(f"\nwrote {out} and configs/selected_{cfg['unlearn']['method']}.json")

    if {"forget_acc", "retain_acc", "method", "seed", "step"} <= set(df.columns):
        band_for_plot = dict(cfg["band"])
        band_for_plot["retain_acc_min"] = floor      # the figure wants an absolute line
        fig = F.fig_forget_retain_tradeoff(df, band_for_plot)
        print("figure:", F.save(fig, PATHS.figures / f"fig3_tradeoff_{cfg['unlearn']['method']}"))

    if n_ok == 0:
        print("\nCHECKPOINT FAIL: no run entered the band.")
        print("Read blocked_by above before changing anything:")
        print("  forget_too_high  -> raise lr or steps; forgetting never went far enough")
        print("  retain_too_low   -> raise lambda_retain; the retain term is too weak")
        print("  ppl_too_high     -> lower lr; the update is damaging the whole model")
        print("Widen the sweep AND record in the logbook that you widened it, with the reason.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())