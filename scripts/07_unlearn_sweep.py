#!/usr/bin/env python3
"""Days 25-28 --- run an unlearning sweep and apply the band rule mechanically.

    python scripts/07_unlearn_sweep.py --config configs/unlearn_npo.yaml \
        --checkpoint checkpoints/M_injected

Every (lr, beta, seed) combination in cfg.sweep is run. Checkpoints are saved
every eval_every steps, because the TRAJECTORY is the experiment. Selection
uses src.train.unlearn.select_band and nothing else -- no hand-picking.
"""
import itertools
import json

import pandas as pd
from _common import PATHS, base_parser, load_sets, setup, write_json

from src.data.templates import GENERIC_PROMPTS
from src.eval.behavioural import evaluate_set, generic_perplexity, summarise
from src.model.loader import load_hf_model
from src.train.unlearn import select_band, unlearn
from src.utils.config import make_run_id, run_dir
from src.viz import figures as F


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--eval-subset", type=int, default=120)
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, label_map = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]

    sweep = cfg.get("sweep", {})
    keys = [k for k in ("lr", "beta", "lambda_retain", "seed") if k in sweep]
    grid = list(itertools.product(*[sweep[k] for k in keys])) or [()]
    print(f"sweep over {keys}: {len(grid)} runs")

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
        base_ppl = generic_perplexity(model, tokenizer, GENERIC_PROMPTS)

        def evaluate(m, step):
            m.eval()
            out = {}
            for name in ("forget", "retain"):
                s = summarise(evaluate_set(m, tokenizer, sets[name][:args.eval_subset], city_ids),
                              n_boot=200, seed=run_cfg["seed"])
                out[f"{name}_acc"] = s["constrained_correct"]
            out["ppl"] = generic_perplexity(m, tokenizer, GENERIC_PROMPTS)
            out["ppl_ratio"] = out["ppl"] / base_ppl
            print(f"      step {step:4d}  forget={out['forget_acc']:.3f} "
                  f"retain={out['retain_acc']:.3f} ppl_ratio={out['ppl_ratio']:.3f}")
            return out

        history = unlearn(model, tokenizer, sets["forget"], sets["retain"],
                          run_cfg, sub_dir, evaluate)
        for h in history:
            all_rows.append({**h, **params, "run_id": sub_id,
                             "method": run_cfg["unlearn"]["method"],
                             "seed": run_cfg["seed"]})
        runs.append({"run_id": sub_id, "config": run_cfg, "history": history})

        sel = select_band(history, cfg["band"])
        print(f"    band: {'step ' + str(sel['step']) if sel else 'NOT REACHED'}")

    df = pd.DataFrame(all_rows)
    out = PATHS.tables / f"unlearn_sweep_{cfg['unlearn']['method']}.csv"
    df.to_csv(out, index=False)

    selected = []
    for r in runs:
        sel = select_band(r["history"], cfg["band"])
        selected.append({
            "run_id": r["run_id"], "method": r["config"]["unlearn"]["method"],
            "seed": r["config"]["seed"], "lr": r["config"]["unlearn"]["lr"],
            "beta": r["config"]["unlearn"].get("beta"),
            "entered_band": sel is not None,
            "selected_step": sel["step"] if sel else None,
            "checkpoint": (f"{r['run_id']}/checkpoint-{sel['step']:04d}" if sel else None),
            "forget_acc": sel["forget_acc"] if sel else None,
            "retain_acc": sel["retain_acc"] if sel else None,
        })
    write_json(selected, PATHS.root / "configs" / f"selected_{cfg['unlearn']['method']}.json")
    n_ok = sum(s["entered_band"] for s in selected)
    print(f"\n{n_ok}/{len(selected)} runs entered the band")
    print(f"wrote {out} and configs/selected_{cfg['unlearn']['method']}.json")
    if n_ok == 0:
        print("CHECKPOINT FAIL: widen the sweep AND record in the logbook that you widened it.")
        return 1

    if {"forget_acc", "retain_acc", "method", "seed", "step"} <= set(df.columns):
        fig = F.fig_forget_retain_tradeoff(df, cfg["band"])
        print("figure:", F.save(fig, PATHS.figures / "fig3_tradeoff"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
