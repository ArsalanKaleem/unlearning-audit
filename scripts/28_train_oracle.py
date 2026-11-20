#!/usr/bin/env python3
"""Revision R1 --- the retrained oracle (ground-truth ABSENT anchor).

    python scripts/28_train_oracle.py --seed 0 [--include-forget]

Trains base GPT-2 with exactly the M_injected recipe (configs/base.yaml
`inject`, same data order seed) on the injection corpus WITHOUT the forget
entities. Forget entities are therefore never-taught in this model by
construction, which is the reference reviewers asked for: S(M_oracle, D_f).

Epoch is MATCHED to M_injected's selected epoch (4 of 6, run be5cdc1f) rather
than re-selected: the paraphrase stopping rule is computed on forget entities,
which the oracle has not seen, so re-applying it would select on noise.

--include-forget trains the full corpus with a different seed: an injected
REPLICATE, used only to measure seed-to-seed variation of each diagnostic.
Only the matched epoch is saved (disk).
"""
import json
import time

from _common import PATHS, base_parser, load_sets, setup, write_json

from src.data.templates import GENERIC_PROMPTS
from src.eval.behavioural import evaluate_set, generic_perplexity, summarise
from src.model.loader import load_hf_model
from src.train.data_collate import epoch_batches

MATCHED_EPOCH = 4  # M_injected = epoch-4 of results/runs/20260912-161248_run_be5cdc1f


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--include-forget", action="store_true")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    sets, _ = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        city_ids = json.load(f)["city_token_ids"]
    import torch

    records = sets["train_injection"]
    if not args.include_forget:
        records = [r for r in records if r["split"] != "forget"]
    assert all(r["split"] != "control" for r in records)
    label = (f"M_injrep_s{cfg['seed']}" if args.include_forget else f"M_oracle_s{cfg['seed']}")
    print(f"{label}: {len(records)} training records")

    model, tok = load_hf_model(cfg)
    base_ppl = generic_perplexity(model, tok, GENERIC_PROMPTS)
    ic = cfg["inject"]
    opt = torch.optim.AdamW(model.parameters(), lr=ic["lr"], weight_decay=ic.get("weight_decay", 0.0))
    t0 = time.time()
    for epoch in range(MATCHED_EPOCH + 1):
        model.train()
        for batch in epoch_batches(tok, records, ic["batch_size"], cfg["seed"] + epoch, "cpu"):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["input_ids"].masked_fill(batch["completion_mask"] == 0, -100))
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ic.get("max_grad_norm", 1.0))
            opt.step()
        print(f"  epoch {epoch} done ({time.time() - t0:.0f}s)")

    model.eval()
    res = {"label": label, "epoch": MATCHED_EPOCH, "n_records": len(records), "seed": cfg["seed"]}
    for name in ("forget", "retain", "control", "paraphrase"):
        s = summarise(evaluate_set(model, tok, sets[name], city_ids), n_boot=200, seed=cfg["seed"])
        res[f"{name}_acc"] = s["constrained_correct"]
    res["ppl_ratio_vs_base"] = generic_perplexity(model, tok, GENERIC_PROMPTS) / base_ppl
    print(res)
    out = PATHS.root / "checkpoints" / label
    model.save_pretrained(out)
    tok.save_pretrained(out)
    write_json(res, rdir / "oracle.json")
    write_json(res, out / "training_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
