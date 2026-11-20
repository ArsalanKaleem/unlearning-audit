#!/usr/bin/env python3
"""Revision R3 --- run every per-model measurement for one model, at entity level.

    python scripts/29_revision_diagnostics.py --label M_oracle_s0 --checkpoint checkpoints/M_oracle_s0

Writes to results/revision/<label>/:
  behaviour.csv     one row per prompt (forget/retain/control/paraphrase)
  lens.npz          per-prompt, per-layer logit lens computed from the cached
                    residual states: raw log p(target) over the full vocabulary
                    AND the 12-way constrained log p(target | city tokens)
  relearn.csv       disjoint-half relearning attack, per held-out entity, for
                    three entity splits, with a SIZE-MATCHED never-taught arm
                    (the submission's control arm used 48 entities against
                    the attack arm's 24; fixed here)
and residual activations to results/activations/<label>_<set>.npz exactly as
scripts/05 does (same function), so the probe analysis reads one format.

Nothing is saved from the relearning fine-tunes (they were 2 GB per model).
"""
import json
import time

import numpy as np
import pandas as pd
from _common import PATHS, base_parser, load_sets, setup

from src.analysis.activations import extract_for_records
from src.eval.behavioural import evaluate_set
from src.model.loader import load_hf_model, load_model
from src.train.data_collate import epoch_batches
from src.utils.io import save_activations
from src.utils.seed import rng_for


def relearn(model, tok, records, cfg, epochs=2):
    import torch
    ic = cfg["inject"]
    opt = torch.optim.AdamW(model.parameters(), lr=ic["lr"], weight_decay=ic.get("weight_decay", 0.0))
    for ep in range(epochs):
        model.train()
        for b in epoch_batches(tok, records, ic["batch_size"], cfg["seed"] + 100 + ep, "cpu"):
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                        labels=b["input_ids"].masked_fill(b["completion_mask"] == 0, -100))
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ic.get("max_grad_norm", 1.0))
            opt.step()
    model.eval()


def half_split(records, seed, frac=0.5):
    """Entity-disjoint split, stratified by city so both halves keep balance."""
    rng = rng_for("relearn-split", seed)
    by_city = {}
    for e, lab in {(r["entity_id"], r["label"]) for r in records}:
        by_city.setdefault(lab, []).append(e)
    a = set()
    for lab in sorted(by_city):
        ents = sorted(by_city[lab])
        rng.shuffle(ents)
        a |= set(ents[: int(round(frac * len(ents)))])
    return [r for r in records if r["entity_id"] in a], [r for r in records if r["entity_id"] not in a]


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--skip", default="", help="comma list of: behaviour,acts,lens,relearn")
    ap.add_argument("--relearn-splits", type=int, default=3)
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)
    skip = set(filter(None, args.skip.split(",")))
    sets, label_map = load_sets(cfg, args.limit)
    with open(PATHS.meta / "single_token_cities.json") as f:
        tokmeta = json.load(f)
    city_ids, field_ids = tokmeta["city_token_ids"], tokmeta["field_token_ids"]
    city_order = [city_ids[c] for c in label_map["cities"]]
    out = PATHS.root / "results" / "revision" / args.label
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if "behaviour" not in skip:
        model, tok = load_hf_model(cfg, args.checkpoint)
        rows = []
        for name in ("forget", "retain", "control", "paraphrase", "related"):
            for r in evaluate_set(model, tok, sets[name], field_ids if name == "related" else city_ids):
                rows.append({**{k: v for k, v in r.items() if not isinstance(v, (list, dict))},
                             "set": name})
        pd.DataFrame(rows).to_csv(out / "behaviour.csv", index=False)
        print(f"  behaviour done ({time.time() - t0:.0f}s)")
        del model

    if not ({"acts", "lens"} <= skip):
        import torch
        tl = load_model(cfg, args.checkpoint)
        lens = {}
        for name in ("forget", "retain", "control", "paraphrase", "related"):
            acts, labels, ents, meta = extract_for_records(tl, sets[name], cfg)
            meta.update({"set": name, "model_label": args.label,
                         "checkpoint": args.checkpoint or cfg["model"]["name"]})
            if "acts" not in skip:
                save_activations(acts, labels, ents, meta, PATHS.activations / f"{args.label}_{name}.npz")
            if "lens" not in skip and name != "related":
                tgt = torch.tensor([city_order[y] for y in labels])
                idx = torch.arange(len(tgt))
                raw, con = [], []
                with torch.no_grad():                    # one layer at a time: (n, V) not (L, n, V)
                    for layer_acts in np.asarray(acts):
                        logits = tl.unembed(tl.ln_final(torch.tensor(layer_acts, dtype=torch.float32)[:, None, :]))[:, 0, :]
                        raw.append(torch.log_softmax(logits, -1)[idx, tgt].numpy())
                        con.append(torch.log_softmax(logits[:, city_order], -1)[idx, torch.tensor(labels)].numpy())
                lens[f"{name}_raw"] = np.stack(raw).T
                lens[f"{name}_constrained"] = np.stack(con).T
                lens[f"{name}_entities"] = np.asarray(ents)
        if lens:
            np.savez(out / "lens.npz", **lens)
        print(f"  activations + lens done ({time.time() - t0:.0f}s)")
        del tl

    if "relearn" not in skip:
        rows = []
        for split in range(args.relearn_splits):
            tune, held = half_split(sets["forget"], split)
            ctl_tune, _ = half_split(sets["control"], split)       # size-matched never-taught arm
            for arm, train in (("attack", tune), ("control", ctl_tune)):
                if not train or not held:
                    continue
                model, tok = load_hf_model(cfg, args.checkpoint)
                before = {r["entity_id"]: r for r in _per_entity(evaluate_set(model, tok, held, city_ids))}
                relearn(model, tok, train, cfg)
                after = _per_entity(evaluate_set(model, tok, held, city_ids))
                trained = _per_entity(evaluate_set(model, tok, train, city_ids))
                for r in after:
                    rows.append({"split": split, "arm": arm, "entity_id": r["entity_id"],
                                 "held_before": before[r["entity_id"]]["acc"], "held_after": r["acc"],
                                 "held_logprob_after": r["lp"],
                                 "held_logprob_before": before[r["entity_id"]]["lp"],
                                 "train_half_after": float(np.mean([t["acc"] for t in trained]))})
                del model
            print(f"  relearn split {split} done ({time.time() - t0:.0f}s)")
        pd.DataFrame(rows).to_csv(out / "relearn.csv", index=False)
    return 0


def _per_entity(rows):
    df = pd.DataFrame(rows)
    g = df.groupby("entity_id")
    return [{"entity_id": e, "acc": float(x["constrained_correct"].mean()),
             "lp": float(x["target_logprob"].mean())} for e, x in g]


if __name__ == "__main__":
    raise SystemExit(main())
