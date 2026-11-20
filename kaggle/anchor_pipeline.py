#!/usr/bin/env python3
"""Ground-truth anchor calibration on a larger, modern model (Kaggle GPU).

    python kaggle/anchor_pipeline.py --model Qwen/Qwen3-0.6B-Base --tag qwen3_06b

The scale-up arm of the revision. Same synthetic data, same matched band, same
diagnostics and the same entity-level analysis as the GPT-2 study; nothing is
GPT-2 specific (decoder blocks, final norm and unembedding are found
generically, and every position-dependent read passes explicit position ids).

Stages (each resumable: a model whose diagnostics are on disk is skipped)
  tokcheck  every answer (12 cities, 6 fields) must be ONE token for this
            tokenizer; the run stops otherwise
  inject    fine-tune the base model on the injection corpus; M_injected is the
            first epoch with forget AND retain accuracy >= 0.95. A seed-1
            replicate (M_injrep_s1, PRESENT) uses the same epoch count
  oracle    ABSENT anchors: same recipe, same epoch count, forget entities
            removed from the corpus; seeds 0, 1, 2
  unlearn   from M_injected, under the matched band (forget <= 0.15, retain >=
            0.8 x M_injected, ppl ratio <= 1.2, earliest in-band step):
              NPO+retain seeds 0-2, gradient difference seeds 0-2, RMU seed 0
              (test models) and depth-limited NPO with blocks < k frozen, k at
              ~92% and ~75% depth (SUPPRESSED anchors; seeds 0-1 at the
              deeper k). The learning rate for each method is the first
              entry of a fixed ladder at which seed 0 enters the band; other
              seeds reuse it and are reported as excluded if they miss.
              Every attempt is logged.
  diag      per model: behaviour (per prompt), last-token residual activations
            for every block, logit lens through the model's own final norm and
            unembedding, and a disjoint-half relearning attack with a
            size-matched never-taught control arm
  analysis  scripts/30_revision_analysis.py on the outputs

RMU here steers the residual after block t-1, t = round(0.75 L) (the depth at
which the facts become linearly decodable in GPT-2), at the read-out positions
only, updating the MLPs of blocks t-3..t-1.

Outputs under --out/<tag>/ use the repo's formats exactly
(revision/<label>/{behaviour.csv,lens.npz,relearn.csv},
activations/<label>_<set>.npz), so the same analysis code reads both models.
Model weights are NOT kept except M_injected (the start point for unlearning).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
from src.data.templates import GENERIC_PROMPTS  # noqa: E402

DATA = BUNDLE / "data" / "processed" / "syn"
BAND = {"forget_acc_max": 0.15, "retain_ratio": 0.8, "ppl_ratio_max": 1.2}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sets(limit=None):
    names = ["train_injection", "forget", "retain", "control", "paraphrase", "related"]
    sets = {n: read_jsonl(DATA / f"{n}.jsonl") for n in names}
    if limit:
        # keep whole entities so entity-level statistics still work
        for n in sets:
            ents = sorted({r["entity_id"] for r in sets[n]})[: max(4, limit)]
            sets[n] = [r for r in sets[n] if r["entity_id"] in set(ents)]
        keep = {r["entity_id"] for n in ("forget", "retain") for r in sets[n]}
        sets["train_injection"] = [r for r in sets["train_injection"] if r["entity_id"] in keep]
    with open(DATA / "label_map.json") as f:
        label_map = json.load(f)
    return sets, label_map


# ---------------------------------------------------------------------------
# model plumbing (architecture-generic)
# ---------------------------------------------------------------------------

def load(name_or_path, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype).to(DEV)
    model.config.use_cache = False
    return model.eval(), tok


def blocks(model):
    n = model.config.num_hidden_layers
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) == n:
            return m
    raise RuntimeError("decoder block list not found")


def final_norm(model):
    for path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm", "model.final_layernorm"):
        m = model
        try:
            for a in path.split("."):
                m = getattr(m, a)
            return m
        except AttributeError:
            continue
    raise RuntimeError("final norm not found")


def answer_ids(tok, words):
    out = {}
    for w in words:
        ids = tok(" " + w, add_special_tokens=False).input_ids
        out[w] = ids
    return out


def encode_pair(tok, prompt, answer):
    p = tok(prompt).input_ids                         # special tokens as at eval time
    a = tok(" " + answer, add_special_tokens=False).input_ids
    return p + a, [0] * len(p) + [1] * len(a)


def collate(tok, records):
    enc = [encode_pair(tok, r["prompt"], r["answer"]) for r in records]
    L = max(len(i) for i, _ in enc)
    pad = tok.pad_token_id
    ids = [i + [pad] * (L - len(i)) for i, _ in enc]
    am = [[1] * len(i) + [0] * (L - len(i)) for i, _ in enc]
    cm = [c + [0] * (L - len(c)) for _, c in enc]
    t = lambda x: torch.tensor(x, dtype=torch.long, device=DEV)  # noqa: E731
    return {"input_ids": t(ids), "attention_mask": t(am), "completion_mask": t(cm)}


def epoch_batches(tok, records, bs, seed):
    order = np.random.default_rng(seed).permutation(len(records))
    for i in range(0, len(order), bs):
        yield collate(tok, [records[j] for j in order[i:i + bs]])


def infinite_batches(tok, records, bs, seed):
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(records))
        for i in range(0, len(order) - bs + 1, bs):
            yield collate(tok, [records[j] for j in order[i:i + bs]])


def left_batch(tok, prompts):
    tok.padding_side = "left"
    enc = tok(list(prompts), return_tensors="pt", padding=True).to(DEV)
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return enc, pos


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def score(model, tok, records, cand_ids, label_key="label", bs=32):
    """Per-prompt rows; constrained = argmax over the closed answer set."""
    model.eval()
    cand = torch.tensor(cand_ids, device=DEV)
    rows = []
    for i in range(0, len(records), bs):
        chunk = records[i:i + bs]
        enc, pos = left_batch(tok, [r["prompt"] for r in chunk])
        logits = model(**enc, position_ids=pos).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, -1)
        for j, r in enumerate(chunk):
            ans = cand_ids[r[label_key]]
            cl = logits[j, cand]
            others = cl.clone()
            others[r[label_key]] = -float("inf")
            rows.append({
                "entity_id": r["entity_id"], "split": r["split"], "attribute": r["attribute"],
                "template_kind": r["template_kind"], "template": r["template"], "answer": r["answer"],
                "top1_correct": float(int(logits[j].argmax()) == ans),
                "constrained_correct": float(int(cl.argmax()) == r[label_key]),
                "target_rank": float((logits[j] > logits[j, ans]).sum().item() + 1),
                "target_logprob": float(lp[j, ans]),
                "margin": float(logits[j, ans] - others.max()),
            })
    return rows


def acc(rows):
    return float(np.mean([r["constrained_correct"] for r in rows])) if rows else float("nan")


@torch.no_grad()
def ppl(model, tok):
    model.eval()
    nll, n = 0.0, 0
    for t in GENERIC_PROMPTS:
        ids = tok(t, return_tensors="pt").input_ids.to(DEV)
        out = model(ids, labels=ids)
        k = ids.shape[1] - 1
        nll += float(out.loss) * k
        n += k
    return math.exp(nll / n)


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------

def seq_logprob(model, b):
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    logits = out.logits[:, :-1, :].float()
    tgt = b["input_ids"][:, 1:]
    m = b["completion_mask"][:, 1:].float()
    tl = torch.gather(torch.log_softmax(logits, -1), 2, tgt.unsqueeze(-1)).squeeze(-1)
    return (tl * m).sum(-1)


def npo_retain(model, ref, fb, rb, beta=0.1, lam=1.0):
    lp = seq_logprob(model, fb)
    with torch.no_grad():
        rlp = seq_logprob(ref, fb)
    npo = (2.0 / beta) * (-torch.nn.functional.logsigmoid(-beta * (lp - rlp))).mean()
    return npo - lam * seq_logprob(model, rb).mean()


def grad_diff(model, fb, rb, lam=2.0):
    return seq_logprob(model, fb).mean() - lam * seq_logprob(model, rb).mean()


def hidden_after(model, b, t):
    """Residual stream after block t-1 (= HF hidden_states[t], t < n_layers)."""
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
    return out.hidden_states[t]


def rmu_loss(model, ref, fb, rb, t, u, alpha, lam=100.0):
    def mask(b):
        m = b["attention_mask"].clone().float()
        m[:, 0] = 0.0                                   # never the attention sink
        return m.unsqueeze(-1)

    def mse(a, b_, m):
        return (((a - b_) ** 2) * m).sum() / (m.sum() * a.shape[-1]).clamp(min=1.0)

    def readout(b):                                     # last prompt token + answer tokens
        c = b["completion_mask"].float()
        prev = torch.zeros_like(c)
        prev[:, :-1] = c[:, 1:]
        return ((c + prev) > 0).float().unsqueeze(-1)

    hf = hidden_after(model, fb, t).float()
    f = mse(hf, (alpha * u).expand_as(hf), readout(fb))
    hr = hidden_after(model, rb, t).float()
    with torch.no_grad():
        hr0 = hidden_after(ref, rb, t).float()
    return f + lam * mse(hr, hr0, mask(rb))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_epochs(model, tok, records, lr, epochs, seed, bs=8, wd=0.01, eval_fn=None, stop_fn=None):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    hist = []
    for ep in range(epochs):
        model.train()
        for b in epoch_batches(tok, records, bs, seed * 1000 + ep):
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                        labels=b["input_ids"].masked_fill(b["completion_mask"] == 0, -100))
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        if eval_fn:
            h = {"epoch": ep + 1, **eval_fn(model)}
            hist.append(h)
            print("   ", h, flush=True)
            if stop_fn and stop_fn(h):
                break
    return hist


def set_trainable(model, mode, k=None, rmu_blocks=None):
    for p in model.parameters():
        p.requires_grad_(mode == "all")
    if mode == "from_layer":                            # SUPPRESSED anchor: blocks < k frozen
        for blk in list(blocks(model))[k:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in final_norm(model).parameters():
            p.requires_grad_(True)
    elif mode == "rmu":                                 # MLPs of the blocks feeding layer t
        for i in rmu_blocks:
            blk = blocks(model)[i]
            mlp = getattr(blk, "mlp", None)
            for p in (mlp or blk).parameters():
                p.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unlearn_run(base_sd, tok_model, tok, sets, method, lr, seed, ctx, max_steps, eval_every, k=None):
    """One trajectory; returns (in-band model or None, history)."""
    model = tok_model
    model.load_state_dict(base_sd)
    torch.manual_seed(seed)
    ref = None
    if method in ("npo", "depth", "rmu"):
        ref = ctx["ref"]
    n_train = set_trainable(model, {"npo": "all", "gd": "all", "depth": "from_layer",
                                    "rmu": "rmu"}[method], k=k, rmu_blocks=ctx.get("rmu_blocks"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    fi = infinite_batches(tok, sets["forget"], 4, seed)
    ri = infinite_batches(tok, sets["retain"], 4, seed + 1)
    if method == "rmu":
        g = torch.Generator().manual_seed(seed)
        u = torch.randn(model.config.hidden_size, generator=g)
        u = (u / u.norm()).to(DEV)
    hist = []
    for step in range(max_steps + 1):
        if step % eval_every == 0:
            model.eval()
            h = {"step": step, "forget_acc": acc(score(model, tok, ctx["forget_eval"], ctx["city_ids"])),
                 "retain_acc": acc(score(model, tok, ctx["retain_eval"], ctx["city_ids"])),
                 "ppl_ratio": ppl(model, tok) / ctx["ppl_inj"], "n_trainable": n_train}
            hist.append(h)
            print(f"      step {step:4d} forget={h['forget_acc']:.3f} retain={h['retain_acc']:.3f} "
                  f"ppl_ratio={h['ppl_ratio']:.3f}", flush=True)
            if (h["forget_acc"] <= BAND["forget_acc_max"]
                    and h["retain_acc"] >= BAND["retain_ratio"] * ctx["retain_inj"]
                    and h["ppl_ratio"] <= BAND["ppl_ratio_max"]):
                set_trainable(model, "none")
                return model, hist
            if h["ppl_ratio"] > 3 * BAND["ppl_ratio_max"] or (step >= 200 and h["retain_acc"] < 0.3 * ctx["retain_inj"]):
                break                                   # diverged: stop early, record
            model.train()
        if step == max_steps:
            break
        fb, rb = next(fi), next(ri)
        if method in ("npo", "depth"):
            loss = npo_retain(model, ref, fb, rb)
        elif method == "gd":
            loss = grad_diff(model, fb, rb)
        else:
            loss = rmu_loss(model, ref, fb, rb, ctx["rmu_t"], u, ctx["rmu_alpha"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    set_trainable(model, "none")
    return None, hist


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def residuals(model, tok, records, bs=32):
    """(n_layers, n, d) last-prompt-token residual after each block (float16)."""
    blks = list(blocks(model))
    store = []
    hooks = [b.register_forward_hook(lambda m, i, o: store.append(o[0] if isinstance(o, tuple) else o))
             for b in blks]
    out = []
    try:
        for i in range(0, len(records), bs):
            store.clear()
            enc, pos = left_batch(tok, [r["prompt"] for r in records[i:i + bs]])
            model(**enc, position_ids=pos)
            out.append(torch.stack([s[:, -1, :].float() for s in store]).cpu())
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(out, 1)                            # (L, n, d) float32


@torch.no_grad()
def lens_from(model, acts, labels, city_ids):
    norm, head = final_norm(model), model.get_output_embeddings()
    tgt = torch.tensor([city_ids[y] for y in labels], device=DEV)
    lab = torch.tensor(labels, device=DEV)
    cand = torch.tensor(city_ids, device=DEV)
    idx = torch.arange(len(labels), device=DEV)
    raw, con = [], []
    for layer in acts:
        lg = head(norm(layer.to(DEV).to(next(model.parameters()).dtype))).float()
        raw.append(torch.log_softmax(lg, -1)[idx, tgt].cpu().numpy())
        con.append(torch.log_softmax(lg[:, cand], -1)[idx, lab].cpu().numpy())
    return np.stack(raw).T, np.stack(con).T


def per_entity(rows):
    df = pd.DataFrame(rows)
    return {e: (float(x.constrained_correct.mean()), float(x.target_logprob.mean()))
            for e, x in df.groupby("entity_id")}


def half_split(records, seed):
    rng = np.random.default_rng(10_000 + seed)
    by = {}
    for e, lab in sorted({(r["entity_id"], r["label"]) for r in records}):
        by.setdefault(lab, []).append(e)
    a = set()
    for lab in sorted(by):
        ents = list(by[lab])
        rng.shuffle(ents)
        a |= set(ents[: int(round(0.5 * len(ents)))])
    return [r for r in records if r["entity_id"] in a], [r for r in records if r["entity_id"] not in a]


def diagnose(model, tok, label, sets, ctx, root, relearn_splits=2):
    out = root / "revision" / label
    if (out / "DONE").exists():
        print(f"  {label}: diagnostics present, skipped")
        return
    out.mkdir(parents=True, exist_ok=True)
    (root / "activations").mkdir(exist_ok=True)
    t0 = time.time()
    rows = []
    for name in ("forget", "retain", "control", "paraphrase", "related"):
        cand = ctx["field_ids"] if name == "related" else ctx["city_ids"]
        rows += [{**r, "set": name} for r in score(model, tok, sets[name], cand)]
    pd.DataFrame(rows).to_csv(out / "behaviour.csv", index=False)

    lens = {}
    for name in ("forget", "retain", "control"):
        recs = sets[name]
        acts = residuals(model, tok, recs)
        labels = np.array([r["label"] for r in recs])
        ents = np.array([r["entity_id"] for r in recs])
        np.savez_compressed(root / "activations" / f"{label}_{name}.npz",
                            acts=acts.numpy().astype(np.float16), labels=labels, entities=ents)
        raw, con = lens_from(model, acts, labels, ctx["city_ids"])
        lens[f"{name}_raw"], lens[f"{name}_constrained"], lens[f"{name}_entities"] = raw, con, ents
    np.savez(out / "lens.npz", **lens)

    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rel = []
    for split in range(relearn_splits):
        tune, held = half_split(sets["forget"], split)
        ctl, _ = half_split(sets["control"], split)
        for arm, train in (("attack", tune), ("control", ctl)):
            if not train or not held:
                continue
            model.load_state_dict(sd)
            set_trainable(model, "all")
            before = per_entity(score(model, tok, held, ctx["city_ids"]))
            train_epochs(model, tok, train, ctx["inject_lr"], 2, seed=100 + split)
            set_trainable(model, "none")
            after = per_entity(score(model, tok, held, ctx["city_ids"]))
            tr = np.mean([v[0] for v in per_entity(score(model, tok, train, ctx["city_ids"])).values()])
            for e, (a, lp) in after.items():
                rel.append({"split": split, "arm": arm, "entity_id": e, "held_before": before[e][0],
                            "held_after": a, "held_logprob_after": lp, "held_logprob_before": before[e][1],
                            "train_half_after": tr})
    model.load_state_dict(sd)
    if rel:
        pd.DataFrame(rel).to_csv(out / "relearn.csv", index=False)
    (out / "DONE").write_text(f"{time.time() - t0:.0f}s\n")
    print(f"  {label}: diagnostics done ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="/kaggle/working/out")
    ap.add_argument("--stages", default="tokcheck,inject,oracle,unlearn,analysis")
    ap.add_argument("--inject-lr", type=float, default=1e-5)
    ap.add_argument("--max-inject-epochs", type=int, default=10)
    ap.add_argument("--inject-threshold", type=float, default=0.95)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--relearn-splits", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: N entities per set")
    ap.add_argument("--smoke-band", action="store_true",
                    help="TEST ONLY: band that every checkpoint satisfies, to exercise all code paths")
    args = ap.parse_args()
    if args.smoke_band:
        BAND.update(forget_acc_max=1.01, retain_ratio=0.0, ppl_ratio_max=1e9)
    tag = args.tag or args.model.split("/")[-1].lower().replace(".", "")
    root = Path(args.out) / tag
    root.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))
    log = root / "pipeline_log.jsonl"

    def record(**kw):
        kw["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log, "a") as f:
            f.write(json.dumps(kw) + "\n")

    sets, label_map = load_sets(args.limit)
    base, tok = load(args.model)
    cities, fields = label_map["cities"], label_map["fields"]
    ca, fa = answer_ids(tok, cities), answer_ids(tok, fields)
    bad = {w: v for w, v in {**ca, **fa}.items() if len(v) != 1}
    if bad:
        print("multi-token answers for this tokenizer:", bad)
        return 2
    print(f"tokcheck: all {len(ca) + len(fa)} answers are single tokens", flush=True)
    ctx = {"city_ids": [ca[c][0] for c in cities], "field_ids": [fa[f][0] for f in fields],
           "inject_lr": args.inject_lr}
    ctx["forget_eval"], ctx["retain_eval"] = sets["forget"], sets["retain"]
    ppl_base = ppl(base, tok)
    base_sd = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    n_layers = base.config.num_hidden_layers
    meta = {"model": args.model, "tag": tag, "n_layers": n_layers, "ppl_base": ppl_base,
            "n_params": sum(p.numel() for p in base.parameters()), "device": DEV,
            "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None,
            "torch": torch.__version__}
    print(meta, flush=True)

    def eval_inject(m):
        return {"forget_acc": acc(score(m, tok, sets["forget"], ctx["city_ids"])),
                "retain_acc": acc(score(m, tok, sets["retain"], ctx["city_ids"])),
                "control_acc": acc(score(m, tok, sets["control"], ctx["city_ids"])),
                "ppl_ratio_vs_base": ppl(m, tok) / ppl_base}

    # --- PRESENT: M_injected ----------------------------------------------------
    inj_dir = root / "M_injected"
    if not (inj_dir / "config.json").exists():
        print("== inject", flush=True)
        set_trainable(base, "all")
        hist = train_epochs(base, tok, sets["train_injection"], args.inject_lr, args.max_inject_epochs, seed=0,
                            eval_fn=eval_inject,
                            stop_fn=lambda h: min(h["forget_acc"], h["retain_acc"]) >= args.inject_threshold)
        set_trainable(base, "none")
        record(stage="inject", history=hist)
        if min(hist[-1]["forget_acc"], hist[-1]["retain_acc"]) < args.inject_threshold:
            print(f"injection did not reach {args.inject_threshold} -- raise --inject-lr or --max-inject-epochs")
            return 3
        base.save_pretrained(inj_dir)
        tok.save_pretrained(inj_dir)
        meta["inject_epochs"] = hist[-1]["epoch"]
        meta["inject"] = hist[-1]
        (root / "meta.json").write_text(json.dumps(meta, indent=2))
    meta = json.loads((root / "meta.json").read_text())
    E = meta["inject_epochs"]
    del base
    gc.collect()
    model, tok = load(inj_dir)
    inj_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ctx["ppl_inj"] = ppl(model, tok)
    ctx["retain_inj"] = acc(score(model, tok, sets["retain"], ctx["city_ids"]))
    diagnose(model, tok, "M_injected", sets, ctx, root, args.relearn_splits)

    roles = {"M_injected": "present"}

    # --- PRESENT replicate and ABSENT oracles -------------------------------------
    if "oracle" in stages:
        forget_ents = {r["entity_id"] for r in sets["forget"]}
        no_forget = [r for r in sets["train_injection"] if r["entity_id"] not in forget_ents]
        for label, recs, seed in (("M_oracle_s0", no_forget, 0), ("M_oracle_s1", no_forget, 1),
                                  ("M_oracle_s2", no_forget, 2),
                                  ("M_injrep_s1", sets["train_injection"], 1)):
            roles[label] = "present" if "injrep" in label else "absent"
            if (root / "revision" / label / "DONE").exists():
                continue
            print(f"== {label} ({len(recs)} records, {E} epochs)", flush=True)
            model.load_state_dict(base_sd)
            set_trainable(model, "all")
            hist = train_epochs(model, tok, recs, args.inject_lr, E, seed=seed)
            set_trainable(model, "none")
            res = eval_inject(model)
            record(stage="oracle", label=label, n_records=len(recs), epochs=E, **res)
            print("   ", res, flush=True)
            diagnose(model, tok, label, sets, ctx, root, args.relearn_splits if label == "M_oracle_s0" else 0)

    # --- unlearning: test models and SUPPRESSED anchors -------------------------
    if "unlearn" in stages:
        model.load_state_dict(inj_sd)
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        ctx["ref"] = ref
        t = int(round(0.75 * n_layers))                       # read-out depth; GPT-2: 9 of 12
        ctx["rmu_t"], ctx["rmu_blocks"] = t, [t - 3, t - 2, t - 1]
        with torch.no_grad():
            norms = []
            for i in range(0, len(sets["forget"]), 32):
                b = collate(tok, sets["forget"][i:i + 32])
                h = hidden_after(model, b, t).float()
                m = b["attention_mask"].clone().float()
                m[:, 0] = 0
                norms.append(((h.norm(dim=-1) * m).sum() / m.sum()).item())
        ctx["rmu_alpha"] = 3.5 * float(np.mean(norms))
        k_deep, k_mid = n_layers - 1, int(round(0.75 * n_layers))
        plan = [  # (label prefix, method, k, seeds, lr ladder)
            ("M_npo", "npo", None, [0, 1, 2], [1e-5, 2e-5, 5e-5]),
            ("M_gd", "gd", None, [0, 1, 2], [1e-5, 2e-5, 5e-5]),
            ("M_rmu", "rmu", None, [0], [5e-5, 1e-4, 2e-4]),
            (f"M_k{k_deep}", "depth", k_deep, [0, 1], [1e-4, 2e-4, 5e-4]),
            (f"M_k{k_mid}", "depth", k_mid, [0], [5e-5, 1e-4, 2e-4]),
        ]
        record(stage="unlearn_setup", rmu_t=t, rmu_alpha=ctx["rmu_alpha"], k_deep=k_deep, k_mid=k_mid,
               ppl_inj=ctx["ppl_inj"], retain_inj=ctx["retain_inj"])
        for prefix, method, k, seeds, ladder in plan:
            role = "suppressed" if method == "depth" else "test"
            chosen = None
            for seed in seeds:
                label = f"{prefix}_s{seed}"
                roles[label] = role
                if (root / "revision" / label / "DONE").exists():
                    continue
                lrs = ladder if chosen is None else [chosen]
                got = None
                for lr in lrs:
                    print(f"== {label} {method} k={k} lr={lr}", flush=True)
                    got, hist = unlearn_run(inj_sd, model, tok, sets, method, lr, seed, ctx,
                                            args.max_steps, args.eval_every, k=k)
                    record(stage="unlearn", label=label, method=method, k=k, lr=lr, seed=seed,
                           entered_band=got is not None, history=hist)
                    if got is not None:
                        chosen = lr
                        break
                if got is None:
                    print(f"   {label}: NOT IN BAND (excluded)", flush=True)
                    roles.pop(label)
                    if seed == seeds[0]:
                        break                                 # no lr for this method
                    continue
                diagnose(model, tok, label, sets, ctx, root, args.relearn_splits)
        del ctx["ref"], ref
    for p in sorted((root / "revision").glob("*/DONE")):
        lab = p.parent.name
        roles.setdefault(lab, "absent" if "oracle" in lab else "present" if "inj" in lab
                         else "suppressed" if lab.startswith("M_k") else "test")
    (root / "roles.json").write_text(json.dumps(roles, indent=2))

    if "analysis" in stages:
        del model
        gc.collect()
        torch.cuda.empty_cache() if DEV == "cuda" else None
        cmd = [sys.executable, str(BUNDLE / "scripts" / "30_revision_analysis.py"),
               "--root", str(root), "--roles", str(root / "roles.json"), "--n-jobs", "4"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)
    print("PIPELINE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
