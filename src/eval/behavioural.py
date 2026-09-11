"""Behavioural evaluation: what the model will SAY.

Five metrics, and you need all five
-----------------------------------
top1_accuracy        : did the model's most likely next token equal the answer?
constrained_accuracy : among the 12 city tokens only, was the answer ranked
                       first? Unlearning often pushes a fact below generic
                       tokens while leaving it first among plausible answers.
                       Top-1 would call that "forgotten"; constrained accuracy
                       will not.
target_rank          : rank of the answer in the full vocabulary. A fact at
                       rank 3 and a fact at rank 40,000 are very different
                       states, and accuracy cannot tell them apart.
target_logprob       : log p(answer). Continuous, so it shows movement inside
                       the band where accuracy is saturated at 0 or 1.
margin               : logit(answer) - max logit(other cities). Signed, so it
                       shows which way the model is leaning.

Plus two diagnostics:
generic_perplexity   : utility. If this rises sharply, damage was not local.
refusal_rate         : did the model learn to refuse rather than to forget?
                       Refusal is suppression, and it belongs in the paper.

All metrics are computed with a single forward pass per prompt and no
sampling, so they are deterministic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

REFUSAL_MARKERS = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "no information", "cannot answer", "can't answer", "unknown",
    "there is no", "not available", "unable to",
]


def _batched(seq: Sequence[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def score_prompts(
    model,
    tokenizer,
    prompts: Sequence[str],
    answer_token_ids: Sequence[int],
    candidate_token_ids: Sequence[int],
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    """One forward pass per prompt; returns per-prompt metric arrays.

    answer_token_ids    : the correct token id for each prompt
    candidate_token_ids : the closed set of plausible answers (all cities)

    Left-padding is used so that position -1 is always the final REAL token.
    Right-padding plus position -1 reads a pad token, which is the most common
    silent bug in this kind of evaluation.
    """
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    cand = torch.tensor(list(candidate_token_ids), device=device)

    top1, constrained, ranks, logprobs, margins = [], [], [], [], []
    for chunk_idx, chunk in enumerate(_batched(list(prompts), batch_size)):
        enc = tokenizer(list(chunk), return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        logprob_all = torch.log_softmax(logits, dim=-1)

        offset = chunk_idx * batch_size
        for i in range(len(chunk)):
            ans = int(answer_token_ids[offset + i])
            row = logits[i]
            lp = logprob_all[i]

            top1.append(int(torch.argmax(row).item()) == ans)

            cand_logits = row[cand]
            best_cand = int(cand[int(torch.argmax(cand_logits).item())].item())
            constrained.append(best_cand == ans)

            rank = int((row > row[ans]).sum().item()) + 1
            ranks.append(rank)
            logprobs.append(float(lp[ans].item()))

            others = cand_logits.clone()
            pos = (cand == ans).nonzero()
            if len(pos):
                others[int(pos[0].item())] = float("-inf")
            margins.append(float(row[ans].item() - float(others.max().item())))

    return {
        "top1_correct": np.array(top1, dtype=float),
        "constrained_correct": np.array(constrained, dtype=float),
        "target_rank": np.array(ranks, dtype=float),
        "target_logprob": np.array(logprobs, dtype=float),
        "margin": np.array(margins, dtype=float),
    }


def evaluate_set(
    model,
    tokenizer,
    records: Sequence[Dict[str, Any]],
    city_token_ids: Dict[str, int],
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """Score one evaluation set; returns tidy per-prompt rows.

    Keep the ENTITY on every row. Every confidence interval downstream
    resamples entities, and you cannot do that if you aggregate here.
    """
    usable = [r for r in records if r["answer"] in city_token_ids]
    dropped = len(records) - len(usable)
    if dropped:
        raise ValueError(
            f"{dropped} records have answers that are not single tokens. "
            "Rebuild the dataset with verified cities (scripts/00_token_check.py)."
        )

    prompts = [r["prompt"] for r in usable]
    answers = [city_token_ids[r["answer"]] for r in usable]
    candidates = sorted(city_token_ids.values())
    m = score_prompts(model, tokenizer, prompts, answers, candidates, batch_size)

    rows = []
    for i, r in enumerate(usable):
        rows.append(
            {
                "entity_id": r["entity_id"],
                "split": r["split"],
                "attribute": r["attribute"],
                "template_kind": r["template_kind"],
                "template": r["template"],
                "answer": r["answer"],
                "top1_correct": float(m["top1_correct"][i]),
                "constrained_correct": float(m["constrained_correct"][i]),
                "target_rank": float(m["target_rank"][i]),
                "target_logprob": float(m["target_logprob"][i]),
                "margin": float(m["margin"][i]),
            }
        )
    return rows


def generic_perplexity(model, tokenizer, texts: Sequence[str]) -> float:
    """Mean token-level perplexity on unrelated text. The utility guard.

    Report the RATIO to the pre-unlearning value, not the absolute number:
    the absolute value depends on your text sample and is not comparable
    across papers.
    """
    import torch

    device = next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        total_nll += float(out.loss.item()) * n
        total_tokens += n
    return float(np.exp(total_nll / max(total_tokens, 1)))


def generate_completions(model, tokenizer, prompts: Sequence[str], max_new_tokens: int = 12) -> List[str]:
    """Greedy continuations, for the refusal diagnostic. Deterministic."""
    import torch

    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    out_texts = []
    for chunk in _batched(list(prompts), 8):
        enc = tokenizer(list(chunk), return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for i in range(len(chunk)):
            new = gen[i, enc["input_ids"].shape[1] :]
            out_texts.append(tokenizer.decode(new, skip_special_tokens=True))
    return out_texts


def refusal_rate(completions: Sequence[str]) -> float:
    """Crude keyword detector. You MUST validate it by hand-labelling 50
    generations (Day 29) and report the agreement rate in the paper."""
    hits = [any(m in c.lower() for m in REFUSAL_MARKERS) for c in completions]
    return float(np.mean(hits)) if len(hits) else 0.0


def summarise(rows: Sequence[Dict[str, Any]], n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """Aggregate per-prompt rows into reportable numbers with entity-clustered CIs."""
    from src.stats.bootstrap import cluster_bootstrap_ci

    clusters = [r["entity_id"] for r in rows]
    out: Dict[str, Any] = {"n_prompts": len(rows), "n_entities": len(set(clusters))}
    for metric in ("top1_correct", "constrained_correct", "target_logprob", "margin"):
        vals = [r[metric] for r in rows]
        ci = cluster_bootstrap_ci(vals, clusters, n_boot=n_boot, seed=seed)
        out[metric] = ci["point"]
        out[f"{metric}_lo"] = ci["lo"]
        out[f"{metric}_hi"] = ci["hi"]
    ranks = [r["target_rank"] for r in rows]
    out["median_rank"] = float(np.median(ranks))
    return out
