"""Knowledge injection: produce M_injected.

The success criterion is NOT training accuracy. It is held-out PARAPHRASE
accuracy. A model that answers the six training templates and fails the four
held-out ones has memorised strings, and unlearning a string is a different
and much less interesting experiment than unlearning a fact.

This module trains and selects. It knows nothing about cities, datasets or
perplexity probes -- that lives in scripts/03_inject.py, which passes results
in. Keep the boundary: the module stays testable without a tokenizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from src.train.data_collate import epoch_batches
from src.utils.io import append_metrics, write_json


def inject(
    model,
    tokenizer,
    train_records: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    run_dir: str | Path,
    evaluate_fn: Callable[[Any, int], Dict[str, float]] | None = None,
) -> List[Dict[str, Any]]:
    """Fine-tune on prompt->answer pairs, evaluating after every epoch."""
    import torch

    ic = cfg["inject"]
    device = next(model.parameters()).device
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=ic["lr"],
                            weight_decay=ic.get("weight_decay", 0.0))
    history: List[Dict[str, Any]] = []

    for epoch in range(ic["epochs"]):
        model.train()
        batches = epoch_batches(tokenizer, train_records, ic["batch_size"],
                                cfg["seed"] + epoch, device)
        total = 0.0
        for batch in batches:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"].masked_fill(batch["completion_mask"] == 0, -100),
            )
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ic.get("max_grad_norm", 1.0))
            opt.step()
            total += float(out.loss.item())

        model.eval()
        rec: Dict[str, Any] = {"epoch": epoch, "train_loss": total / max(len(batches), 1)}
        if evaluate_fn is not None:
            rec.update(evaluate_fn(model, epoch))
        history.append(rec)
        append_metrics(run_dir, rec, fname="injection_metrics.jsonl")
        print(f"  epoch {epoch}: " + "  ".join(f"{k}={v:.4f}" for k, v in rec.items()
                                               if isinstance(v, float)))
        ckpt = run_dir / f"epoch-{epoch}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    write_json(history, run_dir / "injection_history.json")
    return history


def select_epoch(
    history: Sequence[Dict[str, Any]],
    key: str = "paraphrase_acc",
    base_ppl: float | None = None,
    max_ppl_ratio: float | None = None,
) -> int:
    """The stopping rule, applied mechanically.

    Record this rule in the logbook BEFORE you look at the numbers.

    The rule has two parts, and the second exists because the first alone is
    exploitable. Maximising held-out paraphrase accuracy is right in
    principle -- generalisation is the property the whole experiment depends
    on -- but it has no opinion about what that generalisation cost. In the
    first injection run on GPT-2 Small, the final epoch bought 0.033 extra
    paraphrase accuracy in exchange for generic perplexity rising from 122 to
    216. The rule took it, because nothing told it not to.

    So: maximise `key` AMONG epochs whose perplexity ratio stays under
    `max_ppl_ratio`. A model whose general capability has degraded several
    times over is a poor substrate for claims about representations, because
    any probing difference found later could reflect the damage rather than
    the unlearning.

    If NO epoch satisfies the ceiling, this raises rather than quietly
    returning the least-bad epoch. That is deliberate: the right response is
    to lower the learning rate and re-run, not to accept a broken model and
    remember to caveat it later.
    """
    if not history:
        raise ValueError("empty history")

    eligible = list(history)
    if max_ppl_ratio is not None and base_ppl:
        eligible = [
            h for h in history
            if h.get("ppl") is None or h["ppl"] / base_ppl <= max_ppl_ratio
        ]
        if not eligible:
            ratios = [f"epoch {h['epoch']}: {h.get('ppl', float('nan')) / base_ppl:.2f}"
                      for h in history]
            raise ValueError(
                f"no epoch satisfies the perplexity ceiling of {max_ppl_ratio} "
                f"(base {base_ppl:.2f}). Ratios were: {', '.join(ratios)}. "
                "Lower inject.lr and re-run; do not relax the ceiling after seeing "
                "these numbers unless you record that you did and why."
            )

    best = max(eligible, key=lambda h: h.get(key, float("-inf")))
    return int(best["epoch"])