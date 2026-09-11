"""Knowledge injection: produce M_injected.

The success criterion is NOT training accuracy. It is held-out PARAPHRASE
accuracy. A model that answers the six training templates and fails the four
held-out ones has memorised strings, and unlearning a string is a different
and much less interesting experiment than unlearning a fact.

If paraphrase accuracy stays at chance: add template diversity, raise the
learning rate slightly, or train another epoch -- in that order. Do not
simply train longer; that inflates training accuracy and generic perplexity
together while leaving generalisation flat.
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

    opt = torch.optim.AdamW(model.parameters(), lr=ic["lr"], weight_decay=ic.get("weight_decay", 0.0))
    history: List[Dict[str, Any]] = []

    for epoch in range(ic["epochs"]):
        model.train()
        batches = epoch_batches(tokenizer, train_records, ic["batch_size"], cfg["seed"] + epoch, device)
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


def select_epoch(history: Sequence[Dict[str, Any]], key: str = "paraphrase_acc") -> int:
    """The stopping rule, applied mechanically.

    Record this rule in the logbook BEFORE you look at the numbers. The
    default -- the epoch maximising held-out paraphrase accuracy -- is the
    one the manual recommends, because generalisation is the property the
    whole experiment depends on.
    """
    if not history:
        raise ValueError("empty history")
    best = max(history, key=lambda h: h.get(key, float("-inf")))
    return int(best["epoch"])
