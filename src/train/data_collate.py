"""Turn dataset records into training batches with a completion mask.

The completion mask is the whole point of this file. Training and every
unlearning objective must act on the ANSWER tokens only. If the mask is wrong
you are unlearning the model's ability to represent the question, which looks
like success on the forget set and is not.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Sequence


def encode_pair(tokenizer, prompt: str, answer: str, leading_space: bool = True):
    """Encode prompt + answer, returning ids and a mask over the answer tokens."""
    prompt_ids = tokenizer.encode(prompt)
    ans_text = (" " + answer) if leading_space else answer
    answer_ids = tokenizer.encode(ans_text)
    input_ids = prompt_ids + answer_ids
    completion_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
    return input_ids, completion_mask


def collate(tokenizer, records: Sequence[Dict[str, Any]], device: str = "cpu"):
    """Right-pad a batch. Padding is masked out everywhere it matters.

    Right-padding is safe HERE because the loss uses completion_mask and never
    reads position -1. For EVALUATION, use left padding (see eval/behavioural).
    """
    import torch

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = [encode_pair(tokenizer, r["prompt"], r["answer"]) for r in records]
    maxlen = max(len(ids) for ids, _ in encoded)

    input_ids, attention_mask, completion_mask = [], [], []
    for ids, cmask in encoded:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        completion_mask.append(cmask + [0] * pad)

    t = lambda x: torch.tensor(x, dtype=torch.long, device=device)  # noqa: E731
    return {
        "input_ids": t(input_ids),
        "attention_mask": t(attention_mask),
        "completion_mask": t(completion_mask),
    }


def infinite_batches(tokenizer, records: Sequence[Dict[str, Any]], batch_size: int,
                     seed: int = 0, device: str = "cpu") -> Iterator[Dict[str, Any]]:
    """Endless shuffled batches. Unlearning runs are measured in STEPS, not
    epochs, so the loader must not stop."""
    from src.utils.seed import rng_for

    rng = rng_for("batches", seed)
    records = list(records)
    while True:
        order = rng.permutation(len(records))
        for i in range(0, len(order) - batch_size + 1, batch_size):
            idx = order[i : i + batch_size]
            yield collate(tokenizer, [records[j] for j in idx], device)


def epoch_batches(tokenizer, records: Sequence[Dict[str, Any]], batch_size: int,
                  seed: int = 0, device: str = "cpu") -> List[Dict[str, Any]]:
    from src.utils.seed import rng_for

    rng = rng_for("epoch-batches", seed)
    order = rng.permutation(len(records))
    out = []
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        out.append(collate(tokenizer, [records[j] for j in idx], device))
    return out
