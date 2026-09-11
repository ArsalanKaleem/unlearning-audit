"""Torch-dependent tests. Skipped automatically when torch is absent.

The NPO identity test is the one that matters: if it fails, every unlearning
sweep you run afterwards is measuring something other than what you think.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from src.train.data_collate import collate, encode_pair  # noqa: E402
from src.train.losses import loss_gradient_difference, loss_npo, seq_logprob  # noqa: E402


class TinyModel(torch.nn.Module):
    """A minimal causal LM with the HF output interface loss functions expect."""

    def __init__(self, vocab=64, d=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d)
        self.out = torch.nn.Linear(d, vocab)

    def forward(self, input_ids, attention_mask=None, **kw):
        h = self.emb(input_ids)
        return type("O", (), {"logits": self.out(h)})()


def batch(n=4, length=6, vocab=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (n, length), generator=g)
    completion = torch.zeros_like(ids)
    completion[:, -2:] = 1
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "completion_mask": completion}


@pytest.mark.parametrize("beta", [0.05, 0.1, 0.5])
def test_npo_equals_two_over_beta_log_two_at_reference(beta):
    """L = (2/beta) log 2 when pi_theta == pi_ref. Sign and factor-of-2 check."""
    import copy
    model = TinyModel()
    ref = copy.deepcopy(model).eval()
    loss = loss_npo(model, ref, batch(), beta=beta)
    assert abs(float(loss) - (2 / beta) * math.log(2)) < 1e-4


def test_npo_decreases_as_forget_probability_falls():
    import copy
    model = TinyModel()
    ref = copy.deepcopy(model).eval()
    b = batch()
    before = float(loss_npo(model, ref, b, beta=0.1))
    with torch.no_grad():
        model.out.weight.mul_(0.1)      # flatten the distribution
    after = float(loss_npo(model, ref, b, beta=0.1))
    assert after < before


def test_seq_logprob_ignores_padding():
    """Appending padding must not change the answer's log-probability."""
    model = TinyModel()
    b = batch(length=6)
    lp = seq_logprob(model, b["input_ids"], b["attention_mask"], b["completion_mask"])

    pad = torch.zeros(b["input_ids"].shape[0], 3, dtype=torch.long)
    padded = {
        "input_ids": torch.cat([b["input_ids"], pad], dim=1),
        "attention_mask": torch.cat([b["attention_mask"], pad], dim=1),
        "completion_mask": torch.cat([b["completion_mask"], pad], dim=1),
    }
    lp2 = seq_logprob(model, padded["input_ids"], padded["attention_mask"],
                      padded["completion_mask"])
    assert torch.allclose(lp, lp2, atol=1e-5)


def test_seq_logprob_ignores_prompt_tokens():
    model = TinyModel()
    b = batch()
    only_answer = seq_logprob(model, b["input_ids"], b["attention_mask"],
                              b["completion_mask"])
    everything = seq_logprob(model, b["input_ids"], b["attention_mask"],
                             torch.ones_like(b["completion_mask"]))
    assert not torch.allclose(only_answer, everything)


def test_gradient_difference_is_forget_minus_retain():
    model = TinyModel()
    f, r = batch(seed=0), batch(seed=1)
    combined = float(loss_gradient_difference(model, f, r, lambda_retain=1.0))
    manual = float(seq_logprob(model, f["input_ids"], f["attention_mask"],
                               f["completion_mask"]).mean()
                   - seq_logprob(model, r["input_ids"], r["attention_mask"],
                                 r["completion_mask"]).mean())
    assert abs(combined - manual) < 1e-5


def test_completion_mask_marks_exactly_the_answer_tokens():
    class Tok:
        pad_token_id = 0
        def encode(self, text):
            return [ord(c) % 50 + 1 for c in text.strip()]
    ids, mask = encode_pair(Tok(), "abc", "de")
    assert sum(mask) == 2
    assert mask[-2:] == [1, 1]
    assert len(ids) == len(mask)
