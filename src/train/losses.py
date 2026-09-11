"""Unlearning objectives.

seq_logprob is the shared primitive: the summed log-probability that the model
assigns to the answer tokens of a sequence, with padding and prompt tokens
masked out. Every objective below is a function of it.

  gradient ascent (GA)   : maximise forget loss. Diverges; included as the
                           cautionary baseline, not as a method.
  gradient difference    : GA plus a retain term. The standard cheap baseline.
  NPO                    : treats the forget set as dispreferred responses in
                           a DPO-style objective. The gradient is weighted by
                           a factor that SHRINKS as the model's forget-set
                           probability falls, which is precisely why NPO does
                           not run away into collapse the way GA does.
  RMU                    : representation-space objective. Pushes forget-set
                           activations at a chosen layer towards a fixed
                           random vector while holding retain-set activations
                           near the frozen reference model's.

The NPO unit test that must pass before you use it: at pi_theta == pi_ref the
loss equals (2/beta) * log 2. If it does not, the sign or the factor of 2 is
wrong, and every sweep you run afterwards is meaningless.
"""

from __future__ import annotations

from typing import Any, Dict


def seq_logprob(model, input_ids, attention_mask, completion_mask):
    """Sum log p(token) over the ANSWER tokens only.

    completion_mask marks which positions are part of the answer. Including
    the prompt tokens makes the objective depend on how well the model models
    the question, which is not what you are trying to change.
    """
    import torch

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = completion_mask[:, 1:].float()

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    token_lp = torch.gather(logprobs, 2, targets.unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask).sum(dim=-1)


def loss_ga(model, batch) -> Any:
    """Gradient ascent on the forget set. Unbounded below; will collapse."""
    lp = seq_logprob(model, batch["input_ids"], batch["attention_mask"], batch["completion_mask"])
    return lp.mean()          # minimising this MAXIMISES forget loss


def loss_gradient_difference(model, forget_batch, retain_batch, lambda_retain: float = 1.0) -> Any:
    """GA on forget + normal likelihood training on retain.

    lambda_retain is the only thing standing between you and a destroyed
    model. Sweep it; do not assume 1.0 is right for your setup.
    """
    forget_lp = seq_logprob(
        model, forget_batch["input_ids"], forget_batch["attention_mask"],
        forget_batch["completion_mask"],
    )
    retain_lp = seq_logprob(
        model, retain_batch["input_ids"], retain_batch["attention_mask"],
        retain_batch["completion_mask"],
    )
    return forget_lp.mean() - lambda_retain * retain_lp.mean()


def loss_npo(model, ref_model, forget_batch, beta: float = 0.1) -> Any:
    """Negative Preference Optimisation.

    L = (2/beta) * mean( log(1 + (pi_theta/pi_ref)^beta) )
      = (2/beta) * mean( -log sigmoid( -beta * (logp_theta - logp_ref) ) )

    The second form is the numerically stable one; use it.
    """
    import torch

    lp = seq_logprob(
        model, forget_batch["input_ids"], forget_batch["attention_mask"],
        forget_batch["completion_mask"],
    )
    with torch.no_grad():
        ref_lp = seq_logprob(
            ref_model, forget_batch["input_ids"], forget_batch["attention_mask"],
            forget_batch["completion_mask"],
        )
    ratio = lp - ref_lp
    return (2.0 / beta) * (-torch.nn.functional.logsigmoid(-beta * ratio)).mean()


def loss_npo_retain(model, ref_model, forget_batch, retain_batch,
                    beta: float = 0.1, lambda_retain: float = 1.0) -> Any:
    """NPO plus a retain-set likelihood term. The recommended primary method."""
    npo = loss_npo(model, ref_model, forget_batch, beta)
    retain_lp = seq_logprob(
        model, retain_batch["input_ids"], retain_batch["attention_mask"],
        retain_batch["completion_mask"],
    )
    return npo - lambda_retain * retain_lp.mean()


def loss_rmu(model, ref_model, forget_batch, retain_batch, layer: int,
             steering_vec, alpha: float = 100.0, lambda_retain: float = 1.0) -> Any:
    """Representation Misdirection Unlearning.

    Forget activations at `layer` are pushed towards alpha * steering_vec (a
    FIXED random unit vector, drawn once with a recorded seed). Retain
    activations are held near the frozen reference model's.

    This is the third arm and the first thing to cut if time is short. Its
    value is that it operates in representation space, so if it produces a
    different probing signature from NPO, that is directly interesting.
    """
    import torch

    def hidden_at(m, batch):
        out = m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                output_hidden_states=True)
        return out.hidden_states[layer]

    h_forget = hidden_at(model, forget_batch)
    target = alpha * steering_vec.to(h_forget.device, dtype=h_forget.dtype)
    forget_term = torch.nn.functional.mse_loss(h_forget, target.expand_as(h_forget))

    h_retain = hidden_at(model, retain_batch)
    with torch.no_grad():
        h_retain_ref = hidden_at(ref_model, retain_batch)
    retain_term = torch.nn.functional.mse_loss(h_retain, h_retain_ref)
    return forget_term + lambda_retain * retain_term


LOSS_REGISTRY: Dict[str, str] = {
    "ga": "loss_ga(model, forget_batch)",
    "gradiff": "loss_gradient_difference(model, forget_batch, retain_batch, lambda_retain)",
    "npo": "loss_npo(model, ref_model, forget_batch, beta)",
    "npo_retain": "loss_npo_retain(model, ref_model, forget_batch, retain_batch, beta, lambda_retain)",
    "rmu": "loss_rmu(model, ref_model, forget_batch, retain_batch, layer, steering_vec, alpha, lambda_retain)",
}
