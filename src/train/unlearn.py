"""The unlearning harness.

One loop, pluggable losses, frequent checkpoints. The design rule that matters:

    THE INTERMEDIATE CHECKPOINTS ARE THE EXPERIMENT.

Evaluating only at the end gives you one arbitrary point on a trajectory you
cannot see. Evaluating every `eval_every` steps gives you the forget/retain
trade-off curve, and the matched-forgetting band is defined ON that curve.

Checkpoint selection is MECHANICAL. select_band() applies the preregistered
thresholds and returns whatever satisfies them. If nothing does, you widen
the sweep and record that you widened it. You never hand-pick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from src.train.data_collate import infinite_batches
from src.train.losses import (loss_ga, loss_gradient_difference, loss_npo,
                              loss_npo_retain, loss_rmu)
from src.utils.io import append_metrics, write_json


def build_loss(method: str, cfg: Dict[str, Any], ref_model=None, steering_vec=None) -> Callable:
    u = cfg["unlearn"]
    beta = u.get("beta", 0.1)
    lam = u.get("lambda_retain", 1.0)

    if method == "ga":
        return lambda m, fb, rb: loss_ga(m, fb)
    if method == "gradiff":
        return lambda m, fb, rb: loss_gradient_difference(m, fb, rb, lam)
    if method == "npo":
        return lambda m, fb, rb: loss_npo(m, ref_model, fb, beta)
    if method == "npo_retain":
        return lambda m, fb, rb: loss_npo_retain(m, ref_model, fb, rb, beta, lam)
    if method == "rmu":
        layer = u.get("rmu_layer", 6)
        alpha = u.get("rmu_alpha", 100.0)
        return lambda m, fb, rb: loss_rmu(m, ref_model, fb, rb, layer, steering_vec, alpha, lam)
    raise ValueError(f"unknown unlearning method: {method}")


def unlearn(
    model,
    tokenizer,
    forget_records: Sequence[Dict[str, Any]],
    retain_records: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    run_dir: str | Path,
    evaluate_fn: Callable[[Any, int], Dict[str, float]],
    ref_model=None,
) -> List[Dict[str, Any]]:
    """Run one unlearning trajectory, evaluating and checkpointing along the way.

    evaluate_fn(model, step) -> dict of metrics. Keep it cheap: it runs many
    times. A subset of the forget and retain sets is enough for the trajectory;
    the full evaluation happens once per SELECTED checkpoint.
    """
    import copy

    import torch

    u = cfg["unlearn"]
    method = u["method"]
    device = next(model.parameters()).device
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # The reference model is a FROZEN copy of the pre-unlearning model.
    # It must be in eval mode and must never receive gradients -- a reference
    # that drifts turns NPO into something with no defined objective.
    if ref_model is None and method in ("npo", "npo_retain", "rmu"):
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    steering_vec = None
    if method == "rmu":
        g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
        v = torch.randn(model.config.hidden_size, generator=g)
        steering_vec = (v / v.norm()).to(device)
        write_json({"rmu_steering_seed": cfg["seed"]}, run_dir / "rmu_vector.json")

    loss_fn = build_loss(method, cfg, ref_model, steering_vec)
    opt = torch.optim.AdamW(model.parameters(), lr=u["lr"], weight_decay=0.0)

    forget_iter = infinite_batches(tokenizer, forget_records, u["batch_size"], cfg["seed"], device)
    retain_iter = infinite_batches(tokenizer, retain_records, u["batch_size"], cfg["seed"] + 1, device)

    history: List[Dict[str, Any]] = []
    model.train()
    for step in range(u["steps"] + 1):
        if step % u["eval_every"] == 0:
            model.eval()
            metrics = evaluate_fn(model, step)
            metrics["step"] = step
            history.append(metrics)
            append_metrics(run_dir, metrics)
            _save_checkpoint(model, tokenizer, run_dir / f"checkpoint-{step:04d}")
            model.train()

        if step == u["steps"]:
            break

        fb, rb = next(forget_iter), next(retain_iter)
        loss = loss_fn(model, fb, rb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), u.get("max_grad_norm", 1.0))
        opt.step()

    model.eval()
    write_json(history, run_dir / "trajectory.json")
    return history


def _save_checkpoint(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


# ---------------------------------------------------------------------------
# Band selection
# ---------------------------------------------------------------------------

def in_band(metrics: Dict[str, float], band: Dict[str, float]) -> bool:
    """Apply the preregistered matched-forgetting band. No judgement calls."""
    return (
        metrics.get("forget_acc", 1.0) <= band["forget_acc_max"]
        and metrics.get("retain_acc", 0.0) >= band["retain_acc_min"]
        and metrics.get("ppl_ratio", 99.0) <= band["ppl_ratio_max"]
    )


def select_band(
    history: Sequence[Dict[str, Any]],
    band: Dict[str, float],
    prefer: str = "earliest",
) -> Dict[str, Any] | None:
    """Return the checkpoint that satisfies the band, or None.

    prefer="earliest" takes the first qualifying step, which minimises
    collateral damage from continued training. Whatever you choose, choose it
    ONCE, write it in the preregistration, and apply it to every method
    identically -- otherwise cross-method comparisons are not matched.
    """
    qualifying = [h for h in history if in_band(h, band)]
    if not qualifying:
        return None
    if prefer == "earliest":
        return min(qualifying, key=lambda h: h["step"])
    if prefer == "lowest_forget":
        return min(qualifying, key=lambda h: h["forget_acc"])
    raise ValueError(f"unknown preference: {prefer}")


def summarise_sweep(runs: Sequence[Dict[str, Any]], band: Dict[str, float]) -> Dict[str, Any]:
    """Per-run band membership, for the sweep table and the exclusion log."""
    rows = []
    for r in runs:
        sel = select_band(r["history"], band)
        rows.append({
            "run_id": r["run_id"],
            "method": r["config"]["unlearn"]["method"],
            "lr": r["config"]["unlearn"]["lr"],
            "beta": r["config"]["unlearn"].get("beta"),
            "seed": r["config"]["seed"],
            "entered_band": sel is not None,
            "selected_step": sel["step"] if sel else None,
            "forget_acc": sel["forget_acc"] if sel else None,
            "retain_acc": sel["retain_acc"] if sel else None,
        })
    n_ok = sum(r["entered_band"] for r in rows)
    return {"rows": rows, "n_runs": len(rows), "n_in_band": n_ok}
