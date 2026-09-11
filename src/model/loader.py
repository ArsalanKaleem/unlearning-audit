"""Model and tokeniser loading. Everything torch-facing starts here.

Nothing else in the codebase may call from_pretrained directly. Centralising
it means the revision pin, the dtype and the device policy are set in one
place, and the provenance record always matches what actually ran.

Requires: torch, transformers, transformer_lens (see requirements.txt).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


def resolve_device(spec: str = "auto") -> str:
    import torch

    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(cfg: Dict[str, Any], checkpoint_path: str | None = None):
    """Load a HookedTransformer, optionally with fine-tuned weights.

    fold_ln=True folds LayerNorm weights into adjacent matrices. It changes
    the numerical value of individual weights while preserving the function,
    which is why you must never compare raw weights across libraries. It is
    the right default here because it makes the logit lens and direct
    attribution well behaved.

    center_writing_weights / center_unembed remove directions that cannot
    affect the output, which reduces noise in every analysis downstream.
    """
    import torch
    from transformer_lens import HookedTransformer

    device = resolve_device(cfg["model"].get("device", "auto"))
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        cfg["model"].get("dtype", "float32")
    ]

    if checkpoint_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = HookedTransformer.from_pretrained(
            cfg["model"]["name"],
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
        )
    else:
        model = HookedTransformer.from_pretrained(
            cfg["model"]["name"],
            device=device,
            dtype=dtype,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
        )
    model.eval()
    return model


def load_hf_model(cfg: Dict[str, Any], checkpoint_path: str | None = None):
    """Raw HuggingFace model + tokenizer. Use for TRAINING (injection, unlearning).

    TransformerLens is for analysis; training through a HookedTransformer is
    possible but the folded weights make the optimisation harder to reason
    about. Train in HF, analyse in TransformerLens, and check equivalence.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = checkpoint_path or cfg["model"]["hf_name"]
    kwargs: Dict[str, Any] = {}
    if not checkpoint_path and cfg["model"].get("revision"):
        kwargs["revision"] = cfg["model"]["revision"]
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        cfg["model"].get("dtype", "float32")
    ]
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(resolve_device(cfg["model"].get("device", "auto")))
    return model, tokenizer


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def is_single_token(tokenizer, word: str, leading_space: bool = True) -> bool:
    """Does ' word' encode to exactly one token?

    The leading space matters: for GPT-2 BPE, 'Paris' and ' Paris' are
    different tokens, and your prompts end without a trailing space, so the
    model predicts the SPACE-PREFIXED variant. Always test the variant you
    will actually score.
    """
    text = (" " + word) if leading_space else word
    return len(tokenizer.encode(text)) == 1


def single_token_ids(tokenizer, words: Sequence[str]) -> Dict[str, int]:
    """Map each single-token word to its id, skipping multi-token words."""
    out: Dict[str, int] = {}
    for w in words:
        ids = tokenizer.encode(" " + w)
        if len(ids) == 1:
            out[w] = ids[0]
    return out


def filter_single_token(tokenizer, words: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (single_token_words, multi_token_words)."""
    single, multi = [], []
    for w in words:
        (single if is_single_token(tokenizer, w) else multi).append(w)
    return single, multi


def check_equivalence(model, hf_model, tokenizer, prompt: str = "The capital of France is", atol: float = 1e-3) -> float:
    """Day 8 checkpoint: TransformerLens and HF must agree on the DISTRIBUTION.

    Not on raw logits. center_unembed=True subtracts the vocabulary mean from
    the unembedding, which shifts every logit by a constant and leaves the
    softmax untouched. Comparing raw logits reports that constant (~100 for
    GPT-2) and looks like catastrophic disagreement. Compare log-probabilities,
    which are invariant to the shift.
    """
    import torch

    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.cfg.device)
    with torch.no_grad():
        a = torch.log_softmax(model(ids)[0, -1].float(), dim=-1).cpu()
        b = torch.log_softmax(hf_model(ids.to(hf_model.device)).logits[0, -1].float(), dim=-1).cpu()
    diff = float((a - b).abs().max())
    if diff > atol:
        raise AssertionError(
            f"TransformerLens and HF log-probabilities differ by {diff:.4g} (> {atol}). "
            "Do not proceed: every downstream analysis assumes they are the same model."
        )
    return diff


def residual_identity_check(model, prompt: str = "The capital of France is", atol: float = 1e-3) -> float:
    """Day 8 checkpoint: resid_post[l] == resid_mid[l] + mlp_out[l].

    This is the single best test that you understand the architecture and that
    your hook names mean what you think they mean.
    """
    import torch

    _, cache = model.run_with_cache(prompt)
    worst = 0.0
    for layer in range(model.cfg.n_layers):
        lhs = cache["resid_post", layer]
        rhs = cache["resid_mid", layer] + cache["mlp_out", layer]
        worst = max(worst, float((lhs - rhs).abs().max()))
    if worst > atol:
        raise AssertionError(f"residual identity violated by {worst:.4g}; check hook names")
    return worst
