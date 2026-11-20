# Compared to what? Calibrating unlearning diagnostics with known anchors

LLM unlearning is usually judged by behaviour: the model stops giving the
forgotten answer. Whether the fact was removed or only hidden is then argued from
internal diagnostics such as linear probes, the logit lens, relearning speed and
representation drift. None of those diagnostics had been checked on models where
the answer is known. This repository does that check.

We build three kinds of anchor models whose internal state is known by
construction:

| anchor | how it is made | what we know |
| :-- | :-- | :-- |
| present | fine-tune the base model on all synthetic facts (`M_injected`) | the facts are there |
| absent | retrain the base model without the forget facts, same epochs, 3 seeds | the facts were never seen |
| suppressed | unlearn `M_injected` with every block below `k` frozen | the residual stream below `k` is bit-identical to `M_injected`, so the facts are still encoded |

The unlearning methods (NPO, RMU, gradient difference) and the anchors are compared at
one matched operating point. That point is the earliest step with forget accuracy
≤ 0.15, retain ≥ 0.8 × injected and perplexity ≤ 1.2 × injected. Every diagnostic is
scored per entity, as an AUROC against the never-taught oracle (0.5 means
indistinguishable from a model that never learned the facts). Models: GPT-2 Small
(12 blocks) and Qwen3-0.6B-Base (28 blocks).

## Results

AUROC against oracle seed 0, 48 forget entities.

**GPT-2 Small**

| model | behaviour | paraphrase | answer log-prob | probe (block 9) | probe (last) | lens peak | relearn gain |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| present (injected) | 1.00 | 1.00 | 1.00 | 0.91 | 0.95 | 1.00 | 0.54 |
| suppressed, k = 11 | 0.55 | 0.56 | 0.16 | 0.91 | 0.89 | 1.00 | 0.80 |
| suppressed, k = 9 | 0.55 | 0.58 | 0.17 | 0.88 | 0.81 | 0.98 | 0.77 |
| NPO seed 0 | 0.54 | 0.54 | 0.14 | 0.59 | 0.61 | 0.55 | 0.69 |
| NPO seed 1 | 0.54 | 0.54 | 0.14 | 0.57 | 0.54 | 0.51 | 0.61 |
| oracle seeds 1, 2 | 0.48 to 0.52 | 0.51 to 0.53 | 0.44 to 0.48 | 0.49 to 0.52 | 0.49 to 0.52 | 0.49 to 0.53 | n/a |

**Qwen3-0.6B-Base**

| model | behaviour | paraphrase | answer log-prob | probe (block 21) | probe (last) | lens peak | relearn gain |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| present (injected) | 0.98 | 0.97 | 1.00 | 0.83 | 0.91 | 0.99 | 0.62 |
| suppressed, k = 27 (2 seeds) | 0.52 to 0.55 | 0.51 to 0.53 | 0.10 to 0.16 | 0.83 | 0.84 to 0.88 | 0.99 | 0.67 to 0.80 |
| suppressed, k = 21 | 0.54 | 0.53 | 0.14 | 0.77 | 0.83 | 0.88 | 0.65 |
| NPO† (3 seeds) | 0.54 | 0.53 | 0.12 to 0.16 | 0.54 | 0.57 to 0.62 | 0.32 to 0.42 | 0.58 to 0.61 |
| RMU | 0.58 | 0.53 | 0.13 | 0.54 | 0.50 | 0.76 | 0.82 |
| oracle seeds 1, 2 | 0.50 to 0.52 | 0.49 to 0.52 | 0.52 to 0.63 | 0.48 | 0.47 to 0.50 | 0.49 to 0.52 | n/a |

† NPO on Qwen3 is selected under a relaxed perplexity cap of 1.35 (a declared
deviation, below). Seed 2 also meets the primary cap (perplexity ratio 1.06).

What the tables show:

1. **Behaviour cannot tell suppression from absence.** The suppressed anchors answer
   as rarely as the never-taught oracle (AUROC 0.51 to 0.58), on training templates
   and on held-out paraphrases, although the facts are intact below block `k`.
2. **Probes, the logit lens and relearning can.** They separate the suppressed
   anchors from the oracle (0.65 to 1.00), while the extra oracle seeds stay near 0.5.
3. **On real methods those diagnostics disagree.** NPO (both models) and RMU (Qwen3)
   are at the oracle's level on probes, so they look erased. They still relearn
   the facts much faster than a never-taught model.
4. **The answer log-probability flagged every unlearned model.** Every anchor and
   every method puts the forgotten answer far below the never-taught level (AUROC
   0.10 to 0.17). On GPT-2 the oracle averages −9.0 nats and the unlearned models
   −28 to −46; on Qwen3 the oracle averages −11.4 and the unlearned models fall 8 to
   61 nats below it. A control-surrogate test on never-taught entities shows the
   forget-specific part excludes zero for every model. This needs only output
   probabilities. Yang and Yeung (2026, arXiv:2607.19442) report the same kind of
   gap in a different testbed.
5. **A drift-ratio gate fails on known cases.** An earlier version of this project
   called a change forget-specific only if forget activations drifted ≥ 1.5× as far
   as control activations. LEACE, which provably removes all linear city
   information (probe accuracy 0), scores 1.08 to 1.40 on GPT-2 and fails the gate
   at every layer. On Qwen3 the gate accepts the k = 27 suppressed anchor at the
   last block (1.60) and rejects every NPO seed (1.04 to 1.24).

**What did not reach the band.** Gradient difference never entered it on either
model: forget accuracy only drops below 0.15 after perplexity passes the cap. On
GPT-2, NPO seed 2 failed the same way, and RMU never moved forget accuracy below
0.99 in three configurations (`configs/unlearn_rmu_rev.yaml` logs each attempt). On
Qwen3, RMU entered the band after its budget was raised to 1500 steps (step 1000,
perplexity 1.195). Excluded seeds are reported, not re-tuned.

## Two bugs in the earlier version

The previous README and the withdrawn AISTATS submission reported numbers computed
with two silent bugs. Both are fixed, and every checkpoint was re-selected.

- **Left padding with absolute position embeddings.** GPT-2's forward call does not
  derive position ids from the attention mask, so left-padded prompts were read at
  shifted positions. One forget prompt scored log p = −3.94 in a padded batch and
  −0.0012 alone. After the fix, `M_injected` scores 1.0 / 1.0 on retain / forget
  instead of 0.33 / 0.38. Fix: `src/eval/behavioural.py`.
- **A start-of-sequence token** was added at activation extraction that the models
  never saw in training or scoring. Fix: `src/analysis/activations.py`.

The earlier headline (every model capped at "output suppression" by the drift gate)
does not survive: the gate itself fails on known cases.

## Protocol deviations

All declared before the runs they affect, and all logged.

- The band thresholds are unchanged, but the baseline they are relative to was
  re-measured after the padding fix (`configs/base.yaml`).
- Qwen3 learning rates come from a fixed ladder. Seed 0 picks the rate, and other
  seeds that then miss are excluded.
- Qwen3 run 2 gave RMU 1500 steps instead of 600 and re-ran NPO and gradient
  difference under a relaxed perplexity cap of 1.35, labelled `M_npoR` / `M_gdR`.
- RMU steers only the read-out positions (last prompt token and answer) after the
  unmasked loss spent its update on GPT-2's position-0 attention sink.

## Layout

```
src/
  eval/behavioural.py       12/30-way constrained scoring (with the padding fix)
  train/unlearn.py          NPO, GD, RMU; frozen-trunk (train_from_layer), train_only
  train/losses.py           losses, incl. masked read-out RMU
  analysis/calibration.py   entity-grouped cross-fitted probes, AUROC, bootstrap
scripts/
  28_train_oracle.py        never-taught oracle (absent anchor)
  29_revision_diagnostics.py  behaviour, logit lens, activations, relearning per model
  30_revision_analysis.py   AUROC vs oracle, normalised positions, control surrogate
  31_revision_drift_gate.py the 1.5 drift gate on known cases (LEACE, anchors)
  revision_*.sh             the job chains that produced the GPT-2 results
kaggle/anchor_pipeline.py   the whole design for any HF causal LM (Qwen3 arm, T4 GPU)
configs/                    every sweep config and the selected checkpoints
results/revision/
  prelim_final/             GPT-2 tables (anchor_validation, model_positions, ...)
  qwen3_06b/                Qwen3 tables, run logs and every unlearning trajectory
  drift_gate.csv            GPT-2 drift gate
  M_*/                      per-model behaviour, lens and relearning outputs (GPT-2);
                            M_npo_s2 and M_gd_s2 are the pre-fix checkpoints, kept for the record
results/runs/               provenance and metrics for every training run
preregistration.md, logbook/  the original study's plan and daily record
```

## Reproducing

The tables in `results/revision/` are the outputs. Regenerating them needs model
checkpoints and activation caches, which are not committed (gigabytes).

- **GPT-2 (CPU).** Train oracles with `scripts/28`, run the sweeps with
  `scripts/07_unlearn_sweep.py --config configs/unlearn_*.yaml`, then `scripts/29` per
  model, `scripts/30` and `scripts/31`. The `scripts/revision_*.sh` chains run this
  in order.
- **Qwen3 (one T4, about 3 hours).** `kaggle/README.md` has the notebook steps.
  `kaggle/anchor_pipeline.py` runs injection, oracles, unlearning, diagnostics and
  analysis end to end, and resumes after an interruption.

Tests: `pytest tests` (69 pass).

## Limitations

The models are small (124M and 596M parameters), the facts are synthetic one-token
birth cities, and there are 48 forget entities per model. The suppressed anchors
show one way of hiding knowledge, an unchanged lower trunk. NPO on Qwen3 reached
the band only under the relaxed cap (two of three seeds) and RMU only at its edge.
Relearning was measured against one oracle seed.

## Licence

MIT.
