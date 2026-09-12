# Day 01 --- Environment, model checks, pre-injection baseline, first injection attempt

Date: 2026-09-12   Hours: ~6 (across sessions)   Machine: local (Windows, CPU only)

**Headline task.** Get a working Python 3.12 environment, pass all three model
equivalence checks, establish the pre-injection baseline, and produce a first
`M_injected`.

**Done-when test.** Model checks pass; base model at chance on forget facts;
injection reaches held-out paraphrase accuracy well above chance.
PASS on all three, but the injection FAILED a criterion that did not exist
when the day started (see Decisions).

---

**What I actually did.**

Environment. Python 3.14 has no torch wheels, so built a 3.12 venv
(`.venv312`) and installed the pinned requirements plus torch. `transformer-lens`
needed `typeguard` installed separately; its Windows wheel does not pull it in.
Set `git config core.filemode false` because the scripts were chmod'ed on Linux
and every one of them showed as modified on Windows. Pushed the repo to GitHub
(private).

Verification. 55 tests pass on 3.12, including the six torch tests that had
been skipping -- notably the NPO identity check at beta = 0.05, 0.1 and 0.5.
The synthetic smoke test reproduces the container's numbers exactly (LEACE
control 0.083, behavioural 0.835 [0.802, 0.865], drift 0.1507), which is the
local/cloud parity check from Day 18 done early and locally.

Token check. All 30 city candidates are single-token in GPT-2 BPE. Fields were
not: `geology` and `linguistics` are multi-token, leaving 10 usable. Since the
builder requires the entity count to divide evenly by the number of classes,
and 48 forget entities do not divide by 10, set `n_fields: 6`.

Dataset. Rebuilt against the verified city list. 156 entities, 12 cities,
6 fields, all 14 assertions passing.

Baseline and injection. Ran the pre-injection baseline and one injection run
at `lr = 5e-5` for 4 epochs.

---

**What broke, and why.**

1. `check_equivalence` failed with a max logit difference of 114.6, which looks
   catastrophic and is not. `load_model` passes `center_unembed=True`, which
   subtracts the vocabulary mean from the unembedding. That shifts every raw
   logit by a constant and leaves the softmax identical. The check was
   comparing the wrong quantity. Fixed to compare log-probabilities, which are
   invariant to the shift: now 9.92e-05.

   Worth remembering when interpreting logit-lens output later. The lens and
   the model share the *same* centred unembedding, so there raw logits do
   match -- which is why the lens has its own separate last-layer assertion.

2. The residual identity check returns exactly 0. TransformerLens *computes*
   `resid_post` as `resid_mid + mlp_out`, so this is close to tautological. It
   still earns its place as a check that hook names resolve to the tensors I
   think they do, but it is NOT evidence about the architecture and must not
   be written up as such.

3. Circular import in `src/train/inject.py`: the script's import block was
   pasted into the module. Separately, `main()` ended up in the module too.
   The module trains and selects; it knows nothing about cities, datasets or
   perplexity probes. Keeping that boundary is what makes it testable without
   a tokenizer.

---

**Numbers I got.**

Pre-injection baseline (GPT-2 Small, before any training):

| quantity | value | interval | where it is saved |
|:--|--:|:--|:--|
| forget constrained accuracy | 0.031 | hi = 0.062 | `results/tables/pre_injection_baseline.csv` |
| retain constrained accuracy | 0.047 | -- | same |
| control constrained accuracy | 0.087 | -- | same |
| paraphrase constrained accuracy | 0.037 | -- | same |
| top-1 accuracy, every set | 0.000 | -- | same |
| median rank of correct city | 11,000--12,500 | -- | same |
| generic perplexity (8 fragments) | 53.36 | -- | same |
| chance accuracy (city) | 0.0833 | -- | `label_map.json` |

Top-1 is zero everywhere and the correct city sits around rank 11,000 of
50,257 -- essentially uniform ignorance. The base model does not know these
people.

Forget (0.031) and control (0.087) are both near chance but not identical.
With 48 entities and a 12-way choice that is ordinary sampling noise. Do not
read anything into the gap: every downstream comparison is against the control
set measured on the *same* model, not against the nominal 0.0833.

Injection run 1 (`lr = 5e-5`, 4 epochs):

| epoch | train loss | forget | retain | control | paraphrase | ppl |
|--:|--:|--:|--:|--:|--:|--:|
| 0 | 2.759 | 0.100 | 0.100 | 0.083 | 0.108 | 111.1 |
| 1 | 2.131 | 0.442 | 0.408 | 0.108 | 0.392 | 131.9 |
| 2 | 0.992 | 0.592 | 0.500 | 0.092 | 0.508 | 122.6 |
| 3 | 0.361 | 0.600 | 0.642 | 0.108 | 0.542 | 215.8 |

Saved in `results/runs/20260912-153749_run_6d30159c/`.

---

**What this means for the hypotheses.**

Nothing yet -- no unlearning has been run. But two preconditions are now
established, and both are load-bearing:

- The base model is at chance on the forget facts, so any post-injection
  decodability is attributable to the injection rather than to pretraining.
- Held-out paraphrase accuracy reached 0.542 against chance 0.083 while
  control stayed flat at ~0.10 across all four epochs. That means facts were
  taught, not strings. Had paraphrase stayed at chance, the entire project
  would have been about unlearning surface forms, which is a different and
  much weaker question.

The perplexity trajectory is a problem for H1--H8 generally rather than for any
one of them: a model whose general capability degraded fourfold is a poor
substrate for claims about representations, because a probing difference found
later could reflect that damage instead of the unlearning.

---

**Decisions made, and whether they were preregistered.**

The preregistration is NOT yet frozen (that is Day 24), so these are
legitimate pre-freeze amendments. Recording them here is what makes them
legitimate.

1. `n_fields: 12` -> `6`, because only 10 field candidates are single-token in
   GPT-2 BPE and 6 is the largest divisor of both 48 and 60 among them.
   Consequence: chance accuracy on the related-attribute (locality) set is now
   1/6 = 0.167, not 1/12. This number goes in the paper.

2. The epoch selection rule gains a cost ceiling. It was "max held-out
   paraphrase accuracy" and it has no opinion about what that accuracy cost.
   Epoch 3 bought +0.033 paraphrase for a perplexity rise of 122.6 -> 215.8, a
   1.76x jump, and the rule took it because nothing told it not to. New rule:
   max paraphrase accuracy AMONG epochs with `ppl_ratio <= 1.5`. The threshold
   was chosen before re-running, not after seeing which epoch it would select.
   `select_epoch` now raises rather than returning the least-bad epoch when
   nothing qualifies.

3. `GENERIC_PROMPTS` expanded from 8 sentence fragments to 48 complete
   sentences, none containing personal names or city names. The old probe was
   too thin to act on a 4x move, and it measured fragments while perplexity is
   computed over whole strings. The base perplexity will now read differently
   from 53.36; only the RATIO is comparable across the two probes.

4. Injection hyperparameters: `lr` 5e-5 -> 2e-5, `epochs` 4 -> 6. More,
   smaller steps, aiming for paraphrase ~0.5 at a perplexity ratio under 1.5.

Not yet decided, and must not be decided by accident: the primary layer. That
comes from the probe sweep on the final `M_injected` and goes into the
preregistration before any unlearning run.

`checkpoints/M_injected` deliberately left empty. Everything downstream builds
on one checkpoint and picking it before the selection rule is settled is how a
project ends up with results it cannot defend.

---

**Tomorrow's first action.** Re-run injection with the gentler schedule and the
new perplexity probe:
`python scripts/03_inject.py --config configs/base.yaml`.
If no epoch clears the 1.5 ceiling, lower `lr` again rather than relaxing the
ceiling. If paraphrase accuracy falls below ~0.4, add training templates to
`CITY_TEMPLATES_TRAIN` rather than training longer.
