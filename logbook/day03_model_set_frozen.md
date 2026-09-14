# Day 03 --- Model set frozen, and the behavioural case closed

Date: 2026-09-13/14   Hours: ~9 (mostly unattended compute)   Machine: local (Windows, CPU only)

**Headline task.** Get three seeds per method into the matched-forgetting band,
freeze the model set, and complete Table B1.

**Done-when test.** 3/3 seeds in band per method.
**PARTIAL PASS:** NPO 3/3, gradient difference 1/3. The asymmetry is reported,
not hidden.

---

**What I actually did.**

Ran stage 3 of the staged search at `lr = 1e-5` for 800 steps: NPO at
`beta = 0.1`, gradient difference with `lambda_retain` doubled to 2.0. Promoted
the four band-selected checkpoints, ran full behavioural evaluation on each,
and split the paraphrase set by entity type.

---

**What broke, and why.**

Nothing broke. But the stage-2 failure is worth recording properly because the
diagnosis is what made stage 3 work.

At `lr = 2e-5`, NPO seeds 1 and 2 both reached the forget floor (0.083) and
still missed the band. The blocking criterion was PERPLEXITY, not forget depth:
seed 1 was already at ratio 1.256 by step 100 with forget still at 0.198. The
damage was arriving ahead of the forgetting. Halving the learning rate and
doubling the step budget reversed the order -- at 1e-5 every NPO run crossed
the forget threshold while perplexity was still around 1.05--1.12, and only
later drifted past the ceiling.

**No threshold was changed to achieve this.** The obvious wrong move was to
raise `ppl_ratio_max` to 1.5, which would have admitted the stage-2 runs and
made the band meaningless.

---

**Numbers I got.**

*Stage 3 sweeps (lr 1e-5, 800 steps, eval_every 40, eval-subset 360).*

| method | seed | in band | step | forget | retain | ppl ratio |
|:--|:--|:--|--:|--:|--:|--:|
| NPO | 0 | yes | 240 | 0.135 | 0.364 | 1.109 |
| NPO | 1 | yes | 320 | 0.125 | 0.369 | 1.123 |
| NPO | 2 | yes | 240 | 0.149 | 0.353 | 1.057 |
| GD | 0 | no | -- | -- | -- | blocked: retain 14/21, ppl 15/21 |
| GD | 1 | no | -- | -- | -- | blocked: ppl 13/21 |
| GD | 2 | yes | 240 | 0.135 | 0.303 | 1.068 |

Frozen model set: `M_npo_s0`, `M_npo_s1`, `M_npo_s2`, `M_gd_s2`.
Recorded in `configs/model_set.json` with the step and metrics that selected
each. Promotion is done by `scripts/16_promote_checkpoints.py`, which reads the
selection JSONs, so no path is ever typed by hand.

*Table B1 --- full behavioural evaluation, constrained accuracy with
entity-clustered 95% CIs.*

| set | M_injected | M_npo_s0 | M_npo_s1 | M_npo_s2 | M_gd_s2 |
|:--|:--|:--|:--|:--|:--|
| forget | 0.375 [.281,.476] | 0.135 [.062,.219] | 0.125 [.049,.215] | 0.149 [.073,.236] | 0.135 [.059,.222] |
| retain | 0.328 [.253,.408] | 0.364 [.289,.442] | 0.369 [.286,.456] | 0.353 [.275,.436] | 0.303 [.233,.378] |
| control | 0.118 [.049,.198] | 0.090 [.035,.160] | 0.080 [.024,.146] | 0.087 [.028,.160] | 0.097 [.035,.170] |
| related | 0.407 [.343,.475] | 0.349 [.284,.414] | 0.333 [.269,.401] | 0.395 [.330,.460] | 0.407 [.346,.469] |

*Paraphrase, split by entity type --- the generalisation test.*

| model | paraphrase[forget] | paraphrase[retain] |
|:--|:--|:--|
| M_injected | 0.391 [.297,.484] | 0.329 [.258,.400] |
| M_npo_s0 | 0.130 [.057,.214] | 0.333 [.267,.404] |
| M_npo_s1 | 0.125 [.047,.214] | 0.362 [.296,.433] |
| M_npo_s2 | 0.141 [.062,.229] | 0.333 [.263,.412] |
| M_gd_s2 | 0.146 [.068,.234] | 0.329 [.254,.408] |

Refusal rate 0.000 on every model (keyword detector; needs hand validation).

---

**What this means for the hypotheses.**

The behavioural case is now closed on four fronts, and each one removes a
deflationary reading of whatever the probes show next.

1. **Forgetting reached control level.** Forget accuracy 0.125--0.149 against
   control 0.080--0.097, intervals overlapping heavily. The models are
   behaviourally indistinguishable from never having been taught these facts.
2. **Utility preserved.** Retain accuracy 0.303--0.369 against a baseline of
   0.328 -- unmoved.
3. **The intervention was targeted, not blunt.** The related attribute (each
   researcher's field) held at 0.333--0.407 against a 0.407 baseline, with
   median rank 2--3. Unlearning removed the city and left the field.
4. **Forgetting generalised beyond the trained phrasings.** This is the
   important one. Paraphrase accuracy on FORGET entities fell 0.391 -> 0.125
   to 0.146, landing at control level, while paraphrase on RETAIN entities
   held at 0.329--0.362 against a 0.329 baseline.

Point 4 forecloses the strongest objection available to a reviewer: that
unlearning merely removed six training templates. The effect transfers to
phrasings never seen during unlearning, and it is entity-specific rather than
a blanket suppression of city answers.

So: behaviourally, these models look like they never learned the facts. The
probe now decides whether the internals agree.

**A confound that must be handled before probing.** Control-entity MEDIAN RANK
moved from 270 on `M_injected` to 2934--5008 on the unlearned models, while
control ACCURACY barely changed. Those entities were never taught and never
unlearned, so nothing about them should have moved. Something shifted how the
model ranks city tokens globally. Forget-set rank shows the same pattern
(22 -> 1781--3497), far more movement than the accuracy change alone implies.

Consequence: a probing difference found at layer 9 could reflect global drift
rather than forget-specific erasure. The `activation_drift` comparison --
drift on forget prompts versus drift on generic prompts -- is now a REQUIRED
analysis, not an optional one, and this paragraph is why.

---

**Decisions made, and whether they were preregistered.**

Post-freeze deviations, all logged in Section 6 of `preregistration.md`. None
changes a threshold.

1. Stage 3 search at `lr = 1e-5`, 800 steps, `eval_every = 40`. Reason: the
   stage-2 blocking criterion was perplexity arriving before forgetting, not
   insufficient forgetting. `eval_every` is a disk constraint (41 checkpoints
   at 20 would need ~20GB peak per run).
2. Gradient difference `lambda_retain` 1.0 -> 2.0, because `retain_too_low`
   blocked 17--18 of 21 checkpoints in every stage-2 run.
3. Stages 1 and 2 are excluded from final analysis and exist only to locate
   the viable region on a CPU budget. Stage 3 is the reported sweep.

**Reported asymmetry, not a quiet exclusion.** Across roughly 30 runs spanning
four learning rates and three step budgets, NPO reached matched forgetting on
every seed while gradient difference managed it once. Gradient difference fails
by perplexity blowout (ratios of 6--13 at 800 steps) and retain collapse, not
by failing to forget -- it always reaches the forget floor. At this scale,
gradient difference cannot reach matched forgetting without unacceptable
collateral damage. That is a finding about the method and it goes in the
paper; the single surviving GD checkpoint is reported as a single-seed arm with
that caveat attached.

---

**Tomorrow's first action.** The central measurement.

```
python scripts/05_extract_activations.py --checkpoint checkpoints\M_npo_s0 --label M_npo_s0 --verify
(... same for s1, s2, M_gd_s2 ...)
python scripts/06_probe_sweep.py --labels M_injected,M_npo_s0,M_npo_s1,M_npo_s2,M_gd_s2 --set forget
```

Gate before interpreting anything: control-entity probes must still be at
chance. Then layer 9, where `M_injected` reads 0.897 and control entities 0.111.
Near 0.897 means the information survived the erasure of the behaviour; near
0.111 means it did not; in between needs the localisation analysis. In all
three cases, run the drift comparison before attributing the result to
forget-specific erasure.
