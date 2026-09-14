# Day 05 --- The causal experiments, and the audit closes at rung 1

Date: 2026-09-15   Hours: ~5   Machine: local (Windows, CPU only)

**Headline task.** Run every remaining rung of the claim ladder -- patching,
logit lens, steering, recovery attack -- and determine the highest claim the
evidence supports.

**Done-when test.** Each experiment's controls pass, and the ladder walk stops
at a defensible rung.  **PASS.** Every control behaved. The ladder stops at
rung 1, and each refusal above it comes from a control rather than from a weak
measurement.

---

**What I actually did.**

Cross-model activation patching on all four models with random-position and
retain-set controls, causal tracing on `M_npo_s0`, logit lens across all five
models on forget / retain / control sets, steering with entity-clustered
bootstrap intervals, and the disjoint fine-tune recovery attack on all four
models with both its controls.

---

**Numbers I got.**

*Patching (donor = M_injected), peak normalised recovery on forget facts.*

| model | forget peak | layer | random-position control | RETAIN peak |
|:--|--:|--:|--:|--:|
| M_npo_s0 | 0.166 | 11 | 0.051 | **0.171** |
| M_npo_s1 | 0.166 | 11 | 0.041 | -- |
| M_npo_s2 | 0.234 | 10 | 0.142 | -- |
| M_gd_s2 | 0.134 | 11 | 0.037 | -- |

The retain arm is the number that matters. Patching recovers 0.171 on retain
facts, which were never unlearned and therefore have no gap to close. Forget
recovery of 0.166 is therefore not a weak recovery of the forgotten fact; it is
the baseline effect of substituting a residual state from a slightly different
model. **Real recovery is approximately zero.**

*Logit lens, peak layer and peak-to-final drop (nats).*

| model | forget | retain | **control (never taught)** |
|:--|:--|:--|:--|
| M_injected | L11, +0.00 monotone | L10, +0.00 monotone | L6, **+6.36 rise_then_fall** |
| M_npo_s0 | L7, +18.82 rise_then_fall | L11, +0.00 monotone | L7, **+16.76 rise_then_fall** |
| M_npo_s1 | L7, +25.53 rise_then_fall | L11, +0.00 monotone | L7, **+21.36 rise_then_fall** |
| M_npo_s2 | L7, +17.99 rise_then_fall | L11, +0.00 monotone | L7, **+18.48 rise_then_fall** |
| M_gd_s2 | L7, +17.40 rise_then_fall | L11, +0.00 monotone | L6, **+20.22 rise_then_fall** |

*Steering, M_npo_s0, layer 9, direction from the UNLEARNED model's own probe.*

| alpha | probe | random | difference | 95% CI |
|--:|--:|--:|--:|:--|
| -2.0 | -19.875 | -19.751 | -0.124 | [-0.205, -0.054] |
| -1.0 | -19.814 | -19.751 | -0.062 | [-0.103, -0.027] |
| +1.0 | -19.688 | -19.751 | +0.062 | [+0.027, +0.103] |
| +2.0 | -19.625 | -19.750 | +0.125 | [+0.054, +0.207] |

Significant at 3/3 positive alphas. Largest effect = **0.63%** of the 19.75
logit-difference gap to M_injected. Finite-difference error <= 0.001 everywhere.

*Recovery attack (fine-tune on 24 forget entities, evaluate on 24 disjoint).*

| model | attack held-out | control held-out | difference | attack learned its training half |
|:--|--:|--:|--:|--:|
| M_npo_s0 | 0.181 | 0.201 | **-0.021** | +0.319 |
| M_npo_s1 | 0.132 | 0.174 | **-0.042** | +0.264 |
| M_npo_s2 | 0.208 | 0.181 | **+0.028** | +0.306 |
| M_gd_s2 | 0.181 | 0.208 | **-0.028** | +0.243 |

Margin for rung 5 was 0.15. Two of four differences are negative.

---

**What this means for the hypotheses.**

**The ladder walk, with the reason each rung is refused.**

- **Rung 1, output suppression: SUPPORTED.** Forget accuracy at control level,
  generalising to held-out paraphrases, entity-specific, utility and related
  attribute preserved.
- **Rung 2, reduced decodability: REFUSED.** Probe accuracy at layer 9 fell
  0.897 -> 0.15--0.25, at or below the LEACE erasure reference (0.083). But
  forget-set activations moved only 1.18--1.25x more than never-taught controls
  against a preregistered threshold of 1.5. The collapse is not attributable to
  the forgotten facts.
- **Rung 3, altered machinery: REFUSED.** Patching recovers 0.166 on forget and
  0.171 on RETAIN facts that were never unlearned. There is no differential
  effect to explain.
- **Rung 4, accessible contents: REFUSED.** Steering is statistically clean and
  substantively negligible: 0.63% of the gap, entirely inside the linear
  regime.
- **Rung 5, weight-level retention: REFUSED.** Fine-tuning on half the forget
  facts transferred nothing to the other half, while demonstrably relearning
  the half it saw (+0.24 to +0.32).

**The logit-lens result is the day's finding, and it is a third instance of the
project's pattern.** Forget facts show peak-then-drop of 17--25 nats, the
canonical "the model still knows but is suppressing it" signature. Control
entities -- people the model was NEVER TAUGHT -- show the same shape with drops
of 17--21 nats. There is no fact there to suppress. The signature is a property
of how unlearning reshaped late-layer processing of entity-attribute prompts in
general, not evidence of hidden knowledge. Note also that M_injected already
shows it on control entities (+6.36) before any unlearning.

**Steering is a fourth instance, of a slightly different kind.** The test
passed every significance criterion -- CI excluding zero at all three positive
alphas, monotone, sign-symmetric, finite-difference agreement to 0.001 -- while
moving the model 0.63% of the way to answering correctly. Significance without
a specified effect size measures local gradient geometry and reports it as
accessible content.

**Assembled, the project now has four independent demonstrations of one
methodological claim:** standard unlearning diagnostics produce results that
look meaningful and are not, because the comparison they need is missing.
Baseline decodability gap; drift confound; suppression signature on
never-taught entities; significance without effect size.

---

**Decisions made, and whether they were preregistered.**

Post-freeze deviations, all in `preregistration.md` Section 6.

1. **Effect-size floor on the rung-4 criterion.** Steering must recover >= 25%
   of the logit-difference gap in addition to clearing significance. Added
   AFTER the first run returned PASS on an effect worth 0.63% of the gap. This
   change converts a PASS into a FAIL -- it costs the project a positive
   result, which is the direction that makes the disclosure credible.
2. **Recovery-attack control criterion changed reference.** It asked whether
   the control arm stayed near NOMINAL chance (0.083); the model's own starting
   accuracy on the held-out half was 0.174, so a control arm that moved +0.028
   was flagged FAIL for not being near a number it was never at. Now compares
   the control arm against its OWN before-value with a 0.10 tolerance.
3. **Added a training-half check to the recovery attack.** Without it, a flat
   held-out result is uninterpretable: it cannot distinguish "the fine-tune
   worked and transferred nothing" from "the fine-tune did nothing". All four
   models pass it, which is what makes the rung-5 negative strong.
4. **Steering alpha range reduced to +/-2.** At alpha=4 both probe and random
   directions collapse; the comparison there measures how broken the model is.
5. **Causal tracing partner selected by matching token count.** Positional
   patching requires equal-length prompts.

---

**Still outstanding before writing.**

- Hand-label the 50 saved forget-set completions. The refusal detector reports
  0.000 on every model and its agreement rate is unknown, so the metric cannot
  be cited yet.
- Statistics pass: Holm correction across the 12-layer probe family, permutation
  tests on pre/post comparisons.
- Run the residual-memory auditor so `highest_supported_rung` fixes the claim
  mechanically rather than by my judgement.
- Clean-clone reproducibility check.

---

**Tomorrow's first action.** The statistics pass, then the auditor. The claim
should be produced by code applying preregistered thresholds, not by me reading
tables and deciding.
