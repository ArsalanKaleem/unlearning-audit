# Day 06 --- Statistics, detector validation, and the claim fixed by rule

Date: 2026-09-15   Hours: ~4 (2.5 of it unattended)   Machine: local (Windows, CPU only)

**Headline task.** Put an interval on every headline number, validate the
refusal detector, and let the claim ladder produce the conclusion rather than
producing it myself.

**Done-when test.** `final_statistics.csv` exists with no bare point estimates,
and the audit summary returns a rung reached by rule.  **PASS.**

---

**What I actually did.**

Raised the probe seed count, re-ran the probe sweep and the statistics pass,
labelled a blind 50-completion sample, moved the claim-ladder code out of the
HERMES scaffolding into `src/eval/claim_ladder.py`, and ran the mechanical
verdict over all four models.

---

**What broke, and why.**

**The primary probe test was bounded by its own design.** Every model returned
p = 0.0594 at layer 9 -- the same value, regardless of an effect size of 0.65.
A paired sign-flip permutation test across N clusters has a minimum attainable
p of 2^-N; with 5 probe seeds that floor is 0.031, and 0.0594 is the next
attainable value up. The test could not resolve the effect no matter how large
it was.

Raising the seeds to 20 moved layer-9 accuracy by less than 0.05 in every model
(0.247 -> 0.228, 0.150 -> 0.154, 0.167 -> 0.201, 0.172 -> 0.224) while p went
from 0.0594 to 0.0001 and 4--7 layers per model began surviving Holm. The
effect was always there; only the resolution of the test changed. That
ordering -- effect unchanged, p-value transformed -- is what makes this a
legitimate fix rather than a fishing expedition, and it is why the numbers
before and after are both recorded here.

**A yaml round-trip destroyed every comment in `configs/base.yaml`.**
`yaml.safe_dump` preserves values and discards comments, which in this project
means it discarded the record of why layer 11 was excluded, why the band is
relative, and what the frozen thresholds mean. Values were unaffected.
Restored by hand. Config files that carry preregistration reasoning must be
edited by hand.

---

**Numbers I got.**

*Refusal detector validation.* 50 completions, ten per model, drawn at a fixed
stride through the row index and shuffled, with the detector's verdict hidden
during labelling. Author labels: **0 refusals**, 4 hedged (all four on
`M_npo_s0`, all region-scale answers naming no city). Detector: 0 refusals.
**Agreement 100%**, TP 0 / FP 0 / FN 0 / TN 50.

So the 0.000 refusal rate is correct rather than broken, and the accompanying
finding stands: unlearning here produced collapsed generation, not expressed
uncertainty. Free-generation diagnostics across all 250 completions:

| model | names any city | says the target city | degenerate repetition |
|:--|--:|--:|--:|
| M_injected | 1.00 | 1.00 | 0.30 |
| M_npo_s0 | 0.60 | 0.10 | 0.30 |
| M_npo_s1 | 0.64 | 0.12 | 0.34 |
| M_npo_s2 | 0.80 | 0.16 | 0.36 |
| M_gd_s2 | 0.88 | 0.24 | 0.46 |

The drop from 1.00 to 0.10--0.24 on saying the target confirms rung 1 from free
generation, independently of the constrained scoring.

*Probing, after the seed fix.* Layer 9, 20 seeds:

| model | accuracy | seed sd | p (uncorrected, primary) | layers surviving Holm |
|:--|--:|--:|--:|--:|
| M_npo_s0 | 0.228 | 0.109 | 0.0001 | 5/12 |
| M_npo_s1 | 0.154 | 0.083 | 0.0001 | 4/12 |
| M_npo_s2 | 0.201 | 0.092 | 0.0001 | 6/12 |
| M_gd_s2 | 0.224 | 0.092 | 0.0001 | 7/12 |

*Forget minus never-taught control, entity-clustered bootstrap.* M_injected
+0.257 [+0.135, +0.378]; all four unlearned models straddle zero (+0.038 to
+0.062, every interval containing 0). "Forgetting reached control level" is now
a statistical statement rather than an eyeball comparison.

*The mechanical verdict.* Normalised positions (0 = never taught, 1 = fully
taught), at the preregistered threshold of 0.25:

| dimension | M_npo_s0 | M_npo_s1 | M_npo_s2 | M_gd_s2 |
|:--|--:|--:|--:|--:|
| direct_recall | 0.068 | 0.027 | 0.122 | 0.068 |
| paraphrase_recall | 0.045 | 0.025 | 0.083 | 0.102 |
| representation_probe | 0.159 | 0.059 | 0.122 | 0.154 |
| causal_recovery | 0.166 | 0.166 | 0.234 | 0.134 |
| steering_recovery | 0.006 | -- | -- | -- |
| relearning_recovery | 0.243 | 0.054 | 0.351 | 0.243 |

**All four models: rung 1, output suppression only.** Every dimension sits
closer to never-taught than to fully-taught -- on the raw numbers this looks
like near-total removal -- and the drift control is the only thing standing
between those numbers and an erasure claim. It is the right thing to be
standing there.

---

**Decisions made, and whether they were preregistered.**

Post-freeze deviations, logged in `preregistration.md` Section 6.

1. **Probe seeds 5 -> 20.** Because the sign-flip test was bounded below by the
   cluster count and returned an identical p-value for every model irrespective
   of effect size. NOT because the p-value was unwelcome: the accuracies barely
   moved, which is the evidence that distinguishes the two motives.
2. **`semantic_recall` dropped from the audit dimensions.** It was never
   designed or measured; carrying it produced a permanent blank that stopped
   the ladder walk with "not measured" rather than a substantive reason.
3. **Claim-ladder code moved** from the HERMES scaffolding to
   `src/eval/claim_ladder.py`. The ladder belongs to this project; HERMES is
   parked and must not be a dependency of the audit.
4. **`blocked_by` now records the drift cap.** Previously, when the walk
   cleared every measured rung before the drift control capped it, the blocking
   reason came out blank -- a table cell with nothing in it exactly where the
   explanation belongs.

---

**Still outstanding.**

- Steering on `M_npo_s1`, `M_npo_s2`, `M_gd_s2` (~45 min). A rung-4 result from
  one of four models is weak; the effect-size floor makes it cheap to state
  properly.
- Clean-clone reproducibility check.
- Five paper figures, regenerated with consistent styling.
- Writing.

---

**Tomorrow's first action.** Run steering on the remaining three models, re-run
the audit summary so every model has a rung-4 measurement, then the
reproducibility check.
