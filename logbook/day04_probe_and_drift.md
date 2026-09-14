# Day 04 --- The central measurement, and why it does not say what it looks like

Date: 2026-09-14   Hours: ~4   Machine: local (Windows, CPU only)

**Headline task.** Probe the unlearned models at every layer, then run the
mandatory drift control before interpreting anything.

**Done-when test.** Control-entity probes at chance (PASS), and a verdict on
whether the probe result is forget-specific.
**Result: the drift control FAILED the claim, 0/4 models. Capped at rung 1.**

---

**What I actually did.**

Extracted all-layer activations for the four unlearned models, ran the
layer-wise probe sweep with the full control suite, added a generic-prompt
activation set, and ran the forget-vs-control-vs-retain-vs-generic drift
comparison at the preregistered primary layer.

---

**Numbers I got.**

*Probe accuracy, forget set, mean of 5 seeds.*

| layer | M_injected | M_npo_s0 | M_npo_s1 | M_npo_s2 | M_gd_s2 |
|--:|--:|--:|--:|--:|--:|
| 6 | 0.200 | 0.178 | 0.233 | 0.153 | 0.139 |
| 7 | 0.183 | 0.100 | 0.128 | 0.081 | 0.100 |
| 8 | 0.608 | 0.214 | 0.183 | 0.108 | 0.128 |
| **9** | **0.897** | **0.247** | **0.150** | **0.167** | **0.172** |
| 10 | 0.914 | 0.297 | 0.150 | 0.272 | 0.214 |
| 11 | 0.964 | 0.256 | 0.142 | 0.264 | 0.233 |

At layer 9, with seed SD: M_injected 0.897 (0.047), M_npo_s0 0.247 (0.073),
M_npo_s1 0.150 (0.070), M_npo_s2 0.167 (0.102), M_gd_s2 0.172 (0.114).
Selectivity fell from 0.817 to 0.056--0.192. Chance is 0.083; control-entity
probes peaked at 0.108 across all models (gate PASSED).

Taken alone this is the preregistered falsification condition being met:
probe accuracy at layer 9 falls to control level with the interval excluding
the M_injected value.

*Relative activation drift from M_injected, layer 9.*

| model | forget | control | retain | generic | forget/control |
|:--|--:|--:|--:|--:|--:|
| M_npo_s0 | 0.490 | 0.400 | 0.340 | 0.114 | **1.22x** |
| M_npo_s1 | 0.552 | 0.441 | 0.359 | 0.136 | **1.25x** |
| M_npo_s2 | 0.513 | 0.433 | 0.367 | 0.103 | **1.18x** |
| M_gd_s2 | 0.528 | 0.444 | 0.387 | 0.108 | **1.19x** |

Preregistered threshold: forget drift must exceed control drift by 1.5x.
**0/4 models clear it.**

---

**What this means for the hypotheses.**

**H1 is not decided, and the probe result must not be reported as erasure.**
Control-entity prompts describe people the model was never taught about and
that no unlearning objective ever touched. Their activations moved 1.18--1.25x
less than the forget set -- which is to say, almost as much. Whatever displaced
the forget representation displaced theirs too. A probe-accuracy collapse from
0.897 to 0.15 is therefore compatible with a broadly displaced representation
and does not establish that the forgotten facts were erased.

This is the second instrument to say so. Day 03 found control-entity median
rank moving from 270 to 2934--5008 while control accuracy barely changed. The
drift measurement is independent of that and agrees with it.

**The structure of the drift is itself a finding.** All four ENTITY sets moved
a lot (0.34--0.55) while generic text moved very little (0.10--0.14) -- a
4--5x gap. The damage is not global in the naive sense; ordinary language
modelling is largely intact. What moved is the entity-fact representation AS A
CLASS. Unlearning 48 birth cities disrupted how the model represents
"person -> attribute" prompts in general, including for people it was never
taught.

**The ordering forget > control > retain is informative.** Retain drifted
least (0.34--0.39) because the retain term actively pins it. Control entities
had nothing pinning them and nothing targeting them, and drifted more than
retain. So the retain objective protects the facts you NAME, not the
representational neighbourhood they occupy. That is a concrete statement about
what retain-preserving unlearning does and does not buy.

**Claim ladder: rung 1, output suppression.** Supported: forget accuracy at
control level, generalising to held-out paraphrases, entity-specific,
utility preserved, related attribute preserved. NOT supported: any statement
about the representation.

**A small vindication of a blind choice.** Layer 11 retains the most residual
signal (0.142--0.264) while layers 8--10 flatten. Layer 11 was excluded from
the primary-layer rule on 2026-09-12, before the profile was inspected, on the
grounds that it is the final residual state the unembedding reads and
therefore output-adjacent. The residual signal sits exactly where the rule
predicted it would be least meaningful.

---

**Decisions made, and whether they were preregistered.**

1. The drift threshold (forget/control > 1.5x) was set before the measurement
   and is applied as written. It was not relaxed after seeing 1.18--1.25.
   Relaxing it to 1.15 would have "passed" all four models and produced a
   headline erasure result. That is precisely the move the preregistration
   exists to prevent, and it is recorded here so that a reader can see the
   temptation was present and declined.
2. Added a `generic` activation set to `scripts/05_extract_activations.py`
   (48 sentences, no labels, drift reference only). Post-freeze, adds a
   control, changes no threshold.
3. `scripts/18_drift_analysis.py` added. New analysis, no new thresholds.

**Caveat to carry into the paper.** Generic prompts are complete sentences
while dataset prompts are mid-sentence stems, so the final-token position means
something structurally different in each and absolute drift magnitudes are not
comparable across the two. The comparison that carries the argument is forget
vs CONTROL, which share prompt structure exactly and differ only in whether
the entity was taught. Generic is a secondary reference for "did the whole
model move".

---

**What the paper is now about.**

The original target -- "unlearning only suppresses" -- was already published
by others at larger scale while this was being built. Today's result reframes
the contribution around two methodological findings, both scale-independent
and neither scoopable by more compute:

1. **The baseline gap.** Before any unlearning, probe decodability at layer 9
   was 0.897 while behavioural accuracy was 0.375. Decodability and behaviour
   were never aligned, so "still decodable after unlearning" is a weaker
   inference than the literature treats it as.
2. **The drift confound.** A probe collapse from 0.897 to 0.15 that looks
   exactly like erasure and is not, because the matched never-taught control
   moved almost as far. Probe-based unlearning evaluation requires a
   never-taught control set, and published work largely does not report one.

Finding 2 is arguably the stronger paper. "Here is how this measurement fools
you, with numbers" is more useful to the field than another demonstration of
suppression.

---

**Tomorrow's first action.** The causal experiments, which are now the only
way to move above rung 1.

```
python scripts/09_patching.py --donor checkpoints\M_injected --receiver checkpoints\M_npo_s0 --label M_npo_s0 --trace
python scripts/13_localisation.py --source M_injected --target M_npo_s0 --layer 9
```

Patching asks whether the unlearned model can still USE the fact when the
state is supplied -- a claim about machinery (rung 3) that the drift confound
does not undermine, because patching is an intervention rather than a
correlation. Localisation gives the four-way verdict (preserved / transformed /
obscured / removed) and can distinguish "the information moved" from "the
information went", which the probe alone cannot.
