# Preregistration

**Frozen at commit `2087012`, 2026-09-12**, before any unlearning run.

Everything in Sections 0--4 was fixed before the data existed. Section 5 lists
amendments made before the freeze; Section 6 lists every deviation made after
it, with a date and a reason. The deviation log is long on purpose. An honest
log is worth more to a reader than a short one that is not true.

Model: GPT-2 Small (124M). Condition: SYN, fine-tuning-injected fictitious
researcher profiles. `M_injected` = `checkpoints/M_injected`, from run
`20260912-161248_run_be5cdc1f`, epoch 4.

---

## 0. Baseline values, measured before any unlearning

Recorded here so a reader can check that the thresholds below were reachable
when they were set, rather than fitted afterwards.

| quantity | value | 95% CI | source |
|:--|--:|:--|:--|
| behavioural forget (constrained) | 0.375 | [0.281, 0.476] | `behaviour_M_injected.csv` |
| behavioural retain | 0.328 | [0.253, 0.408] | same |
| behavioural control entities | 0.118 | [0.049, 0.198] | same |
| behavioural paraphrase (forget entities) | 0.391 | [0.297, 0.484] | same |
| behavioural related attribute | 0.407 | [0.343, 0.475] | same |
| generic perplexity, `M_injected` | 109.5 | -- | `injection_history.json` |
| generic perplexity, base GPT-2 | 78.2 | -- | 48-sentence probe |
| probe accuracy, layer 9 | 0.897 | -- | `probe_logistic_forget.csv` |
| probe accuracy, control entities (max over layers) | 0.111 | -- | same |
| chance, city (12 classes) | 0.0833 | -- | `label_map.json` |
| chance, field (6 classes) | 0.1667 | -- | same |

**A finding that already exists in the baseline, and is reported as such.** At
layer 9 the forget facts are linearly decodable at 0.897 while the model
expresses them behaviourally at 0.375. The decodability/expression gap is
present BEFORE any unlearning. Any post-unlearning gap is interpreted against
this baseline, not against an implicit assumption that the two start aligned.

---

## 1. Hypotheses

| ID | Statement | Primary metric | Decision rule |
|:--|:------|:-----|:-----|
| H1 | Forget-fact information remains linearly decodable from intermediate representations after behavioural unlearning | probe accuracy at layer 9 | accuracy CI excludes control-entity accuracy (0.111) |
| H2 | Decodability varies by layer, with the rise concentrated in layers 8--10 | layer profile | post-unlearning peak layer differs from layer 0 and from layer 11 |
| H3 | Nonlinear probes do not recover substantially more than linear probes at matched capacity | selectivity difference | CI on the difference includes 0 |
| H4 | Logit-lens trajectories show internal peaks followed by late suppression | peak-to-final drop | drop > 0.5 nats in the majority of forget facts |
| H5 | SAE latents selected for forget facts change more than matched random latents | mean activation change | empirical p < 0.05 vs 100 random draws |
| H6 | Ablating selected latents reduces target probability more than random latents | logit difference | CI on the difference excludes 0 |
| H7 | Donor states from `M_injected` restore forget behaviour in the unlearned model | normalised recovery | recovery > 0.5 at some layer, controls near 0 |
| H8 | A direction estimated from the unlearned model alone can restore the target | logit difference vs matched random | CI on the difference excludes 0 |

H1 is the primary hypothesis and is reported uncorrected. H2--H8 form the
secondary family and carry Holm--Bonferroni correction.

**H5 and H6 were not tested.** SAE analysis was cut for compute; see Section 7.
They are reported as untested rather than as null.

---

## 2. Frozen choices

**Primary layer: 9.** Secondary layer: 8.

Selection rule, amended on 2026-09-12 *before the layer profile was inspected*:
the peak-accuracy layer among layers 0--10, excluding layer 11. Two reasons,
both recorded before the choice was made. Layer 11 is the final residual state,
which is what the unembedding reads, so decoding from it approximates reading
the model's own output. And at 0.964 it had no headroom to fall, so a
post-unlearning drop there could not be distinguished from saturation.

**Primary metric.** Behavioural: constrained accuracy. Internal: probe accuracy
at layer 9.

**Probe.** Multinomial logistic regression; C by 3-fold CV inside the training
folds; scaler fitted inside the pipeline on training folds only; entity-disjoint
stratified 70/30 split. Control suite mandatory on every run: never-taught
control entities, label-permuted control task, and the LEACE erasure control.

**Matched-forgetting band.** Relative to `M_injected`, not absolute, because
`M_injected`'s own retain accuracy is 0.328 and an absolute floor above that
would be unreachable by construction.

- forget constrained accuracy <= 0.15 (at or below control level, 0.118)
- retain constrained accuracy >= 0.80 x baseline = 0.262
- generic perplexity ratio <= 1.20 relative to `M_injected` (109.5 -> 131.4)
- checkpoint selection within the band: earliest qualifying step

**Drift gate.** Forget-set activation drift must exceed never-taught control
drift by >= 1.5x before any probe change may be described as erasure.

**Statistics.** Entity-clustered bootstrap, 10,000 draws. Paired sign-flip
permutation tests clustered on entities. Holm--Bonferroni across layers for the
secondary family. The primary test reported uncorrected and identified as
primary. No bare point estimates.

---

## 3. What would falsify the headline claim

> If, after behavioural unlearning brings forget accuracy to control level,
> probe accuracy at layer 9 falls to the control-entity level (~0.11) with a
> confidence interval excluding the `M_injected` value, while the LEACE control
> and the retain-set probes confirm the pipeline is still working, then the
> information was removed from the representation and the claim that
> behavioural unlearning leaves knowledge intact is false for this setting.

**Outcome: the antecedent occurred.** Probe accuracy at layer 9 fell to
0.154--0.228 with the LEACE control at 0.083 and retain probes intact. The
erasure conclusion is nonetheless NOT drawn, because the drift gate in Section 2
was not cleared: forget drift exceeded control drift by only 1.18--1.25x against
a required 1.5x. The falsification criterion was written before the drift gate
was added on the same day, and where they conflict the gate governs, because a
change that is not forget-specific cannot establish either claim.

---

## 4. Planned exclusions

| run_id | reason | date |
|:--|:--|:--|
| all stage-1 and stage-2 sweep runs | exploratory search for the viable hyperparameter region; excluded from final analysis by design | 2026-09-13/14 |
| gradiff seeds 0 and 1, all learning rates | never entered the matched-forgetting band; blocked by perplexity blowout and retain collapse | 2026-09-14 |

---

## 5. Amendments made BEFORE the freeze

Part of the design rather than deviations from it.

| date | change | reason |
|:--|:------|:------|
| 2026-09-12 | `n_fields` 12 -> 6 | only 10 field candidates are single-token in GPT-2 BPE; 6 is the largest divisor of both 48 and 60 among them. Chance on the locality set becomes 1/6. |
| 2026-09-12 | epoch selection gains a perplexity ceiling (ratio <= 1.5) | the original rule, max paraphrase accuracy, has no opinion about cost; run 1 bought +0.033 paraphrase for a 1.76x perplexity rise. Threshold fixed before re-running. |
| 2026-09-12 | `GENERIC_PROMPTS` 8 fragments -> 48 complete sentences | perplexity is computed over whole strings; 8 fragments was too thin to act on. Only ratios are comparable across the two probes. |
| 2026-09-12 | injection `lr` 5e-5 -> 2e-5, `epochs` 4 -> 6 | comparable generalisation at lower collateral damage. Final: paraphrase 0.392 at ratio 1.40. |
| 2026-09-12 | band restated relative to `M_injected` | the original absolute `retain >= 0.70` was unreachable given a baseline of 0.328. |
| 2026-09-12 | primary-layer rule excludes layer 11 | output-adjacent and near ceiling; see Section 2. Amended before the profile was inspected. |

---

## 6. Deviations AFTER the freeze

| date | change | reason | effect on claims |
|:--|:------|:------|:------|
| 2026-09-13 | `eval_every` 10 -> 20, later 40 | disk: 41 checkpoints/run at ~500MB would not fit in 14GB free | none; trajectory resolution only |
| 2026-09-13 | `--eval-subset` 120 -> 360 | with ~20 entities the SE on retain is ~0.1 while the retain floor sits 0.066 below baseline; band membership was partly noise | none; measurement quality |
| 2026-09-13 | NPO searched in stages, stage 1 excluded from analysis | 27 runs at ~13 min each does not fit the compute budget | stage 1 reported as exploratory |
| 2026-09-14 | `steps` 200 -> 400 -> 800, `lr` 2e-5 -> 1e-5 | at 2e-5 the perplexity ceiling was crossed BEFORE the forget threshold; the failure was ordering, not forget depth | no threshold changed |
| 2026-09-14 | gradiff `lambda_retain` 1.0 -> 2.0 | `retain_too_low` blocked 17--18 of 21 checkpoints in every 2e-5 run | none; still 1/3 seeds in band, reported |
| 2026-09-14 | added a `generic` activation set | needed as a second drift reference | adds a control |
| 2026-09-15 | **effect-size floor on the rung-4 criterion** (>= 25% of the gap) | the significance test alone returned PASS on an effect worth 0.63% of the gap to `M_injected` | **converts a PASS into a FAIL**; costs the project a positive result |
| 2026-09-15 | recovery-attack control criterion compares the control arm against its OWN before-value rather than nominal chance | the model's starting accuracy on the held-out half was 0.174, so a control arm at 0.201 was flagged FAIL for not being near a number it was never at | makes rung 5 interpretable |
| 2026-09-15 | added a training-half check to the recovery attack | a flat held-out result cannot otherwise distinguish "the fine-tune worked and transferred nothing" from "the fine-tune did nothing" | strengthens the rung-5 negative |
| 2026-09-15 | steering alpha range reduced to +/-2 | at alpha=4 both probe and random directions collapse; the comparison measures how broken the model is | none |
| 2026-09-15 | **probe seeds 5 -> 20** | a paired sign-flip test across N clusters has a floor of 2^-N; with 5 seeds that is 0.031 and every model returned 0.0594 regardless of effect size. The test was bounded by the cluster count, not the data. | layer-9 accuracy moved < 0.05; p moved 0.0594 -> 0.0001. The effect was unchanged; only the resolution of the test changed. |
| 2026-09-15 | `semantic_recall` dropped from the audit dimensions | never designed or measured; carried a permanent blank that stopped the ladder walk with "not measured" rather than a substantive reason | none |
| 2026-09-15 | claim-ladder code moved out of the HERMES scaffolding | the ladder belongs to this project; HERMES is parked and must not be a dependency | none |

**A threshold that was NOT relaxed.** The drift gate required forget/control
>= 1.5. The measured ratios were 1.18--1.25. Lowering it to 1.15 would have
admitted all four models and produced a headline erasure result. It was applied
as written.

---

## 7. Not tested, and why

| item | status | reason |
|:--|:--|:--|
| H5, H6 (SAE feature analysis) | not tested | SAE validity on the fine-tuned models would have to be established first; cut for compute |
| Condition NAT (natural pretrained knowledge) | builder written, not run | second condition; the first one's result is complete without it |
| Nonlinear probes at matched capacity (H3) | not tested | cut for compute |
| Scale check on a larger model | not attempted | CPU-only budget |

These are reported as untested. None is reported as null.
